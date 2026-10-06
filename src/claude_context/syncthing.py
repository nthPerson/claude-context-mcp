"""Small synchronous Syncthing REST client plus a cached file -> machine resolver."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

DEFAULT_EVENT_TYPES = ("ItemFinished", "LocalChangeDetected", "RemoteChangeDetected")
STATUS_KEYS = ("state", "needFiles", "errors", "globalFiles", "localFiles")
BACKOFF_SECONDS = 60.0


class SyncthingError(Exception):
    """Any failure talking to Syncthing (transport, HTTP status, bad JSON, bad config)."""


def read_api_key(config_xml: Path) -> str:
    """Read ``<gui><apikey>`` from a Syncthing config.xml."""
    try:
        key = ET.parse(config_xml).getroot().findtext("gui/apikey")
    except (OSError, ET.ParseError) as exc:
        raise SyncthingError(f"cannot read Syncthing config: {exc.__class__.__name__}") from exc
    if not key or not key.strip():
        raise SyncthingError("no API key in Syncthing config")
    return key.strip()


def _short(device_id: str) -> str:
    return device_id.split("-", 1)[0]


class SyncthingClient:
    """Thin wrapper over the Syncthing REST API; everything but ``ping`` raises SyncthingError."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.Client(
            base_url=self._api_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
            transport=transport,
        )
        self._names: dict[str, str] | None = None

    def __repr__(self) -> str:
        return f"SyncthingClient(api_url={self._api_url!r})"

    @classmethod
    def from_config_xml(cls, api_url: str, config_xml: Path, **kw: Any) -> SyncthingClient:
        """Build a client using the API key stored in ``config_xml``."""
        return cls(api_url, read_api_key(config_xml), **kw)

    def close(self) -> None:
        self._http.close()

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        allow_404: bool = False,
    ) -> Any:
        try:
            resp = self._http.get(path, params=params, timeout=timeout or self._timeout)
            if allow_404 and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise SyncthingError(f"GET {path}: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise SyncthingError(f"GET {path}: {exc.__class__.__name__}") from exc
        except ValueError as exc:
            raise SyncthingError(f"GET {path}: invalid JSON") from exc

    # -- identity / config ---------------------------------------------------

    def ping(self) -> bool:
        """True if the API answers ``/rest/system/ping``; never raises."""
        try:
            return self._get("/rest/system/ping").get("ping") == "pong"
        except (SyncthingError, AttributeError):
            return False

    def my_id(self) -> str:
        try:
            return str(self._get("/rest/system/status")["myID"])
        except (KeyError, TypeError) as exc:
            raise SyncthingError("status response lacks myID") from exc

    def devices(self) -> dict[str, str]:
        """``{full device id: name}`` for all configured devices (including this one)."""
        try:
            return {d["deviceID"]: d.get("name") or _short(d["deviceID"])
                    for d in self._get("/rest/config/devices")}
        except (KeyError, TypeError) as exc:
            raise SyncthingError("unexpected devices response") from exc

    def short_id_names(self, refresh: bool = False) -> dict[str, str]:
        """``{short id: name}``; cached until ``refresh=True``."""
        if self._names is None or refresh:
            self._names = {_short(i): n for i, n in self.devices().items()}
        return self._names

    def folders(self) -> list[dict]:
        data = self._get("/rest/config/folders")
        if not isinstance(data, list):
            raise SyncthingError("unexpected folders response")
        return data

    def folder_for_path(self, path: Path) -> tuple[str, Path] | None:
        """``(folder id, folder root)`` of the configured folder containing ``path``."""
        target = Path(path).resolve()
        best: tuple[str, Path] | None = None
        for f in self.folders():
            if not f.get("path"):
                continue
            root = Path(f["path"]).expanduser().resolve()
            if target.is_relative_to(root) and (best is None or len(root.parts) > len(best[1].parts)):
                best = (f["id"], root)
        return best

    # -- files / events ------------------------------------------------------

    def file_modified_by(self, folder: str, rel_path: str) -> str | None:
        """Short id of the device that last modified the file (global version), or None."""
        data = self._get(
            "/rest/db/file", {"folder": folder, "file": rel_path}, allow_404=True
        )
        if not data:
            return None
        try:
            return data["global"]["modifiedBy"] or None
        except (KeyError, TypeError):
            return None

    def events(
        self,
        since: int,
        *,
        types: Sequence[str] = DEFAULT_EVENT_TYPES,
        timeout: int = 60,
        limit: int | None = None,
    ) -> list[dict]:
        """Long-poll events with id > ``since``; returns [] if none arrive within ``timeout``."""
        params: dict[str, Any] = {"since": since, "events": ",".join(types), "timeout": timeout}
        if limit is not None:
            params["limit"] = limit
        data = self._get("/rest/events", params, timeout=timeout + self._timeout)
        if not isinstance(data, list):
            raise SyncthingError("unexpected events response")
        return data

    # -- health --------------------------------------------------------------

    def connections(self) -> dict[str, dict]:
        """Per remote device name: ``{"device_id", "connected", "last_seen"}``."""
        names = self.devices()
        conns = self._get("/rest/system/connections").get("connections", {})
        stats = self._get("/rest/stats/device")
        out: dict[str, dict] = {}
        for dev_id, info in conns.items():
            seen = (stats.get(dev_id) or {}).get("lastSeen")
            if not seen or seen.startswith(("0001", "1969", "1970")):  # never seen
                seen = None
            out[names.get(dev_id, _short(dev_id))] = {
                "device_id": dev_id,
                "connected": bool(info.get("connected")),
                "last_seen": seen,
            }
        return out

    def folder_completion(self, folder: str) -> dict[str, float]:
        """Completion percent per remote device name sharing ``folder``."""
        me, names = self.my_id(), self.devices()
        shared = next((f for f in self.folders() if f.get("id") == folder), None)
        if shared is None:
            raise SyncthingError(f"unknown folder {folder!r}")
        out: dict[str, float] = {}
        for d in shared.get("devices", []):
            dev_id = d["deviceID"]
            if dev_id == me:
                continue
            data = self._get("/rest/db/completion", {"folder": folder, "device": dev_id})
            out[names.get(dev_id, _short(dev_id))] = float(data.get("completion", 0.0))
        return out

    def folder_status(self, folder: str) -> dict:
        data = self._get("/rest/db/status", {"folder": folder})
        return {k: data[k] for k in STATUS_KEYS if k in data}


class MachineResolver:
    """Maps a file to the name of the machine that last modified it, with caching."""

    def __init__(
        self,
        client: SyncthingClient | None,
        folder: str | None,
        *,
        unknown: str = "unknown",
        max_cache: int = 50_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._folder = folder
        self._unknown = unknown
        self._max_cache = max_cache
        self._clock = clock
        self._cache: OrderedDict[tuple[str, int], str] = OrderedDict()
        self._retry_at = 0.0

    def machine_for(self, rel_path: str, mtime_ns: int) -> str:
        """Machine name for the file, or ``unknown``; never raises."""
        if self._client is None or not self._folder:
            return self._unknown
        key = (rel_path, mtime_ns)
        if key in self._cache:
            return self._cache[key]
        if self._clock() < self._retry_at:
            return self._unknown
        try:
            name = self._lookup(rel_path)
        except Exception:  # noqa: BLE001 - degrade to unknown, whatever went wrong
            self._retry_at = self._clock() + BACKOFF_SECONDS
            return self._unknown
        self._cache[key] = name
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return name

    def _lookup(self, rel_path: str) -> str:
        assert self._client is not None and self._folder
        short = self._client.file_modified_by(self._folder, rel_path)
        if short is None:
            return self._unknown
        names = self._client.short_id_names()
        if short not in names:
            names = self._client.short_id_names(refresh=True)
        return names.get(short, short)
