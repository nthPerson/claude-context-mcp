from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from claude_context.syncthing import (
    MachineResolver,
    SyncthingClient,
    SyncthingError,
    read_api_key,
)

ME = "AAAAAAA-BBBBBBB-CCCCCCC-DDDDDDD-EEEEEEE-FFFFFFF-GGGGGGG-HHHHHHH"
LAPTOP = "LLLLLLL-BBBBBBB-CCCCCCC-DDDDDDD-EEEEEEE-FFFFFFF-GGGGGGG-HHHHHHH"
SERVER = "SSSSSSS-BBBBBBB-CCCCCCC-DDDDDDD-EEEEEEE-FFFFFFF-GGGGGGG-HHHHHHH"
KEY = "secret-key-123"


class Api:
    """Fake Syncthing: routes by path, records requests."""

    def __init__(self, root: Path | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self.root = root or Path("/data/sessions")
        self.modified_by: dict[str, str] = {"a/b c.jsonl": "LLLLLLL"}
        self.fail = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if self.fail:
            return httpx.Response(500, text="boom")
        p, q = req.url.path, req.url.params
        if p == "/rest/system/ping":
            return httpx.Response(200, json={"ping": "pong"})
        if p == "/rest/system/status":
            return httpx.Response(200, json={"myID": ME})
        if p == "/rest/config/devices":
            return httpx.Response(200, json=[
                {"deviceID": ME, "name": "desktop"},
                {"deviceID": LAPTOP, "name": "laptop"},
                {"deviceID": SERVER, "name": "server"},
            ])
        if p == "/rest/config/folders":
            return httpx.Response(200, json=[
                {"id": "outer", "path": str(self.root), "devices": []},
                {"id": "inner", "path": str(self.root / "inner"), "devices": []},
                {"id": "sessions", "path": str(self.root), "devices": [
                    {"deviceID": ME}, {"deviceID": LAPTOP}, {"deviceID": SERVER}]},
            ])
        if p == "/rest/db/file":
            by = self.modified_by.get(q["file"])
            if by is None:
                return httpx.Response(404, text="no such object")
            return httpx.Response(200, json={"global": {"modifiedBy": by}})
        if p == "/rest/events":
            return httpx.Response(200, json=[{"id": 5, "type": "ItemFinished"}])
        if p == "/rest/system/connections":
            return httpx.Response(200, json={"connections": {
                LAPTOP: {"connected": True}, SERVER: {"connected": False}}})
        if p == "/rest/stats/device":
            return httpx.Response(200, json={
                ME: {"lastSeen": "1969-12-31T16:00:00-08:00"},
                LAPTOP: {"lastSeen": "2026-01-02T03:04:05Z"},
                SERVER: {"lastSeen": "1969-12-31T16:00:00-08:00"}})
        if p == "/rest/db/completion":
            return httpx.Response(200, json={"completion": 100 if q["device"] == LAPTOP else 42.5})
        if p == "/rest/db/status":
            return httpx.Response(200, json={
                "state": "idle", "needFiles": 0, "errors": 0, "globalFiles": 9,
                "localFiles": 9, "extra": "dropped"})
        return httpx.Response(404)


@pytest.fixture
def api() -> Api:
    return Api()


@pytest.fixture
def client(api: Api) -> SyncthingClient:
    return SyncthingClient("http://st.invalid:8384/", KEY, transport=httpx.MockTransport(api))


def failing_client(status: int = 500, *, text: str = "x") -> SyncthingClient:
    return SyncthingClient(
        "http://st.invalid", KEY, transport=httpx.MockTransport(lambda r: httpx.Response(status, text=text))
    )


def test_read_api_key(tmp_path: Path) -> None:
    cfg = tmp_path / "config.xml"
    cfg.write_text("<configuration><gui><apikey> abc123 </apikey></gui></configuration>")
    assert read_api_key(cfg) == "abc123"


@pytest.mark.parametrize("content", [None, "not xml", "<configuration><gui/></configuration>"])
def test_read_api_key_errors(tmp_path: Path, content: str | None) -> None:
    cfg = tmp_path / "config.xml"
    if content is not None:
        cfg.write_text(content)
    with pytest.raises(SyncthingError):
        read_api_key(cfg)


def test_from_config_xml_and_header(tmp_path: Path, api: Api) -> None:
    cfg = tmp_path / "config.xml"
    cfg.write_text(f"<configuration><gui><apikey>{KEY}</apikey></gui></configuration>")
    c = SyncthingClient.from_config_xml("http://x", cfg, transport=httpx.MockTransport(api))
    assert c.ping()
    assert api.calls[0].headers["X-API-Key"] == KEY


def test_repr_hides_key(client: SyncthingClient) -> None:
    assert KEY not in repr(client)


def test_ping(client: SyncthingClient) -> None:
    assert client.ping() is True
    assert failing_client().ping() is False
    assert failing_client(200, text="not json").ping() is False

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert SyncthingClient("http://x", KEY, transport=httpx.MockTransport(boom)).ping() is False


def test_my_id(client: SyncthingClient) -> None:
    assert client.my_id() == ME
    with pytest.raises(SyncthingError):
        failing_client().my_id()


def test_devices_and_short_ids(client: SyncthingClient, api: Api) -> None:
    assert client.devices() == {ME: "desktop", LAPTOP: "laptop", SERVER: "server"}
    names = client.short_id_names()
    assert names == {"AAAAAAA": "desktop", "LLLLLLL": "laptop", "SSSSSSS": "server"}
    n = len(api.calls)
    client.short_id_names()  # cached
    assert len(api.calls) == n
    client.short_id_names(refresh=True)
    assert len(api.calls) == n + 1
    with pytest.raises(SyncthingError):
        failing_client().devices()


def test_folders_and_error(client: SyncthingClient) -> None:
    assert {f["id"] for f in client.folders()} == {"outer", "inner", "sessions"}
    with pytest.raises(SyncthingError):
        failing_client(200, text="not json").folders()


def test_folder_for_path_longest_prefix(client: SyncthingClient, api: Api) -> None:
    assert client.folder_for_path(api.root / "inner" / "x.jsonl") == ("inner", api.root / "inner")
    assert client.folder_for_path(api.root / "other" / "x.jsonl") == ("outer", api.root)
    assert client.folder_for_path(Path("/elsewhere/x")) is None


def test_file_modified_by(client: SyncthingClient, api: Api) -> None:
    assert client.file_modified_by("sessions", "a/b c.jsonl") == "LLLLLLL"
    assert api.calls[-1].url.params["file"] == "a/b c.jsonl"
    assert api.calls[-1].url.params["folder"] == "sessions"
    assert "b%20c" in str(api.calls[-1].url) or "b+c" in str(api.calls[-1].url)
    assert client.file_modified_by("sessions", "missing") is None  # 404
    with pytest.raises(SyncthingError):
        failing_client().file_modified_by("sessions", "x")


def test_events_params(client: SyncthingClient, api: Api) -> None:
    assert client.events(7, timeout=30, limit=10) == [{"id": 5, "type": "ItemFinished"}]
    q = api.calls[-1].url.params
    assert q["since"] == "7"
    assert q["events"] == "ItemFinished,LocalChangeDetected,RemoteChangeDetected"
    assert q["timeout"] == "30" and q["limit"] == "10"
    client.events(0, types=["ItemFinished"])
    q = api.calls[-1].url.params
    assert q["events"] == "ItemFinished" and "limit" not in q and q["timeout"] == "60"


def test_events_read_timeout_exceeds_poll_and_empty() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.extensions["timeout"])
        return httpx.Response(200, json=[])

    c = SyncthingClient("http://x", KEY, timeout=5, transport=httpx.MockTransport(handler))
    assert c.events(3, timeout=20) == []
    assert seen[0]["read"] > 20
    with pytest.raises(SyncthingError):
        failing_client().events(0)


def test_connections(client: SyncthingClient) -> None:
    assert client.connections() == {
        "laptop": {"device_id": LAPTOP, "connected": True, "last_seen": "2026-01-02T03:04:05Z"},
        "server": {"device_id": SERVER, "connected": False, "last_seen": None},
    }
    with pytest.raises(SyncthingError):
        failing_client().connections()


def test_folder_completion(client: SyncthingClient) -> None:
    assert client.folder_completion("sessions") == {"laptop": 100.0, "server": 42.5}
    with pytest.raises(SyncthingError):
        client.folder_completion("nope")
    with pytest.raises(SyncthingError):
        failing_client().folder_completion("sessions")


def test_folder_status(client: SyncthingClient) -> None:
    assert client.folder_status("sessions") == {
        "state": "idle", "needFiles": 0, "errors": 0, "globalFiles": 9, "localFiles": 9}
    with pytest.raises(SyncthingError):
        failing_client().folder_status("sessions")


# -- MachineResolver ---------------------------------------------------------

def lookups(api: Api) -> int:
    return sum(c.url.path == "/rest/db/file" for c in api.calls)


def test_resolver_caches_by_path_and_mtime(client: SyncthingClient, api: Api) -> None:
    r = MachineResolver(client, "sessions")
    assert [r.machine_for("a/b c.jsonl", 1) for _ in range(3)] == ["laptop"] * 3
    assert lookups(api) == 1
    r.machine_for("a/b c.jsonl", 2)
    assert lookups(api) == 2


def test_resolver_local_device_included(client: SyncthingClient, api: Api) -> None:
    api.modified_by["mine"] = "AAAAAAA"
    assert MachineResolver(client, "sessions").machine_for("mine", 1) == "desktop"


def test_resolver_unknown_short_id_refreshes_then_falls_back(client: SyncthingClient, api: Api) -> None:
    api.modified_by["new"] = "ZZZZZZZ"
    r = MachineResolver(client, "sessions")
    assert r.machine_for("new", 1) == "ZZZZZZZ"
    device_calls = sum(c.url.path == "/rest/config/devices" for c in api.calls)
    assert device_calls == 2  # initial load + one refresh


def test_resolver_absent_file_is_unknown(client: SyncthingClient) -> None:
    assert MachineResolver(client, "sessions", unknown="?").machine_for("gone", 1) == "?"


def test_resolver_no_client_or_folder(client: SyncthingClient, api: Api) -> None:
    assert MachineResolver(None, "sessions").machine_for("x", 1) == "unknown"
    assert MachineResolver(client, None).machine_for("x", 1) == "unknown"
    assert api.calls == []


def test_resolver_backoff_and_recovery(client: SyncthingClient, api: Api) -> None:
    now = [1000.0]
    r = MachineResolver(client, "sessions", clock=lambda: now[0])
    api.fail = True
    assert r.machine_for("a/b c.jsonl", 1) == "unknown"
    n = len(api.calls)
    api.fail = False
    now[0] += 30  # still backing off: no API call, not cached
    assert r.machine_for("a/b c.jsonl", 1) == "unknown"
    assert len(api.calls) == n
    now[0] += 31
    assert r.machine_for("a/b c.jsonl", 1) == "laptop"
    assert len(api.calls) > n


def test_resolver_cache_bounded(client: SyncthingClient, api: Api) -> None:
    for i in range(3):
        api.modified_by[f"f{i}"] = "LLLLLLL"
    r = MachineResolver(client, "sessions", max_cache=2)
    for i in range(3):
        r.machine_for(f"f{i}", 1)
    assert lookups(api) == 3
    r.machine_for("f2", 1)  # still cached
    assert lookups(api) == 3
    r.machine_for("f0", 1)  # evicted
    assert lookups(api) == 4
