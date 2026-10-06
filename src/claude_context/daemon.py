"""The long-running indexer process (``claude-context index --watch``).

One worker (the main thread) owns the index.db write connection used for ingestion and
drains a job queue fed by: a watchfiles (inotify) thread, a Syncthing event long-poll
thread, and timers. A separate thread chunks and embeds with its own connection, so
embedding work never delays ingestion.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

from watchfiles import watch

from . import __version__, db, embed, importer
from .config import Config, RootConfig
from .indexer import Indexer, classify
from .roots import root_dir
from .syncthing import SyncthingClient, SyncthingError

log = logging.getLogger(__name__)

RECONCILE_SECONDS = 15 * 60
STATUS_SECONDS = 60
ZIP_STABLE_SECONDS = 30
EVENT_TYPES = ("ItemFinished", "LocalChangeDetected")


def make_syncthing(cfg: Config) -> SyncthingClient | None:
    try:
        return SyncthingClient.from_config_xml(cfg.syncthing.api, cfg.syncthing.config_xml)
    except SyncthingError as e:
        log.warning("Syncthing API unavailable (machine attribution disabled): %s", e)
        return None


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.conn = db.connect(cfg.index_db)
        db.init_schema(self.conn, dim=cfg.embedding.dim)
        self.syncthing = make_syncthing(cfg)
        self.indexer = Indexer(cfg, self.conn, syncthing=self.syncthing)
        self.jobs: queue.Queue[tuple] = queue.Queue()
        self.stop = threading.Event()
        self.started_at = db.utcnow()
        self.last_event: dict[str, str] = {}
        self.errors: list[str] = []
        self.pending_zips: dict[Path, tuple[int, float]] = {}  # path -> (size, first seen stable at)
        self._live_roots = [r for r in cfg.roots if r.kind == "claude-code"]

    # --- producers ---------------------------------------------------------------------
    def _root_for(self, path: Path) -> tuple[RootConfig, str] | None:
        for root in self._live_roots:
            try:
                return root, path.relative_to(root.path).as_posix()
            except ValueError:
                continue
        return None

    def _watch_paths(self) -> list[Path]:
        paths = [r.path for r in self._live_roots if r.path.is_dir()] + [self.cfg.queue_dir]
        drop = self.cfg.imports.claudeai_drop
        if drop is not None:
            drop.mkdir(parents=True, exist_ok=True)
            paths.append(drop)
        return paths

    def _watch_files(self) -> None:
        drop = self.cfg.imports.claudeai_drop
        while not self.stop.is_set():
            try:
                for changes in watch(*self._watch_paths(), debounce=2000, step=200, stop_event=self.stop,
                                     watch_filter=lambda _c, p: "/.stversions/" not in p and "/.stfolder" not in p):
                    for _change, raw in changes:
                        path = Path(raw)
                        if path.parent == self.cfg.queue_dir:
                            if path.suffix == ".json":
                                self.jobs.put(("queue", path))
                        elif drop is not None and path.parent == drop:
                            if path.suffix.lower() == ".zip":
                                self.jobs.put(("zip", path))
                        elif (hit := self._root_for(path)) and classify(hit[1]) is not None:
                            self.jobs.put(("path", hit[0], hit[1]))
            except Exception:
                log.exception("file watcher failed; restarting in 10 s")
                self.stop.wait(10)

    def _watch_syncthing(self) -> None:
        """Fast path for remote changes; inotify and the periodic reconcile are the safety net."""
        folders = {r.syncthing_folder: r for r in self._live_roots if r.syncthing_folder}
        if self.syncthing is None or not folders:
            return
        since = 0
        try:  # start from "now": history is covered by the startup reconcile
            last = self.syncthing.events(0, types=EVENT_TYPES, timeout=1, limit=1)
            since = last[-1]["id"] if last else 0
        except SyncthingError:
            pass
        while not self.stop.is_set():
            try:
                events = self.syncthing.events(since, types=EVENT_TYPES, timeout=60)
            except SyncthingError as e:
                log.debug("Syncthing events unavailable: %s", e)
                self.stop.wait(30)
                continue
            for ev in events:
                since = max(since, ev.get("id", since))
                data = ev.get("data") or {}
                root = folders.get(data.get("folder"))
                rel = data.get("item") or data.get("path")
                if root is not None and rel and classify(rel) is not None:
                    self.jobs.put(("path", root, rel))

    def _embed_loop(self) -> None:
        embedder = embed.embedder_from_config(self.cfg)
        if embedder is None:
            return
        conn = db.connect(self.cfg.index_db)
        try:
            embed.ensure_model(conn, embedder)
            while not self.stop.is_set():
                try:
                    chunked = embed.chunk_pending_sessions(conn, self.cfg)
                    embedded = embed.embed_pending(conn, embedder, batch=self.cfg.embedding.batch,
                                                   max_chunks=self.cfg.embedding.batch * 8)
                except Exception as e:
                    log.exception("embedding pass failed")
                    self._error(f"embedding: {type(e).__name__}: {e}")
                    conn.rollback()
                    self.stop.wait(60)
                    continue
                if not chunked and not embedded:
                    self.stop.wait(10)
        finally:
            conn.close()

    # --- worker ------------------------------------------------------------------------
    def _error(self, text: str) -> None:
        self.errors = (self.errors + [f"{db.utcnow()} {text}"])[-10:]

    def _handle(self, job: tuple) -> None:
        kind = job[0]
        if kind == "path":
            _, root, rel = job
            if self.indexer.ingest_path(root, rel):
                self.last_event[root.label] = db.utcnow()
        elif kind == "queue":
            path: Path = job[1]
            try:
                req = json.loads(path.read_text(encoding="utf-8"))
                root = self.cfg.root(str(req.get("root")))
                rel = str(req.get("rel_path") or "")
                if root is not None and rel and ".." not in rel.split("/"):
                    self.indexer.ingest_path(root, rel)
            except (OSError, ValueError) as e:
                log.warning("bad queue file %s: %s", path.name, e)
            finally:
                path.unlink(missing_ok=True)
        elif kind == "zip":
            self.pending_zips.setdefault(job[1], (-1, 0.0))
        elif kind == "reconcile":
            stats = self.indexer.reconcile()
            log.info("reconcile: scanned=%d ingested=%d missing=%d removed=%d errors=%d in %.1fs", stats.scanned,
                     stats.ingested, stats.missing, stats.removed, stats.errors, stats.seconds)
            if stats.errors:
                self._error(f"reconcile: {stats.errors} file(s) failed to ingest (see journal)")

    def _check_zips(self) -> None:
        """Import a dropped zip once its size has been stable (Syncthing may still be writing)."""
        drop = self.cfg.imports.claudeai_drop
        if drop is not None and drop.is_dir():
            for p in drop.glob("*.zip"):
                self.pending_zips.setdefault(p, (-1, 0.0))
        now = time.monotonic()
        for path, (size, since) in list(self.pending_zips.items()):
            try:
                current = path.stat().st_size
            except OSError:
                del self.pending_zips[path]
                continue
            if current != size:
                self.pending_zips[path] = (current, now)
            elif now - since >= ZIP_STABLE_SECONDS:
                del self.pending_zips[path]
                result = importer.process_drop(self.cfg, self.conn, path)
                if result.error:
                    self._error(f"claude.ai import {result.summary()}")

    def run(self, *, once: bool = False) -> None:
        log.info("indexer starting (version %s, db %s)", __version__, self.cfg.index_db)
        self.indexer.reconcile()
        self.write_status()
        if once:
            return
        for target in (self._watch_files, self._watch_syncthing, self._embed_loop):
            threading.Thread(target=target, name=target.__name__, daemon=True).start()
        for stale in self.cfg.queue_dir.glob("*.json"):
            self.jobs.put(("queue", stale))
        last_reconcile = last_status = time.monotonic()
        while not self.stop.is_set():
            try:
                job = self.jobs.get(timeout=5)
            except queue.Empty:
                job = None
            try:
                if job is not None:
                    self._handle(job)
                now = time.monotonic()
                if now - last_reconcile >= RECONCILE_SECONDS:
                    last_reconcile = now
                    self._handle(("reconcile",))
                if now - last_status >= STATUS_SECONDS:
                    last_status = now
                    self._check_zips()
                    self.write_status()
            except Exception as e:
                log.exception("indexer job failed: %r", job)
                self._error(f"{type(e).__name__}: {e}")
                self.conn.rollback()

    # --- status ------------------------------------------------------------------------
    def _syncthing_status(self) -> dict:
        if self.syncthing is None or not self.syncthing.ping():
            return {"reachable": False}
        out: dict = {"reachable": True, "devices": {}, "folders": {}}
        try:
            out["devices"] = {name: {"connected": d.get("connected"), "last_seen": d.get("last_seen")}
                              for name, d in self.syncthing.connections().items()}
            for root in self._live_roots:
                folder = root.syncthing_folder
                if folder:
                    st = self.syncthing.folder_status(folder)
                    out["folders"][folder] = {"state": st.get("state"), "need_files": st.get("needFiles"),
                                              "completion": self.syncthing.folder_completion(folder)}
        except SyncthingError as e:
            out["error"] = str(e)
        return out

    def write_status(self) -> None:
        """Heartbeat + health snapshot read by the server (see health.py)."""
        conn = self.conn

        def meta_json(key: str):
            raw = db.get_meta(conn, key)
            return json.loads(raw) if raw else None

        last_import = conn.execute("SELECT name, ingested_at FROM imports ORDER BY ingested_at DESC LIMIT 1").fetchone()
        maintenance = meta_json("last_maintenance")
        status = {
            "written_at": db.utcnow(),
            "started_at": self.started_at,
            "version": __version__,
            "pid": os.getpid(),
            "last_reconcile_at": db.get_meta(conn, "last_reconcile_at"),
            "roots": {
                r.label: {
                    "last_event_at": self.last_event.get(r.label),
                    "last_reconcile_at": db.get_meta(conn, f"reconcile:{r.label}"),
                    "files": conn.execute("SELECT COUNT(*) FROM files WHERE root = ?", (r.label,)).fetchone()[0],
                    "path_ok": root_dir(self.cfg, r).is_dir(),
                } for r in self.cfg.roots
            },
            "syncthing": self._syncthing_status(),
            "embedding": {"model": self.cfg.embedding.model if self.cfg.embedding.enabled else None,
                          **embed.backlog(conn, embed_automated=self.cfg.embedding.embed_automated)},
            "conflicts": conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0],
            "disk_free_gb": round(shutil.disk_usage(self.cfg.data_dir).free / 1024**3, 1),
            "last_import": dict(last_import) if last_import else None,
            "last_prune": meta_json("last_prune"),
            "last_maintenance_at": maintenance.get("at") if maintenance else None,
            "errors": self.errors,
        }
        tmp = self.cfg.status_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
        os.replace(tmp, self.cfg.status_file)

