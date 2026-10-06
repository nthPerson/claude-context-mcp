"""Where indexed files live on disk, and the server → indexer reindex queue."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

from .config import Config, RootConfig


def root_dir(cfg: Config, root: RootConfig) -> Path:
    """Directory the indexer scans for a root.

    A ``stversions`` root is never scanned in place: sessions recovered from it are copied
    into ``archive/<label>/`` first, and that copy is what gets indexed.
    """
    return cfg.archive_dir / root.label if root.kind == "stversions" else root.path


def raw_path(cfg: Config, root_label: str, rel_path: str) -> Path | None:
    """Best existing copy of an indexed file: the live one, else the archived one."""
    root = cfg.root(root_label)
    candidates = []
    if root is not None:
        candidates.append(root_dir(cfg, root) / rel_path)
    candidates.append(cfg.archive_dir / root_label / rel_path)
    return next((p for p in candidates if p.is_file()), None)


def enqueue_reindex(cfg: Config, root_label: str, rel_path: str) -> None:
    """Ask the indexer to (re)index one file now instead of waiting for its watcher."""
    cfg.queue_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{secrets.token_hex(4)}.json"
    tmp = cfg.queue_dir / f".{name}.tmp"
    tmp.write_text(json.dumps({"root": root_label, "rel_path": rel_path}), encoding="utf-8")
    os.replace(tmp, cfg.queue_dir / name)
