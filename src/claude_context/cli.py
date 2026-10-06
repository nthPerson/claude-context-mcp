"""``claude-context`` command line."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__, db
from .config import CONFIG_ENV, ConfigError, default_config_path, load_config


def _open(cfg):
    cfg.ensure_dirs()
    conn = db.connect(cfg.index_db)
    db.init_schema(conn, dim=cfg.embedding.dim)
    return conn


def cmd_serve(cfg, args) -> int:
    from .server import serve
    return serve(cfg, insecure_local_test=args.insecure_local_test)


def cmd_index(cfg, args) -> int:
    from .daemon import Daemon
    daemon = Daemon(cfg)
    if args.full:
        for table in ("messages", "sessions", "memories", "notes", "docs", "fts_docs", "chunks", "chunk_vec", "files",
                      "conflicts", "drift"):
            daemon.conn.execute(f"DELETE FROM {table}")
        daemon.conn.commit()
    if args.rebuild_embeddings:
        daemon.conn.execute("DELETE FROM chunk_vec")
        daemon.conn.execute("UPDATE chunks SET embedded = 0")
        daemon.conn.commit()
    if args.watch:
        daemon.run()
        return 0
    daemon.run(once=True)
    from . import embed
    embedder = embed.embedder_from_config(cfg)
    if embedder is not None and not args.no_embed:
        embed.ensure_model(daemon.conn, embedder)
        while embed.chunk_pending_sessions(daemon.conn, cfg) + embed.embed_pending(
                daemon.conn, embedder, batch=cfg.embedding.batch, max_chunks=cfg.embedding.batch * 16):
            print(f"embedding… {embed.backlog(daemon.conn, embed_automated=cfg.embedding.embed_automated)['pending_chunks']} chunks pending", file=sys.stderr)
        daemon.write_status()
    return 0


def cmd_import(cfg, args) -> int:
    from .importer import import_export
    result = import_export(cfg, _open(cfg), Path(args.zip))
    print(result.summary())
    for w in result.warnings:
        print("  warning:", w)
    return 1 if result.error else 0


def cmd_recover(cfg, args) -> int:
    from .maintenance import recover_stversions
    recovered = recover_stversions(cfg, None if args.dry_run else _open(cfg), dry_run=args.dry_run)
    sessions = sum(1 for r in recovered if r.endswith(".jsonl") and "/subagents/" not in r)
    print(f"{'would recover' if args.dry_run else 'recovered'} {len(recovered)} files ({sessions} main sessions)")
    return 0


def cmd_prune(cfg, args) -> int:
    from .maintenance import prune
    sessions, files = prune(cfg, _open(cfg), dry_run=args.dry_run)
    print(f"{'would prune' if args.dry_run else 'pruned'} {sessions} sessions older than {cfg.retention_days} days"
          + ("" if args.dry_run else f" ({files} archived files removed)"))
    return 0


def cmd_merge(cfg, args) -> int:
    from .maintenance import merge_conflicts
    merges = merge_conflicts(cfg, dry_run=args.dry_run)
    for m in merges:
        print(f"{m.conflict_file}: {m.detail}")
    print(f"{len(merges)} MEMORY.md conflict file(s) {'found' if args.dry_run else 'processed'}")
    return 0


def cmd_maintenance(cfg, args) -> int:
    from .maintenance import run_all
    report = run_all(cfg, _open(cfg))
    print(report)
    return 1 if report.errors else 0


def cmd_status(cfg, args) -> int:
    from .service import HubService
    print(HubService(cfg).hub_status())
    return 0


def cmd_search(cfg, args) -> int:
    from . import embed
    from .service import HubService
    service = HubService(cfg, embedder=None if args.mode == "keyword" else embed.embedder_from_config(cfg))
    print(service.search(query=args.query, mode=args.mode, project=args.project, limit=args.limit))
    return 0


def cmd_unit_paths(cfg, args) -> int:
    """Shell-evalable paths for deploy/install-units.sh."""
    import shlex
    writable = " ".join(str(r.path) for r in cfg.roots if r.writable and r.kind == "claude-code")
    print(f"DATA_DIR={shlex.quote(str(cfg.data_dir))}")
    print(f"WRITABLE_ROOTS={shlex.quote(writable)}")
    print(f"DROP_DIR={shlex.quote(str(cfg.imports.claudeai_drop or ''))}")
    return 0


def cmd_doctor(cfg, args) -> int:
    from . import doctor
    return doctor.run(cfg, args.config or default_config_path())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claude-context", description="Claude Context Hub")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help=f"config file (default: {default_config_path()})")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the MCP server")
    p.add_argument("--insecure-local-test", action="store_true",
                   help="disable auth; only on loopback and never while the tunnel is running")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("index", help="build or update the index")
    p.add_argument("--watch", action="store_true", help="keep running and follow changes")
    p.add_argument("--full", action="store_true", help="drop the index and rebuild it from the files")
    p.add_argument("--rebuild-embeddings", action="store_true")
    p.add_argument("--no-embed", action="store_true", help="skip embedding in a one-shot run")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("import-claudeai", help="import a claude.ai data export zip (left in place)")
    p.add_argument("zip")
    p.set_defaults(fn=cmd_import)

    for name, fn, text in (("recover-stversions", cmd_recover, "recover sessions deleted upstream"),
                           ("prune", cmd_prune, "delete sessions past the retention period"),
                           ("merge-conflicts", cmd_merge, "merge MEMORY.md Syncthing conflict files")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(fn=fn)

    sub.add_parser("maintenance", help="run the daily maintenance job").set_defaults(fn=cmd_maintenance)
    sub.add_parser("status", help="print hub_status").set_defaults(fn=cmd_status)

    p = sub.add_parser("search", help="search the index from the shell")
    p.add_argument("query")
    p.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid")
    p.add_argument("--project")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_search)

    sub.add_parser("doctor", help="check the installation").set_defaults(fn=cmd_doctor)
    sub.add_parser("unit-paths", help=argparse.SUPPRESS).set_defaults(fn=cmd_unit_paths)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.config:  # secrets.env and backups are located relative to the config file
        os.environ[CONFIG_ENV] = str(args.config)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
