# Claude Context Hub

A self-hosted remote [MCP](https://modelcontextprotocol.io) server that gives **every Claude surface** —
claude.ai, Claude Desktop, Cowork, mobile, the Office add-ins and Claude Code — read and write access to the
memories, transcripts and notes produced by **every Claude Code instance on every machine you use**.

Claude Code keeps its history as files under `~/.claude/projects`. If you sync that folder between your
machines (for example with Syncthing), this hub indexes the shared folder and serves it as a custom connector:

- *"Give me a brief on the website project"* in claude.ai returns the project's memory index, recent sessions
  from all machines, and notes left by other Claude surfaces.
- *"Find where we discussed the retry logic"* runs a hybrid keyword + semantic search over all transcripts.
- A conversation in PowerPoint or on your phone can leave a note, or save a memory that the next Claude Code
  session on any machine loads automatically.

It is built for one person, low maintenance, and strict lockdown: GitHub sign-in restricted to an allowlist of
numeric user ids, secrets redacted before anything is indexed, thinking blocks never stored.

```
 Claude (cloud) ──HTTPS──► reverse tunnel ──► 127.0.0.1:8790  claude-context serve
                                                 FastMCP (Streamable HTTP, stateless) + GitHub OAuth proxy
                                                 reads index.db · writes memory/note files · audit.db
 claude-context index --watch ── watches the synced folder, Syncthing events and an import drop folder
                                 writes index.db (SQLite FTS5 + sqlite-vec), embeds locally on CPU (fastembed)
 claude-context maintenance (daily) ── archive, recovery of deleted sessions, retention, backups
```

## Tools

| Tool | Purpose |
|---|---|
| `project_brief` | The main entry point: memory index, recent memory changes, latest sessions, notes, warnings. |
| `recent_activity` | Cross-project timeline of sessions, memory changes and notes. |
| `search` | Hybrid search (FTS5 bm25 + local embeddings, fused with RRF and a recency factor). |
| `list_projects`, `list_sessions`, `read_session`, `read_message` | Browse and read transcripts, paged. |
| `list_memories`, `read_memory`, `list_notes`, `read_note` | Read memories and notes. |
| `save_memory`, `archive_memory` | Write memories in Claude Code's own format (optimistic concurrency by sha256). |
| `log_note` | Leave an append-only note from any surface. |
| `hub_status` | Index freshness, counts, sync state, warnings. |

Prompts: `catch_up(project)` and `wrap_up(project)`.

## How it works

**Sources.** One or more *roots*, each a `~/.claude/projects`-style tree:
`<project-key>/<session-id>.jsonl` transcripts, `<session-id>/subagents/…` subagent transcripts,
`memory/MEMORY.md` + one memory per file, and `remote-notes/*.md` (written by `log_note`). An optional root of
kind `stversions` recovers sessions that were deleted upstream from Syncthing's versions folder. claude.ai
conversations come from data-export zips dropped into a watched folder.

**Indexing.** Transcripts are append-only, so the indexer keeps a byte offset per file and parses only new,
complete lines. Every message becomes a row in an FTS5 table; interactive sessions, memories and notes are
also chunked and embedded (`BAAI/bge-small-en-v1.5`, 384-d, CPU). Automated (SDK-driven) sessions get keyword
search only and are hidden from listings unless asked for. Each file is attributed to the machine that last
modified it via Syncthing's REST API.

**Safety properties.**
- Authentication is never optional. The only exception is `serve --insecure-local-test`, which refuses to
  start unless bound to loopback with no tunnel running, and refuses any request that came through a proxy.
- The GitHub allowlist is enforced in three independent layers: at the OAuth callback, in token verification,
  and per MCP request. See [`docs/AUTH-NOTES.md`](docs/AUTH-NOTES.md).
- Secrets (API keys, tokens, private keys, passwords, high-entropy values) are redacted at ingest, so they
  never reach the index or the embeddings, and again on output.
- `thinking` blocks are skipped by the parsers and never stored.
- The server writes only memory and note files, atomically, and never deletes: `archive_memory` moves a file
  into `memory/.archived/`.
- Every tool call and auth event is recorded in `audit.db`.
- Transcript text is labeled as untrusted historical data in the server instructions and tool descriptions.

## Requirements

- Linux with systemd, Python 3.12+, [`uv`](https://docs.astral.sh/uv/), SQLite ≥ 3.45 with loadable extensions.
- A folder of Claude Code sessions (ideally synced from all your machines; Syncthing is supported natively).
- A public HTTPS hostname that forwards to `127.0.0.1:8790` without opening inbound ports — for example a
  Cloudflare Tunnel. Claude's servers must be able to reach it; an authenticating proxy in front will not work.
- A GitHub OAuth App (not a GitHub App) with callback URL `https://<your-host>/auth/callback`.

## Install

```bash
git clone https://github.com/<you>/claude-context-mcp.git && cd claude-context-mcp
uv venv --python /usr/bin/python3.12 && uv sync

mkdir -p ~/.config/claude-context && chmod 700 ~/.config/claude-context
cp config.example.toml ~/.config/claude-context/config.toml && chmod 600 ~/.config/claude-context/config.toml
$EDITOR ~/.config/claude-context/config.toml      # base_url, allowed_github_ids, roots, projects
```

Create the secrets file **in your own terminal** (never paste secrets into a Claude session — transcripts are
exactly what this tool indexes):

```bash
umask 077
read -rp  "GitHub Client ID: "     GCID
read -rsp "GitHub Client secret: " GCS; echo
cat > ~/.config/claude-context/secrets.env <<EOF
GITHUB_CLIENT_ID=$GCID
GITHUB_CLIENT_SECRET=$GCS
JWT_SIGNING_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(64))')
STORAGE_ENCRYPTION_KEY=$(python3 -c 'import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())')
EOF
chmod 600 ~/.config/claude-context/secrets.env; unset GCID GCS
```

Build the index once, then install the services:

```bash
uv run claude-context index            # first run downloads the embedding model (~70 MB)
uv run claude-context doctor
deploy/install-units.sh                # renders units into deploy/rendered/ for review
deploy/install-units.sh --install      # copies them to /etc/systemd/system (sudo)
sudo systemctl enable --now claude-context-indexer claude-context-server claude-context-maintenance.timer
```

`deploy/systemd/cloudflared-claude-context.service` is a dedicated tunnel unit that reads its token from
`/etc/cloudflared/claude-context.token` (root, mode 600). It is separate from any other `cloudflared` service.

Finally add the connector in claude.ai (Customize → Connectors → Add custom connector) with URL
`https://<your-host>/mcp`, and sign in with the allowlisted GitHub account. Suggested permissions: always allow
the read tools and `log_note`; require approval for `save_memory` and `archive_memory`.

## Operations

```
claude-context serve [--insecure-local-test]
claude-context index [--watch | --full | --rebuild-embeddings | --no-embed]
claude-context import-claudeai <zip>
claude-context recover-stversions [--dry-run]
claude-context prune [--dry-run]
claude-context merge-conflicts [--dry-run]
claude-context maintenance
claude-context status
claude-context search "<query>" [--mode hybrid|keyword|semantic] [--project <alias>]
claude-context doctor
```

**Health.** There is no separate alerting stack. The indexer writes a heartbeat (`status.json`) every minute;
when something is wrong (stale heartbeat, sync device offline for a day, conflict files, low disk, embedding
backlog, unknown transcript record types…) `project_brief` and `recent_activity` prepend a "⚠ Hub health" block,
so the problem surfaces the next time any Claude uses the hub. `claude-context status` and `doctor` show the
same from the shell; logs go to the journal (`journalctl -u claude-context-server -u claude-context-indexer`).

**Data locations** (default `~/.local/share/claude-context/`): `index.db` (rebuildable at any time with
`index --full`), `audit.db`, `oauth/` (encrypted OAuth store), `archive/` (copies of transcripts, kept after
the originals are deleted upstream, until retention), `models/`, `imports/processed/`, `backups/` (14 nightly
copies of `audit.db` and the config).

**Retention.** Sessions older than `retention_days` are pruned daily from the index and the archive. Live
files are never deleted by the hub (set Claude Code's own `cleanupPeriodDays` to match). Memories and notes are
never pruned.

**Conflicts.** If two machines edit a `MEMORY.md` at once, Syncthing creates a `MEMORY.sync-conflict-*.md`.
The maintenance job merges these automatically (union of lines, newest wins per memory) and archives the
conflict copy. Conflicts in other files are only reported.

**Key rotation.**
- `JWT_SIGNING_KEY`: rotating it invalidates every issued token; each connector must sign in again.
- `STORAGE_ENCRYPTION_KEY`: rotating it makes the stored OAuth clients and tokens unreadable; delete
  `oauth/` and sign in again.
- GitHub client secret: generate a new one in the OAuth App, update `secrets.env`, restart the server.
- To revoke access for a GitHub account, remove its id from `allowed_github_ids` and restart the server;
  previously issued tokens for that id stop working immediately.

**Upgrading.** Dependencies are pinned in `uv.lock`; nothing upgrades automatically.
`uv lock --upgrade-package <name> && uv sync && uv run pytest && uv run claude-context doctor`, then restart
the services. When upgrading `fastmcp`, re-read [`docs/AUTH-NOTES.md`](docs/AUTH-NOTES.md): the allowlist
tests are designed to fail loudly if its OAuth internals change.

**Format drift.** Claude Code's transcript format changes over time. The parser ignores unknown fields and
counts unknown record types; `hub_status` reports them so drift is visible instead of silently losing data.

## Troubleshooting

| Symptom | Check |
|---|---|
| Connector says "couldn't connect" | `claude-context doctor` (public endpoint line); tunnel unit active; `base_url` matches the public hostname exactly. |
| Sign-in loops or fails | The OAuth App callback URL must be `<base_url>/auth/callback`; the account's numeric id must be in `allowed_github_ids`; see `auth_events` in `audit.db`. |
| Results are stale | `claude-context status`: indexer heartbeat, last reconcile, Syncthing state. |
| Semantic search finds nothing | Embedding backlog in `status`; `index --rebuild-embeddings` after changing the model. |
| A memory edit was rejected | Another machine changed it first: `read_memory` again and retry with the new sha256. |

## Development

```bash
uv sync && uv run pytest
```

Tests are fully offline and use synthetic fixtures; nothing in this repository is derived from real transcripts.

## License

MIT
