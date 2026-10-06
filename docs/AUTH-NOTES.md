# Authentication notes

Verified against **fastmcp 4.0.11** (mcp 2.3.0, py-key-value-aio 0.4.6). Code: `src/claude_context/auth.py`,
`middleware.py`, `audit.py`. Tests: `tests/test_auth.py` (end to end against a fake GitHub), `test_middleware.py`,
`test_audit.py`.

## FastMCP API used

| What | Name |
|---|---|
| Provider | `fastmcp.server.auth.providers.github.GitHubProvider` (subclassed as `AllowlistedGitHubProvider`) |
| Provider params | `client_id`, `client_secret`, `base_url`, `redirect_path="/auth/callback"`, `required_scopes=[]`, `cache_ttl_seconds=300`, `allowed_client_redirect_uris`, `client_storage`, `jwt_signing_key`, `require_authorization_consent=True`, `enable_cimd=True`, `http_client` (an `httpx2.AsyncClient`; used by the token verifier only) |
| DCR | always on in `OAuthProxy` (`/register`) |
| Storage | `key_value.aio.stores.filetree.FileTreeStore` (+ `FileTreeV1KeySanitizationStrategy`, `FileTreeV1CollectionSanitizationStrategy`) under `oauth_dir`, wrapped in `key_value.aio.wrappers.encryption.FernetEncryptionWrapper(raise_on_decryption_error=False)`. This is the store FastMCP itself defaults to; `DiskStore` would need the extra `diskcache` package and is not used. |
| App | `FastMCP(..., auth=provider, middleware=[GuardMiddleware(...)]).http_app(path="/mcp", stateless_http=True)` |
| Middleware | `fastmcp.server.middleware.Middleware`; hooks `on_message` (guard) and `on_call_tool` (audit) |
| Headers | `fastmcp.server.dependencies.get_http_request().headers` (`get_http_headers()` drops `authorization`, `cookie`, `host` and a few others by default; it does not drop `cf-connecting-ip`) |
| Token | `fastmcp.server.dependencies.get_access_token()` |

Redirect-URI patterns: `https://claude.ai/api/mcp/auth_callback`, `https://claude.com/api/mcp/auth_callback`,
`http://localhost:*`, `http://127.0.0.1:*`. A loopback pattern with `:*` and no path matches any port and path.
Matching is per URL component; userinfo (`http://localhost@evil`) and dot-segments are rejected.

GitHub scope: **none**. `GET /user` returns the public profile, including the numeric id, for a token with no
scope, and FastMCP's verifier only calls `/user` and `/user/repos`. Scopes a client requests are dropped in
`authorize()` and never reach GitHub. This works unchanged with a GitHub App, which has no scopes.

## HTTP routes

| Path | Methods | Called by |
|---|---|---|
| `/mcp` | POST, DELETE | Claude's servers (MCP traffic). Bearer token required, otherwise 401. |
| `/.well-known/oauth-protected-resource/mcp` | GET | Claude's servers (discovery) |
| `/.well-known/oauth-authorization-server` | GET | Claude's servers (discovery) |
| `/register` | POST | Claude's servers (DCR; not used when the client uses CIMD) |
| `/token` | POST (form-urlencoded) | Claude's servers (code exchange, refresh) |
| `/authorize` | GET, POST | **user's browser** |
| `/consent` | GET, POST | **user's browser** (consent page and its form) |
| `/auth/callback` | GET | **user's browser** (redirect from github.com) |

There are no static assets. The consent and error pages are self-contained HTML; they show the FastMCP logo,
which the browser loads from `gofastmcp.com`, not from this server. Clients that run on your own machine (for
example Claude Code's loopback flow) call discovery, `/register`, `/token` and possibly `/mcp` from your own IP,
not from Anthropic's.

A firewall rule that admits only Claude's egress ranges must exempt `/authorize`, `/consent` and
`/auth/callback`. Any extra routes the server adds (health checks and so on) are outside this list.

Outbound calls made by the hub: `github.com/login/oauth/access_token`, `api.github.com/user` and
`api.github.com/user/repos`, plus the client's CIMD document URL when a client identifies itself by URL
(FastMCP fetches it with SSRF checks).

## The allowlist (three independent layers)

Ids are compared as integers (`type(id) is int`); logins are never used.

1. **GitHub callback.** `AllowlistedGitHubProvider._create_upstream_oauth_client()` wraps FastMCP's upstream
   token client. Right after GitHub returns tokens (code exchange or refresh) it calls `GET /user` and raises
   unless the id is allowlisted. FastMCP's callback handler then stops before it stores or issues an
   authorization code. `get_routes()` also wraps the `/auth/callback` route: a denial becomes a 403 page, and
   a response that carries a `code` without a passed check is replaced by a 500 page. That second rule fails
   closed if a FastMCP upgrade stops using the hook. The server refuses to start if the callback route is
   missing. Logins, refreshes and denials are written to `auth_events`.
2. **Token verification.** `load_access_token()` runs FastMCP's validation: JWT signature, `iss`, `aud`
   (`<base_url>/mcp`), `exp`, JTI lookup, then the upstream token re-verified with GitHub. It then requires the
   verified `github_user_data.id` to be an allowlisted integer, consistent with `sub` and with the id embedded
   at login (`upstream_claims.github_id`). Any failure returns `None`, which the HTTP layer turns into a 401.
   Changing the allowlist takes effect on the next request.
3. **Per request.** `GuardMiddleware.on_message` re-checks the id from the verified token and applies the
   source-CIDR check and the rate limit.

FastMCP internals this depends on, all pinned by tests that fail loudly:
- `OAuthProxy._create_upstream_oauth_client()`: an underscore method, but its docstring documents it as an
  override point. Covered by `test_fastmcp_callback_uses_upstream_client_hook`, and by the fail-closed
  tripwire `test_callback_refuses_code_if_identity_hook_was_bypassed`.
- `OAuthProxy._extract_upstream_claims()`: a documented override hook. Used only for the consistency check.
- `AsyncOAuth2Client._client`: replaced only when a test injects an HTTP client.
- The claim shape of `GitHubTokenVerifier` (`sub`, `github_user_data`). If it changes, every token is rejected
  (fail closed), and the end-to-end happy-path test fails.

## Middleware behaviour

- `on_message` fires exactly once for **every** inbound JSON-RPC message: `initialize`, `ping`,
  `tools/*`, `prompts/*`, `resources/*`, and notifications. Component methods get it from the interior pass;
  everything else gets it from the root "outer" pass. Tests cover each of these.
- An unauthenticated or invalid-token HTTP request never reaches it. `RequireAuthMiddleware` on the `/mcp` route
  returns 401 before any MCP session exists (tested).
- Denials raise a JSON-RPC error with code **-32003** ("Forbidden: <reason>"). The rate limit raises FastMCP's
  `RateLimitError` (-32000). Both arrive inside an HTTP 200 response: FastMCP middleware cannot set the HTTP
  status. A true HTTP 403 would need an ASGI middleware in front of `/mcp`.
- `allowed_mcp_source_cidrs` (IPv4/IPv6): when the list is non-empty, `Cf-Connecting-Ip` must be present,
  parse as an IP, and fall inside a listed range. Invalid CIDRs stop startup.
- Rate limit: `rate_limit_per_minute` requests per rolling 60 s per GitHub id, held in memory. Notifications are
  not counted. 0 disables it. FastMCP's own `SlidingWindowRateLimitingMiddleware` was not used: its clock is
  hard-wired and its denials can't be audited.
- Audit: each tool call writes one `calls` row (`ok`, `error` or `denied`). Every denied request writes one too,
  with `tool` set to the method name for non-tool requests. Arguments go through
  `claude_context.redact.redact` when that module imports (identity otherwise) and are truncated to 500
  characters. If redaction fails, `[redaction failed]` is stored instead. Audit errors are logged and never
  fail a request.
- `--insecure-local-test`: build the app with `auth=None` and `GuardMiddleware(..., insecure_local_test=True)`,
  and only after `local_test_mode_allowed(cfg, bind_host=...)` returns True. That function requires a loopback
  bind and, when `base_url` is set, no process whose command line contains both `cloudflared` and
  `claude-context`. The middleware logs a warning at startup and refuses any request carrying
  `Cf-Connecting-Ip`.

## Token lifetimes

| Item | Lifetime |
|---|---|
| Authorization transaction (authorize → callback), consent CSRF token | 15 min |
| Authorization code | 5 min, single use |
| Access token (FastMCP JWT, HS256) | Same as GitHub's `expires_in`. A GitHub **OAuth App** sends none and issues no refresh token, so the access token lives **1 year** and no refresh token is issued. A GitHub App with expiring user tokens gives 8 h, plus a refresh token. |
| Refresh token (FastMCP JWT) | Only when GitHub issues one. 1 year, rotated on every use; the old one stops working (`invalid_grant`, which FastMCP returns with HTTP 401). |
| GitHub verification cache | 300 s. The allowlist is checked after the cache on every request; only a revocation done on github.com can take up to 5 min to apply. |
| Registered clients (DCR) | Until storage is wiped |

## Rotating secrets

- `JWT_SIGNING_KEY`: every issued access and refresh token fails its signature check, and connectors must sign
  in again. Registered clients survive.
- `STORAGE_ENCRYPTION_KEY`: stored entries can no longer be decrypted and read as missing. Clients, token
  mappings and upstream GitHub tokens are all gone, so every connector must re-register and sign in. You can
  delete the old files in `oauth_dir`.
- `GITHUB_CLIENT_SECRET`: it also signs the consent cookies, so only sign-ins in progress (15 min) are affected.
- To cut off one user immediately, remove their id from `allowed_github_ids` and restart.

Values in `oauth_dir` are encrypted, but file names are not. File names include client ids, token JTIs,
refresh-token hashes and, for up to 5 minutes, authorization codes. The directory is created with mode 700.

## When upgrading fastmcp

Run `pytest tests/test_auth.py tests/test_middleware.py`, then check:
- `OAuthProxy._handle_idp_callback` still exchanges the code through `_create_upstream_oauth_client()`, and the
  `/auth/callback` route still exists.
- `GitHubTokenVerifier` still sets `sub` and `github_user_data` from `GET /user`.
- `load_access_token` is still what `verify_token` and the HTTP bearer backend call.
- Middleware dispatch: `on_message` still fires once per message, including `initialize` and notifications.
- The route list above, especially any new browser-facing path (firewall exceptions).
- Default token lifetimes (`fastmcp/server/auth/oauth_proxy/models.py`) and the `invalid_grant` status.
