"""Secret redaction applied at ingest and again on every output.

``redact`` replaces anything that looks like a credential with ``[REDACTED:<kind>]``. It is
idempotent, linear-time on large inputs, and deliberately conservative about ordinary text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

REDACTION_KINDS: tuple[str, ...] = (
    "private_key",
    "anthropic_key",
    "openai_key",
    "github_token",
    "aws_access_key",
    "aws_secret",
    "google_api_key",
    "slack_token",
    "stripe_key",
    "jwt",
    "bearer",
    "password",
    "secret",
    "token",
    "api_key",
    "high_entropy",
)

# Patterns start with a literal so the regex engine can skip ahead quickly; the
# "not preceded by a token character" check is a lookbehind placed after the literal.
_W = "[A-Za-z0-9_-]"
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
            r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY-----|[A-Za-z0-9+/=\s\\:,-]*)",
            re.DOTALL,
        ),
    ),
    ("anthropic_key", re.compile(rf"sk-ant-(?<!{_W}sk-ant-){_W}{{20,}}")),
    (
        "openai_key",
        re.compile(rf"sk-(?<!{_W}sk-)(?:proj-|svcacct-|admin-)?(?={_W}{{0,200}}\d){_W}{{20,}}"),
    ),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{22,}")),
    ("aws_access_key", re.compile(r"A[KS]IA(?<![A-Z0-9]A[KS]IA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("google_api_key", re.compile(rf"AIza{_W}{{30,}}")),
    ("slack_token", re.compile(r"xox[abpr]-[0-9A-Za-z-]{10,}")),
    ("stripe_key", re.compile(r"[sr]k_live_[0-9A-Za-z]{10,}")),
    ("jwt", re.compile(rf"eyJ{_W}{{8,}}\.{_W}{{8,}}\.{_W}{{8,}}")),
)

_BEARER_RE = re.compile(r"(?P<pre>(?i:bearer)[ \t]+)(?P<v>[A-Za-z0-9._~+/-]{6,}=*)")
_AUTH_TAIL_RE = re.compile(r"(?i:authorization)\\?[\"']?[ \t]*[:=][ \t]*\\?[\"']?\Z")

# key<sep>value where the key name contains a credential word, e.g. DB_PASSWORD=...,
# "client_secret": "...", accessToken: ... The value alone is replaced.
_CAMEL = r"(?<=[a-z])(?=[A-Z])"
_KEYED_RE = re.compile(
    r"(?P<key>(?i:password|passwd|secret|token|api[_-]?key|apikey)"
    rf"(?:(?:[_-]|{_CAMEL})[A-Za-z0-9_-]*)?)"
    r"\\?[\"']?[ \t]*(?P<sep>:=|=>|=|:)[ \t]*"
    r"(?:(?P<q>\\?[\"'])(?P<qv>[^\s\"'\\]*)(?P=q)"
    r"|(?P<uv>[^\s\"'`,;&<>(){}\[\]=][^\s\"'`,;&<>(){}\[\]]*))"
)

# Long opaque values: in a key=value / key: value / quoted-string position, followed by
# neither more token characters nor "@" (an email local part).
_ENTROPY_RE = re.compile(
    r"(?P<pre>[=:][ \t]*\\?[\"']?|[\"'])(?P<v>[A-Za-z0-9+/_=-]{40,})(?![A-Za-z0-9+/_=.@-])"
)
_ENTROPY_KEY_RE = re.compile(r"(?<![\w-])([A-Za-z_][\w-]+)\\?[\"']?[ \t]*\Z")
_ENTROPY_SKIP_PREFIXES = ("toolu_", "srvtoolu_", "msg_", "req_", "/")
# snake_case, kebab-case and CONSTANT_CASE names
_IDENT_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+){2,}|[A-Z0-9]+(?:_[A-Z0-9]+){2,}")

# Key-name parts that mark metadata rather than a credential, e.g. token_type,
# password_hash, secret_name, TOKEN_EXPIRE_MINUTES, tokenUrl, "id", "next_cursor".
_NON_SECRET_KEY_PARTS = frozenset(
    "hash hashed hashing expires expire expiry expiration url uri href endpoint payload type "
    "kind file path dir name id ids eid uuid guid oid ref etag sha digest checksum cursor "
    "version length len count limit field form input label hint text message error header "
    "policy ttl minutes seconds hours days reset budget usage use format prefix".split()
)
_KEY_PART_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z0-9]+(?![a-z])")
_NON_SECRET_WORDS = frozenset(
    "null none nil true false undefined bearer basic string required optional example "
    "placeholder changeme redacted hidden masked".split()
)
_KEY_WORDS = ("password", "passwd", "secret", "token", "apikey", "api_key", "your_", "your-")
_DOTTED_IDENT_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_NUMERIC_RE = re.compile(r"[\d.,_-]+")
_WORDS_RE = re.compile(r"[A-Za-z]+(?:[_-][A-Za-z]+)*")


def _entropy(s: str) -> float:
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _is_placeholder(v: str) -> bool:
    """True for values that are clearly not secrets (short, empty-ish, templated, numeric)."""
    low = v.lower()
    return (
        len(v) < 6
        or v.startswith(("[REDACTED", "$", "%", "<", "{", "*", "?", "/"))
        or low in _NON_SECRET_WORDS
        or "..." in v
        or "**" in v
        or not set(low) - set("x*._-")
        or _NUMERIC_RE.fullmatch(v) is not None
        or any(w in low for w in _KEY_WORDS)
    )


def _keyed_kind(key: str) -> str:
    k = key.lower()
    if k.startswith("secret_access_key"):
        return "aws_secret"
    if k.startswith("pass"):
        return "password"
    if k.startswith("secret"):
        return "secret"
    if k.startswith("token"):
        return "token"
    return "api_key"


def _metadata_key(key: str, skip_first: bool) -> bool:
    parts = _KEY_PART_RE.findall(key)[1 if skip_first else 0 :]
    return any(p.lower() in _NON_SECRET_KEY_PARTS for p in parts)


def _sub_keyed(m: re.Match[str]) -> str:
    s, key = m.string, m.group("key")
    prev = s[m.start() - 1] if m.start() else ""
    if prev.isalnum() and not (prev.islower() and key[0].isupper()):
        return m.group(0)  # keyword inside a longer word, e.g. "mytoken"
    if _metadata_key(key, skip_first=True):
        return m.group(0)
    quoted = m.group("q") is not None
    value = m.group("qv") if quoted else m.group("uv")
    if not quoted:
        if s[m.end() : m.end() + 1] in ("(", "[") or _DOTTED_IDENT_RE.fullmatch(value):
            return m.group(0)  # code (call, subscript, attribute), not a literal
        value = value.rstrip(".:")
        # Unquoted "key: word" is usually prose; only redact values that look token-like.
        if m.group("sep") == ":" and value.isalpha() and (value.islower() or value.istitle()):
            return m.group(0)
    kind = _keyed_kind(key)
    if _is_placeholder(value) or (kind == "token" and _WORDS_RE.fullmatch(value)):
        return m.group(0)  # a real token has digits; "token: under_load" is vocabulary
    start = m.start("qv") if quoted else m.start("uv")
    end = start + len(value)
    return s[m.start() : start] + f"[REDACTED:{kind}]" + s[end : m.end()]


def _sub_bearer(m: re.Match[str]) -> str:
    v = m.group("v")
    is_header = _AUTH_TAIL_RE.search(m.string, max(0, m.start() - 40), m.start()) is not None
    if _is_placeholder(v) or (not is_header and (len(v) < 20 or not any(c.isdigit() for c in v))):
        return m.group(0)
    return m.group("pre") + "[REDACTED:bearer]"


def _sub_entropy(m: re.Match[str]) -> str:
    v, pre = m.group("v"), m.group("pre")
    if (
        v.startswith(_ENTROPY_SKIP_PREFIXES)
        or not any(c.isdigit() for c in v)
        or not any(c.isalpha() for c in v)
        or _IDENT_RE.fullmatch(v)
        or _entropy(v) <= 4.0
    ):
        return m.group(0)
    if pre[0] in "=:":  # needs a plausible key name that is not an id/cursor/path field
        key = _ENTROPY_KEY_RE.search(m.string, max(0, m.start() - 64), m.start())
        if key is None or _metadata_key(key[1], skip_first=False):
            return m.group(0)
    return pre + "[REDACTED:high_entropy]"


_PASSES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str] | str], ...] = (
    *((p, f"[REDACTED:{kind}]") for kind, p in _TOKEN_PATTERNS),
    (_BEARER_RE, _sub_bearer),
    (_KEYED_RE, _sub_keyed),
    (_ENTROPY_RE, _sub_entropy),
)


def redact(text: str) -> str:
    """Return ``text`` with secrets replaced by ``[REDACTED:<kind>]`` markers."""
    if not text:
        return text
    for pattern, repl in _PASSES:
        text = pattern.sub(repl, text)
    return text
