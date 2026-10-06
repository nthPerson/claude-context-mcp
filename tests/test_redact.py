"""Tests for claude_context.redact.

Fake credentials are assembled at runtime so no literal secret-shaped string sits in the
source (keeps repository secret scanners quiet).
"""

from __future__ import annotations

import random
import time

import pytest

from claude_context.redact import REDACTION_KINDS, redact

ALNUM = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


def fake(prefix: str, n: int, alphabet: str = ALNUM) -> str:
    return prefix + (alphabet * 4)[:n]


PEM = "\n".join(
    [
        "-----BEGIN RSA " + "PRIVATE KEY-----",
        "MIIEowIBAAKCAQEAfakefakefakefakefakefakefakefakefake",
        "Zm9vYmFyYmF6cXV4ZmFrZWZha2VmYWtlZmFrZWZha2U=",
        "-----END RSA " + "PRIVATE KEY-----",
    ]
)

POSITIVE = [
    # (input, kind, text that must disappear)
    (f"key is {fake('sk-' + 'ant-api03-', 40)} ok", "anthropic_key", fake("sk-ant-api03-", 40)),
    (f"export K={fake('sk-' + 'proj-', 40)}", "openai_key", fake("sk-proj-", 40)),
    (f"legacy {fake('sk-', 48)}", "openai_key", fake("sk-", 48)),
    (f"gh auth {fake('gh' + 'p_', 36)}", "github_token", fake("ghp_", 36)),
    (f"{fake('github' + '_pat_', 60)}", "github_token", fake("github_pat_", 60)),
    (f"id {'AK' + 'IA'}{'Q' * 8}{'7' * 8} here", "aws_access_key", "QQQQQQQQ7777"),
    (f"id {'AS' + 'IA'}{'Z' * 8}{'3' * 8}", "aws_access_key", "ZZZZZZZZ3333"),
    (f"aws_secret_access_key = {fake('', 40, 'wJalrXUtn/K7MDENG+bPxRfiCY')}", "aws_secret", "wJalrXUtn"),
    (f"url?key={fake('AI' + 'za', 35)}", "google_api_key", fake("AIza", 35)),
    (f"slack {'xo' + 'xb'}-1234567890-0987654321-{fake('', 24)}", "slack_token", "1234567890-0987654321"),
    (f"stripe {'sk' + '_live_'}{fake('', 24)}", "stripe_key", fake("", 24)),
    (f"stripe {'rk' + '_live_'}{fake('', 24)}", "stripe_key", fake("", 24)),
    (
        "jwt " + "ey" + "JhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlZmFrZQ",
        "jwt",
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
    ),
    (f"prefix\n{PEM}\nsuffix", "private_key", "MIIEowIBAAKCAQEA"),
    ("curl -H 'Authorization: Bearer abc123def456ghi'", "bearer", "abc123def456ghi"),
    ('{"Authorization": "Bearer opaque-value-77"}', "bearer", "opaque-value-77"),
    ("header: Bearer 0123456789abcdefABCDEF.xyz", "bearer", "0123456789abcdefABCDEF"),
    ("DB_PASSWORD=hunter22", "password", "hunter22"),
    ('{"password": "correct horse"}', None, None),  # spaces: not a single token
    ('{"password": "correct-horse-battery"}', "password", "correct-horse-battery"),
    ("passwd: 'Tr0ub4dor'", "password", "Tr0ub4dor"),
    ('client_secret: "s3cr3tValue"', "secret", "s3cr3tValue"),
    ("APP_SECRET=abc123xyz", "secret", "abc123xyz"),
    ("GITHUB_TOKEN=abc123def456", "token", "abc123def456"),
    ("accessToken: Zm9vYmFy1234", "token", "Zm9vYmFy1234"),
    ("x-api-key: k3y-v4lue-001", "api_key", "k3y-v4lue-001"),
    ('apiKey = "abcd1234efgh"', "api_key", "abcd1234efgh"),
    ('\\"password\\": \\"hunter22x\\"', "password", "hunter22x"),
    ("https://host/cb?code=1&token=abc123def456&x=1", "token", "abc123def456"),
    (f"SIGNING_KEY={fake('', 48, 'Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MGFiY2RlZg+/')}", "high_entropy", "Zm9vYmFy"),
    (f'{{"sig": "{fake("", 44, "q9X2mK7pL4vR8sT1wY5zB3nC6hJ0")}"}}', "high_entropy", "q9X2mK7pL4"),
]


@pytest.mark.parametrize(("text", "kind", "gone"), POSITIVE)
def test_positive(text: str, kind: str | None, gone: str | None) -> None:
    out = redact(text)
    if kind is None:
        assert out == text
        return
    assert f"[REDACTED:{kind}]" in out
    assert gone not in out


def test_keyed_rule_keeps_key_name_and_quotes() -> None:
    assert redact("DB_PASSWORD=hunter22") == "DB_PASSWORD=[REDACTED:password]"
    assert redact('{"password": "hunter22"}') == '{"password": "[REDACTED:password]"}'
    assert redact("Authorization: Bearer abc123def456") == "Authorization: Bearer [REDACTED:bearer]"


def test_pem_block_redacted_whole() -> None:
    assert redact(f"a\n{PEM}\nb") == "a\n[REDACTED:private_key]\nb"
    truncated = PEM.split("-----END")[0]
    assert "MIIE" not in redact(truncated)


NEGATIVE = [
    "",
    "the token: is used for authentication",
    "the secret: understanding the system",
    "Bearer tokens are sent in a header",
    'password=""',
    "password: null",
    "token = None",
    "secret: true",
    "password=<token>",
    "api_key=${API_KEY}",
    "TOKEN=$GITHUB_TOKEN",
    "password=...",
    "apikey=xxxxxxxx",
    "password = '********'",
    "password: Optional[str] = None",
    "api_key=self.api_key",
    "token = os.environ['X_TOKEN']",
    "secret = load_secret()",
    "max_tokens=100000",
    '"input_tokens": 123456',
    "tokenizer=bert-base-uncased",
    "token_type: bearer_access",
    "password_hash: pbkdf2-sha256",
    'tokenUrl="/oauth/authorize"',
    "TOKEN_EXPIRE_MINUTES=30",
    'token: "under_load"',
    "src/auth/password_reset.py:12:def reset():",
    "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?password required}",
    "/home/user/projects/some-long-directory-name/with/many/parts/file_0001.py",
    'path="/home/user/projects/some-long-directory-name/with/many/parts/file_0001.py"',
    "see https://example.com/a/very/long/path/segment/0123456789abcdefABCDEF/more",
    "commit=da39a3ee5e6b4b0d3255bfef95601890afd80709",
    'sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"',
    'id: "123e4567-e89b-12d3-a456-426614174000"',
    '"tool_use_id": "toolu_01AbCdEfGhIjKlMnOpQrStUvWxYz0123456789XyZ"',
    '"id": "msg_01AbCdEfGhIjKlMnOpQrStUvWxYz0123456789XyZ"',
    "request=req_011CAbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    '"next_cursor": "Q2FjaGVkQ3Vyc29yMTIzNDU2Nzg5MGFiY2RlZmdoaWprbG1u"',
    "name=supercalifragilisticexpialidociousandmorewordsbesides",
    "test_id=tests/test_parser.py::TestSomething::test_handles_long_names_2",
    "Supercalifragilisticexpialidocious is a long ordinary word in a sentence.",
    "contact <CAFakeMessageIdAbCdEfGh0123456789XyZaBcDeFgHiJk@mail.example.com>",
]


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative(text: str) -> None:
    assert redact(text) == text


@pytest.mark.parametrize("text", [p[0] for p in POSITIVE] + NEGATIVE)
def test_idempotent(text: str) -> None:
    once = redact(text)
    assert redact(once) == once


def test_kinds_cover_all_markers() -> None:
    produced = {k for _, k, _ in POSITIVE if k}
    assert produced <= set(REDACTION_KINDS)
    assert produced == set(REDACTION_KINDS)


def test_performance_on_large_mixed_text() -> None:
    rng = random.Random(7)
    pieces = [
        "ordinary prose about tokens, passwords and secrets in a design document. ",
        '{"type": "tool_result", "tool_use_id": "toolu_01AbCdEfGhIjKlMnOpQrStUvWx", "content": "ok"}\n',
        "/home/user/project/src/module/file.py:123: def function_name(arg): return arg\n",
        "DB_PASSWORD=hunter22 ",
        f"export K={fake('sk-' + 'proj-', 40)}\n",
        "commit da39a3ee5e6b4b0d3255bfef95601890afd80709\n",
        "x" * 300 + "\n",
        "=" * 200 + ":" * 200 + '"' * 200 + "\n",
    ]
    text = "".join(rng.choice(pieces) for _ in range(30_000))
    while len(text) < 2_000_000:
        text += text
    text = text[:2_000_000]
    start = time.perf_counter()
    out = redact(text)
    elapsed = time.perf_counter() - start
    assert "[REDACTED:openai_key]" in out
    assert elapsed < 2.0, f"redact took {elapsed:.2f}s on 2 MB"
