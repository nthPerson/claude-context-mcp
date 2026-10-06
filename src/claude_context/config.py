"""Configuration and secrets loading.

Non-secret settings live in ``~/.config/claude-context/config.toml``; secrets live in
``secrets.env`` next to it (mode 600) and are normally injected by systemd's
``EnvironmentFile=``. Nothing in this module logs or prints a secret value.
"""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_ENV = "CLAUDE_CONTEXT_CONFIG"
DATA_ENV = "CLAUDE_CONTEXT_DATA"
SECRET_KEYS = ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "JWT_SIGNING_KEY", "STORAGE_ENCRYPTION_KEY")


def default_config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "claude-context"


def default_config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV) or default_config_dir() / "config.toml")


def default_data_dir() -> Path:
    return Path(os.environ.get(DATA_ENV) or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "claude-context")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RootConfig(_Model):
    """One directory tree of Claude Code project folders."""

    label: str
    path: Path
    # "syncthing:<folder-id>" attributes each file to the device that last modified it;
    # any other value is used verbatim as the machine name.
    machine: str = "unknown"
    writable: bool = False
    # "claude-code": a live ~/.claude/projects tree. "stversions": a Syncthing versions
    # folder; sessions deleted upstream are recovered from it into the archive.
    kind: Literal["claude-code", "stversions"] = "claude-code"

    @field_validator("path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def syncthing_folder(self) -> str | None:
        return self.machine.split(":", 1)[1] if self.machine.startswith("syncthing:") else None


class ProjectConfig(_Model):
    keys: list[str]  # first key is the primary one (where new memories are written)
    display: str = ""


class ImportsConfig(_Model):
    claudeai_drop: Path | None = None

    @field_validator("claudeai_drop")
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v else v


class EmbeddingConfig(_Model):
    enabled: bool = True
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    threads: int = 8
    batch: int = 64
    embed_automated: bool = False


class SyncthingConfig(_Model):
    api: str = "http://127.0.0.1:8384"
    config_xml: Path = Path("~/.local/state/syncthing/config.xml")

    @field_validator("config_xml")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()


class Config(_Model):
    owner_name: str = "the owner"  # used in the server instructions shown to Claude
    base_url: str = ""  # public https origin, e.g. "https://claude-context.example.com"
    bind: str = "127.0.0.1:8790"
    allowed_github_ids: list[int] = Field(default_factory=list)
    allowed_mcp_source_cidrs: list[str] = Field(default_factory=list)  # empty = no CIDR check
    retention_days: int = 365
    default_max_chars: int = 20_000
    max_max_chars: int = 100_000
    rate_limit_per_minute: int = 120
    timezone: str = "UTC"  # IANA name used when rendering times
    data_dir: Path = Field(default_factory=default_data_dir)

    interactive_entrypoints: list[str] = ["cli", "claude-vscode", "claude-desktop", "claude-desktop-code"]
    automated_entrypoints: list[str] = ["sdk-py", "sdk-cli", "sdk-ts"]

    roots: list[RootConfig] = Field(default_factory=list)
    projects: dict[str, ProjectConfig] = Field(default_factory=dict)
    claudeai_project_map: dict[str, str] = Field(default_factory=dict)  # claude.ai project name -> alias
    imports: ImportsConfig = Field(default_factory=ImportsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    syncthing: SyncthingConfig = Field(default_factory=SyncthingConfig)

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    # --- derived locations -------------------------------------------------------------
    @property
    def index_db(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def audit_db(self) -> Path:
        return self.data_dir / "audit.db"

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "queue"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def oauth_dir(self) -> Path:
        return self.data_dir / "oauth"

    @property
    def imports_processed_dir(self) -> Path:
        return self.data_dir / "imports" / "processed"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def status_file(self) -> Path:
        return self.data_dir / "status.json"

    @property
    def host(self) -> str:
        return self.bind.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.bind.rsplit(":", 1)[1])

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    def root(self, label: str) -> RootConfig | None:
        return next((r for r in self.roots if r.label == label), None)

    @property
    def writable_root(self) -> RootConfig | None:
        return next((r for r in self.roots if r.writable and r.kind == "claude-code"), None)

    def ensure_dirs(self) -> None:
        """Create the data directories (private to the user)."""
        for d in (self.data_dir, self.archive_dir, self.queue_dir, self.models_dir, self.oauth_dir,
                  self.imports_processed_dir, self.backups_dir):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)


class ConfigError(Exception):
    pass


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path} (copy config.example.toml there)") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"config file {path} is not valid TOML: {e}") from e
    try:
        return Config.model_validate(raw)
    except ValueError as e:
        raise ConfigError(f"config file {path} is invalid:\n{e}") from e


# --- secrets ---------------------------------------------------------------------------

def default_secrets_path() -> Path:
    return default_config_path().parent / "secrets.env"


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (same subset systemd's EnvironmentFile accepts)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def load_secrets(path: Path | None = None) -> dict[str, str]:
    """Secrets from the process environment, falling back to ``secrets.env``.

    Returns only the keys in SECRET_KEYS that are present and non-empty.
    """
    found = {k: os.environ[k] for k in SECRET_KEYS if os.environ.get(k)}
    if len(found) < len(SECRET_KEYS):
        path = path or default_secrets_path()
        if path.is_file():
            if file_mode(path) & 0o077:
                raise ConfigError(f"{path} must not be readable by group/others (chmod 600)")
            for k, v in parse_env_file(path).items():
                if k in SECRET_KEYS and v and k not in found:
                    found[k] = v
    return found
