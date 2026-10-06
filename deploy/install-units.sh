#!/usr/bin/env bash
# Render the systemd unit templates for this machine and install them.
#   deploy/install-units.sh            render into deploy/rendered/ and show what would be installed
#   deploy/install-units.sh --install  also copy to /etc/systemd/system and daemon-reload (uses sudo)
# Nothing is enabled or started: do that yourself, e.g.
#   sudo systemctl enable --now claude-context-indexer claude-context-server claude-context-maintenance.timer
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user="$(id -un)"
config="${CLAUDE_CONTEXT_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/claude-context/config.toml}"
out="$repo/deploy/rendered"

# Ask the application for the paths it needs write access to (from the config file).
eval "$(CLAUDE_CONTEXT_CONFIG="$config" "$repo/.venv/bin/claude-context" unit-paths)"

mkdir -p "$out"
for template in "$repo"/deploy/systemd/*.in; do
  name="$(basename "${template%.in}")"
  sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" -e "s|@CONFIG@|$config|g" \
      -e "s|@CONFIG_DIR@|$(dirname "$config")|g" -e "s|@DATA_DIR@|$DATA_DIR|g" \
      -e "s|@WRITABLE_ROOTS@|$WRITABLE_ROOTS|g" -e "s|@DROP_DIR@|$DROP_DIR|g" "$template" > "$out/$name"
done
cp "$repo"/deploy/systemd/*.timer "$repo"/deploy/systemd/cloudflared-claude-context.service "$out/"
echo "Rendered units in $out:"; ls -1 "$out"

if [[ "${1:-}" == "--install" ]]; then
  sudo install -m 644 "$out"/*.service "$out"/*.timer /etc/systemd/system/
  sudo systemctl daemon-reload
  echo "Installed. Nothing was enabled or started."
fi
