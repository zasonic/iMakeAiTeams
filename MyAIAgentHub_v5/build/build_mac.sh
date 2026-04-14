#!/usr/bin/env bash
# build_mac.sh — Build MyAI Agent Hub for macOS
#
# Produces: dist/MyAI Agent Hub.app and (optionally) dist/MyAIAgentHub.dmg
#
# Requirements:
#   brew install create-dmg   (optional, for .dmg)
#   pip install pyinstaller

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing/updating dependencies..."
pip install -r app/requirements.txt --quiet
pip install pyinstaller --quiet

echo "==> Running PyInstaller..."
pyinstaller build/MyAIAgentHub.spec --noconfirm --clean

echo "==> Build output: dist/MyAI Agent Hub.app"

# Optional: wrap in a .dmg for distribution
if command -v create-dmg &>/dev/null; then
  echo "==> Creating .dmg installer..."
  create-dmg \
    --volname "MyAI Agent Hub" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "MyAI Agent Hub.app" 175 190 \
    --hide-extension "MyAI Agent Hub.app" \
    --app-drop-link 425 190 \
    "dist/MyAIAgentHub-1.0.0-mac.dmg" \
    "dist/"
  echo "==> Installer: dist/MyAIAgentHub-1.0.0-mac.dmg"
else
  echo ""
  echo "Tip: install create-dmg to produce a .dmg installer:"
  echo "  brew install create-dmg"
fi

echo ""
echo "Done. To run directly: open 'dist/MyAI Agent Hub.app'"
