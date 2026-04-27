#!/usr/bin/env bash
# build_sidecar.sh — POSIX equivalent of build_sidecar.bat for CI.
#
# Outputs backend/dist/server/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --timeout=1000 --retries=20 --no-cache-dir --upgrade pip wheel setuptools
python -m pip install --timeout=1000 --retries=20 --no-cache-dir -r requirements.txt
python -m pip install --timeout=1000 --retries=20 --no-cache-dir "pyinstaller==6.11.1"

python -m PyInstaller pyinstaller.spec --noconfirm --clean
