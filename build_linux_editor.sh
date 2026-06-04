#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python3" -m PyInstaller \
    --noconfirm \
    --clean \
    --specpath "$SCRIPT_DIR/build" \
    --onefile \
    --windowed \
    --name mosaika-pre-csv-editor \
    "$SCRIPT_DIR/pre_csv_editor.py"
