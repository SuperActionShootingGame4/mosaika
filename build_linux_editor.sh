#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

echo "=== Mosaika Linux editor build ==="

if [ ! -x "$PYTHON" ]; then
    echo ".venv が見つかりません。作成します..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

echo "依存関係を確認・インストール中..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$PYTHON" -m pip install pyinstaller

echo "mosaika-pre-csv-editor をビルド中..."
exec "$PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    --specpath "$SCRIPT_DIR/build" \
    --onefile \
    --windowed \
    --name mosaika-pre-csv-editor \
    "$SCRIPT_DIR/pre_csv_editor.py"
