#!/bin/bash
# pre_csv_editor.py をvenv環境で実行するラッパー
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/pre_csv_editor.py" "$@"
