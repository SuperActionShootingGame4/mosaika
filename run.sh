#!/bin/bash
# mosaic_censor.py をvenv環境で実行するラッパー
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python3" "$SCRIPT_DIR/mosaic_censor.py" "$@"
