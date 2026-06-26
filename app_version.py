"""Application version definition."""
import subprocess
from pathlib import Path

# バージョン本体。コミットハッシュ先頭4桁を付与した文字列が APP_VERSION になる。
_BASE_VERSION = "0.1.37"


def _commit_hash() -> str:
    """コミットハッシュ先頭7桁（大文字）を返す。取得できなければ空文字。"""
    # ビルド時に build スクリプトが生成する _build_version.py を優先（凍結配布物向け）。
    try:
        from _build_version import BUILD_HASH  # type: ignore
    except Exception:
        BUILD_HASH = ""
    if BUILD_HASH:
        return str(BUILD_HASH).upper()
    # ソース実行時は git から取得。
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().upper()
    except Exception:
        return ""


def _resolve_version() -> str:
    commit = _commit_hash()
    return f"{_BASE_VERSION}_{commit}" if commit else _BASE_VERSION


APP_VERSION = _resolve_version()
__version__ = APP_VERSION
