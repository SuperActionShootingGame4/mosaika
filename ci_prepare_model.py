"""CI helper: find or download 640m.onnx (NudeNet v3.4 model).

nudenet を直接 import せず pip show でパッケージディレクトリを探す。
モデルが見つからない場合はサブプロセス経由で NudeDetector() を呼び出しダウンロードを試みる。

Usage: python ci_prepare_model.py
Exit 0 = model ready at ./640m.onnx
Exit 1 = failed
"""

import shutil
import subprocess
import sys
from pathlib import Path


def pip_package_dir(package: str) -> Path | None:
    """pip show でパッケージのインストール先ディレクトリを返す（import 不要）。"""
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", package],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"pip show {package}: not installed")
        return None
    for line in r.stdout.splitlines():
        if line.startswith("Location:"):
            loc = line.split(":", 1)[1].strip()
            return Path(loc) / package
    return None


def find_onnx(root: Path) -> Path | None:
    """1 MB 以上の .onnx ファイルを再帰検索して最初に見つかったものを返す。"""
    try:
        for p in root.rglob("*.onnx"):
            if p.is_file() and p.stat().st_size > 1_000_000:
                return p
    except (PermissionError, OSError):
        pass
    return None


def main() -> None:
    dest = Path("640m.onnx")
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"640m.onnx already present ({dest.stat().st_size:,} bytes)")
        return

    # 1. pip show でパッケージディレクトリを確認（import 不要）
    pkg_dir = pip_package_dir("nudenet")
    if pkg_dir:
        print(f"nudenet package dir: {pkg_dir}")
        model = find_onnx(pkg_dir)
        if model:
            shutil.copy(model, dest)
            print(f"Copied from package dir: {model}")
            return
    else:
        print("nudenet is not installed — attempting install...")
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "nudenet>=3.4.0"],
            check=False,
        )
        if ret.returncode != 0:
            print("ERROR: nudenet install failed", file=sys.stderr)
            sys.exit(1)
        pkg_dir = pip_package_dir("nudenet")
        if pkg_dir:
            model = find_onnx(pkg_dir)
            if model:
                shutil.copy(model, dest)
                print(f"Copied: {model}")
                return

    # 2. サブプロセスで NudeDetector() を呼び出し、モデルの自動ダウンロードを試みる
    print("Calling NudeDetector() in subprocess to trigger model download...")
    subprocess.run(
        [sys.executable, "-c",
         "from nudenet import NudeDetector; NudeDetector()"],
        check=False,
    )

    # 3. パッケージディレクトリを再確認
    if pkg_dir:
        model = find_onnx(pkg_dir)
        if model:
            shutil.copy(model, dest)
            print(f"Copied after download: {model}")
            return

    # 4. ホームディレクトリ等を広く探す
    home = Path.home()
    search_dirs = [
        home / ".nudenet",
        home / "AppData" / "Roaming" / "nudenet",
        home / "AppData" / "Local" / "nudenet",
        home / ".cache" / "nudenet",
    ]
    if pkg_dir:
        search_dirs.append(pkg_dir.parent)  # site-packages 直下も探す

    for d in search_dirs:
        if d.exists():
            model = find_onnx(d)
            if model:
                shutil.copy(model, dest)
                print(f"Copied from {d}: {model}")
                return

    print("ERROR: 640m.onnx not found in any known location.", file=sys.stderr)
    print("解決策: 640m.onnx (NudeNet v3.4-weights) をプロジェクトルートに手動で置いてください。", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
