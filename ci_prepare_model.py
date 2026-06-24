"""CI helper: find or download 640m.onnx (NudeNet v3.4 model).

Usage: python ci_prepare_model.py
Exit 0 = model ready at ./640m.onnx
Exit 1 = failed
"""

import os
import shutil
import site
import sys
from pathlib import Path


def find_onnx(root: Path) -> Path | None:
    """Return first .onnx file >= 1 MB under root, or None."""
    try:
        for p in root.rglob("*.onnx"):
            if p.is_file() and p.stat().st_size > 1_000_000:
                return p
    except PermissionError:
        pass
    return None


def main() -> None:
    dest = Path("640m.onnx")
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"640m.onnx already present ({dest.stat().st_size:,} bytes)")
        return

    import nudenet
    from nudenet import NudeDetector

    pkg_dir = Path(nudenet.__file__).parent
    print(f"nudenet package: {pkg_dir}")

    # 1. Check package dir before triggering download
    model = find_onnx(pkg_dir)
    if model:
        shutil.copy(model, dest)
        print(f"Copied from package dir: {model}")
        return

    # 2. Trigger auto-download via NudeDetector()
    print("Calling NudeDetector() to trigger model download...")
    try:
        _ = NudeDetector()
    except Exception as e:
        print(f"  (raised: {e})")

    # 3. Re-check package dir
    model = find_onnx(pkg_dir)
    if model:
        shutil.copy(model, dest)
        print(f"Copied after init: {model}")
        return

    # 4. Search common cache locations
    home = Path.home()
    search_dirs = [
        home / ".nudenet",
        home / "AppData" / "Roaming" / "nudenet",
        home / "AppData" / "Local" / "nudenet",
        home / ".cache" / "nudenet",
        pkg_dir.parent,  # site-packages root
    ]
    for sp in site.getsitepackages():
        search_dirs.append(Path(sp))

    for d in search_dirs:
        if d.exists():
            model = find_onnx(d)
            if model:
                shutil.copy(model, dest)
                print(f"Copied from {d}: {model}")
                return

    print("ERROR: 640m.onnx not found in any known location.", file=sys.stderr)
    print("Place 640m.onnx (NudeNet v3.4-weights) in the project root.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
