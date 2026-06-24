# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for mosaika-cli (CLI)

block_cipher = None

a = Analysis(
    ['mosaic_censor.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'nudenet',
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'cv2',
        'numpy',
        'tqdm',
        'ultralytics',
        'ultralytics.engine',
        'ultralytics.engine.model',
        'ultralytics.engine.predictor',
        'ultralytics.engine.results',
        'ultralytics.nn',
        'ultralytics.nn.modules',
        'ultralytics.nn.tasks',
        'ultralytics.utils',
        'ultralytics.utils.ops',
        'ultralytics.utils.plotting',
        'ultralytics.utils.torch_utils',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.detect',
        'ultralytics.models.yolo.pose',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'PyQt6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mosaika-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='mosaika-cli',
)
