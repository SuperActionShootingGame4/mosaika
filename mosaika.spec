# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — editor と CLI を dist/mosaika/ にまとめてビルド

block_cipher = None

COMMON_HIDDENIMPORTS = [
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
]

COMMON_EXCLUDES = [
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
]

a_editor = Analysis(
    ['pre_csv_editor.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=COMMON_HIDDENIMPORTS + [
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        'mosaic_censor',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_cli = Analysis(
    ['mosaic_censor.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=COMMON_HIDDENIMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES + ['PyQt6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_editor = PYZ(a_editor.pure, a_editor.zipped_data, cipher=block_cipher)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

exe_editor = EXE(
    pyz_editor,
    a_editor.scripts,
    [],
    exclude_binaries=True,
    name='mosaika-editor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='mosaika-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_editor,
    a_editor.binaries,
    a_editor.zipfiles,
    a_editor.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mosaika',
)
