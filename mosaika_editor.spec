# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for mosaika-editor (GUI)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

a = Analysis(
    ['pre_csv_editor.py'],
    pathex=[],
    binaries=[],
    datas=collect_data_files('PyQt6'),
    hiddenimports=(
        collect_submodules('PyQt6')
        + [
            'nudenet',
            'onnxruntime',
            'onnxruntime.capi',
            'onnxruntime.capi.onnxruntime_pybind11_state',
            'cv2',
            'numpy',
            'tqdm',
            'mosaic_censor',
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
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter'],
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
    name='mosaika-editor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
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
    name='mosaika-editor',
)
