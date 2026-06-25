@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

set SCRIPT_DIR=%~dp0

echo === Mosaika Windows ビルド ===
echo.

set PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo .venv が見つかりません。作成します...
    python -m venv "%SCRIPT_DIR%.venv"
    if errorlevel 1 (
        echo エラー: .venv の作成に失敗しました。
        echo Python がインストールされ、PATH に登録されているか確認してください。
        pause
        exit /b 1
    )
)

echo 依存関係を確認・インストール中...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 ( echo pip の更新に失敗しました & pause & exit /b 1 )

"%PYTHON%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 ( echo requirements.txt のインストールに失敗しました & pause & exit /b 1 )

"%PYTHON%" -m pip install pyinstaller
if errorlevel 1 ( echo pyinstaller のインストールに失敗しました & pause & exit /b 1 )

echo CPU 版 PyTorch をインストール中（CUDA 版は不要で巨大なため）...
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 ( echo PyTorch のインストールに失敗しました & pause & exit /b 1 )

"%PYTHON%" -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo PyInstaller が見つかりません。インストールします...
    "%PYTHON%" -m pip install pyinstaller
)

if not exist "%SCRIPT_DIR%640m.onnx" (
    echo エラー: 640m.onnx が見つかりません。
    echo NudeNet のモデルファイルをスクリプトと同じフォルダに置いてください。
    pause
    exit /b 1
)

echo [1/2] mosaika-editor ^(GUI^) をビルド中...
"%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_editor.spec"
if errorlevel 1 ( echo ビルド失敗 & pause & exit /b 1 )

echo PyQt6 を _internal\ にコピー中...
for /f "delims=" %%i in ('"%PYTHON%" -c "import pathlib, PyQt6; print(pathlib.Path(PyQt6.__path__[0]))"') do set PYQT6=%%i
if not exist "%PYQT6%" (
    echo エラー: PyQt6 パッケージが見つかりません: %PYQT6%
    pause
    exit /b 1
)
if exist "%SCRIPT_DIR%dist\mosaika-editor\_internal\PyQt6" rmdir /s /q "%SCRIPT_DIR%dist\mosaika-editor\_internal\PyQt6"
xcopy /e /i /y "%PYQT6%" "%SCRIPT_DIR%dist\mosaika-editor\_internal\PyQt6" > nul

echo.
echo [2/2] mosaika-cli ^(CLI^) をビルド中...
"%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_cli.spec"
if errorlevel 1 ( echo ビルド失敗 & pause & exit /b 1 )

echo.
echo dist\mosaika\ にまとめています...
if exist "%SCRIPT_DIR%dist\mosaika" rmdir /s /q "%SCRIPT_DIR%dist\mosaika"
rename "%SCRIPT_DIR%dist\mosaika-editor" mosaika
copy "%SCRIPT_DIR%dist\mosaika-cli\mosaika-cli.exe" "%SCRIPT_DIR%dist\mosaika\" > nul
copy "%SCRIPT_DIR%640m.onnx" "%SCRIPT_DIR%dist\mosaika\" > nul

echo.
echo === ビルド完了 ===
echo.
echo 出力先: dist\mosaika\
echo   mosaika-editor.exe  ... GUI エディタ
echo   mosaika-cli.exe     ... CLI ツール
echo.
echo 注意:
echo   - ffmpeg を PATH に通すか dist\mosaika\ に ffmpeg.exe を置いてください
echo   - YOLO モデルは初回実行時に自動ダウンロードされます
echo.
pause
