@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

set SCRIPT_DIR=%~dp0

echo === Mosaika Windows ビルド ===
echo.

set PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    echo エラー: .venv が見つかりません。
    echo 先に以下を実行してください:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo   .venv\Scripts\pip install pyinstaller
    pause
    exit /b 1
)

echo CPU 版 PyTorch をインストール中（CUDA 版は不要で巨大なため）...
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

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
for /f "delims=" %%i in ('"%PYTHON%" -c "import site; print(site.getsitepackages()[0])"') do set SITE=%%i
xcopy /e /i /y "%SITE%\PyQt6" "%SCRIPT_DIR%dist\mosaika-editor\_internal\PyQt6" > nul

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
