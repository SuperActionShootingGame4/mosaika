@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

set SCRIPT_DIR=%~dp0

echo === Mosaika Windows ビルド ===
echo.

REM venv の Python を使う
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

REM PyInstaller のバージョン確認
"%PYTHON%" -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo PyInstaller が見つかりません。インストールします...
    "%PYTHON%" -m pip install pyinstaller
)

REM 640m.onnx の存在確認
if not exist "%SCRIPT_DIR%640m.onnx" (
    echo エラー: 640m.onnx が見つかりません。
    echo NudeNet のモデルファイルをスクリプトと同じフォルダに置いてください。
    echo 取得先: https://github.com/notAI-tech/NudeNet/releases ^(v3.4-weights^)
    pause
    exit /b 1
)

echo ビルド中 ^(editor + CLI → dist\mosaika\^)...
"%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika.spec"
if errorlevel 1 (
    echo ビルドに失敗しました。
    pause
    exit /b 1
)

echo.
echo モデルファイルを dist\mosaika\ にコピー中...
copy "%SCRIPT_DIR%640m.onnx" "%SCRIPT_DIR%dist\mosaika\" > nul

echo.
echo === ビルド完了 ===
echo.
echo 出力先: dist\mosaika\
echo   mosaika-editor.exe  ... GUI エディタ
echo   mosaika-cli.exe     ... CLI ツール
echo.
echo 注意事項:
echo   - ffmpeg を PATH に通すか dist\mosaika\ 内に ffmpeg.exe を置いてください
echo   - YOLO ポーズモデル ^(yolo11l-pose.pt など^) は初回実行時に自動ダウンロードされます
echo   - config.toml は mosaika-editor.exe と同じフォルダに自動生成されます
echo   - 640m.onnx は dist\mosaika\ にコピー済みです
echo.
pause
