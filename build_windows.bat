@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

set SCRIPT_DIR=%~dp0
set "PYINSTALLER_UPX_DIR="
set "SEVEN_ZIP="

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

if defined UPX_DIR (
    if exist "%UPX_DIR%\upx.exe" (
        set "PYINSTALLER_UPX_DIR=%UPX_DIR%"
        echo UPX_DIR を使用します: %UPX_DIR%
    )
)
if defined PYINSTALLER_UPX_DIR (
    echo UPX 圧縮を有効にします。
) else (
    where upx > nul 2>&1
    if errorlevel 1 (
        echo UPX が見つかりません。UPX 圧縮なしでビルドします。
    ) else (
        echo UPX が見つかりました。実行ファイルとDLLの圧縮を有効にします。
    )
)

if not exist "%SCRIPT_DIR%640m.onnx" (
    echo エラー: 640m.onnx が見つかりません。
    echo NudeNet のモデルファイルをスクリプトと同じフォルダに置いてください。
    pause
    exit /b 1
)

echo [1/2] mosaika-editor ^(GUI^) をビルド中...
if defined PYINSTALLER_UPX_DIR (
    "%PYTHON%" -m PyInstaller --noconfirm --clean --upx-dir "%PYINSTALLER_UPX_DIR%" "%SCRIPT_DIR%mosaika_editor.spec"
) else (
    "%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_editor.spec"
)
if errorlevel 1 ( echo ビルド失敗 & pause & exit /b 1 )

echo.
echo [2/2] mosaika-cli ^(CLI^) をビルド中...
if defined PYINSTALLER_UPX_DIR (
    "%PYTHON%" -m PyInstaller --noconfirm --clean --upx-dir "%PYINSTALLER_UPX_DIR%" "%SCRIPT_DIR%mosaika_cli.spec"
) else (
    "%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_cli.spec"
)
if errorlevel 1 ( echo ビルド失敗 & pause & exit /b 1 )

echo.
echo dist\mosaika\ にまとめています...
if exist "%SCRIPT_DIR%dist\mosaika" rmdir /s /q "%SCRIPT_DIR%dist\mosaika"
rename "%SCRIPT_DIR%dist\mosaika-editor" mosaika
copy "%SCRIPT_DIR%dist\mosaika-cli\mosaika-cli.exe" "%SCRIPT_DIR%dist\mosaika\" > nul
copy "%SCRIPT_DIR%640m.onnx" "%SCRIPT_DIR%dist\mosaika\" > nul

echo.
echo 配布アーカイブを作成中...
if exist "%SCRIPT_DIR%dist\mosaika.7z" del "%SCRIPT_DIR%dist\mosaika.7z"
for /f "delims=" %%i in ('where 7z 2^>nul') do if not defined SEVEN_ZIP set "SEVEN_ZIP=%%i"
if not defined SEVEN_ZIP if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVEN_ZIP=%ProgramFiles%\7-Zip\7z.exe"
if not defined SEVEN_ZIP if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVEN_ZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
if not defined SEVEN_ZIP (
    echo 7z が見つかりません。アーカイブ作成をスキップします。
    echo 7-Zip をインストールして PATH に追加すると dist\mosaika.7z を作成できます。
) else (
    pushd "%SCRIPT_DIR%dist"
    "%SEVEN_ZIP%" a -t7z -mx=9 -m0=lzma2 -ms=on "mosaika.7z" "mosaika\"
    if errorlevel 1 (
        popd
        echo 7z アーカイブの作成に失敗しました。
        pause
        exit /b 1
    )
    popd
)

echo.
echo === ビルド完了 ===
echo.
echo 出力先: dist\mosaika\
if exist "%SCRIPT_DIR%dist\mosaika.7z" echo 配布用: dist\mosaika.7z
echo   mosaika-editor.exe  ... GUI エディタ
echo   mosaika-cli.exe     ... CLI ツール
echo.
echo 注意:
echo   - ffmpeg を PATH に通すか dist\mosaika\ に ffmpeg.exe を置いてください
echo   - YOLO モデルは初回実行時に自動ダウンロードされます
echo.
pause
