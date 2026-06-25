@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYINSTALLER_UPX_DIR="
set "SEVEN_ZIP="

echo === Mosaika Windows build ===
echo.

set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo .venv not found. Creating virtual environment...
    python -m venv "%SCRIPT_DIR%.venv"
    if errorlevel 1 (
        echo ERROR: Failed to create .venv.
        echo Check that Python is installed and available on PATH.
        pause
        exit /b 1
    )
)

echo Checking and installing dependencies...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

"%PYTHON%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt.
    pause
    exit /b 1
)

"%PYTHON%" -m pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install pyinstaller.
    pause
    exit /b 1
)

echo Installing CPU PyTorch...
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

"%PYTHON%" -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    "%PYTHON%" -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

if defined UPX_DIR (
    if exist "%UPX_DIR%\upx.exe" (
        set "PYINSTALLER_UPX_DIR=%UPX_DIR%"
        echo Using UPX_DIR: %UPX_DIR%
    )
)

if defined PYINSTALLER_UPX_DIR (
    echo UPX compression enabled.
) else (
    where upx > nul 2>&1
    if errorlevel 1 (
        echo UPX not found. Building without UPX compression.
    ) else (
        echo UPX found on PATH. PyInstaller may use UPX.
    )
)

if not exist "%SCRIPT_DIR%640m.onnx" (
    echo ERROR: 640m.onnx not found.
    echo Put the NudeNet model file in the repository root.
    pause
    exit /b 1
)

echo [1/2] Building mosaika-editor GUI...
if defined PYINSTALLER_UPX_DIR (
    "%PYTHON%" -m PyInstaller --noconfirm --clean --upx-dir "%PYINSTALLER_UPX_DIR%" "%SCRIPT_DIR%mosaika_editor.spec"
) else (
    "%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_editor.spec"
)
if errorlevel 1 (
    echo ERROR: mosaika-editor build failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Building mosaika-cli CLI...
if defined PYINSTALLER_UPX_DIR (
    "%PYTHON%" -m PyInstaller --noconfirm --clean --upx-dir "%PYINSTALLER_UPX_DIR%" "%SCRIPT_DIR%mosaika_cli.spec"
) else (
    "%PYTHON%" -m PyInstaller --noconfirm --clean "%SCRIPT_DIR%mosaika_cli.spec"
)
if errorlevel 1 (
    echo ERROR: mosaika-cli build failed.
    pause
    exit /b 1
)

echo.
echo Combining into dist\mosaika...
if exist "%SCRIPT_DIR%dist\mosaika" rmdir /s /q "%SCRIPT_DIR%dist\mosaika"
rename "%SCRIPT_DIR%dist\mosaika-editor" mosaika
if errorlevel 1 (
    echo ERROR: Failed to rename dist\mosaika-editor.
    pause
    exit /b 1
)
copy "%SCRIPT_DIR%dist\mosaika-cli\mosaika-cli.exe" "%SCRIPT_DIR%dist\mosaika\" > nul
if errorlevel 1 (
    echo ERROR: Failed to copy mosaika-cli.exe.
    pause
    exit /b 1
)
copy "%SCRIPT_DIR%640m.onnx" "%SCRIPT_DIR%dist\mosaika\" > nul
if errorlevel 1 (
    echo ERROR: Failed to copy 640m.onnx.
    pause
    exit /b 1
)

echo.
echo Creating distribution archive...
if exist "%SCRIPT_DIR%dist\mosaika.7z" del "%SCRIPT_DIR%dist\mosaika.7z"
for /f "delims=" %%i in ('where 7z 2^>nul') do if not defined SEVEN_ZIP set "SEVEN_ZIP=%%i"
if not defined SEVEN_ZIP if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVEN_ZIP=%ProgramFiles%\7-Zip\7z.exe"
if not defined SEVEN_ZIP if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVEN_ZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
if not defined SEVEN_ZIP (
    echo 7z not found. Skipping archive creation.
    echo Install 7-Zip and add it to PATH to create dist\mosaika.7z.
) else (
    pushd "%SCRIPT_DIR%dist"
    "%SEVEN_ZIP%" a -t7z -mx=9 -m0=lzma2 -ms=on "mosaika.7z" "mosaika\"
    if errorlevel 1 (
        popd
        echo ERROR: Failed to create 7z archive.
        pause
        exit /b 1
    )
    popd
)

echo.
echo === Build complete ===
echo.
echo Output: dist\mosaika\
if exist "%SCRIPT_DIR%dist\mosaika.7z" echo Archive: dist\mosaika.7z
echo   mosaika-editor.exe ... GUI editor
echo   mosaika-cli.exe    ... CLI tool
echo.
echo Notes:
echo   - Put ffmpeg on PATH or place ffmpeg.exe in dist\mosaika\
echo   - YOLO model is downloaded automatically on first run
echo.
pause
