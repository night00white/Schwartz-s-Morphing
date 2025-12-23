@echo off
setlocal

echo ==========================================
echo      Starting Identity Shift App...
echo ==========================================

:: 1. Try global python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_EXE=python
    goto :INSTALL_DEPS
)

:: 2. Try Blender Python
set BLENDER_PYTHON="C:\Program Files\Blender Foundation\Blender 4.5\4.5\python\bin\python.exe"
if exist %BLENDER_PYTHON% (
    echo [INFO] Using Blender Python...
    set PYTHON_EXE=%BLENDER_PYTHON%
    goto :INSTALL_DEPS
)

:: 3. Try Houdini Python
set HOUDINI_PYTHON="C:\Program Files\Side Effects Software\Houdini 21.0.440\python311\python.exe"
if exist %HOUDINI_PYTHON% (
    echo [INFO] Using Houdini Python...
    set PYTHON_EXE=%HOUDINI_PYTHON%
    goto :INSTALL_DEPS
)

echo [ERROR] Python not found.
pause
exit /b

:INSTALL_DEPS
echo [INFO] Installing dependencies to ./site-packages...
if not exist "site-packages" mkdir "site-packages"

%PYTHON_EXE% -m pip install flask opencv-python-headless mediapipe==0.10.9 scipy numpy==1.26.4 --only-binary=:all: --target=.\site-packages --no-warn-script-location --upgrade

echo [INFO] Dependencies installed.

:START_APP
echo [INFO] Launching Server...
start http://127.0.0.1:5000
%PYTHON_EXE% app.py
pause
