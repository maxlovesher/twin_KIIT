@echo off
setlocal EnableExtensions
title GlucoTwin
cd /d "%~dp0"

echo.
echo   GlucoTwin - hybrid digital twin for Type 2 Diabetes
echo   ---------------------------------------------------
echo.

rem ---- 1. Find Python 3.10+ (prefer the "py" launcher; plain "python" may be the Microsoft Store stub)
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   [!] Python 3.10 or newer was not found.
    echo.
    echo       Install Python 3.12 from https://www.python.org/downloads/
    echo       and tick "Add python.exe to PATH" on the first installer screen.
    echo       Then double-click start.bat again.
    echo.
    pause
    exit /b 1
)

rem ---- 2. Create a private environment in .venv (first run only)
if not exist ".venv\Scripts\python.exe" (
    echo   [1/3] Creating a private Python environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   [!] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

rem ---- 3. Install / update packages when requirements.txt has changed
fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>&1
if errorlevel 1 (
    echo   [2/3] Installing packages - first run takes a few minutes ...
    "%VPY%" -m pip install --disable-pip-version-check -q --upgrade pip
    "%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [!] Package install failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    copy /y requirements.txt ".venv\installed-requirements.txt" >nul
) else (
    echo   [2/3] Packages already installed.
)

rem ---- 4. Launch the dashboard (opens in your browser)
echo   [3/3] Starting the dashboard at http://localhost:8501
echo         The first page load takes ~30 s while the patient twins compile.
echo         Close this window (or press Ctrl+C) to stop.
echo.
"%VPY%" -m streamlit run dashboard\app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
