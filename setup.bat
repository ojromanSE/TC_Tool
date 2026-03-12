@echo off
setlocal
set PYTHON_VERSION=3.11.9
set PYTHON_ZIP=python-%PYTHON_VERSION%-embed-amd64.zip
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_ZIP%
set EMBED_DIR=%~dp0python_embed

echo =============================================
echo  SE Tool - First-Time Setup
echo =============================================
echo.

REM Skip if already set up
IF EXIST "%EMBED_DIR%\python.exe" (
    echo Setup already complete. Run launch.bat to start the app.
    pause
    exit /b 0
)

REM Check for internet and curl
curl --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: curl not found. Please use Windows 10 version 1803 or later.
    pause
    exit /b 1
)

echo Step 1/4: Downloading embedded Python %PYTHON_VERSION%...
mkdir "%EMBED_DIR%" 2>nul
curl -L "%PYTHON_URL%" -o "%EMBED_DIR%\%PYTHON_ZIP%"
IF ERRORLEVEL 1 (
    echo ERROR: Failed to download Python. Check your internet connection.
    pause
    exit /b 1
)

echo Step 2/4: Extracting Python...
powershell -Command "Expand-Archive -Path '%EMBED_DIR%\%PYTHON_ZIP%' -DestinationPath '%EMBED_DIR%' -Force"
del "%EMBED_DIR%\%PYTHON_ZIP%"

REM Enable site-packages so pip-installed packages are found
for %%f in ("%EMBED_DIR%\python*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site', 'import site' | Set-Content '%%f'"
)

echo Step 3/4: Installing pip...
curl -L https://bootstrap.pypa.io/get-pip.py -o "%EMBED_DIR%\get-pip.py"
"%EMBED_DIR%\python.exe" "%EMBED_DIR%\get-pip.py" --quiet
del "%EMBED_DIR%\get-pip.py"

echo Step 4/4: Installing SE Tool dependencies (this may take a few minutes)...
"%EMBED_DIR%\python.exe" -m pip install -r "%~dp0requirements.txt" --quiet

echo.
echo =============================================
echo  Setup complete! Run launch.bat to start.
echo =============================================
pause
