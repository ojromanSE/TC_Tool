@echo off
setlocal
set EMBED_DIR=%~dp0python_embed

echo Starting SE Tool...

REM Check setup has been run
IF NOT EXIST "%EMBED_DIR%\python.exe" (
    echo Setup not complete. Please run setup.bat first.
    pause
    exit /b 1
)

IF NOT EXIST "%~dp0app.py" (
    echo ERROR: app.py not found.
    echo launch.bat must be inside the TC_Tool project folder.
    pause
    exit /b 1
)

REM Launch app using bundled Python (local only, not public)
echo Launching SE Tool in your browser...
"%EMBED_DIR%\python.exe" -m streamlit run "%~dp0app.py" --server.address localhost --server.headless false
pause
