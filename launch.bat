@echo off
echo Starting SE Tool...

REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check that project files are present
IF NOT EXIST "%~dp0requirements.txt" (
    echo ERROR: requirements.txt not found.
    echo launch.bat must be inside the TC_Tool project folder, not run standalone.
    pause
    exit /b 1
)
IF NOT EXIST "%~dp0app.py" (
    echo ERROR: app.py not found.
    echo launch.bat must be inside the TC_Tool project folder, not run standalone.
    pause
    exit /b 1
)

REM Install dependencies (skips packages already installed)
echo Installing dependencies...
pip install -r "%~dp0requirements.txt" --quiet

REM Launch the app
echo Launching SE Tool in your browser...
streamlit run "%~dp0app.py"
pause
