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

REM Install dependencies (skips packages already installed)
echo Installing dependencies...
pip install -r "%~dp0requirements.txt" --quiet

REM Launch the app
echo Launching SE Tool in your browser...
streamlit run "%~dp0app.py"
pause
