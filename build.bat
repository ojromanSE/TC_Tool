@echo off
setlocal
set EMBED_DIR=%~dp0python_embed

echo =============================================
echo  SE Tool - Build .exe
echo =============================================
echo.

REM Use embedded Python if available, otherwise fall back to system Python
IF EXIST "%EMBED_DIR%\python.exe" (
    set PYTHON="%EMBED_DIR%\python.exe"
    echo Using embedded Python...
) ELSE (
    python --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo ERROR: Python not found. Run setup.bat first, or install Python.
        pause
        exit /b 1
    )
    set PYTHON=python
    echo Using system Python...
)

echo Installing PyInstaller...
%PYTHON% -m pip install pyinstaller --quiet

echo Building SE_Tool.exe...
%PYTHON% -m PyInstaller ^
    --noconfirm ^
    --onedir ^
    --windowed ^
    --name "SE_Tool" ^
    --add-data "app.py;." ^
    --add-data "core.py;." ^
    --add-data "static;static" ^
    --collect-all streamlit ^
    --collect-all altair ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all reportlab ^
    --hidden-import streamlit.web.cli ^
    --hidden-import streamlit.runtime.scriptrunner ^
    --hidden-import reportlab.lib.pagesizes ^
    --hidden-import reportlab.platypus ^
    --hidden-import reportlab.lib.styles ^
    "%~dp0launcher.py"

echo.
IF EXIST "%~dp0dist\SE_Tool\SE_Tool.exe" (
    echo =============================================
    echo  Build complete!
    echo  Executable: dist\SE_Tool\SE_Tool.exe
    echo  Share the entire dist\SE_Tool\ folder.
    echo =============================================
) ELSE (
    echo Build may have failed. Check output above for errors.
)
pause
