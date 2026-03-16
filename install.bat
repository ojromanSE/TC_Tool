@echo off
setlocal enabledelayedexpansion

echo =============================================
echo  SE Tool - Desktop Installer
echo =============================================
echo.

set EXE=%~dp0SE_Tool.exe

IF NOT EXIST "%EXE%" (
    echo ERROR: SE_Tool.exe not found in this folder.
    echo Make sure you are running install.bat from inside the SE_Tool folder.
    pause
    exit /b 1
)

REM Get the real Desktop path (handles OneDrive-redirected desktops)
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set DESKTOP=%%D

echo Where would you like to save the shortcut?
echo.
echo  [1] Desktop          (%DESKTOP%)
echo  [2] Custom location  (you will type the path)
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="1" (
    set SHORTCUT_DIR=%DESKTOP%
) else if "%CHOICE%"=="2" (
    echo.
    set /p SHORTCUT_DIR="Enter full folder path: "
) else (
    echo Invalid choice. Defaulting to Desktop.
    set SHORTCUT_DIR=%DESKTOP%
)

set SHORTCUT=!SHORTCUT_DIR!\SE Tool.lnk

IF NOT EXIST "!SHORTCUT_DIR!" (
    echo.
    echo ERROR: Folder does not exist: !SHORTCUT_DIR!
    pause
    exit /b 1
)

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('!SHORTCUT!'); $sc.TargetPath = '%EXE%'; $sc.WorkingDirectory = '%~dp0'; $sc.IconLocation = '%EXE%,0'; $sc.Description = 'SE Tool - Tube Curve Tool'; $sc.Save()"

IF EXIST "!SHORTCUT!" (
    echo.
    echo  Shortcut created successfully!
    echo  Location: !SHORTCUT!
    echo.
) ELSE (
    echo.
    echo  Something went wrong. Shortcut was not created.
    echo  You can still run SE_Tool.exe directly from this folder.
    echo.
)

pause
