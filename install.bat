@echo off
setlocal

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

set SHORTCUT=%USERPROFILE%\Desktop\SE Tool.lnk

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$sc = $ws.CreateShortcut('%SHORTCUT%');" ^
    "$sc.TargetPath = '%EXE%';" ^
    "$sc.WorkingDirectory = '%~dp0';" ^
    "$sc.IconLocation = '%EXE%,0';" ^
    "$sc.Description = 'SE Tool - Tube Curve Tool';" ^
    "$sc.Save()"

IF EXIST "%SHORTCUT%" (
    echo.
    echo  Shortcut created on your Desktop!
    echo  You can now launch SE Tool from there.
    echo.
) ELSE (
    echo.
    echo  Something went wrong. Shortcut was not created.
    echo  You can still run SE_Tool.exe directly from this folder.
    echo.
)

pause
