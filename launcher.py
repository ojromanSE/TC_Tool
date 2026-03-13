"""
Entry point for PyInstaller .exe build.
Starts Streamlit as a subprocess and opens the browser automatically.
A native Windows dialog is shown; closing it shuts down Streamlit and exits.
"""
import os
import sys
import subprocess
import threading
import webbrowser
import time
import ctypes


def _open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


def _run():
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
        python_exe = sys.executable
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        python_exe = sys.executable

    app_path = os.path.join(base, "app.py")

    cmd = [
        python_exe, "-m", "streamlit", "run", app_path,
        "--server.address=localhost",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]

    # Hide the subprocess console window on Windows
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    proc = subprocess.Popen(cmd, startupinfo=si)

    threading.Thread(target=_open_browser, daemon=True).start()

    # Show a native Windows dialog — blocks until user clicks OK
    ctypes.windll.user32.MessageBoxW(
        0,
        "SE Tool is running in your browser.\n\nClick OK to close SE Tool.",
        "SE Tool",
        0x00000040  # MB_ICONINFORMATION
    )

    # User clicked OK — shut down Streamlit
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    sys.exit(0)


if __name__ == "__main__":
    _run()
