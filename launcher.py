"""
Entry point for PyInstaller .exe build.
Starts Streamlit in a background thread and opens the browser automatically.
A native Windows dialog is shown; clicking OK shuts down everything cleanly.
"""
import os
import sys
import threading
import webbrowser
import time
import ctypes


def _open_browser():
    time.sleep(8)
    webbrowser.open("http://localhost:8501")


def _run_streamlit(app_path):
    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.address=localhost",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    stcli.main()


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base, "app.py")

    # Run Streamlit in a daemon thread (dies automatically when main thread exits)
    threading.Thread(target=_run_streamlit, args=(app_path,), daemon=True).start()
    threading.Thread(target=_open_browser, daemon=True).start()

    # Show a native Windows dialog — blocks until user clicks OK
    ctypes.windll.user32.MessageBoxW(
        0,
        "SE Tool is running in your browser.\n\nClick OK to close SE Tool.",
        "SE Tool",
        0x00000040  # MB_ICONINFORMATION
    )

    # Force-kill all threads (including Streamlit) immediately
    os._exit(0)
