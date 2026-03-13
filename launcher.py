"""
Entry point for PyInstaller .exe build.
Starts Streamlit on localhost and opens the browser automatically.
"""
import os
import sys
import threading
import webbrowser
import time


def _open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    # When frozen by PyInstaller, app files are in sys._MEIPASS
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base, "app.py")

    threading.Thread(target=_open_browser, daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.address=localhost",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
