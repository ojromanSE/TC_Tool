"""
Entry point for PyInstaller .exe build.
Starts Streamlit on the main thread (required), opens the browser once
the server is ready, and exits automatically when the browser is closed.
"""
import os
import sys
import threading
import webbrowser
import time
import urllib.request
import subprocess


def _hidden_si():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return si


def _open_browser_when_ready():
    """Poll the health endpoint, then open the browser."""
    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1)
            break
        except Exception:
            continue
    webbrowser.open("http://localhost:8501")


def _has_connections():
    try:
        result = subprocess.run(
            ["netstat", "-an"],
            capture_output=True, text=True,
            startupinfo=_hidden_si()
        )
        return ":8501" in result.stdout and "ESTABLISHED" in result.stdout
    except Exception:
        return True  # assume connected if check fails


def _watch_connections():
    """Exit automatically once all browser tabs are closed."""
    # Wait up to 120 s for the first connection
    for _ in range(120):
        time.sleep(1)
        if _has_connections():
            break
    else:
        return  # browser never connected — don't auto-exit

    # Monitor until disconnected
    while True:
        time.sleep(2)
        if not _has_connections():
            time.sleep(3)  # grace period for page reloads
            if not _has_connections():
                os._exit(0)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base, "app.py")

    # Daemon threads — both die automatically if main thread exits
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    threading.Thread(target=_watch_connections, daemon=True).start()

    # Streamlit MUST run on the main thread
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
