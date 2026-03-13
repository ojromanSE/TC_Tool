"""
Entry point for PyInstaller .exe build.
Starts Streamlit in a background thread, opens the browser, and automatically
exits when the browser window/tab is closed.
"""
import os
import sys
import threading
import webbrowser
import time
import subprocess


def _hidden_si():
    """STARTUPINFO to suppress console windows for subprocesses."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return si


def _has_browser_connections():
    """Return True if any browser is connected to port 8501."""
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
    """Exit automatically once all browser connections drop."""
    # Wait up to 60 s for the first connection
    connected = False
    for _ in range(60):
        time.sleep(1)
        if _has_browser_connections():
            connected = True
            break

    if not connected:
        return  # no browser ever connected — don't auto-exit

    # Monitor until all tabs are closed
    while True:
        time.sleep(2)
        if not _has_browser_connections():
            time.sleep(3)  # short grace period for page reloads
            if not _has_browser_connections():
                os._exit(0)


def _open_browser():
    """Wait until Streamlit is ready, then open the browser."""
    import urllib.request
    for _ in range(120):  # try for up to 120 seconds
        time.sleep(1)
        try:
            urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1)
            break  # server is ready
        except Exception:
            continue
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

    threading.Thread(target=_run_streamlit, args=(app_path,), daemon=True).start()
    threading.Thread(target=_open_browser, daemon=True).start()
    threading.Thread(target=_watch_connections, daemon=True).start()

    # Keep main thread alive (daemon threads die when main exits)
    while True:
        time.sleep(1)
