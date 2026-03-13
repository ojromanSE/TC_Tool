"""
Entry point for PyInstaller .exe build.
Starts Streamlit as a subprocess and opens the browser automatically.
A small tkinter window is shown; closing it shuts down Streamlit and exits.
"""
import os
import sys
import subprocess
import threading
import webbrowser
import time
import tkinter as tk
from tkinter import ttk


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

    # On Windows, hide the subprocess console window
    kwargs = {}
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = si

    proc = subprocess.Popen(cmd, **kwargs)

    threading.Thread(target=_open_browser, daemon=True).start()

    # Build a small control window
    root = tk.Tk()
    root.title("SE Tool")
    root.resizable(False, False)
    root.geometry("300x100")

    # Centre on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 300) // 2
    y = (root.winfo_screenheight() - 100) // 2
    root.geometry(f"300x100+{x}+{y}")

    ttk.Label(root, text="SE Tool is running in your browser.",
              padding=10).pack(expand=True)
    ttk.Button(root, text="Close SE Tool",
               command=root.destroy).pack(pady=(0, 10))

    def _on_close():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()

    # Ensure process is gone after window closes (handles both X and button)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    sys.exit(0)


if __name__ == "__main__":
    _run()
