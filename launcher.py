"""PyInstaller entry point — thin wrapper so the exe launches the CLI main().

Windowed (noconsole) builds get sys.stdout/stderr == None; anything that writes
to them (uvicorn's logging handlers, tracebacks) would kill the process on a
real double-click launch. Route both to a log file BEFORE any other import.
"""
import os
import sys

if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
    try:
        _log_dir = os.path.join(os.path.expanduser("~"), ".audio-visualizer")
        os.makedirs(_log_dir, exist_ok=True)
        _log = open(os.path.join(_log_dir, "app.log"), "a",
                    buffering=1, encoding="utf-8", errors="replace")
    except OSError:
        _log = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

from visualizer.__main__ import main

if __name__ == "__main__":
    main()
