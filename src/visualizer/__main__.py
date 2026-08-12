"""CLI entry point: port scan with instance-identity probe, browser auto-open."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import urllib.request
import webbrowser

import uvicorn

DEFAULT_PORT = 8765
PORT_SCAN_RANGE = 11


def _fatal(msg: str) -> None:
    """Frozen/noconsole builds have no visible stderr — surface fatals in a dialog."""
    if getattr(sys, "frozen", False) and os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "Audio Visualizer", 0x10)  # MB_ICONERROR
    raise SystemExit(msg)


def _pick_port(pref: int) -> tuple[int, bool]:
    """Returns (port, already_running). Probes occupied ports with /api/ping so a
    second launch reuses the running instance instead of double-launching a GL
    context (and exhausting NVENC sessions mid-export)."""
    for p in range(pref, pref + PORT_SCAN_RANGE):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
            return p, False
        except OSError:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/ping", timeout=0.3) as r:
                    if json.load(r).get("app") == "audio-visualizer":
                        return p, True
            except Exception:  # noqa: BLE001 — occupant is some other program
                pass
    raise SystemExit(f"No free port in {pref}-{pref + PORT_SCAN_RANGE - 1}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="visualizer",
                                 description="GPU audio visualizer (MP3 in, editor-ready .webm out)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="preferred port (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _fatal("ffmpeg를 찾을 수 없습니다.\n\n"
               "관리자 권한이 없어도 되는 설치 방법 (PowerShell에서):\n"
               "    winget install Gyan.FFmpeg\n\n"
               "설치 후 Audio Visualizer를 다시 실행해주세요.")

    try:
        port, already_running = _pick_port(args.port)
    except SystemExit as e:
        _fatal(str(e))
        raise
    url = f"http://127.0.0.1:{port}/"
    if already_running:
        print(f"Audio Visualizer is already running at {url} — opening browser")
        webbrowser.open(url)
        return

    from .server import app  # deferred: heavy imports only when we actually serve

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    app.state.server = server  # /api/shutdown flips server.should_exit
    print(f"Audio Visualizer running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=[url]).start()
    server.run()


if __name__ == "__main__":
    main()
