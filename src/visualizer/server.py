"""FastAPI app: frontend hosting, upload+analysis, live preview WS, export jobs.

Threading model (the one paragraph of truth):
- uvicorn event loop runs every HTTP/WS/SSE handler;
- ONE dedicated GL thread (RenderService) owns the moderngl Renderer and runs
  both preview-frame closures and entire blocking export loops;
- the FastAPI threadpool runs upload ingest (ffmpeg transcode + PCM decode) and
  Features precompute — never GL.
Previews are refused at submit time while an export runs (never queued behind it).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Callable

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ValidationError

from .audio import AudioData, decode_audio
from .features import FeatureConfig, Features, compute_features, waveform_overview
from .jobs import TERMINAL_STATES, JobBusy, JobManager, JobState
from .render import RenderConfig, Renderer
from .settings import RenderSettings, build_configs, preview_render_config

if getattr(sys, "frozen", False):  # PyInstaller bundle
    FRONTEND_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "frontend"
    OUTPUT_DIR = Path.home() / "Videos" / "Audio Visualizer"  # user-facing, survives updates
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    FRONTEND_DIR = PROJECT_ROOT / "frontend"
    OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "audio_visualizer_uploads"

PREVIEW_JPEG_Q = 80
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_UPLOAD = 512 * 2**20
MAX_IMAGE_UPLOAD = 64 * 2**20
MAX_BG_DIM = 4096  # backgrounds are downscaled to fit this on upload
# Cap decode size well below PIL's default bomb threshold: bounds the transient
# RGBA allocation for hostile/degenerate images (64 MP * 4 B = 256 MB worst case).
Image.MAX_IMAGE_PIXELS = 64_000_000
UPLOAD_TTL_S = 24 * 3600
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------- audio store

@dataclass
class AudioEntry:
    audio_id: str
    path: Path          # original upload — export muxes audio from this
    preview_path: Path  # browser-safe .m4a for the <audio> element
    filename: str
    audio: AudioData


class AudioRegistry:
    """LRU of loaded tracks (PCM is heavy). Eviction deletes temp files."""

    def __init__(self, cap: int = 3) -> None:
        self._lock = threading.Lock()
        self._d: OrderedDict[str, AudioEntry] = OrderedDict()
        self._cap = cap

    def put(self, entry: AudioEntry) -> None:
        evicted: list[AudioEntry] = []
        with self._lock:
            self._d[entry.audio_id] = entry
            self._d.move_to_end(entry.audio_id)
            while len(self._d) > self._cap:
                evicted.append(self._d.popitem(last=False)[1])
        for old in evicted:
            FEATURES.drop(old.audio_id)
            for p in (old.path, old.preview_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def get(self, audio_id: str) -> AudioEntry | None:
        with self._lock:
            entry = self._d.get(audio_id)
            if entry is not None:
                self._d.move_to_end(audio_id)
            return entry


class BackgroundRegistry:
    """Uploaded background images: content-hash id -> normalized PNG in UPLOAD_DIR.

    Ids are content hashes, so re-uploading the same image (even after a server
    restart) yields the same id, and localStorage-restored ids keep working as
    long as the file survives the 24 h TTL sweep. scan() re-registers files
    from a previous run on startup.
    """

    # Cap is generous: eviction deletes the PNG, and an in-flight render still
    # holds only the PATH (see ws frame / start_export) — a burst of uploads
    # during that window would fail that one render (cleanly, retryable).
    def __init__(self, cap: int = 16) -> None:
        self._lock = threading.Lock()
        self._d: OrderedDict[str, Path] = OrderedDict()
        self._cap = cap

    def put(self, bg_id: str, path: Path) -> None:
        evicted: list[Path] = []
        with self._lock:
            self._d[bg_id] = path
            self._d.move_to_end(bg_id)
            while len(self._d) > self._cap:
                evicted.append(self._d.popitem(last=False)[1])
        for p in evicted:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def get(self, bg_id: str | None) -> Path | None:
        if not bg_id:
            return None
        with self._lock:
            path = self._d.get(bg_id)
            if path is not None:
                self._d.move_to_end(bg_id)
        if path is not None and not path.exists():  # TTL sweep beat us to it
            with self._lock:
                self._d.pop(bg_id, None)
            return None
        return path

    def scan(self, directory: Path) -> None:
        for p in sorted(directory.glob("*.bg.png"), key=lambda p: p.stat().st_mtime):
            self.put(p.name.split(".")[0], p)


class FeatureStore:
    """Features cache keyed by (audio_id, FeatureConfig); builds run in the
    threadpool, deduplicated so concurrent requesters await the same build.
    Event-loop-only access."""

    def __init__(self, cap: int = 6) -> None:
        self._cache: OrderedDict[tuple[str, FeatureConfig], Features] = OrderedDict()
        self._pending: dict[tuple[str, FeatureConfig], asyncio.Future] = {}
        self._cap = cap

    async def get(self, entry: AudioEntry, cfg: FeatureConfig) -> Features:
        key = (entry.audio_id, cfg)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        fut = self._pending.get(key)
        if fut is None:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, compute_features, entry.audio, cfg)
            self._pending[key] = fut
        try:
            feats = await asyncio.shield(fut)
        except asyncio.CancelledError:
            raise
        except BaseException:
            if self._pending.get(key) is fut:
                del self._pending[key]
            raise
        if self._pending.get(key) is fut:
            del self._pending[key]
            self._cache[key] = feats
            while len(self._cache) > self._cap:
                self._cache.popitem(last=False)
        return feats

    def drop(self, audio_id: str) -> None:
        for key in [k for k in self._cache if k[0] == audio_id]:
            del self._cache[key]


# ---------------------------------------------------------------- GL service

class RenderService:
    """Single moderngl context on a single dedicated thread; export-exclusive."""

    def __init__(self) -> None:
        self._q: SimpleQueue[tuple[str, Callable, Future | None]] = SimpleQueue()
        self._export_active = threading.Event()
        self.last_applied: RenderConfig | None = None  # touched only on the GL thread
        self._thread = threading.Thread(target=self._loop, name="gl-render", daemon=True)
        self._thread.start()

    def submit_preview(self, fn: Callable[[Renderer], bytes]) -> Future | None:
        """None while an export runs — previews are refused, never queued behind it."""
        if self._export_active.is_set():
            return None
        fut: Future = Future()
        self._q.put(("preview", fn, fut))
        return fut

    def submit_export(self, fn: Callable[[Renderer | None], None]) -> None:
        self._export_active.set()  # BEFORE enqueue: closes the race with submit_preview
        self._q.put(("export", fn, None))

    @property
    def export_active(self) -> bool:
        return self._export_active.is_set()

    def _loop(self) -> None:
        renderer: Renderer | None = None
        init_error: Exception | None = None
        while True:
            kind, fn, fut = self._q.get()
            if renderer is None and init_error is None:
                try:
                    # Context created HERE, lazily: thread affinity is absolute,
                    # and importing this module stays GL-free.
                    renderer = Renderer(_boot_render_config())
                except Exception as e:  # noqa: BLE001
                    init_error = e
            if kind == "export":
                try:
                    fn(renderer)  # jobs.run marks the job errored if renderer is None
                finally:
                    self.last_applied = None  # export changed the renderer's config
                    self._export_active.clear()
            elif renderer is None:
                fut.set_exception(RuntimeError(f"GL init failed: {init_error}"))
            else:
                try:
                    fut.set_result(fn(renderer))
                except Exception as e:  # noqa: BLE001
                    fut.set_exception(e)


def _boot_render_config() -> RenderConfig:
    rs = RenderSettings()
    _, render_cfg, _ = build_configs(rs, None)
    return preview_render_config(rs, render_cfg)


_CHECKER_CACHE: dict[tuple[int, int], np.ndarray] = {}  # GL-thread only


def _checkerboard(w: int, h: int) -> np.ndarray:
    """(h, w, 3) float32 16px checkerboard — the preview's alpha indicator."""
    board = _CHECKER_CACHE.get((w, h))
    if board is None:
        if len(_CHECKER_CACHE) > 6:
            _CHECKER_CACHE.clear()
        yy, xx = np.mgrid[0:h, 0:w]
        cells = ((xx // 16 + yy // 16) % 2).astype(np.float32)
        board = np.repeat((42.0 + cells[..., None] * 16.0), 3, axis=2)  # 0x2a / 0x3a
        _CHECKER_CACHE[(w, h)] = board
    return board


def _render_preview_jpeg(renderer: Renderer, svc: RenderService, cfg: RenderConfig,
                         features: Features, t: float, transparent: bool) -> bytes:
    """Runs on the GL thread. The JPEG is fully materialized before returning,
    so the renderer's double-buffered readback can never be overwritten mid-encode."""
    if svc.last_applied != cfg:
        renderer.update_config(cfg)
        svc.last_applied = cfg
    buf = renderer.draw(features.at_time(t))
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(cfg.height, cfg.width, 4)
    if transparent:
        a = rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (rgba[:, :, :3].astype(np.float32) * a
               + _checkerboard(cfg.width, cfg.height) * (1.0 - a)).astype(np.uint8)
    else:
        rgb = np.ascontiguousarray(rgba[:, :, :3])  # renderer baked the solid background
    out = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(out, "JPEG", quality=PREVIEW_JPEG_Q, subsampling=2)
    return out.getvalue()


# ---------------------------------------------------------------- app + state

AUDIO = AudioRegistry()
BACKGROUNDS = BackgroundRegistry()
FEATURES = FeatureStore()
JOBS = JobManager()
RENDER = RenderService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _purge_old_uploads()
    BACKGROUNDS.scan(UPLOAD_DIR)  # background ids survive restarts (content-hashed)
    yield
    active = JOBS.active
    if active is not None:
        JOBS.cancel(active.id)
        for _ in range(50):  # let the export loop unwind and release the .part
            if JOBS.active is None:
                break
            await asyncio.sleep(0.1)
    for p in UPLOAD_DIR.glob("*"):
        if p.name.endswith(".bg.png"):
            continue  # backgrounds persist across restarts (localStorage keeps their ids)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _purge_old_uploads() -> None:
    now = time.time()
    for p in UPLOAD_DIR.iterdir():
        try:
            if now - p.stat().st_mtime > UPLOAD_TTL_S:
                p.unlink(missing_ok=True)
        except OSError:
            pass


app = FastAPI(lifespan=lifespan)  # same-origin only; bound to 127.0.0.1 — no CORS


@app.middleware("http")
async def _no_stale_frontend(request, call_next):
    """Frontend assets must never go stale (a cached app.js from an older build
    silently desyncs the UI from the server). no-cache still allows ETag 304s."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/ping")
def ping() -> dict:
    return {"app": "audio-visualizer", "version": "1"}


# ---------------------------------------------------------------- upload

def _ingest(audio_id: str, path: Path, filename: str) -> tuple[AudioEntry, list]:
    """Threadpool: transcode a browser-safe playback copy, decode PCM, overview."""
    preview = UPLOAD_DIR / f"{audio_id}.preview.m4a"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-map", "a:0",
         "-ac", "2", "-c:a", "aac", "-b:a", "160k", str(preview)],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg transcode failed: {tail}")
    audio = decode_audio(path)
    overview = np.asarray(waveform_overview(audio, 1000), dtype=np.float64).round(4).tolist()
    return AudioEntry(audio_id, path, preview, filename, audio), overview


@app.post("/api/audio")
async def upload_audio(file: UploadFile, settings: str | None = Form(None)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"unsupported audio format: {ext or '(none)'}")
    audio_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{audio_id}{ext}"
    size = 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "file exceeds the 512 MB limit")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    loop = asyncio.get_running_loop()
    try:
        entry, overview = await loop.run_in_executor(
            None, _ingest, audio_id, dest, file.filename or dest.name)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"could not decode audio: {e}")
    AUDIO.put(entry)

    rs = RenderSettings()
    if settings:
        try:
            rs = RenderSettings.model_validate_json(settings)
        except ValidationError:
            pass  # pre-warm with defaults instead
    feature_cfg, _, _ = build_configs(rs, entry.audio)
    await FEATURES.get(entry, feature_cfg)  # pre-warm so the first preview is instant

    return {"audio_id": audio_id, "filename": entry.filename,
            "duration": entry.audio.duration, "sample_rate": entry.audio.sample_rate,
            "overview": overview}


@app.get("/api/audio/{audio_id}/file")
def audio_file(audio_id: str) -> FileResponse:
    entry = AUDIO.get(audio_id)
    if entry is None or not entry.preview_path.exists():
        raise HTTPException(404, "unknown audio_id")
    return FileResponse(entry.preview_path, media_type="audio/mp4")


# ---------------------------------------------------------------- backgrounds

def _ingest_background(raw: bytes) -> tuple[str, Path, int, int]:
    """Threadpool: validate + normalize an uploaded image to a flattened PNG.

    The id is a hash of the RAW upload so identical files re-upload to the same
    id; the stored file is opaque RGB (alpha flattened over black) so the
    renderer's cover-fit pass never has to composite a translucent background.
    """
    bg_id = hashlib.sha256(raw).hexdigest()[:12]
    dest = UPLOAD_DIR / f"{bg_id}.bg.png"
    if not dest.exists():
        tmp = dest.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGBA")
                if max(im.size) > MAX_BG_DIM:
                    im.thumbnail((MAX_BG_DIM, MAX_BG_DIM), Image.LANCZOS)
                flat = Image.new("RGB", im.size, (0, 0, 0))
                flat.paste(im, mask=im.split()[3])
                flat.save(tmp, "PNG")
            # atomic publish: a crash mid-save must never leave a truncated PNG
            # at the content-hash path (dest.exists() would then skip rewrites forever)
            try:
                tmp.replace(dest)
            except OSError:
                if not dest.exists():
                    raise
                # concurrent identical upload won the race (same content hash) — theirs is fine
                tmp.unlink(missing_ok=True)
        except Exception as e:  # PIL raises Exception subclasses beyond OSError/ValueError
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"not a decodable image: {e}") from e
    with Image.open(dest) as im:
        w, h = im.size
    return bg_id, dest, w, h


@app.post("/api/background")
async def upload_background(file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(415, f"unsupported image format: {ext or '(none)'}")
    raw = await file.read(MAX_IMAGE_UPLOAD + 1)
    if len(raw) > MAX_IMAGE_UPLOAD:
        raise HTTPException(413, "image exceeds the 64 MB limit")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    try:
        bg_id, dest, w, h = await loop.run_in_executor(None, _ingest_background, raw)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    BACKGROUNDS.put(bg_id, dest)
    return {"bg_id": bg_id, "width": w, "height": h}


@app.get("/api/background/{bg_id}")
def get_background(bg_id: str) -> FileResponse:
    path = BACKGROUNDS.get(bg_id)
    if path is None:
        raise HTTPException(404, "unknown background — re-upload the image")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------- preview WS

@app.websocket("/ws/preview")
async def ws_preview(ws: WebSocket) -> None:
    await ws.accept()
    settings = RenderSettings()  # per-connection state — two tabs never fight
    entry: AudioEntry | None = None
    try:
        while True:
            try:
                data = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "malformed message"})
                continue
            if not isinstance(data, dict):
                await ws.send_json({"type": "error", "message": "malformed message"})
                continue
            mtype = data.get("type")

            if mtype == "use_audio":
                candidate = AUDIO.get(str(data.get("audio_id", "")))
                if candidate is None:
                    await ws.send_json({"type": "error", "message": "unknown audio_id — re-upload"})
                    continue
                entry = candidate
                feature_cfg, _, _ = build_configs(settings, entry.audio)
                await FEATURES.get(entry, feature_cfg)
                await ws.send_json({"type": "ready"})

            elif mtype == "settings":
                try:
                    settings = RenderSettings.model_validate(data.get("settings") or {})
                except ValidationError as ex:
                    first = ex.errors()[0]
                    await ws.send_json({"type": "error",
                                        "message": f"invalid settings: {first.get('loc')}: {first.get('msg')}"})

            elif mtype == "frame":
                if entry is None:
                    continue
                if RENDER.export_active:
                    await ws.send_json({"type": "export_active"})
                    continue
                try:
                    t = float(data.get("t", 0.0))
                except (TypeError, ValueError):
                    await ws.send_json({"type": "error", "message": "malformed message"})
                    continue
                bg_path = BACKGROUNDS.get(settings.bg_image_id)
                if settings.bg_mode == "image" and bg_path is None:
                    # expired/unknown id: tell the client so it can fall back to solid
                    await ws.send_json({"type": "bg_missing"})
                    continue
                feature_cfg, render_cfg, _ = build_configs(settings, entry.audio,
                                                           bg_image_path=bg_path)
                pcfg = preview_render_config(settings, render_cfg)
                feats = await FEATURES.get(entry, feature_cfg)
                transparent = settings.bg_mode == "transparent"
                fut = RENDER.submit_preview(
                    lambda r: _render_preview_jpeg(r, RENDER, pcfg, feats, t, transparent))
                if fut is None:
                    await ws.send_json({"type": "export_active"})
                    continue
                try:
                    jpeg = await asyncio.wrap_future(fut)
                except Exception as ex:  # noqa: BLE001
                    await ws.send_json({"type": "error", "message": f"render failed: {ex}"})
                    continue
                await ws.send_bytes(jpeg)
    except (WebSocketDisconnect, RuntimeError):
        return


# ---------------------------------------------------------------- export

class ExportRequest(BaseModel):
    audio_id: str
    settings: RenderSettings


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@app.post("/api/export", status_code=202)
async def start_export(req: ExportRequest) -> dict:
    entry = AUDIO.get(req.audio_id)
    if entry is None:
        raise HTTPException(404, "unknown audio_id — re-upload the file")
    if JOBS.active is not None:
        raise HTTPException(409, "An export is already running")
    bg_path = BACKGROUNDS.get(req.settings.bg_image_id)
    if req.settings.bg_mode == "image" and bg_path is None:
        raise HTTPException(404, "background image expired — re-upload the image")
    feature_cfg, render_cfg, enc_settings = build_configs(req.settings, entry.audio,
                                                          bg_image_path=bg_path)
    features = await FEATURES.get(entry, feature_cfg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _UNSAFE_FILENAME.sub("_", Path(entry.filename).stem)[:40].strip() or "audio"
    name = (f"{stem}_{req.settings.style}_{req.settings.width}x{req.settings.height}_"
            f"{req.settings.fps}fps_{time.strftime('%Y%m%d-%H%M%S')}.webm")
    try:
        job = JOBS.create(req.audio_id, req.settings, OUTPUT_DIR / name)
    except JobBusy:
        raise HTTPException(409, "An export is already running")
    audio_path = entry.path
    RENDER.submit_export(
        lambda renderer: JOBS.run(job, renderer, features, audio_path, render_cfg, enc_settings))
    return {"job_id": job.id}


@app.get("/api/export/{job_id}")
def export_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.snapshot()


def _terminal_event(job) -> str:
    if job.state is JobState.DONE:
        try:
            size = job.out_path.stat().st_size
        except OSError:
            size = 0
        payload = {"job_id": job.id, "file": {"name": job.out_path.name, "size": size,
                                              "url": f"/api/renders/{job.out_path.name}"}}
        return f"event: done\ndata: {json.dumps(payload)}\n\n"
    if job.state is JobState.ERROR:
        return f"event: error\ndata: {json.dumps({'job_id': job.id, 'message': job.error or 'export failed'})}\n\n"
    return f"event: cancelled\ndata: {json.dumps({'job_id': job.id})}\n\n"


@app.get("/api/export/{job_id}/events")
async def export_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        while True:
            if job.state in TERMINAL_STATES:
                # Replayed immediately for finished jobs: makes EventSource
                # auto-reconnect and page reload harmless.
                yield _terminal_event(job)
                return
            yield f"event: progress\ndata: {json.dumps(job.snapshot())}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/api/export/{job_id}")
def cancel_export(job_id: str) -> dict:
    if JOBS.get(job_id) is None:
        raise HTTPException(404, "unknown job")
    return {"cancelled": JOBS.cancel(job_id)}


# ---------------------------------------------------------------- renders

def _safe_output_path(name: str) -> Path:
    if name != Path(name).name or not name.endswith(".webm"):
        raise HTTPException(403, "invalid name")
    p = (OUTPUT_DIR / name).resolve()
    if p.parent != OUTPUT_DIR.resolve():
        raise HTTPException(403, "invalid name")
    return p


@app.get("/api/renders")
def list_renders() -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime,
             "url": f"/api/renders/{p.name}"} for p in files]


@app.get("/api/renders/{name}")
def get_render(name: str) -> FileResponse:
    p = _safe_output_path(name)
    if not p.exists():
        raise HTTPException(404, "no such render")
    return FileResponse(p, media_type="video/webm", filename=name)


class RevealRequest(BaseModel):
    name: str | None = None


@app.post("/api/reveal")
def reveal(req: RevealRequest) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if req.name:
        p = _safe_output_path(req.name)
        if not p.exists():
            raise HTTPException(404, "no such render")
        subprocess.Popen(["explorer", "/select,", str(p)], creationflags=CREATE_NO_WINDOW)
    else:
        subprocess.Popen(["explorer", str(OUTPUT_DIR)], creationflags=CREATE_NO_WINDOW)
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown() -> dict:
    server = getattr(app.state, "server", None)
    if server is not None:
        server.should_exit = True
    else:  # launched some other way (e.g. bare uvicorn): exit after the response flushes
        import os
        threading.Timer(0.3, os._exit, args=[0]).start()
    return {"ok": True}


# Mounted LAST so /api and /ws routes are never shadowed.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
