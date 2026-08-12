"""ffmpeg encode subprocess for .webm export.

Wire invariants (cross-review binding — do not "fix" on this side):
- Frames arrive TOP-DOWN, straight (non-premultiplied) alpha RGBA8, tightly
  packed. The renderer's y-down projection guarantees the row order, so there
  is NO vflip filter here and no numpy flip anywhere.
- write_frame() blocks until the whole frame is in the ffmpeg stdin pipe.
  That blocking is what makes the zero-copy memoryview handoff from
  Renderer.draw() valid: the renderer's double buffer cannot be overwritten
  while a write is pending, provided the caller runs draw -> write in lockstep
  and never buffers frames ahead (see pipeline.run_export).
- Video duration is governed solely by writing exactly total_frames frames and
  closing stdin. Never add -shortest (it races the muxer and can drop the
  final frame). total_frames comes from Features.n_frames — never ffprobe.
"""
from __future__ import annotations

import collections
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

FFMPEG_EXE = "ffmpeg"
FFPROBE_EXE = "ffprobe"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

Quality = Literal["fast", "balanced", "best"]
OpaqueCodec = Literal["av1_nvenc", "vp9"]

# (crf, cpu_used)
VP9_QUALITY: dict[str, tuple[int, int]] = {
    "fast": (31, 5),
    "balanced": (23, 4),
    "best": (16, 2),
}
# (cq, preset, multipass, rc_lookahead)
AV1_QUALITY: dict[str, tuple[int, str, str, int]] = {
    "fast": (33, "p4", "disabled", 0),
    "balanced": (27, "p6", "qres", 32),
    "best": (21, "p7", "fullres", 32),
}


@dataclass(frozen=True)
class EncodeSettings:
    width: int  # must be even (4:2:0 chroma subsampling)
    height: int  # must be even
    fps: int
    transparent: bool  # True -> libvpx-vp9 yuva420p; False -> opaque_codec
    quality: Quality = "balanced"
    opaque_codec: OpaqueCodec = "av1_nvenc"
    audio_bitrate_k: int = 160  # libopus
    title: Optional[str] = None  # container metadata; default derived from audio stem

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height must be positive, got {self.width}x{self.height}")
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"width and height must be even for 4:2:0 chroma subsampling, "
                f"got {self.width}x{self.height}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.quality not in VP9_QUALITY:
            raise ValueError(f"unknown quality {self.quality!r}")
        if self.opaque_codec not in ("av1_nvenc", "vp9"):
            raise ValueError(f"unknown opaque_codec {self.opaque_codec!r}")


@dataclass(frozen=True)
class Progress:
    frames_done: int  # encoder-side count (truth for ETA), not frames written to the pipe
    total_frames: int
    fps: float  # EMA-smoothed encoder fps
    eta_seconds: Optional[float]  # None until fps stabilizes; 0.0 on the final emit


ProgressFn = Callable[[Progress], None]


class EncodeError(RuntimeError):
    """ffmpeg failed; the message carries the last ~60 stderr lines."""


class EncodeCancelled(Exception):
    """The encode was cancelled; the partial output file has been deleted."""


def build_ffmpeg_cmd(s: EncodeSettings, audio_path: Path, out_path: Path) -> list[str]:
    """Pure argv builder for all three codec paths. No vflip — frames are top-down.

    -f webm is explicit because out_path is normally a *.webm.part file, from
    which ffmpeg cannot infer the container.
    """
    pix = "yuva420p" if s.transparent else "yuv420p"
    vf = (
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:"
        f"flags=accurate_rnd+full_chroma_int,format={pix}"
    )
    cmd = [
        FFMPEG_EXE, "-hide_banner", "-y", "-loglevel", "error", "-nostats",
        "-progress", "pipe:1", "-stats_period", "0.25",
        "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", f"{s.width}x{s.height}", "-framerate", str(s.fps), "-i", "pipe:0",
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1",
        "-vf", vf,
    ]
    gop = str(2 * s.fps)
    if s.transparent or s.opaque_codec == "vp9":
        crf, cpu_used = VP9_QUALITY[s.quality]
        tile_columns = max(0, min(4, int(math.log2(max(1, s.width // 256)))))
        threads = min(os.cpu_count() or 8, 16)
        cmd += [
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
            "-deadline", "good", "-cpu-used", str(cpu_used),
            "-row-mt", "1", "-threads", str(threads), "-tile-columns", str(tile_columns),
        ]
        if s.transparent:
            # libvpx rejects alpha with alt-ref frames
            cmd += ["-auto-alt-ref", "0", "-lag-in-frames", "0", "-g", gop,
                    "-metadata:s:v:0", "alpha_mode=1"]
        else:
            cmd += ["-auto-alt-ref", "1", "-lag-in-frames", "25", "-g", gop]
    else:
        cq, preset, multipass, lookahead = AV1_QUALITY[s.quality]
        cmd += [
            "-c:v", "av1_nvenc", "-preset", preset, "-tune", "hq",
            "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
            "-multipass", multipass, "-rc-lookahead", str(lookahead), "-g", gop,
        ]
    title = s.title if s.title is not None else f"{audio_path.stem} visualizer"
    cmd += [
        "-color_range", "tv", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "libopus", "-b:a", f"{s.audio_bitrate_k}k", "-ar", "48000", "-ac", "2",
        "-metadata", f"title={title}",
        "-f", "webm",
        str(out_path),
    ]
    return cmd


class Encoder:
    """Owns one ffmpeg subprocess writing a .webm (normally to a *.part path).

    Context manager: __enter__ -> start(); __exit__ -> finish() on clean exit,
    cancel() on exception. cancel() deletes the partial output file.
    """

    def __init__(
        self,
        settings: EncodeSettings,
        audio_path: Path,
        out_path: Path,
        total_frames: int,
        on_progress: ProgressFn | None = None,
    ) -> None:
        audio_path = Path(audio_path)
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")
        if not audio_path.is_file():
            raise ValueError(f"audio file not found: {audio_path}")
        self.settings = settings
        self.audio_path = audio_path
        self.out_path = Path(out_path)
        self.total_frames = int(total_frames)
        self.on_progress = on_progress
        self._frame_bytes = settings.width * settings.height * 4
        self._proc: subprocess.Popen[bytes] | None = None
        self._t_prog: threading.Thread | None = None
        self._t_err: threading.Thread | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=60)
        self._enc_frames = 0
        self._frames_written = 0
        self._fps_ema = 0.0
        self._cancelled = threading.Event()

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_cmd(self.settings, self.audio_path, self.out_path)
        # Never shell=True; all three std handles piped (inherited console
        # handles under uvicorn on Windows cause hangs and window flashes).
        # bufsize=0: raw pipes, so the blocking stdin write IS the backpressure.
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=CREATE_NO_WINDOW,
        )
        # Both drain threads must exist before the first frame write, or ffmpeg
        # blocks on a full stdout/stderr pipe while we block on stdin: deadlock.
        self._t_prog = threading.Thread(target=self._drain_progress, daemon=True)
        self._t_err = threading.Thread(target=self._drain_stderr, daemon=True)
        self._t_prog.start()
        self._t_err.start()

    def write_frame(self, rgba: bytes | bytearray | memoryview) -> None:
        """Write exactly width*height*4 bytes of top-down straight-alpha RGBA.

        Blocks until the frame is fully in the pipe — this blocking makes the
        zero-copy memoryview handoff from the renderer valid (the caller may
        pass a view into the renderer's reusable buffer, e.g. numpy arr.data).
        """
        if self._cancelled.is_set():
            raise EncodeCancelled()
        mv = memoryview(rgba).cast("B")
        if mv.nbytes != self._frame_bytes:
            raise ValueError(f"expected {self._frame_bytes} bytes, got {mv.nbytes}")
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EncodeError("encoder not started")
        try:
            while mv.nbytes:
                # bufsize=0 -> raw FileIO: PARTIAL writes possible; loop is mandatory.
                n = proc.stdin.write(mv)
                if n is None:
                    break
                mv = mv[n:]
        except (OSError, ValueError) as e:
            # Windows raises OSError errno 22 (EINVAL) on a dead pipe, not
            # always BrokenPipeError; ValueError if cancel() closed stdin.
            if self._cancelled.is_set():
                raise EncodeCancelled() from e
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if self._t_err is not None:
                self._t_err.join(timeout=2)
            raise EncodeError(
                "ffmpeg exited during encode:\n" + "\n".join(self._stderr_tail)
            ) from e
        self._frames_written += 1

    def finish(self) -> Path:
        """Close stdin (EOF finalizes the webm cues), wait, verify exit code."""
        if self._cancelled.is_set():
            raise EncodeCancelled()
        proc = self._proc
        if proc is None:
            raise EncodeError("encoder not started")
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            rc = proc.wait(timeout=300)  # generous: VP9 'best' flush tail
        except subprocess.TimeoutExpired:
            proc.kill()
            raise EncodeError("ffmpeg did not exit within 300s after EOF")
        if self._t_prog is not None:
            self._t_prog.join(timeout=5)
        if self._t_err is not None:
            self._t_err.join(timeout=5)
        if rc != 0:
            raise EncodeError(f"ffmpeg exited {rc}:\n" + "\n".join(self._stderr_tail))
        self._emit(final=True)
        return self.out_path

    def cancel(self) -> None:
        """Kill ffmpeg and delete the partial output. Thread-safe, idempotent.

        Sole owner of partial-file deletion; retries the unlink 5x200ms because
        Windows can briefly hold the file handle after TerminateProcess.
        """
        self._cancelled.set()
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except (OSError, ValueError):
                    pass
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if self._t_prog is not None:
                self._t_prog.join(timeout=5)
            if self._t_err is not None:
                self._t_err.join(timeout=5)
        for attempt in range(5):
            try:
                self.out_path.unlink(missing_ok=True)
                return
            except OSError:
                time.sleep(0.2)
        # Best-effort: never raise out of cancel() over a lingering handle.

    def __enter__(self) -> "Encoder":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is None:
            self.finish()
        else:
            self.cancel()
        return False

    # -- progress / stderr drain (daemon threads) --

    def _drain_progress(self) -> None:
        # -progress pipe:1 emits key=value blocks ending with progress=continue|end.
        assert self._proc is not None and self._proc.stdout is not None
        fps_ema = 0.0
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            key, _, val = line.partition("=")
            if key == "frame":
                try:
                    self._enc_frames = int(val)
                except ValueError:
                    pass
            elif key == "fps":
                try:
                    f = float(val)
                except ValueError:
                    continue
                if f > 0:
                    fps_ema = f if fps_ema == 0.0 else 0.7 * fps_ema + 0.3 * f
            elif key == "progress":  # end of block -> emit once per period
                self._fps_ema = fps_ema
                self._emit()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for raw in self._proc.stderr:
            self._stderr_tail.append(raw.decode("utf-8", "replace").rstrip())

    def _emit(self, final: bool = False) -> None:
        # Fires on the progress drain thread (or the finishing thread when
        # final=True): the GUI layer must store-and-poll, never touch its
        # event loop from here.
        if self.on_progress is None:
            return
        done = self.total_frames if final else min(self._enc_frames, self.total_frames)
        fps = self._fps_ema
        eta = 0.0 if final else ((self.total_frames - done) / fps if fps > 0.1 else None)
        try:
            self.on_progress(Progress(done, self.total_frames, fps, eta))
        except Exception:
            pass  # UI bugs must never kill the encode


def default_output_name(audio_path: Path, style: str, s: EncodeSettings) -> str:
    stem = re.sub(r'[<>:"/\\|?*\s]+', "_", audio_path.stem).strip("_") or "audio"
    mode = "alpha" if s.transparent else ("av1" if s.opaque_codec == "av1_nvenc" else "vp9")
    return f"{stem}__{style}__{s.width}x{s.height}@{s.fps}__{mode}.webm"


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    for i in range(1, 1000):
        cand = p.with_stem(f"{p.stem}-{i}")
        if not cand.exists():
            return cand
    raise FileExistsError(p)


def verify_alpha(webm_path: Path, sample_time: float = 1.0) -> bool:
    """True if the file is VP9 with a real alpha channel.

    ffprobe reports pix_fmt=yuv420p even for valid VP9-alpha files (the native
    decoder ignores Matroska BlockAdditions alpha) — that is NOT a bug in the
    encode. Stage 1 checks the codec, stage 2 force-decodes one frame with
    libvpx-vp9 (the only decoder that reconstructs alpha) and inspects bytes.
    """
    tag = subprocess.run(
        [FFPROBE_EXE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name:stream_tags=alpha_mode",
         "-of", "default=noprint_wrappers=1", str(webm_path)],
        capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
    ).stdout
    if "codec_name=vp9" not in tag:
        return False
    r = subprocess.run(
        [FFMPEG_EXE, "-v", "error", "-c:v", "libvpx-vp9", "-ss", str(sample_time),
         "-i", str(webm_path), "-frames:v", "1", "-pix_fmt", "rgba",
         "-f", "rawvideo", "pipe:1"],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )
    a = r.stdout[3::4]
    return len(a) > 0 and min(a) < 255
