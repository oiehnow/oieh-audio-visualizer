"""Encoding + export pipeline tests. Real ffmpeg, real (tiny) encodes."""
import math
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from visualizer.encode import (
    AV1_QUALITY,
    VP9_QUALITY,
    EncodeCancelled,
    EncodeSettings,
    Encoder,
    Progress,
    build_ffmpeg_cmd,
    default_output_name,
    unique_path,
    verify_alpha,
)
from visualizer.features import FrameFeatures
from visualizer.pipeline import ExportCancelled, run_export

W, H, FPS = 320, 180, 30
ASSETS = Path(__file__).parent / "assets"


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def short_audio(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """~1 s mp3 clip + its frame count at FPS, so A/V durations are comparable."""
    out = tmp_path_factory.mktemp("audio") / "short.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-i", str(ASSETS / "test.mp3"), "-t", "1.0",
         "-c:a", "libmp3lame", "-q:a", "4", str(out)],
        check=True, capture_output=True)
    dur = float(_probe("-show_entries", "format=duration", "-of", "csv=p=0", str(out)))
    return out, math.ceil(dur * FPS)


class FakeFeatures:
    """Duck-typed stand-in for the analysis Features object."""

    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames
        self._bands = np.zeros(64, np.float32)
        self._waveform = np.zeros(256, np.float32)

    def frame(self, k: int) -> FrameFeatures:
        return FrameFeatures(bands=self._bands, waveform=self._waveform, rms=0.5)


class FakeRenderer:
    """Duck-typed renderer: animated gradient with a real alpha ramp, top-down.

    Mimics the real renderer's reusable buffer: draw() returns a memoryview
    into an internal array that the next draw() overwrites.
    """

    def __init__(self, width: int = W, height: int = H,
                 cancel_event: threading.Event | None = None,
                 cancel_at_draw: int | None = None) -> None:
        self.width, self.height = width, height
        self.configs: list[object] = []
        self.draw_calls = 0
        self._cancel_event = cancel_event
        self._cancel_at_draw = cancel_at_draw
        self._x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
        self._y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
        self._buf = np.empty((height, width, 4), np.uint8)

    def update_config(self, config: object) -> None:
        self.configs.append(config)

    def draw(self, features: object) -> memoryview:
        self.draw_calls += 1
        if self._cancel_at_draw is not None and self.draw_calls >= self._cancel_at_draw:
            assert self._cancel_event is not None
            self._cancel_event.set()
        k = self.draw_calls - 1
        self._buf[..., 0] = ((self._x + 5 * k) % 256).astype(np.uint8)
        self._buf[..., 1] = self._y.astype(np.uint8)
        self._buf[..., 2] = 128
        self._buf[..., 3] = ((self._x + self._y) / 2).astype(np.uint8)  # alpha ramp
        return self._buf.data


# ---------------------------------------------------------------- helpers

def _has_pair(cmd: list[str], flag: str, value: str) -> bool:
    return any(cmd[i] == flag and cmd[i + 1] == value for i in range(len(cmd) - 1))


def _probe(*args: str) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _stream_end(path: Path, spec: str, fallback_dur: float) -> float:
    """End time (s) of a stream = last packet pts_time + its duration."""
    out = _probe("-select_streams", spec, "-show_entries",
                 "packet=pts_time,duration_time", "-of", "csv=p=0", str(path))
    rows = [ln.strip().rstrip(",") for ln in out.splitlines() if ln.strip()]
    fields = rows[-1].split(",")
    pts = float(fields[0])
    try:
        d = float(fields[1])
    except (IndexError, ValueError):
        d = fallback_dur
    return pts + d


def _assert_playable(path: Path) -> None:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr.strip() == "", r.stderr


def _assert_av_sync(path: Path, n_frames: int) -> None:
    v_end = _stream_end(path, "v:0", 1.0 / FPS)
    a_end = _stream_end(path, "a:0", 0.02)
    assert abs(v_end - a_end) < 0.100, f"A/V delta {abs(v_end - a_end):.3f}s"
    assert abs(v_end - n_frames / FPS) < 0.100
    container = float(_probe("-show_entries", "format=duration", "-of", "csv=p=0", str(path)))
    assert abs(container - n_frames / FPS) < 0.150


def _run(mode: str, audio: Path, n_frames: int, out_dir: Path,
         progress: list[Progress] | None = None) -> Path:
    settings = {
        "transparent": EncodeSettings(width=W, height=H, fps=FPS, transparent=True,
                                      quality="fast"),
        "opaque_fast": EncodeSettings(width=W, height=H, fps=FPS, transparent=False,
                                      quality="fast", opaque_codec="av1_nvenc"),
        "opaque_compat": EncodeSettings(width=W, height=H, fps=FPS, transparent=False,
                                        quality="fast", opaque_codec="vp9"),
    }[mode]
    sink: list[Progress] = progress if progress is not None else []
    renderer = FakeRenderer()
    cfg = object()
    final = run_export(
        renderer, FakeFeatures(n_frames), audio, cfg, settings,
        out_dir / f"{mode}.webm", sink.append, threading.Event())
    assert renderer.configs == [cfg]  # export applied its config (WYSIWYG)
    assert renderer.draw_calls == n_frames
    assert final == out_dir / f"{mode}.webm" and final.exists()
    assert not (out_dir / f"{mode}.webm.part").exists()
    return final


# ---------------------------------------------------------------- build_ffmpeg_cmd (pure)

ALL_SETTINGS = [
    EncodeSettings(width=1920, height=1080, fps=60, transparent=t,
                   quality=q, opaque_codec=c)
    for t in (True, False) for q in ("fast", "balanced", "best")
    for c in ("av1_nvenc", "vp9")
]


@pytest.mark.parametrize("s", ALL_SETTINGS)
def test_cmd_invariants(s: EncodeSettings) -> None:
    cmd = build_ffmpeg_cmd(s, Path("a.mp3"), Path("o.webm.part"))
    assert not any("vflip" in a for a in cmd)  # frames are top-down on the wire
    assert "-shortest" not in cmd
    assert "-pix_fmt" not in cmd  # the format filter is authoritative
    for flag, value in (("-map", "0:v:0"), ("-map", "1:a:0"), ("-map_metadata", "-1"),
                        ("-f", "webm"), ("-c:a", "libopus"), ("-ar", "48000"),
                        ("-colorspace", "bt709"), ("-color_range", "tv"),
                        ("-g", str(2 * s.fps))):
        assert _has_pair(cmd, flag, value), (flag, value)
    assert cmd[-1] == "o.webm.part"
    assert cmd[-3:-1] == ["-f", "webm"]  # .part hides the container: -f is load-bearing


@pytest.mark.parametrize("q", ["fast", "balanced", "best"])
def test_cmd_transparent(q: str) -> None:
    s = EncodeSettings(width=1920, height=1080, fps=60, transparent=True, quality=q)
    cmd = build_ffmpeg_cmd(s, Path("a.mp3"), Path("o.webm.part"))
    crf, cpu = VP9_QUALITY[q]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.endswith("format=yuva420p") and "out_color_matrix=bt709" in vf
    for flag, value in (("-c:v", "libvpx-vp9"), ("-crf", str(crf)), ("-cpu-used", str(cpu)),
                        ("-auto-alt-ref", "0"), ("-lag-in-frames", "0"), ("-row-mt", "1"),
                        ("-metadata:s:v:0", "alpha_mode=1"), ("-b:v", "0")):
        assert _has_pair(cmd, flag, value), (flag, value)
    assert "-threads" in cmd and "-tile-columns" in cmd
    assert cmd[cmd.index("-tile-columns") + 1] == "2"  # 1920 -> log2(7)->2


@pytest.mark.parametrize("q", ["fast", "balanced", "best"])
def test_cmd_opaque_fast(q: str) -> None:
    s = EncodeSettings(width=1920, height=1080, fps=60, transparent=False, quality=q)
    cmd = build_ffmpeg_cmd(s, Path("a.mp3"), Path("o.webm.part"))
    cq, preset, multipass, lookahead = AV1_QUALITY[q]
    assert cmd[cmd.index("-vf") + 1].endswith("format=yuv420p")
    for flag, value in (("-c:v", "av1_nvenc"), ("-preset", preset), ("-cq", str(cq)),
                        ("-multipass", multipass), ("-rc-lookahead", str(lookahead)),
                        ("-rc", "vbr"), ("-tune", "hq")):
        assert _has_pair(cmd, flag, value), (flag, value)
    assert "libvpx-vp9" not in cmd


@pytest.mark.parametrize("q", ["fast", "balanced", "best"])
def test_cmd_opaque_compat(q: str) -> None:
    s = EncodeSettings(width=1920, height=1080, fps=60, transparent=False,
                       quality=q, opaque_codec="vp9")
    cmd = build_ffmpeg_cmd(s, Path("a.mp3"), Path("o.webm.part"))
    crf, cpu = VP9_QUALITY[q]
    assert cmd[cmd.index("-vf") + 1].endswith("format=yuv420p")
    for flag, value in (("-c:v", "libvpx-vp9"), ("-crf", str(crf)), ("-cpu-used", str(cpu)),
                        ("-auto-alt-ref", "1"), ("-lag-in-frames", "25")):
        assert _has_pair(cmd, flag, value), (flag, value)
    assert "alpha_mode=1" not in cmd


# ---------------------------------------------------------------- real encodes

def test_transparent_export(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, n_frames = short_audio
    progress: list[Progress] = []
    out = _run("transparent", audio, n_frames, tmp_path, progress)

    info = _probe("-select_streams", "v:0",
                  "-show_entries", "stream=codec_name:stream_tags=alpha_mode",
                  "-of", "default=noprint_wrappers=1", str(out))
    assert "codec_name=vp9" in info
    assert "alpha_mode=1" in info.lower()  # webm writes the tag key uppercase
    # Native ffprobe reports yuv420p even for valid alpha files; the forced
    # libvpx-vp9 decoder is the one that reconstructs the alpha plane.
    forced = _probe("-c:v", "libvpx-vp9", "-select_streams", "v:0",
                    "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(out))
    assert forced == "yuva420p"
    assert verify_alpha(out, sample_time=0.5)

    audio_info = _probe("-select_streams", "a:0", "-show_entries",
                        "stream=codec_name,sample_rate", "-of", "csv=p=0", str(out))
    assert audio_info == "opus,48000"
    n_packets = int(_probe("-select_streams", "v:0", "-count_packets",
                           "-show_entries", "stream=nb_read_packets",
                           "-of", "csv=p=0", str(out)))
    assert n_packets == n_frames
    _assert_av_sync(out, n_frames)
    _assert_playable(out)

    assert progress, "progress callback never fired"
    assert all(0 <= p.frames_done <= p.total_frames == n_frames for p in progress)
    assert all(p.fps >= 0.0 for p in progress)
    last = progress[-1]
    assert last.frames_done == n_frames and last.eta_seconds == 0.0


def test_opaque_fast_export(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, n_frames = short_audio
    out = _run("opaque_fast", audio, n_frames, tmp_path)
    assert _probe("-select_streams", "v:0", "-show_entries", "stream=codec_name",
                  "-of", "csv=p=0", str(out)) == "av1"
    assert _probe("-select_streams", "a:0", "-show_entries", "stream=codec_name",
                  "-of", "csv=p=0", str(out)) == "opus"
    n_packets = int(_probe("-select_streams", "v:0", "-count_packets",
                           "-show_entries", "stream=nb_read_packets",
                           "-of", "csv=p=0", str(out)))
    assert n_packets == n_frames
    _assert_av_sync(out, n_frames)
    _assert_playable(out)
    assert not verify_alpha(out)  # av1 -> stage 1 rejects


def test_opaque_compat_export(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, n_frames = short_audio
    out = _run("opaque_compat", audio, n_frames, tmp_path)
    info = _probe("-select_streams", "v:0",
                  "-show_entries", "stream=codec_name,pix_fmt",
                  "-of", "csv=p=0", str(out))
    assert info == "vp9,yuv420p"
    assert _probe("-select_streams", "a:0", "-show_entries", "stream=codec_name",
                  "-of", "csv=p=0", str(out)) == "opus"
    _assert_av_sync(out, n_frames)
    _assert_playable(out)
    assert not verify_alpha(out, sample_time=0.5)  # opaque: alpha decodes as 255


# ---------------------------------------------------------------- cancel semantics

def test_pipeline_cancel_mid_export(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, n_frames = short_audio
    assert n_frames > 12
    cancel = threading.Event()
    renderer = FakeRenderer(cancel_event=cancel, cancel_at_draw=10)
    s = EncodeSettings(width=W, height=H, fps=FPS, transparent=True, quality="fast")
    out = tmp_path / "cancelled.webm"
    with pytest.raises(EncodeCancelled):
        run_export(renderer, FakeFeatures(n_frames), audio, None, s, out,
                   lambda p: None, cancel)
    assert renderer.draw_calls == 10  # stopped at the next per-frame check
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists(), "cancel left a .part behind"


def test_encoder_cancel_direct(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, n_frames = short_audio
    s = EncodeSettings(width=W, height=H, fps=FPS, transparent=True, quality="fast")
    part = tmp_path / "direct.webm.part"
    renderer = FakeRenderer()
    enc = Encoder(s, audio, part, total_frames=n_frames)
    enc.start()
    for k in range(5):
        enc.write_frame(renderer.draw(None))
    enc.cancel()
    with pytest.raises(EncodeCancelled):
        enc.write_frame(renderer.draw(None))
    assert not part.exists()
    enc.cancel()  # idempotent


# ---------------------------------------------------------------- validation + naming

def test_settings_validation(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, _ = short_audio
    with pytest.raises(ValueError, match="even"):
        EncodeSettings(width=321, height=180, fps=30, transparent=False)
    with pytest.raises(ValueError, match="even"):
        EncodeSettings(width=320, height=181, fps=30, transparent=False)
    ok = EncodeSettings(width=W, height=H, fps=FPS, transparent=True)
    with pytest.raises(ValueError, match="total_frames"):
        Encoder(ok, audio, tmp_path / "x.webm.part", total_frames=0)
    with pytest.raises(ValueError, match="audio"):
        Encoder(ok, tmp_path / "missing.mp3", tmp_path / "x.webm.part", total_frames=1)


def test_write_frame_size_validation(short_audio: tuple[Path, int], tmp_path: Path) -> None:
    audio, _ = short_audio
    s = EncodeSettings(width=W, height=H, fps=FPS, transparent=True)
    enc = Encoder(s, audio, tmp_path / "x.webm.part", total_frames=10)
    with pytest.raises(ValueError, match="bytes"):
        enc.write_frame(b"\x00" * 100)


def test_export_exception_aliases() -> None:
    from visualizer.encode import EncodeError
    from visualizer.pipeline import ExportError
    assert ExportCancelled is EncodeCancelled and ExportError is EncodeError


def test_default_output_name() -> None:
    s = EncodeSettings(width=1920, height=1080, fps=60, transparent=True)
    assert (default_output_name(Path(r"C:\m\My Song (mix).mp3"), "radial_bars", s)
            == "My_Song_(mix)__radial_bars__1920x1080@60__alpha.webm")
    s2 = EncodeSettings(width=1920, height=400, fps=30, transparent=False)
    assert (default_output_name(Path("a.mp3"), "bars", s2)
            == "a__bars__1920x400@30__av1.webm")
    s3 = EncodeSettings(width=640, height=360, fps=30, transparent=False,
                        opaque_codec="vp9")
    assert (default_output_name(Path("???.mp3"), "wave", s3)
            == "audio__wave__640x360@30__vp9.webm")


def test_unique_path(tmp_path: Path) -> None:
    p = tmp_path / "out.webm"
    assert unique_path(p) == p
    p.touch()
    (tmp_path / "out-1.webm").touch()
    assert unique_path(p) == tmp_path / "out-2.webm"
