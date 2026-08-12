"""Renderer smoke + contract tests. Needs a real GPU/GL 3.3 context (this machine has one)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualizer.features import FrameFeatures
from visualizer.render import GradientAxis, RenderConfig, Renderer, Style
from visualizer.render.config import RebuildLevel, config_diff
from visualizer.render.geometry import (
    build_circle_points,
    catmull_rom,
    polyline_to_segments,
)
from visualizer.render.passes import gauss_weights

SMOKE_DIR = Path(
    r"C:\Users\oiehn\AppData\Local\Temp\claude"
    r"\C--Users-oiehn-Desktop-Audio-Visualizer"
    r"\dec1fb6d-b3c7-46af-b66c-a0b03d61bd77\scratchpad\render_smoke"
)
W, H = 1280, 720
COLOR = (0.10, 0.95, 0.80)
COLOR2 = (0.85, 0.20, 0.90)
BG_SOLID = (0.07, 0.07, 0.10)
STYLES = [Style.BARS, Style.WAVEFORM, Style.LINE, Style.RADIAL_BARS, Style.RADIAL_WAVE]


def smoke_config(style: Style, background: tuple[float, float, float] | None) -> RenderConfig:
    return RenderConfig(
        width=W, height=H, msaa=4, style=style, background=background,
        color=COLOR, color2=COLOR2, gradient_axis=GradientAxis.ALONG_SPECTRUM,
        glow=True,
    )


def make_features(cfg: RenderConfig, phase: float = 0.0) -> FrameFeatures:
    bands = np.abs(np.sin(np.linspace(0.0, np.pi, cfg.bar_count) + phase)).astype(np.float32)
    t = np.linspace(0.0, 4.0 * np.pi, cfg.wave_points, dtype=np.float32)
    waveform = (0.8 * np.sin(t + phase)).astype(np.float32)
    return FrameFeatures(bands=bands, waveform=waveform, rms=0.6)


def as_array(buf: memoryview, w: int, h: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    return arr.reshape(h, w, 4)


@pytest.fixture(scope="module")
def renderer():
    r = Renderer(smoke_config(Style.BARS, None))
    yield r
    r.close()


# -- pure-python unit tests (no GL) -----------------------------------------

def test_style_alias_circle_wave():
    assert Style.CIRCLE_WAVE is Style.RADIAL_WAVE
    assert Style("circle_wave") is Style.RADIAL_WAVE
    assert {Style.BARS, Style.WAVEFORM, Style.LINE, Style.RADIAL_BARS,
            Style.CIRCLE_WAVE} == set(STYLES)


def test_validate_rejects_odd_dimensions():
    with pytest.raises(ValueError, match="even"):
        RenderConfig(width=1919, height=400).validate()


def test_config_diff_levels():
    import dataclasses
    base = RenderConfig()
    assert config_diff(base, dataclasses.replace(base)) is RebuildLevel.UNIFORMS
    assert config_diff(base, dataclasses.replace(base, amplitude=2.0)) is RebuildLevel.UNIFORMS
    assert config_diff(base, dataclasses.replace(base, color=(1, 0, 0), glow=False)) \
        is RebuildLevel.UNIFORMS
    assert config_diff(base, dataclasses.replace(base, bar_count=128)) is RebuildLevel.GEOMETRY
    assert config_diff(base, dataclasses.replace(base, style=Style.LINE)) is RebuildLevel.GEOMETRY
    assert config_diff(base, dataclasses.replace(base, width=1280, height=720)) \
        is RebuildLevel.FRAMEBUFFERS
    assert config_diff(base, dataclasses.replace(base, msaa=0, bar_count=64)) \
        is RebuildLevel.FRAMEBUFFERS


def test_catmull_rom_interpolates_inputs():
    y = np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float32)
    out = catmull_rom(y, 8)
    assert out.shape == (3 * 8 + 1,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[::8], y, atol=1e-6)


def test_polyline_to_segments_layout():
    pts = np.array([[0, 0], [10, 0], [10, 5]], dtype=np.float32)
    segs = polyline_to_segments(pts)
    assert segs.shape == (2, 6)
    np.testing.assert_allclose(segs[0], [0, 0, 10, 0, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(segs[1], [10, 0, 10, 5, 0.5, 1.0], atol=1e-6)


def test_circle_points_close_the_loop():
    cfg = smoke_config(Style.RADIAL_WAVE, None)
    wave = (0.7 * np.sin(np.linspace(0, 11.0, cfg.wave_points))).astype(np.float32)
    (pts,) = build_circle_points(wave, cfg)
    assert pts.shape == (cfg.wave_points + 1, 2)
    np.testing.assert_allclose(pts[0], pts[-1], atol=1e-4)      # closed
    center = np.array([cfg.width / 2, cfg.height / 2])
    r = np.linalg.norm(pts - center, axis=1)
    assert abs(r[-2] - r[0]) < 1.0                              # seam crossfade: no jump


def test_gauss_weights_normalized():
    w = gauss_weights(4.0)
    assert w.shape == (13,)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert w[6] == w.max()


# -- GL smoke tests ----------------------------------------------------------

@pytest.mark.parametrize("bg_name,background", [("transparent", None), ("solid", BG_SOLID)])
@pytest.mark.parametrize("style", STYLES, ids=[s.name.lower() for s in STYLES])
def test_smoke_style(renderer, style, bg_name, background):
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = smoke_config(style, background)
    renderer.update_config(cfg)
    for k, phase in enumerate((0.0, 0.3, 0.6)):
        buf = renderer.draw(make_features(cfg, phase))
        assert len(buf) == W * H * 4
        arr = as_array(buf, W, H)
        img = Image.frombuffer("RGBA", (W, H), bytes(buf), "raw", "RGBA", 0, 1)
        img.save(SMOKE_DIR / f"{style.value}_{bg_name}_{k}.png")
        alpha = arr[:, :, 3]
        if background is None:
            assert alpha[0, 0] == 0, "transparent bg: corner pixel must have alpha 0"
            assert alpha.max() == 255, "transparent bg: content interior must be opaque"
            assert (alpha > 0).sum() > 1000, "drawing should cover a meaningful area"
        else:
            assert alpha.min() == 255, "solid bg: every pixel must be fully opaque"
            bg255 = np.array([round(c * 255) for c in background], dtype=np.int16)
            diff = np.abs(arr[:, :, :3].astype(np.int16) - bg255).max(axis=2)
            assert (diff > 10).sum() > 1000, "content must differ from the solid background"


def test_top_down_orientation(renderer):
    """Bars anchor at the bottom edge; in a TOP-DOWN buffer that coverage must land
    in the LAST rows. If someone flips the projection or adds np.flipud, this fails."""
    cfg = smoke_config(Style.BARS, None)
    cfg = __import__("dataclasses").replace(cfg, amplitude=0.3)  # max bar height ~216px
    renderer.update_config(cfg)
    arr = as_array(renderer.draw(make_features(cfg, 0.3)), W, H)
    covered = arr[:, :, 3] > 0
    top_quarter = covered[: H // 4].sum()
    bottom_quarter = covered[3 * H // 4:].sum()
    assert bottom_quarter > 50_000, "bottom rows must hold the bars"
    assert top_quarter == 0, "top rows must be empty — buffer would be upside-down"


def test_straight_alpha_edges(renderer):
    """At partially covered pixels straight alpha keeps RGB at the source color;
    premultiplied leakage would scale RGB down by alpha."""
    cfg = smoke_config(Style.BARS, None)
    cfg = __import__("dataclasses").replace(cfg, color2=None, glow=False)
    renderer.update_config(cfg)
    arr = as_array(renderer.draw(make_features(cfg, 0.3)), W, H).astype(np.int16)
    edge = (arr[:, :, 3] >= 60) & (arr[:, :, 3] <= 200)
    assert edge.sum() > 50, "expected antialiased edge pixels"
    expected = np.array([round(c * 255) for c in COLOR], dtype=np.int16)
    err = np.abs(arr[:, :, :3][edge] - expected).max()
    assert err <= 3, f"edge RGB deviates from source color by {err} (premultiplied leak?)"


def test_feature_shape_validation(renderer):
    cfg = smoke_config(Style.BARS, None)
    renderer.update_config(cfg)
    bad = make_features(cfg)
    with pytest.raises(ValueError, match="bands"):
        renderer.draw(FrameFeatures(bands=bad.bands[:-1], waveform=bad.waveform, rms=0.5))
    cfg_w = smoke_config(Style.WAVEFORM, None)
    renderer.update_config(cfg_w)
    with pytest.raises(ValueError, match="waveform"):
        renderer.draw(FrameFeatures(bands=bad.bands, waveform=bad.waveform[:-3], rms=0.5))


def test_readback_double_buffering(renderer):
    cfg = smoke_config(Style.BARS, None)
    renderer.update_config(cfg)
    f = make_features(cfg)
    m1, m2, m3 = renderer.draw(f), renderer.draw(f), renderer.draw(f)
    assert m1.obj is not m2.obj, "consecutive draws must alternate readback buffers"
    assert m3.obj is m1.obj, "buffer is recycled two draws later (documented contract)"


def test_fps_1080p(renderer):
    """Measure draw+readback throughput at 1920x1080 (exercises FBO resize too)."""
    cfg = RenderConfig(width=1920, height=1080, msaa=4, style=Style.BARS,
                       color=COLOR, color2=COLOR2, glow=True)
    renderer.update_config(cfg)
    frames = [make_features(cfg, 0.05 * i) for i in range(20)]
    for i in range(10):
        renderer.draw(frames[i])
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        renderer.draw(frames[i % len(frames)])
    dt = time.perf_counter() - t0
    fps = n / dt
    print(f"\n[render perf] 1920x1080 draw+readback: {fps:.1f} fps ({dt / n * 1000:.2f} ms/frame)")
    assert fps > 60, f"renderer unexpectedly slow: {fps:.1f} fps"
