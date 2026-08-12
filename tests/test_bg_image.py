"""Background-image feature: settings mapping, config classification, GL compositing,
and the server-side image ingest. GL tests need a real GPU context (this machine has one)."""
from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from visualizer.render import RenderConfig, Renderer, Style
from visualizer.render.config import RebuildLevel, config_diff
from visualizer.settings import RenderSettings, build_configs

W, H = 1280, 720


def _bg_png(tmp_path, name: str, w: int, h: int) -> str:
    """Top half pure red, bottom half pure blue — orientation- and crop-sensitive."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, :, 0] = 255
    arr[h // 2:, :, 2] = 255
    p = tmp_path / name
    Image.fromarray(arr, "RGB").save(p)
    return str(p)


def quiet_config(**kw) -> RenderConfig:
    """Config whose drawing covers almost nothing, so the bg dominates the frame."""
    return RenderConfig(width=W, height=H, style=Style.BARS, amplitude=0.001,
                        glow=False, **kw)


class TestSettingsMapping:
    def test_bg_image_id_validation(self):
        assert RenderSettings().bg_image_id is None
        assert RenderSettings(bg_image_id="0123456789ab").bg_image_id == "0123456789ab"
        for bad in ("xyz", "0123456789ABCD", "0123456789ZZ", "../etc/passwd"):
            with pytest.raises(ValidationError):
                RenderSettings(bg_image_id=bad)

    def test_image_mode_sets_path(self):
        rs = RenderSettings(bg_mode="image", bg_image_id="0123456789ab")
        _, render_cfg, enc = build_configs(rs, None, bg_image_path="C:/x/bg.png")
        assert render_cfg.background_image == "C:/x/bg.png"
        assert render_cfg.background is not None      # opaque mode keeps a solid fallback
        assert enc.transparent is False

    def test_image_mode_without_resolved_path(self):
        rs = RenderSettings(bg_mode="image", bg_image_id="0123456789ab")
        _, render_cfg, _ = build_configs(rs, None)
        assert render_cfg.background_image is None

    @pytest.mark.parametrize("mode", ["solid", "transparent"])
    def test_other_modes_ignore_stale_path(self, mode):
        rs = RenderSettings(bg_mode=mode)
        _, render_cfg, enc = build_configs(rs, None, bg_image_path="C:/x/bg.png")
        assert render_cfg.background_image is None
        assert enc.transparent == (mode == "transparent")
        assert (render_cfg.background is None) == (mode == "transparent")


def test_config_diff_background_image_is_uniform_level():
    base = RenderConfig()
    changed = dataclasses.replace(base, background_image="C:/x/bg.png")
    assert config_diff(base, changed) is RebuildLevel.UNIFORMS


@pytest.fixture(scope="module")
def renderer():
    r = Renderer(quiet_config(background=(0, 0, 0)))
    yield r
    r.close()


class TestRendererBackground:
    def _features(self, cfg: RenderConfig):
        from visualizer.features import FrameFeatures
        return FrameFeatures(
            bands=np.zeros(cfg.bar_count, np.float32),
            waveform=np.zeros(cfg.wave_points, np.float32),
            rms=0.0,
        )

    def _draw(self, renderer, cfg) -> np.ndarray:
        renderer.update_config(cfg)
        buf = renderer.draw(self._features(cfg))
        return np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4).copy()

    def test_image_fills_frame_top_down(self, renderer, tmp_path):
        path = _bg_png(tmp_path, "same_ar.png", 320, 180)  # same 16:9 -> no crop
        arr = self._draw(renderer, quiet_config(background_image=path))
        assert arr[:, :, 3].min() == 255, "image background must be fully opaque"
        # top rows red, bottom rows blue — proves the image is not vertically flipped
        assert arr[5, W // 2, 0] > 200 and arr[5, W // 2, 2] < 50
        assert arr[H - 5, W // 2, 2] > 200 and arr[H - 5, W // 2, 0] < 50

    def test_cover_fit_crops_square_image(self, renderer, tmp_path):
        # square image on a 16:9 canvas: cover-fit crops top/bottom but the frame
        # must still be red on top and blue at bottom (center crop, no letterbox)
        path = _bg_png(tmp_path, "square.png", 200, 200)
        arr = self._draw(renderer, quiet_config(background_image=path))
        assert arr[5, W // 2, 0] > 200
        assert arr[H - 5, W // 2, 2] > 200
        # left/right edges identical to center columns (no horizontal squeeze)
        np.testing.assert_allclose(arr[5, 5, :3], arr[5, W - 5, :3], atol=2)

    def test_swap_back_to_solid_and_transparent(self, renderer, tmp_path):
        path = _bg_png(tmp_path, "swap.png", 320, 180)
        self._draw(renderer, quiet_config(background_image=path))
        solid = self._draw(renderer, quiet_config(background=(0.5, 0.5, 0.5)))
        assert abs(int(solid[5, W // 2, 0]) - 128) <= 2, "solid bg must return after image"
        transparent = self._draw(renderer, quiet_config(background=None))
        assert transparent[5, W // 2, 3] == 0, "transparent bg must return after image"

    def test_missing_file_raises_cleanly(self, renderer, tmp_path):
        with pytest.raises(ValueError, match="background image"):
            self._draw(renderer, quiet_config(background_image=str(tmp_path / "nope.png")))
        # renderer must remain usable afterwards
        ok = self._draw(renderer, quiet_config(background=(0, 0, 0)))
        assert ok.shape == (H, W, 4)

    def test_failed_load_retries_same_path(self, renderer, tmp_path):
        """A failed load must NOT latch the path: once the file appears (content-hash
        ids mean a re-upload recreates the SAME path), the same config must render it."""
        path = tmp_path / "late.png"
        with pytest.raises(ValueError, match="background image"):
            self._draw(renderer, quiet_config(background_image=str(path)))
        _bg_png(tmp_path, "late.png", 320, 180)  # file appears at the identical path
        arr = self._draw(renderer, quiet_config(background_image=str(path)))
        assert arr[5, W // 2, 0] > 200, "retry with the same path must load the image"


class TestServerIngest:
    def test_ingest_is_content_addressed(self):
        from visualizer.server import _ingest_background
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 200, 30)).save(buf, "PNG")
        raw = buf.getvalue()
        id1, path1, w, h = _ingest_background(raw)
        id2, path2, _, _ = _ingest_background(raw)
        assert id1 == id2 and path1 == path2
        assert (w, h) == (64, 64)
        assert path1.exists()

    def test_ingest_downscales_and_flattens(self):
        from visualizer.server import MAX_BG_DIM, _ingest_background
        img = Image.new("RGBA", (5000, 500), (255, 0, 0, 128))  # translucent red
        buf = io.BytesIO()
        img.save(buf, "PNG")
        _, path, w, h = _ingest_background(buf.getvalue())
        assert max(w, h) <= MAX_BG_DIM
        with Image.open(path) as out:
            assert out.mode == "RGB"  # alpha flattened: renderer gets an opaque bg
            r, g, b = out.getpixel((w // 2, h // 2))
        assert abs(r - 128) <= 2 and g == 0 and b == 0  # composited over black

    def test_ingest_rejects_non_image(self):
        from visualizer.server import _ingest_background
        with pytest.raises(RuntimeError, match="image"):
            _ingest_background(b"definitely not an image")
