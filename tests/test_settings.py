"""Unit tests for the pure parts of visualizer.settings.

build_configs wiring (FeatureConfig/RenderConfig/EncodeSettings) is exercised in
integration — importing sibling modules here would couple this suite to code
being written concurrently.
"""
import pytest
from pydantic import ValidationError

from visualizer.settings import (
    STYLE_MAP,
    RenderSettings,
    parse_hex,
    preview_size,
)


class TestParseHex:
    def test_parses_channels(self):
        assert parse_hex("#ff0000") == (1.0, 0.0, 0.0)
        assert parse_hex("#00ff00") == (0.0, 1.0, 0.0)
        assert parse_hex("#0000ff") == (0.0, 0.0, 1.0)

    def test_black_and_white(self):
        assert parse_hex("#000000") == (0.0, 0.0, 0.0)
        assert parse_hex("#ffffff") == (1.0, 1.0, 1.0)

    def test_mixed_value(self):
        r, g, b = parse_hex("#3ce6a8")
        assert r == pytest.approx(0x3C / 255)
        assert g == pytest.approx(0xE6 / 255)
        assert b == pytest.approx(0xA8 / 255)

    def test_uppercase_accepted(self):
        assert parse_hex("#3CE6A8") == parse_hex("#3ce6a8")

    @pytest.mark.parametrize("bad", ["3ce6a8", "#3ce6a", "#3ce6a8f", "#zzzzzz", "", "#", "red"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_hex(bad)


class TestStyleMap:
    def test_covers_all_ui_styles(self):
        assert set(STYLE_MAP) == {"bars", "waveform", "line", "radial_bars", "circle_wave"}

    def test_targets_are_render_enum_names(self):
        assert set(STYLE_MAP.values()) == {"BARS", "WAVEFORM", "LINE", "RADIAL_BARS", "RADIAL_WAVE"}

    def test_circle_wave_maps_to_radial_wave(self):
        assert STYLE_MAP["circle_wave"] == "RADIAL_WAVE"
        assert STYLE_MAP["radial_bars"] == "RADIAL_BARS"


class TestRenderSettingsValidation:
    def test_defaults_valid(self):
        rs = RenderSettings()
        assert rs.style == "bars"
        assert rs.bg_mode == "solid"
        assert (rs.width, rs.height, rs.fps) == (1920, 1080, 60)
        assert rs.codec == "av1_nvenc"

    def test_frontend_shaped_payload_roundtrip(self):
        payload = {
            "style": "circle_wave", "color1": "#FF00aa", "color2": "#012345",
            "gradient": False, "bg_mode": "transparent", "bg_color": "#101014",
            "bg_image_id": None,
            "sensitivity": 2.5, "width": 1920, "height": 400, "fps": 30,
            "quality": "best", "codec": "vp9",
        }
        rs = RenderSettings.model_validate(payload)
        assert rs.model_dump() == {**payload}

    @pytest.mark.parametrize("field,value", [
        ("style", "spiral"),
        ("color1", "3ce6a8"),
        ("color1", "#12345"),
        ("color2", "#gggggg"),
        ("bg_mode", "gradient"),
        ("bg_color", "not-a-color"),
        ("sensitivity", 0.05),
        ("sensitivity", 4.5),
        ("width", 15),        # below minimum
        ("width", 1919),      # odd
        ("height", 5000),     # above maximum
        ("height", 401),      # odd
        ("fps", 24),
        ("quality", "ultra"),
        ("codec", "h264"),
    ])
    def test_rejects_invalid(self, field, value):
        with pytest.raises(ValidationError):
            RenderSettings(**{field: value})

    def test_json_string_parsing(self):
        rs = RenderSettings.model_validate_json(
            '{"style":"line","width":1920,"height":400,"fps":30}')
        assert (rs.style, rs.width, rs.height, rs.fps) == ("line", 1920, 400, 30)


class TestPreviewSize:
    def _size(self, w, h):
        return preview_size(RenderSettings(width=w, height=h))

    def test_1080p(self):
        assert self._size(1920, 1080) == (960, 540)

    def test_strip_preserves_export_aspect(self):
        assert self._size(1920, 400) == (960, 200)

    def test_4k_fits_same_box(self):
        assert self._size(3840, 2160) == (960, 540)

    def test_no_upscale_below_box(self):
        assert self._size(640, 360) == (640, 360)

    def test_portrait(self):
        pw, ph = self._size(1080, 1920)
        assert ph == 540
        assert pw < ph

    @pytest.mark.parametrize("w,h", [(1920, 1080), (1920, 400), (3840, 2160),
                                     (1080, 1920), (2560, 1440), (16, 4096)])
    def test_always_even_and_positive(self, w, h):
        pw, ph = self._size(w, h)
        assert pw % 2 == 0 and ph % 2 == 0
        assert pw >= 2 and ph >= 2
        assert pw <= 960 and ph <= 540
