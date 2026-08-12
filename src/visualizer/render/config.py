"""Render configuration, style enum, and config-change classification."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class Style(str, Enum):
    BARS = "bars"                # bar spectrum
    WAVEFORM = "waveform"        # thick polyline over PCM
    LINE = "line"                # smooth line spectrum (Catmull-Rom over bands)
    RADIAL_BARS = "radial_bars"  # bars radiating from a circle
    RADIAL_WAVE = "radial_wave"  # waveform wrapped around a circle
    CIRCLE_WAVE = "radial_wave"  # alias — the GUI's name for RADIAL_WAVE

    @classmethod
    def _missing_(cls, value: object) -> "Style | None":
        if value == "circle_wave":
            return cls.RADIAL_WAVE
        return None


#: Styles that consume FrameFeatures.bands; the rest consume .waveform.
BAND_STYLES = frozenset({Style.BARS, Style.LINE, Style.RADIAL_BARS})
#: Styles drawn with the rect (bar) primitive; the rest use capsule segments.
RECT_STYLES = frozenset({Style.BARS, Style.RADIAL_BARS})


class GradientAxis(str, Enum):
    ALONG_BAR = "along_bar"            # gradient over bar length axis
    ALONG_SPECTRUM = "along_spectrum"  # gradient over frequency / sample-position axis


@dataclass
class RenderConfig:
    # canvas
    width: int = 1920            # must be even (yuva420p subsampling)
    height: int = 1080           # must be even
    fps: int = 60
    msaa: int = 4                # 0 disables; clamped to ctx.max_samples
    style: Style = Style.BARS
    # colors (floats 0..1, RGB; alpha is never user-set — output alpha comes from coverage)
    color: tuple[float, float, float] = (0.10, 0.95, 0.80)
    color2: tuple[float, float, float] | None = None   # None => solid color
    gradient_axis: GradientAxis = GradientAxis.ALONG_BAR
    background: tuple[float, float, float] | None = None  # None => transparent; renderer BAKES it
    # Absolute path of a background image, drawn cover-fit UNDER glow+layer.
    # Overrides `background` when set (both preview and export — WYSIWYG).
    # Uniform-level change: the renderer swaps the GPU texture when the path changes.
    background_image: str | None = None
    # bars / radial bars
    bar_count: int = 96          # 8..512; analysis delivers exactly this many bands
    bar_gap_frac: float = 0.25   # gap as fraction of one slot (0..0.9)
    corner_radius_frac: float = 0.5   # 0..0.5 of bar width (0.5 = pill)
    mirror: bool = False         # bars: centered ±; radial: extends inward too;
                                 # waveform/line: second reflected polyline
    baseline_frac: float = 1.0   # bars/line: baseline y as fraction of height (1.0 = bottom edge)
    # polyline styles
    thickness: float = 4.0       # stroke thickness px (full-res)
    wave_points: int = 1024      # samples per frame the analysis delivers, 256..4096
    line_subdiv: int = 8         # Catmull-Rom subdivisions per band (LINE style)
    # radial styles
    inner_radius_frac: float = 0.22   # of min(width, height)
    radial_extent_frac: float = 0.28  # max bar length / wave amplitude, of min(w, h)
    # amplitude (sensitivity slider maps here — per-frame scale, never a geometry rebuild)
    amplitude: float = 1.0
    # glow
    glow: bool = True
    glow_strength: float = 1.6   # 0..4; multiplier on blurred layer at composite
    glow_radius: float = 24.0    # blur radius px at full res, 4..96

    def validate(self) -> None:
        def _check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"RenderConfig: {msg}")

        _check(self.width % 2 == 0 and self.height % 2 == 0,
               f"width/height must be even for 4:2:0 subsampling, got {self.width}x{self.height}")
        _check(16 <= self.width <= 4096 and 16 <= self.height <= 4096,
               f"width/height must be in [16, 4096], got {self.width}x{self.height}")
        _check(self.fps > 0, f"fps must be positive, got {self.fps}")
        _check(self.msaa in (0, 2, 4, 8, 16), f"msaa must be one of 0/2/4/8/16, got {self.msaa}")
        _check(isinstance(self.style, Style), f"style must be a Style, got {self.style!r}")
        for name, col in (("color", self.color), ("color2", self.color2),
                          ("background", self.background)):
            if col is None:
                continue
            _check(len(col) == 3 and all(0.0 <= c <= 1.0 for c in col),
                   f"{name} must be 3 floats in [0, 1], got {col!r}")
        _check(self.background_image is None or isinstance(self.background_image, str),
               f"background_image must be a path string or None, got {self.background_image!r}")
        _check(8 <= self.bar_count <= 512, f"bar_count must be in [8, 512], got {self.bar_count}")
        _check(0.0 <= self.bar_gap_frac <= 0.9,
               f"bar_gap_frac must be in [0, 0.9], got {self.bar_gap_frac}")
        _check(0.0 <= self.corner_radius_frac <= 0.5,
               f"corner_radius_frac must be in [0, 0.5], got {self.corner_radius_frac}")
        _check(0.0 <= self.baseline_frac <= 1.0,
               f"baseline_frac must be in [0, 1], got {self.baseline_frac}")
        _check(self.thickness > 0.0, f"thickness must be positive, got {self.thickness}")
        _check(256 <= self.wave_points <= 4096,
               f"wave_points must be in [256, 4096], got {self.wave_points}")
        _check(1 <= self.line_subdiv <= 64,
               f"line_subdiv must be in [1, 64], got {self.line_subdiv}")
        _check(0.0 < self.inner_radius_frac < 0.5,
               f"inner_radius_frac must be in (0, 0.5), got {self.inner_radius_frac}")
        _check(0.0 < self.radial_extent_frac <= 0.5,
               f"radial_extent_frac must be in (0, 0.5], got {self.radial_extent_frac}")
        _check(self.amplitude > 0.0, f"amplitude must be positive, got {self.amplitude}")
        _check(0.0 <= self.glow_strength <= 4.0,
               f"glow_strength must be in [0, 4], got {self.glow_strength}")
        _check(4.0 <= self.glow_radius <= 96.0,
               f"glow_radius must be in [4, 96], got {self.glow_radius}")


class RebuildLevel(Enum):
    UNIFORMS = 0       # color, gradient, glow params, amplitude, thickness, corner radius
    GEOMETRY = 1       # style, bar_count, wave_points, mirror, gaps, radii, baseline
    FRAMEBUFFERS = 2   # width, height, msaa


_FRAMEBUFFER_FIELDS = frozenset({"width", "height", "msaa"})
_GEOMETRY_FIELDS = frozenset({
    "style", "bar_count", "bar_gap_frac", "mirror", "baseline_frac",
    "wave_points", "line_subdiv", "inner_radius_frac", "radial_extent_frac",
})


def config_diff(old: RenderConfig, new: RenderConfig) -> RebuildLevel:
    """Most expensive rebuild required to go from `old` to `new`.

    Returns UNIFORMS when nothing (or only uniform-level fields) changed —
    re-applying uniforms is always cheap and always done.
    """
    level = RebuildLevel.UNIFORMS
    for f in fields(RenderConfig):
        if getattr(old, f.name) == getattr(new, f.name):
            continue
        if f.name in _FRAMEBUFFER_FIELDS:
            return RebuildLevel.FRAMEBUFFERS
        if f.name in _GEOMETRY_FIELDS:
            level = RebuildLevel.GEOMETRY
    return level
