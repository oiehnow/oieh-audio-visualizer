"""User-facing render settings and the single mapper to subsystem configs.

RenderSettings is the flat JSON schema shared by the frontend, the preview
WebSocket, and the export request.  build_configs() is the ONLY place that
translates it into the analysis/render/encode config objects — every rule
that keeps preview and export identical (WYSIWYG) lives here.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .encode import EncodeSettings
    from .features import FeatureConfig
    from .render import RenderConfig

HEX_PATTERN = r"^#[0-9a-fA-F]{6}$"

# style string (RenderSettings.style) -> render.Style member name.
# Kept as plain strings so this table is testable without the render package.
STYLE_MAP: dict[str, str] = {
    "bars": "BARS",
    "waveform": "WAVEFORM",
    "line": "LINE",
    "radial_bars": "RADIAL_BARS",
    "circle_wave": "RADIAL_WAVE",
}

PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 540


class RenderSettings(BaseModel):
    style: Literal["bars", "waveform", "line", "radial_bars", "circle_wave"] = "bars"
    color1: str = Field("#3ce6a8", pattern=HEX_PATTERN)
    color2: str = Field("#1e5cff", pattern=HEX_PATTERN)
    gradient: bool = True  # False -> color1 only
    bg_mode: Literal["transparent", "solid", "image"] = "solid"
    bg_color: str = Field("#101014", pattern=HEX_PATTERN)  # used only when bg_mode == "solid"
    # server-issued id of an uploaded background image (bg_mode == "image");
    # the SERVER resolves it to a file path — settings never carry paths.
    bg_image_id: str | None = Field(None, pattern=r"^[0-9a-f]{12}$")
    sensitivity: float = Field(1.0, ge=0.1, le=4.0)
    width: int = Field(1920, ge=16, le=4096, multiple_of=2)
    height: int = Field(1080, ge=16, le=4096, multiple_of=2)
    fps: Literal[30, 60] = 60
    quality: Literal["fast", "balanced", "best"] = "balanced"
    # Opaque-export codec only (transparent always uses VP9+alpha). Exposed in the
    # GUI because AV1-in-WebM is unreadable by several NLEs.
    codec: Literal["av1_nvenc", "vp9"] = "av1_nvenc"


def parse_hex(color: str) -> tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) floats in [0, 1]. Raises ValueError on anything else."""
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"expected '#RRGGBB', got {color!r}")
    try:
        r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        raise ValueError(f"expected '#RRGGBB', got {color!r}") from None
    return (r / 255.0, g / 255.0, b / 255.0)


def preview_size(rs: RenderSettings) -> tuple[int, int]:
    """Fit the export resolution into 960x540 preserving aspect, even dims, no upscale."""
    scale = min(PREVIEW_MAX_W / rs.width, PREVIEW_MAX_H / rs.height, 1.0)
    return (
        max(2, round(rs.width * scale / 2) * 2),
        max(2, round(rs.height * scale / 2) * 2),
    )


def build_configs(rs: RenderSettings, audio: Any,
                  bg_image_path: "str | None" = None) -> "tuple[FeatureConfig, RenderConfig, EncodeSettings]":
    """Map RenderSettings (+ the loaded AudioData) to the three subsystem configs.

    Invariants enforced here and nowhere else:
    - EncodeSettings.transparent == (bg_mode == 'transparent') == (RenderConfig.background is None)
    - RenderConfig.background_image is set only when bg_mode == 'image'; the caller
      (server) resolves rs.bg_image_id to a real file path — bg_image_path is ignored
      in the other modes, so a stale id can never tint a solid/transparent render.
    - FeatureConfig band/point counts come FROM RenderConfig (renderer never resamples)
    - sensitivity maps to RenderConfig.amplitude (a uniform), never into FeatureConfig,
      so slider changes never trigger a Features rebuild.

    Sibling imports are deferred so the pure helpers above stay importable alone.
    """
    from .encode import EncodeSettings
    from .features import FeatureConfig
    from .render import RenderConfig, Style

    transparent = rs.bg_mode == "transparent"
    use_image = rs.bg_mode == "image" and bg_image_path is not None
    render_cfg = RenderConfig(
        width=rs.width,
        height=rs.height,
        fps=rs.fps,
        style=Style[STYLE_MAP[rs.style]],
        color=parse_hex(rs.color1),
        color2=parse_hex(rs.color2) if rs.gradient else None,
        background=None if transparent else parse_hex(rs.bg_color),
        background_image=str(bg_image_path) if use_image else None,
        amplitude=rs.sensitivity,
    )
    feature_cfg = FeatureConfig(
        fps=float(rs.fps),
        n_bands=render_cfg.bar_count,
        wf_points=render_cfg.wave_points,
    )
    src = getattr(audio, "path", None)
    enc_settings = EncodeSettings(
        width=rs.width,
        height=rs.height,
        fps=rs.fps,
        transparent=transparent,
        quality=rs.quality,
        opaque_codec=rs.codec,
        title=Path(src).stem if src else None,
    )
    return feature_cfg, render_cfg, enc_settings


def preview_render_config(rs: RenderSettings, render_cfg: "RenderConfig") -> "RenderConfig":
    """Derive the preview-resolution RenderConfig from the export one.

    Pixel-denominated knobs (stroke thickness, glow radius) are scaled with the
    canvas so the 960x540 preview is a true miniature of the export frame.
    """
    pw, ph = preview_size(rs)
    scale = pw / rs.width
    return dataclasses.replace(
        render_cfg,
        width=pw,
        height=ph,
        thickness=max(1.0, render_cfg.thickness * scale),
        glow_radius=min(96.0, max(4.0, render_cfg.glow_radius * scale)),
    )
