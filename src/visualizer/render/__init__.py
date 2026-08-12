"""GPU renderer: offscreen moderngl pipeline producing straight-alpha RGBA frames.

Public API::

    from visualizer.render import Renderer, RenderConfig, Style, GradientAxis, FrameFeatures

All Renderer calls (construction included) must happen on ONE dedicated thread.
"""
from ..features import FrameFeatures
from .config import GradientAxis, RebuildLevel, RenderConfig, Style, config_diff
from .renderer import Renderer

__all__ = [
    "FrameFeatures",
    "GradientAxis",
    "RebuildLevel",
    "RenderConfig",
    "Renderer",
    "Style",
    "config_diff",
]
