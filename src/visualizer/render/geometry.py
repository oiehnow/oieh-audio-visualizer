"""Pure numpy tessellators — no GL, all float32, all pixel-space y-DOWN.

Polyline builders return a list of open polylines (each an (m, 2) float32 array);
mirrored variants append a second reflected polyline rather than joining it to the
first (joining would draw a bogus connecting segment).
"""
from __future__ import annotations

import numpy as np

from .config import RenderConfig


def bar_layout(cfg: RenderConfig) -> tuple[np.ndarray, float]:
    """Linear bar center xs (bar_count,) and bar width in px."""
    slot = cfg.width / cfg.bar_count
    centers = ((np.arange(cfg.bar_count, dtype=np.float32) + 0.5) * slot).astype(np.float32)
    return centers, float(slot * (1.0 - cfg.bar_gap_frac))


def radial_layout(cfg: RenderConfig) -> tuple[np.ndarray, float, float]:
    """Radial bar angles (bar_count,), inner radius px, and arc-based bar width px."""
    n = cfg.bar_count
    angles = ((np.arange(n, dtype=np.float32) / n) * 2.0 * np.pi - np.pi / 2.0).astype(np.float32)
    r0 = cfg.inner_radius_frac * min(cfg.width, cfg.height)
    bar_w = max((2.0 * np.pi * r0 / n) * (1.0 - cfg.bar_gap_frac), 1.0)
    return angles, float(r0), float(bar_w)


def catmull_rom(y: np.ndarray, subdiv: int) -> np.ndarray:
    """Uniform Catmull-Rom through y (n,), endpoints duplicated.

    Returns ((n - 1) * subdiv + 1,) float32 passing exactly through every input value.
    """
    y = np.asarray(y, dtype=np.float32)
    n = y.shape[0]
    if n < 2 or subdiv <= 1:
        return y.copy()
    yp = np.concatenate([y[:1], y, y[-1:]])
    p0, p1, p2, p3 = yp[0:n - 1], yp[1:n], yp[2:n + 1], yp[3:n + 2]
    s = np.linspace(0.0, 1.0, subdiv, endpoint=False, dtype=np.float32)
    s2, s3 = s * s, s * s * s
    out = 0.5 * (2.0 * p1[:, None]
                 + (-p0 + p2)[:, None] * s
                 + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)[:, None] * s2
                 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3)[:, None] * s3)
    return np.concatenate([out.ravel(), y[-1:]]).astype(np.float32)


def polyline_to_segments(pts: np.ndarray) -> np.ndarray:
    """pts (m, 2) px -> (m-1, 6) float32: [p0x, p0y, p1x, p1y, t0, t1], t = arange(m)/(m-1)."""
    m = pts.shape[0]
    t = (np.arange(m, dtype=np.float32) / max(m - 1, 1)).astype(np.float32)
    segs = np.empty((m - 1, 6), dtype=np.float32)
    segs[:, 0:2] = pts[:-1]
    segs[:, 2:4] = pts[1:]
    segs[:, 4] = t[:-1]
    segs[:, 5] = t[1:]
    return segs


def build_wave_points(wave: np.ndarray, cfg: RenderConfig) -> list[np.ndarray]:
    n = wave.shape[0]
    x = np.linspace(0.0, cfg.width, n, dtype=np.float32)
    amp_px = cfg.amplitude * cfg.height * 0.45
    cy = cfg.height * 0.5
    y = (cy - wave * amp_px).astype(np.float32)          # y-down: positive wave goes up
    polylines = [np.stack([x, y], axis=1)]
    if cfg.mirror:
        polylines.append(np.stack([x, (cy + wave * amp_px).astype(np.float32)], axis=1))
    return polylines


def build_line_points(bands: np.ndarray, cfg: RenderConfig) -> list[np.ndarray]:
    h = catmull_rom(bands, cfg.line_subdiv)
    x = np.linspace(0.0, cfg.width, h.shape[0], dtype=np.float32)
    if cfg.mirror:
        baseline_y = cfg.height * 0.5
        amp_px = cfg.amplitude * cfg.height * 0.5
    else:
        baseline_y = cfg.height * cfg.baseline_frac
        amp_px = cfg.amplitude * cfg.height * cfg.baseline_frac
    y = (baseline_y - h * amp_px).astype(np.float32)
    polylines = [np.stack([x, y], axis=1)]
    if cfg.mirror:
        polylines.append(np.stack([x, (baseline_y + h * amp_px).astype(np.float32)], axis=1))
    return polylines


def build_circle_points(wave: np.ndarray, cfg: RenderConfig) -> list[np.ndarray]:
    """Waveform wrapped around a circle, closed loop.

    SEAM FIX (load-bearing): the per-frame waveform window doesn't loop, so the last
    n//16 samples are linearly crossfaded toward wave[0] before closing — without it
    the circle shows a jumping gap at 12 o'clock.
    """
    n = wave.shape[0]
    w = np.asarray(wave, dtype=np.float32).copy()
    k = max(n // 16, 2)
    fade = np.linspace(0.0, 1.0, k, dtype=np.float32)
    w[-k:] = w[-k:] * (1.0 - fade) + w[0] * fade
    theta = (np.arange(n, dtype=np.float32) / n) * 2.0 * np.pi - np.pi / 2.0
    r0 = cfg.inner_radius_frac * min(cfg.width, cfg.height)
    amp_px = cfg.amplitude * cfg.radial_extent_frac * min(cfg.width, cfg.height)
    cx, cy = cfg.width * 0.5, cfg.height * 0.5

    def _loop(r: np.ndarray) -> np.ndarray:
        pts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
        return np.vstack([pts, pts[:1]]).astype(np.float32)   # close the loop

    polylines = [_loop(r0 + w * amp_px)]
    if cfg.mirror:
        polylines.append(_loop(np.maximum(r0 - w * amp_px, 1.0)))
    return polylines


def max_segments(cfg: RenderConfig) -> int:
    """Capacity bound for the dynamic segment buffer (x2 covers mirror)."""
    return 2 * max(cfg.wave_points, cfg.bar_count * cfg.line_subdiv) + 8
