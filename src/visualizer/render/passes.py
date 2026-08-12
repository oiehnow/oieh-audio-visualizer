"""Glow: two-step downsample + separable gaussian blur at quarter res."""
from __future__ import annotations

import numpy as np

import moderngl

from .context import GLContext
from .programs import Programs

_TAPS = 13


def gauss_weights(sigma: float, taps: int = _TAPS, spacing: float | None = None) -> np.ndarray:
    """Normalized gaussian weights at tap offsets x = i * spacing, i in [-taps//2, taps//2].

    spacing defaults to sigma/2 (the kernel's natural stride).
    """
    if spacing is None:
        spacing = sigma / 2.0
    half = taps // 2
    x = np.arange(-half, half + 1, dtype=np.float32) * spacing
    w = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (w / w.sum()).astype(np.float32)


class GlowPass:
    """Blur chain on premultiplied color; blending stays disabled throughout."""

    def __init__(self, glc: GLContext, progs: Programs) -> None:
        self.glc = glc
        self.progs = progs
        ctx = glc.ctx
        self.fs_vbo = ctx.buffer(
            np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype=np.float32).tobytes()
        )
        self._blit_vao = ctx.vertex_array(progs.blit, [(self.fs_vbo, "2f", "in_pos")])
        self._blur_vao = ctx.vertex_array(progs.blur, [(self.fs_vbo, "2f", "in_pos")])
        self._spacing = 1.0
        self.set_radius(24.0)

    def set_radius(self, glow_radius: float) -> None:
        sigma_q = float(np.clip(glow_radius / 4.0, 1.0, 8.0))  # quarter-res px
        # Tap spacing capped at 2 texels: bilinear taps spaced wider leave coverage
        # gaps that show as visible comb banding in the boosted glow.
        self._spacing = min(sigma_q / 2.0, 2.0)
        # Weights must be computed at the ACTUAL tap offsets, or the kernel shape lies.
        self.progs.blur["u_w"].write(gauss_weights(sigma_q, spacing=self._spacing).tobytes())

    def run(self) -> None:
        """layer -> half -> quarter_a -> (H blur) quarter_b -> (V blur) quarter_a."""
        glc, progs = self.glc, self.progs
        progs.blit["u_tex"].value = 0
        glc.half.use()
        glc.layer_tex.use(location=0)
        self._blit_vao.render(moderngl.TRIANGLES, vertices=3)
        glc.quarter_a.use()
        glc.half_tex.use(location=0)
        self._blit_vao.render(moderngl.TRIANGLES, vertices=3)

        progs.blur["u_tex"].value = 0
        progs.blur["u_spacing"].value = self._spacing
        glc.quarter_b.use()
        glc.quarter_a_tex.use(location=0)
        progs.blur["u_dir"].value = (1.0, 0.0)
        self._blur_vao.render(moderngl.TRIANGLES, vertices=3)
        glc.quarter_a.use()
        glc.quarter_b_tex.use(location=0)
        progs.blur["u_dir"].value = (0.0, 1.0)
        self._blur_vao.render(moderngl.TRIANGLES, vertices=3)
