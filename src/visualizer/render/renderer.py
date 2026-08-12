"""Per-frame orchestration: geometry upload, layer pass, glow, finalize, readback.

Alpha invariant: every internal pass is PREMULTIPLIED; finalize.frag unpremultiplies
so draw() hands out STRAIGHT-alpha RGBA — do not remove the finalize pass or change
the blend func from (ONE, ONE_MINUS_SRC_ALPHA), or transparent exports get fringes.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image

import moderngl

from ..features import FrameFeatures
from .config import (
    BAND_STYLES,
    RECT_STYLES,
    GradientAxis,
    RebuildLevel,
    RenderConfig,
    Style,
    config_diff,
)
from .context import GLContext
from .geometry import (
    bar_layout,
    build_circle_points,
    build_line_points,
    build_wave_points,
    max_segments,
    polyline_to_segments,
    radial_layout,
)
from .passes import GlowPass
from .programs import load_programs, set_uniform

_AA_PAD = 1.5  # px of quad padding so analytic AA is never clipped


class Renderer:
    """NOT thread-safe. Construct and call from ONE dedicated thread (GL affinity).

    draw() returns straight-alpha RGBA8 bytes, rows TOP-DOWN, tightly packed —
    pipe directly to `ffmpeg -f rawvideo -pix_fmt rgba`. The returned memoryview
    stays valid during the NEXT draw() and is overwritten by the one after
    (double-buffered): never buffer more than one frame ahead.
    """

    def __init__(self, config: RenderConfig) -> None:
        config.validate()
        self.cfg = dataclasses.replace(config)
        self.glc = GLContext(config.width, config.height, config.msaa)
        self.progs = load_programs(self.glc.ctx)
        self.glow = GlowPass(self.glc, self.progs)
        self._unit_vbo = self.glc.ctx.buffer(
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32).tobytes()
        )
        self._final_vao = self.glc.ctx.vertex_array(
            self.progs.finalize, [(self.glow.fs_vbo, "2f", "in_pos")]
        )
        self._rect_static_vbo: moderngl.Buffer | None = None
        self._rect_h_vbo: moderngl.Buffer | None = None
        self._rect_vao: moderngl.VertexArray | None = None
        self._seg_vbo: moderngl.Buffer | None = None
        self._seg_vao: moderngl.VertexArray | None = None
        self._bg_tex: moderngl.Texture | None = None
        self._bg_tex_path: str | None = None   # path the current _bg_tex was loaded from
        self._bg_tex_size: tuple[int, int] = (1, 1)
        self._build_geometry()
        self._apply_uniforms()

    # -- construction / reconfiguration -------------------------------------

    def _release_geometry(self) -> None:
        for obj in (self._rect_vao, self._rect_static_vbo, self._rect_h_vbo,
                    self._seg_vao, self._seg_vbo):
            if obj is not None:
                obj.release()
        self._rect_static_vbo = self._rect_h_vbo = self._rect_vao = None
        self._seg_vbo = self._seg_vao = None

    def _build_geometry(self) -> None:
        cfg, ctx = self.cfg, self.glc.ctx
        self._release_geometry()
        if cfg.style in RECT_STYLES:
            n = cfg.bar_count
            static = np.zeros((n, 4), dtype=np.float32)   # [x, angle, r0, t]
            static[:, 3] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            if cfg.style is Style.BARS:
                centers, self._bar_w = bar_layout(cfg)
                static[:, 0] = centers
                self._extent_px = (cfg.height * 0.5 if cfg.mirror
                                   else cfg.height * cfg.baseline_frac)
                self._anchor = (0.0, cfg.height * 0.5 if cfg.mirror
                                else cfg.height * cfg.baseline_frac)
                self._rect_mode = 0
            else:  # RADIAL_BARS
                angles, r0, self._bar_w = radial_layout(cfg)
                static[:, 1] = angles
                static[:, 2] = r0
                self._extent_px = cfg.radial_extent_frac * min(cfg.width, cfg.height)
                self._anchor = (cfg.width * 0.5, cfg.height * 0.5)
                self._rect_mode = 1
            self._rect_static_vbo = ctx.buffer(static.tobytes())
            self._rect_h_vbo = ctx.buffer(reserve=n * 4, dynamic=True)
            self._rect_vao = ctx.vertex_array(self.progs.rect, [
                (self._unit_vbo, "2f", "in_unit"),
                (self._rect_static_vbo, "1f 1f 1f 1f/i", "in_x", "in_angle", "in_r0", "in_t"),
                (self._rect_h_vbo, "1f/i", "in_h"),
            ])
        else:
            self._seg_capacity = max_segments(cfg)
            self._seg_vbo = ctx.buffer(reserve=self._seg_capacity * 24, dynamic=True)
            self._seg_vao = ctx.vertex_array(self.progs.segment, [
                (self._unit_vbo, "2f", "in_unit"),
                (self._seg_vbo, "4f 2f/i", "in_seg", "in_t01"),
            ])

    def _apply_uniforms(self) -> None:
        cfg, progs = self.cfg, self.progs
        res = (float(cfg.width), float(cfg.height))
        color_a = tuple(cfg.color)
        color_b = tuple(cfg.color2) if cfg.color2 is not None else color_a
        has_grad = cfg.color2 is not None

        if cfg.style in RECT_STYLES:
            p = progs.rect
            p["u_res"].value = res
            p["u_anchor"].value = self._anchor
            p["u_bar_w"].value = self._bar_w
            p["u_pad"].value = _AA_PAD
            p["u_mode"].value = self._rect_mode
            p["u_mirror"].value = 1 if cfg.mirror else 0
            p["u_color_a"].value = color_a
            p["u_color_b"].value = color_b
            if not has_grad:
                p["u_grad"].value = 0
            else:
                p["u_grad"].value = 1 if cfg.gradient_axis is GradientAxis.ALONG_SPECTRUM else 2
            p["u_corner_r"].value = cfg.corner_radius_frac * self._bar_w
        else:
            p = progs.segment
            p["u_res"].value = res
            p["u_half_w"].value = cfg.thickness * 0.5
            p["u_pad"].value = _AA_PAD
            p["u_color_a"].value = color_a
            p["u_color_b"].value = color_b
            # thickness-axis gradients are pointless on a stroke — only the
            # along-the-line gradient is meaningful for polyline styles
            p["u_grad"].value = (
                1 if has_grad and cfg.gradient_axis is GradientAxis.ALONG_SPECTRUM else 0
            )

        f = progs.finalize
        f["u_layer"].value = 0
        f["u_glow"].value = 1
        f["u_bg_tex"].value = 2
        if cfg.background is None:
            f["u_bg"].value = (0.0, 0.0, 0.0, 0.0)
        else:
            r, g, b = cfg.background
            f["u_bg"].value = (r, g, b, 1.0)   # premultiplied opaque
        if cfg.background_image != self._bg_tex_path:
            self._load_bg_texture(cfg.background_image)
        f["u_bg_mode"].value = 1 if self._bg_tex is not None else 0
        f["u_bg_uv_scale"].value = self._bg_uv_scale()
        f["u_glow_strength"].value = cfg.glow_strength if cfg.glow else 0.0
        self.glow.set_radius(cfg.glow_radius)

    def _load_bg_texture(self, path: str | None) -> None:
        """(Re)load the background image texture. Runs on the GL thread.

        _bg_tex_path is recorded only AFTER a successful load: a failed load must
        leave it None so the next update_config with the same path RETRIES instead
        of silently rendering the solid fallback forever (content-hash ids mean a
        re-upload of the same image produces the identical path).
        """
        if self._bg_tex is not None:
            self._bg_tex.release()
            self._bg_tex = None
        self._bg_tex_path = None
        if path is None:
            return
        try:
            with Image.open(path) as im:
                im = im.convert("RGBA")
                # Bound GPU memory; the server downscales on upload, this is the backstop.
                if max(im.size) > 4096:
                    im.thumbnail((4096, 4096), Image.LANCZOS)
                data, size = im.tobytes(), im.size
        except Exception as e:  # PIL raises Exception subclasses beyond OSError/ValueError
            raise ValueError(f"could not load background image {path!r}: {e}") from e
        tex = self.glc.ctx.texture(size, 4, data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        self._bg_tex = tex
        self._bg_tex_size = size
        self._bg_tex_path = path

    def _bg_uv_scale(self) -> tuple[float, float]:
        """Visible texture fraction per axis for cover-fit (aspect-fill) cropping."""
        if self._bg_tex is None:
            return (1.0, 1.0)
        tw, th = self._bg_tex_size
        s = max(self.cfg.width / tw, self.cfg.height / th)
        return (self.cfg.width / (tw * s), self.cfg.height / (th * s))

    def update_config(self, new: RenderConfig) -> None:
        """Cheap live changes: uniforms are free, geometry ~1 ms, size/msaa rebuilds FBOs."""
        new.validate()
        level = config_diff(self.cfg, new)
        self.cfg = dataclasses.replace(new)
        if level is RebuildLevel.FRAMEBUFFERS:
            self.glc.resize(new.width, new.height, new.msaa)
        if level in (RebuildLevel.FRAMEBUFFERS, RebuildLevel.GEOMETRY):
            self._build_geometry()
        self._apply_uniforms()

    # -- per-frame ----------------------------------------------------------

    def _validated(self, features: FrameFeatures) -> np.ndarray:
        cfg = self.cfg
        if cfg.style in BAND_STYLES:
            arr, want, name = features.bands, (cfg.bar_count,), "bands"
        else:
            arr, want, name = features.waveform, (cfg.wave_points,), "waveform"
        if arr is None:
            raise ValueError(f"FrameFeatures.{name} is required for style {cfg.style.value}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != want:
            raise ValueError(
                f"FrameFeatures.{name} shape {arr.shape} != required {want} "
                f"for style {cfg.style.value} (renderer never resamples)"
            )
        return arr

    def draw(self, features: FrameFeatures) -> memoryview:
        cfg, ctx, glc = self.cfg, self.glc.ctx, self.glc
        data = self._validated(features)
        rms = float(np.clip(features.rms, 0.0, 1.0))

        # 1) dynamic upload
        if cfg.style in RECT_STYLES:
            h_px = (np.clip(data, 0.0, 1.0) * (cfg.amplitude * self._extent_px)
                    ).astype(np.float32)
            self._rect_h_vbo.orphan()
            self._rect_h_vbo.write(h_px.tobytes())
            n_inst, vao, prog = cfg.bar_count, self._rect_vao, self.progs.rect
        else:
            if cfg.style is Style.WAVEFORM:
                polylines = build_wave_points(data, cfg)
            elif cfg.style is Style.LINE:
                polylines = build_line_points(data, cfg)
            else:  # RADIAL_WAVE
                polylines = build_circle_points(data, cfg)
            segs = np.concatenate([polyline_to_segments(p) for p in polylines])
            n_inst = segs.shape[0]
            self._seg_vbo.orphan()
            self._seg_vbo.write(np.ascontiguousarray(segs).tobytes())
            vao, prog = self._seg_vao, self.progs.segment
        set_uniform(prog, "u_rms", rms)
        set_uniform(self.progs.finalize, "u_rms", rms)

        # 2) layer pass — premultiplied over
        glc.layer_ms.use()
        glc.layer_ms.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.blend_equation = moderngl.FUNC_ADD
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n_inst)
        ctx.disable(moderngl.BLEND)

        # 3) MSAA resolve (averages premultiplied samples — the correct resolve)
        ctx.copy_framebuffer(glc.layer, glc.layer_ms)

        # 4) glow
        if cfg.glow:
            self.glow.run()

        # 5) finalize: composite glow under layer over bg, then unpremultiply
        glc.out_fbo.use()
        glc.layer_tex.use(location=0)
        (glc.quarter_a_tex if cfg.glow else glc.black_tex).use(location=1)
        (self._bg_tex if self._bg_tex is not None else glc.black_tex).use(location=2)
        self._final_vao.render(moderngl.TRIANGLES, vertices=3)

        # 6) straight-alpha RGBA8, rows top-down
        return glc.read()

    def close(self) -> None:
        if self._bg_tex is not None:
            self._bg_tex.release()
            self._bg_tex = None
        self.glc.release()
