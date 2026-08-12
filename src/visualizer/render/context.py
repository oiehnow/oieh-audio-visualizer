"""Standalone GL context, framebuffer inventory, and pixel readback.

ORIENTATION INVARIANT (do not "fix"): all geometry is built in pixel space with
y pointing DOWN and projected with ``ndc = px / res * 2 - 1``, which puts image-top
at GL-window-bottom. ``fbo.read_into`` returns rows bottom-up in GL terms, so the
bytes stream out TOP-ROW-FIRST — exactly what ffmpeg rawvideo expects. There must
never be an np.flipud, a projection flip, or an ffmpeg vflip anywhere in the app.
"""
from __future__ import annotations

import warnings

import moderngl

_SOFTWARE_RENDERER_MARKERS = ("llvmpipe", "software", "gdi generic", "swiftshader")


class GLContext:
    """Owns the moderngl standalone context, all FBOs, and readback buffers.

    NOT thread-safe — create and use on one thread only (GL context affinity).
    """

    def __init__(self, width: int, height: int, msaa: int) -> None:
        try:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        except Exception as exc:  # moderngl raises various backend errors
            raise RuntimeError(
                "Could not create a standalone OpenGL 3.3 context. On Windows this "
                "requires a local desktop session with GPU drivers (fails over RDP "
                f"or as a service). Underlying error: {exc}"
            ) from exc
        # Manual object lifetime: every rebuilt object is released explicitly.
        # (gc_mode="context_gc" + explicit release() double-frees in moderngl 5.12:
        # released objects re-enter the gc queue as InvalidObject and crash ctx.gc().)
        self.ctx.gc_mode = None
        renderer_name = self.ctx.info.get("GL_RENDERER", "")
        if any(m in renderer_name.lower() for m in _SOFTWARE_RENDERER_MARKERS):
            warnings.warn(
                f"OpenGL is using a software renderer ({renderer_name!r}); "
                "rendering will be extremely slow.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.width = 0
        self.height = 0
        self.msaa = 0
        self._build_fbos(width, height, msaa)

    def _build_fbos(self, width: int, height: int, msaa: int) -> None:
        ctx = self.ctx
        self.width, self.height = width, height
        self.msaa = min(msaa, ctx.max_samples)

        def _tex(w: int, h: int) -> moderngl.Texture:
            t = ctx.texture((max(w, 1), max(h, 1)), 4, dtype="f1")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False  # clamp-to-edge: blur must not wrap at edges
            t.repeat_y = False
            return t

        self._layer_rb = ctx.renderbuffer((width, height), 4, samples=self.msaa, dtype="f1")
        self.layer_ms = ctx.framebuffer(color_attachments=[self._layer_rb])
        self.layer_tex = _tex(width, height)
        self.layer = ctx.framebuffer(color_attachments=[self.layer_tex])
        self.half_tex = _tex(width // 2, height // 2)
        self.half = ctx.framebuffer(color_attachments=[self.half_tex])
        self.quarter_a_tex = _tex(width // 4, height // 4)
        self.quarter_a = ctx.framebuffer(color_attachments=[self.quarter_a_tex])
        self.quarter_b_tex = _tex(width // 4, height // 4)
        self.quarter_b = ctx.framebuffer(color_attachments=[self.quarter_b_tex])
        self.out_tex = _tex(width, height)
        self.out_fbo = ctx.framebuffer(color_attachments=[self.out_tex])
        self.black_tex = ctx.texture((1, 1), 4, data=b"\x00\x00\x00\x00", dtype="f1")
        self.black_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Double-buffered readback: the memoryview returned by read() stays valid
        # while the NEXT frame is drawn and is overwritten by the one after.
        self._bufs = [bytearray(width * height * 4), bytearray(width * height * 4)]
        self._buf_idx = 0

    def _release_fbos(self) -> None:
        for name in ("layer_ms", "_layer_rb", "layer", "layer_tex", "half", "half_tex",
                     "quarter_a", "quarter_a_tex", "quarter_b", "quarter_b_tex",
                     "out_fbo", "out_tex", "black_tex"):
            getattr(self, name).release()

    def resize(self, width: int, height: int, msaa: int) -> None:
        self._release_fbos()
        self._build_fbos(width, height, msaa)

    def read(self) -> memoryview:
        """Read out_fbo as tightly packed RGBA8, rows top-down (see module docstring)."""
        self._buf_idx ^= 1
        buf = self._bufs[self._buf_idx]
        self.out_fbo.read_into(buf, components=4, alignment=1)
        return memoryview(buf)

    def release(self) -> None:
        self.ctx.release()
