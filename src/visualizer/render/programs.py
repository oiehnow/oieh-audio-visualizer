"""Shader loading and compilation into a Programs bundle."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl

_SHADER_DIR = Path(__file__).parent / "shaders"


def _src(name: str) -> str:
    return (_SHADER_DIR / name).read_text(encoding="utf-8")


@dataclass
class Programs:
    rect: moderngl.Program
    segment: moderngl.Program
    blit: moderngl.Program
    blur: moderngl.Program
    finalize: moderngl.Program

    def release(self) -> None:
        for prog in (self.rect, self.segment, self.blit, self.blur, self.finalize):
            prog.release()


def load_programs(ctx: moderngl.Context) -> Programs:
    fullscreen = _src("fullscreen.vert")
    return Programs(
        rect=ctx.program(vertex_shader=_src("rect.vert"), fragment_shader=_src("rect.frag")),
        segment=ctx.program(vertex_shader=_src("segment.vert"),
                            fragment_shader=_src("segment.frag")),
        blit=ctx.program(vertex_shader=fullscreen, fragment_shader=_src("blit.frag")),
        blur=ctx.program(vertex_shader=fullscreen, fragment_shader=_src("blur.frag")),
        finalize=ctx.program(vertex_shader=fullscreen, fragment_shader=_src("finalize.frag")),
    )


def set_uniform(prog: moderngl.Program, name: str, value: object) -> None:
    """Set a uniform if the (possibly optimized-out) member exists."""
    try:
        prog[name].value = value
    except KeyError:
        pass
