"""Export orchestration: the lockstep draw -> write_frame loop.

run_export executes on the CALLER'S thread, which must be the dedicated GL
thread that owns the Renderer (moderngl standalone contexts have thread
affinity). The renderer and features objects are duck-typed on purpose — this
module never imports from visualizer.render.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

from visualizer.encode import (
    EncodeCancelled,
    EncodeError,
    Encoder,
    EncodeSettings,
    Progress,
)

# GUI-layer aliases (cross-review): the export exceptions ARE the encode ones.
ExportCancelled = EncodeCancelled
ExportError = EncodeError


class RendererLike(Protocol):
    def update_config(self, config: Any) -> None: ...
    def draw(self, features: Any) -> memoryview: ...


class FeaturesLike(Protocol):
    n_frames: int

    def frame(self, k: int) -> Any: ...


def run_export(
    renderer: RendererLike,
    features: FeaturesLike,
    audio_path: Path,
    render_cfg: Any,
    enc_settings: EncodeSettings,
    out_path: Path,
    progress_cb: Callable[[Progress], None],
    cancel_event: threading.Event,
) -> Path:
    """Render every frame of `features` into a finished .webm at `out_path`.

    - total_frames is features.n_frames — the single source of truth; never
      derived from ffprobe or the muxed audio.
    - Encodes into `<out_path>.part`; on success THIS function renames
      .part -> .webm and returns the final path. run_export is the single
      owner of the caller-visible rename (a .part path passed as out_path is
      tolerated: its .part suffix is stripped to get the final name).
    - Checks cancel_event once per frame; on cancel the Encoder is cancelled
      (which kills ffmpeg and deletes the .part) and EncodeCancelled is
      raised. Any other failure likewise cancels the encoder, so no .part
      survives an unsuccessful export.
    - progress_cb receives encode.Progress on the encoder's drain thread at
      ~4 Hz; it must not touch an event loop directly.
    """
    final = Path(out_path)
    if final.suffix == ".part":
        final = final.with_suffix("")
    part = final.with_name(final.name + ".part")

    total = int(features.n_frames)
    renderer.update_config(render_cfg)  # WYSIWYG: export uses exactly this config

    enc = Encoder(
        enc_settings,
        audio_path=Path(audio_path),
        out_path=part,
        total_frames=total,
        on_progress=progress_cb,
    )
    # Context manager: finish() on clean exit; cancel() (kills ffmpeg, deletes
    # the .part) on any exception, including the EncodeCancelled raised below.
    with enc:
        for k in range(total):
            if cancel_event.is_set():
                raise EncodeCancelled("export cancelled")
            # Zero-copy handoff: draw()'s memoryview points into the
            # renderer's double buffer; it is safe here only because
            # write_frame blocks until the frame is fully piped and this loop
            # never buffers ahead (lockstep draw -> write -> draw).
            enc.write_frame(renderer.draw(features.frame(k)))

    os.replace(part, final)
    return final
