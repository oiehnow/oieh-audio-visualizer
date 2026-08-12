"""Export job bookkeeping: one active job, terminal history, progress snapshots.

JobManager.run() executes ON THE GL THREAD (via RenderService.submit_export);
everything else is called from the event loop.  Progress crosses threads as
plain attribute writes on ExportJob, polled by the SSE generator.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import pipeline
from .encode import EncodeCancelled

if TYPE_CHECKING:
    from .encode import EncodeSettings
    from .features import Features
    from .render import RenderConfig, Renderer
    from .settings import RenderSettings

MAX_JOB_HISTORY = 20


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.DONE, JobState.ERROR, JobState.CANCELLED})


class JobBusy(RuntimeError):
    """Raised by create() while another export is active."""


@dataclass
class ExportJob:
    id: str
    audio_id: str
    settings: "RenderSettings"
    out_path: Path
    state: JobState = JobState.PENDING
    frames_done: int = 0
    frames_total: int = 0
    render_fps: float = 0.0
    eta_s: float | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: float | None = None

    def on_progress(self, p: Any) -> None:
        """progress_cb for pipeline.run_export; fires on the encoder's drain thread.

        Reads encode.Progress fields (frames_done/total_frames/fps/eta_seconds);
        the getattr fallbacks tolerate the alternate names that appeared in
        earlier drafts of the encoding contract.
        """
        self.frames_done = int(getattr(p, "frames_done", 0) or 0)
        self.frames_total = int(getattr(p, "total_frames", getattr(p, "frames_total", 0)) or 0)
        self.render_fps = float(getattr(p, "fps", 0.0) or 0.0)
        eta = getattr(p, "eta_seconds", getattr(p, "eta_s", None))
        self.eta_s = float(eta) if eta is not None else None

    def snapshot(self) -> dict:
        pct = 100.0 * self.frames_done / self.frames_total if self.frames_total else 0.0
        return {
            "job_id": self.id,
            "state": self.state.value,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "percent": round(pct, 1),
            "render_fps": round(self.render_fps, 1),
            "eta_s": round(self.eta_s, 1) if self.eta_s is not None else None,
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active: ExportJob | None = None
        self._jobs: OrderedDict[str, ExportJob] = OrderedDict()

    def create(self, audio_id: str, settings: "RenderSettings", out_path: Path) -> ExportJob:
        with self._lock:
            if self.active is not None:
                raise JobBusy("an export is already running")
            job = ExportJob(id=uuid.uuid4().hex[:8], audio_id=audio_id,
                            settings=settings, out_path=out_path)
            self.active = job
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOB_HISTORY:
                oldest = next(iter(self._jobs))
                if oldest == job.id:
                    break
                del self._jobs[oldest]
            return job

    def get(self, job_id: str) -> ExportJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Sets the job's cancel event; True if the job was pending/running."""
        job = self.get(job_id)
        if job is None:
            return False
        was_live = job.state not in TERMINAL_STATES
        job.cancel_event.set()
        return was_live

    def run(self, job: ExportJob, renderer: "Renderer | None", features: "Features",
            audio_path: Path, render_cfg: "RenderConfig", enc_settings: "EncodeSettings") -> None:
        """Executed ON THE GL THREAD via RenderService.submit_export."""
        job.state, job.started_at = JobState.RUNNING, time.monotonic()
        part = job.out_path.with_suffix(".webm.part")
        try:
            if renderer is None:
                raise RuntimeError("GPU renderer unavailable (GL context failed to initialize)")
            renderer.update_config(render_cfg)
            # run_export owns the .part -> .webm rename and returns the final path.
            job.out_path = pipeline.run_export(
                renderer, features, audio_path, render_cfg, enc_settings,
                job.out_path, progress_cb=job.on_progress, cancel_event=job.cancel_event)
            job.state = JobState.DONE
        except EncodeCancelled:
            job.state = JobState.CANCELLED
            _unlink_retry(part)  # Encoder.cancel() owns deletion; this mops up if it lost the race
        except Exception as e:  # noqa: BLE001 — job must always reach a terminal state
            job.state, job.error = JobState.ERROR, f"{type(e).__name__}: {e}"
            _unlink_retry(part)
        finally:
            with self._lock:
                if self.active is job:
                    self.active = None


def _unlink_retry(p: Path, tries: int = 5, delay: float = 0.2) -> None:
    """Delete with backoff — Windows can hold the ffmpeg handle briefly after kill."""
    for attempt in range(tries):
        try:
            p.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == tries - 1:
                return
            time.sleep(delay)
