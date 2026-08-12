"""Audio decoding via ffmpeg: any input format -> 48 kHz mono float32 PCM."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AudioDecodeError(RuntimeError):
    """ffmpeg/ffprobe failed or produced no samples. Message includes stderr."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray  # float32 mono, shape (n_samples,), DC-removed
    sample_rate: int
    duration: float  # seconds, == len(samples) / sample_rate
    path: Path  # original file — the exporter muxes audio from THIS path, never the PCM


def probe_duration(path: str | Path) -> float:
    """Duration in seconds via ffprobe container metadata (no decode)."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        capture_output=True,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise AudioDecodeError(f"ffprobe failed: {proc.stderr.decode('utf-8', 'replace')}")
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise AudioDecodeError(f"ffprobe returned no audio duration for {path}") from exc


def decode_audio(path: str | Path, sample_rate: int = 48000) -> AudioData:
    """Decode any ffmpeg-readable audio file to mono float32 PCM at `sample_rate`."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-map", "a:0",  # first audio stream — never an embedded cover-art video stream
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    # capture_output=True drains both pipes fully — no manual PIPE reads, no Windows deadlock.
    proc = subprocess.run(cmd, capture_output=True, creationflags=_CREATE_NO_WINDOW)
    if proc.returncode != 0 or len(proc.stdout) < 4:
        raise AudioDecodeError(f"ffmpeg decode failed: {proc.stderr.decode('utf-8', 'replace')}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()  # frombuffer is read-only
    samples -= np.float32(samples.mean())  # DC-offset removal (some MP3 encoders leak DC)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=len(samples) / sample_rate,
        path=Path(path),
    )
