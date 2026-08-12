"""Audio feature extraction. The FrameFeatures dataclass below is the canonical
per-frame payload shared by the analysis and render subsystems — its field names
and shapes are a cross-module contract; do not rename or move it."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .audio import AudioData


@dataclass(frozen=True)
class FrameFeatures:
    bands: np.ndarray  # (n_bands,) float32 in [0, 1] — smoothed, normalized log-band magnitudes
    waveform: np.ndarray  # (wave_points,) float32 in [-1, 1] — display waveform for this frame
    rms: float  # smoothed loudness envelope in [0, 1]


@dataclass(frozen=True)
class FeatureConfig:
    """Analysis parameters. fps/n_bands/wf_points are assigned by settings.build_configs
    so they always equal the export fps and the renderer's bar_count/wave_points."""

    fps: float = 60.0  # MUST equal export fps (WYSIWYG)
    n_bands: int = 64
    fmin: float = 40.0
    fmax: float = 16000.0
    fft_size: int = 4096
    range_db: float = 50.0
    ref_percentile: float = 95.0
    slope_db_per_octave: float = 4.5
    attack_tau: float = 0.015  # seconds
    release_tau: float = 0.250  # seconds
    wf_points: int = 1024
    wf_window_s: float = 0.040
    rms_window_s: float = 0.050
    rms_range_db: float = 40.0


_STFT_BATCH = 1024  # 1024 x 4096 float32 = 16 MB per batch — bounded peak memory


def build_filterbank(
    sr: int, n_fft: int, n_bands: int, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced triangular filterbank. Returns (W, band_freqs): W is (n_bands, n_bins)
    float32 with row-normalized weights (band value = weighted MEAN of bin magnitudes),
    band_freqs is (n_bands,) float32 band center frequencies in Hz."""
    n_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = fmin * (fmax / fmin) ** (np.arange(n_bands + 2) / (n_bands + 1))
    W = np.zeros((n_bands, n_bins), dtype=np.float32)
    for i in range(n_bands):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        tri = np.clip(
            np.minimum((freqs - lo) / (ctr - lo), (hi - freqs) / (hi - ctr)), 0.0, None
        )
        if tri.sum() == 0.0:  # band narrower than one FFT bin (lowest bands)
            tri[np.argmin(np.abs(freqs - ctr))] = 1.0
        W[i] = (tri / tri.sum()).astype(np.float32)
    return W, edges[1:-1].astype(np.float32)


def waveform_overview(audio: AudioData, n: int = 1000) -> np.ndarray:
    """Whole-track overview: (n, 2) float32 of per-bucket [min, max] in [-1, 1].
    Serialized by the server into the upload response for the GUI's seek strip."""
    samples = audio.samples
    if n <= 0:
        return np.zeros((0, 2), np.float32)
    if len(samples) == 0:
        return np.zeros((n, 2), np.float32)
    starts = (np.arange(n, dtype=np.int64) * len(samples)) // n
    mins = np.minimum.reduceat(samples, starts)
    maxs = np.maximum.reduceat(samples, starts)
    return np.clip(np.stack([mins, maxs], axis=1), -1.0, 1.0).astype(np.float32)


def _smoothing_coeff(tau: float, fps: float) -> float:
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-1.0 / (tau * fps))


def _smooth_attack_release(
    x: np.ndarray, fps: float, attack_tau: float, release_tau: float
) -> np.ndarray:
    """Asymmetric one-pole IIR over axis 0: fast coefficient while rising, slow while
    falling. Coefficients derive from time constants, so the look is fps-invariant.
    Stateful and strictly sequential — this is why features must be fully precomputed
    before any random-access preview lookup."""
    a = np.float32(_smoothing_coeff(attack_tau, fps))
    r = np.float32(_smoothing_coeff(release_tau, fps))
    y = np.zeros(x.shape[1], np.float32)
    out = np.empty_like(x)
    for k in range(x.shape[0]):
        xi = x[k]
        y = y + np.where(xi > y, a, r) * (xi - y)
        out[k] = y
    return out


def _adaptive_ref_db(values_db: np.ndarray, percentile: float) -> float:
    """95th-percentile adaptive reference with a -60 dB silence guard so near-silent
    tracks don't get their noise floor amplified to full scale."""
    active = values_db[values_db > -70.0]
    if active.size == 0:
        return -60.0
    return max(float(np.percentile(active, percentile)), -60.0)


class Features:
    """Precomputed per-frame features for one (track, config) pair. Immutable after
    construction — safe for concurrent FastAPI preview reads while an export thread
    iterates frames. `n_frames` is the single authoritative video frame count."""

    def __init__(
        self,
        cfg: FeatureConfig,
        samples: np.ndarray,
        sample_rate: int,
        duration: float,
        centers: np.ndarray,
        bands: np.ndarray,
        rms: np.ndarray,
        band_freqs: np.ndarray,
        wf_gain: float,
    ) -> None:
        self.cfg = cfg
        self.fps = float(cfg.fps)
        self.duration = float(duration)
        self.n_frames = int(bands.shape[0])
        self.band_freqs = band_freqs
        self._samples = samples
        self._sr = int(sample_rate)
        self._centers = centers
        self._bands = bands
        self._rms = rms
        self._wf_gain = float(wf_gain)

    def frame(self, k: int) -> FrameFeatures:
        """Features for video frame k, clamped to [0, n_frames - 1]."""
        k = min(max(int(k), 0), self.n_frames - 1)
        return FrameFeatures(
            bands=self._bands[k],  # zero-copy read-only row view
            waveform=self._waveform(k),  # lazy slice from retained PCM
            rms=float(self._rms[k]),
        )

    def at_time(self, t: float) -> FrameFeatures:
        """Preview entry point: features for the frame nearest time t (seconds).
        Rounding (not floor) makes at_time(k / fps) == frame(k) exact despite
        float roundtrip error."""
        return self.frame(int(round(t * self.fps)))

    def _waveform(self, k: int) -> np.ndarray:
        c = int(self._centers[k])
        n = len(self._samples)
        w = int(round(self.cfg.wf_window_s * self._sr))
        idx = np.arange(c - w // 2, c - w // 2 + w)
        seg = np.zeros(w, np.float32)
        valid = (idx >= 0) & (idx < n)
        seg[valid] = self._samples[idx[valid]]
        pos = np.linspace(0.0, w - 1, self.cfg.wf_points)
        return np.clip(
            np.interp(pos, np.arange(w), seg) * self._wf_gain, -1.0, 1.0
        ).astype(np.float32)


def compute_features(
    audio: AudioData,
    config: FeatureConfig,
    progress: Callable[[float], None] | None = None,
) -> Features:
    """Full blocking precompute: batched STFT -> log filterbank -> dB + spectral tilt
    -> adaptive normalization -> attack/release smoothing -> RMS envelope. `progress`
    is called with 0.0–1.0 once per STFT batch (for the GUI analysis bar)."""
    cfg = config
    samples = audio.samples
    sr = audio.sample_rate
    n_samples = len(samples)

    # ceil(duration * fps) is the frame-count authority; the 1e-9 backs off float
    # noise when duration * fps lands within rounding error of an exact integer.
    n_frames = max(int(math.ceil(audio.duration * cfg.fps - 1e-9)), 1)
    # Centers from k directly (never an accumulated hop): zero drift at fractional fps.
    centers = np.round(np.arange(n_frames) * sr / cfg.fps).astype(np.int64)

    W, band_freqs = build_filterbank(sr, cfg.fft_size, cfg.n_bands, cfg.fmin, cfg.fmax)

    half = cfg.fft_size // 2
    padded = np.concatenate(
        [np.zeros(half, np.float32), samples, np.zeros(cfg.fft_size, np.float32)]
    )
    win = np.hanning(cfg.fft_size).astype(np.float32)
    scale = 2.0 / win.sum()  # a full-scale sine reads ~its amplitude in its band

    band_mag = np.empty((n_frames, cfg.n_bands), np.float32)
    offsets = np.arange(cfg.fft_size, dtype=np.int64)
    for s in range(0, n_frames, _STFT_BATCH):
        idx = centers[s : s + _STFT_BATCH, None] + offsets[None, :]
        frames = padded[idx] * win
        # explicit float32 cast: np.fft.rfft may return complex128 for float32 input
        mag = (np.abs(np.fft.rfft(frames, axis=1)) * scale).astype(np.float32)
        band_mag[s : s + _STFT_BATCH] = mag @ W.T
        if progress is not None:
            progress(min(s + _STFT_BATCH, n_frames) / n_frames)

    db = 20.0 * np.log10(band_mag + np.float32(1e-10))
    # Spectral tilt: music has a ~pink slope; +4.5 dB/oct keeps hi-hats/air visible.
    db += (cfg.slope_db_per_octave * np.log2(band_freqs / 1000.0)).astype(np.float32)

    ref_db = _adaptive_ref_db(db.max(axis=1), cfg.ref_percentile)
    norm = np.clip(
        (db - (ref_db - cfg.range_db)) / np.float32(cfg.range_db), 0.0, 1.0
    ).astype(np.float32)
    bands = _smooth_attack_release(norm, cfg.fps, cfg.attack_tau, cfg.release_tau)

    # Windowed RMS envelope, O(1) per frame via cumulative sum.
    csum = np.concatenate([[0.0], np.cumsum(samples.astype(np.float64) ** 2)])
    w = max(int(round(cfg.rms_window_s * sr)), 1)
    starts = np.clip(centers - w // 2, 0, n_samples)
    ends = np.clip(starts + w, 0, n_samples)
    rms = np.sqrt((csum[ends] - csum[starts]) / np.maximum(ends - starts, 1))
    rms_db = (20.0 * np.log10(rms + 1e-10)).astype(np.float32)
    rms_ref = _adaptive_ref_db(rms_db, cfg.ref_percentile)
    rms_norm = np.clip(
        (rms_db - (rms_ref - cfg.rms_range_db)) / np.float32(cfg.rms_range_db), 0.0, 1.0
    ).astype(np.float32)
    rms_env = _smooth_attack_release(
        rms_norm[:, None], cfg.fps, cfg.attack_tau, cfg.release_tau
    )[:, 0]

    # Global waveform gain: 99.9th percentile ignores single clicks; the x32 cap
    # (+30 dB) bounds silence amplification.
    peak = float(np.percentile(np.abs(samples), 99.9)) if n_samples else 0.0
    wf_gain = min(1.0 / max(peak, 1e-4), 32.0)

    bands.flags.writeable = False
    rms_env.flags.writeable = False
    band_freqs.flags.writeable = False

    if progress is not None:
        progress(1.0)

    return Features(
        cfg=cfg,
        samples=samples,
        sample_rate=sr,
        duration=audio.duration,
        centers=centers,
        bands=bands,
        rms=rms_env,
        band_freqs=band_freqs,
        wf_gain=wf_gain,
    )
