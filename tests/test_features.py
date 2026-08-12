"""Feature-extraction tests: real tests/assets/test.mp3 for pipeline behavior,
plus a synthetic sine-burst track for smoothing dynamics."""
import math
from pathlib import Path

import numpy as np
import pytest

from visualizer.audio import AudioData, decode_audio
from visualizer.features import (
    FeatureConfig,
    Features,
    FrameFeatures,
    build_filterbank,
    compute_features,
    waveform_overview,
)

ASSET = Path(__file__).parent / "assets" / "test.mp3"


@pytest.fixture(scope="module")
def audio() -> AudioData:
    return decode_audio(ASSET)


@pytest.fixture(scope="module")
def feats60(audio: AudioData) -> Features:
    return compute_features(audio, FeatureConfig(fps=60.0))


@pytest.fixture(scope="module")
def feats30(audio: AudioData) -> Features:
    return compute_features(audio, FeatureConfig(fps=30.0))


def _all_bands(feats: Features) -> np.ndarray:
    return np.stack([feats.frame(k).bands for k in range(feats.n_frames)])


# --- frame count authority ---------------------------------------------------


def test_n_frames_is_ceil_duration_times_fps(
    audio: AudioData, feats30: Features, feats60: Features
) -> None:
    assert feats30.n_frames == math.ceil(audio.duration * 30.0)
    assert feats60.n_frames == math.ceil(audio.duration * 60.0)
    assert feats60.fps == 60.0
    assert feats30.duration == audio.duration


# --- bands contract ----------------------------------------------------------


def test_bands_shape_dtype_range(feats60: Features) -> None:
    f = feats60.frame(0)
    assert isinstance(f, FrameFeatures)
    assert f.bands.shape == (64,)
    assert f.bands.dtype == np.float32
    all_bands = _all_bands(feats60)
    assert float(all_bands.min()) >= 0.0
    assert float(all_bands.max()) <= 1.0
    # 95th-percentile adaptive reference pegs the loudest frames near full scale.
    assert float(all_bands.max()) > 0.9


def test_bands_are_read_only(feats60: Features) -> None:
    f = feats60.frame(0)
    with pytest.raises(ValueError):
        f.bands[0] = 0.5


def test_band_freqs_log_spaced_in_range(feats60: Features) -> None:
    freqs = feats60.band_freqs
    assert freqs.shape == (64,)
    assert freqs.dtype == np.float32
    assert np.all(np.diff(freqs) > 0)
    assert 40.0 < float(freqs[0]) < float(freqs[-1]) < 16000.0


# --- rms contract ------------------------------------------------------------


def test_rms_scalar_in_range(feats60: Features) -> None:
    vals = np.array([feats60.frame(k).rms for k in range(feats60.n_frames)])
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0
    assert vals.max() > 0.8  # loudest sections reach near full scale


# --- waveform contract -------------------------------------------------------


def test_waveform_shape_dtype_range(feats60: Features) -> None:
    f = feats60.frame(feats60.n_frames // 2)
    assert f.waveform.shape == (1024,)
    assert f.waveform.dtype == np.float32
    assert float(f.waveform.min()) >= -1.0
    assert float(f.waveform.max()) <= 1.0


def test_waveform_has_signal_and_is_deterministic(feats60: Features) -> None:
    k = feats60.n_frames // 2
    w1 = feats60.frame(k).waveform
    w2 = feats60.frame(k).waveform
    assert np.array_equal(w1, w2)
    peak = max(float(np.abs(feats60.frame(k).waveform).max()) for k in range(feats60.n_frames))
    assert peak > 0.05


# --- config plumbing ---------------------------------------------------------


def test_custom_band_and_waveform_counts(audio: AudioData) -> None:
    cfg = FeatureConfig(fps=30.0, n_bands=32, wf_points=256)
    feats = compute_features(audio, cfg)
    f = feats.frame(0)
    assert f.bands.shape == (32,)
    assert f.waveform.shape == (256,)
    assert feats.band_freqs.shape == (32,)


def test_progress_callback_monotonic_to_one(audio: AudioData) -> None:
    seen: list[float] = []
    compute_features(audio, FeatureConfig(fps=30.0), progress=seen.append)
    assert len(seen) >= 1
    assert all(0.0 <= p <= 1.0 for p in seen)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == 1.0


# --- at_time / frame consistency ---------------------------------------------


@pytest.mark.parametrize("fps_fixture", ["feats30", "feats60"])
def test_at_time_matches_frame_round(fps_fixture: str, request: pytest.FixtureRequest) -> None:
    feats: Features = request.getfixturevalue(fps_fixture)
    for k in [0, 1, 7, feats.n_frames // 2, feats.n_frames - 1]:
        by_time = feats.at_time(k / feats.fps)
        by_index = feats.frame(k)
        assert np.array_equal(by_time.bands, by_index.bands)
        assert np.array_equal(by_time.waveform, by_index.waveform)
        assert by_time.rms == by_index.rms


def test_at_time_rounds_to_nearest_frame(feats60: Features) -> None:
    k = 10
    near_before = feats60.at_time((k - 0.4) / feats60.fps)
    near_after = feats60.at_time((k + 0.4) / feats60.fps)
    exact = feats60.frame(k)
    assert np.array_equal(near_before.bands, exact.bands)
    assert np.array_equal(near_after.bands, exact.bands)


def test_lookup_clamped_to_valid_frames(feats60: Features) -> None:
    first = feats60.frame(0)
    last = feats60.frame(feats60.n_frames - 1)
    assert np.array_equal(feats60.frame(-5).bands, first.bands)
    assert np.array_equal(feats60.frame(feats60.n_frames + 100).bands, last.bands)
    assert np.array_equal(feats60.at_time(-1.0).bands, first.bands)
    assert np.array_equal(feats60.at_time(1e6).bands, last.bands)


# --- smoothing dynamics (synthetic sine burst) -------------------------------


@pytest.fixture(scope="module")
def burst_audio() -> AudioData:
    """0.2 s of 440 Hz at 0.8 amplitude, then 2.8 s of silence."""
    sr = 48000
    t = np.arange(int(0.2 * sr), dtype=np.float32) / sr
    burst = (0.8 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    samples = np.concatenate([burst, np.zeros(int(2.8 * sr), np.float32)])
    return AudioData(
        samples=samples, sample_rate=sr, duration=len(samples) / sr, path=Path("burst.synthetic")
    )


def test_attack_fast_then_monotonic_release_decay(burst_audio: AudioData) -> None:
    fps = 30.0
    feats = compute_features(burst_audio, FeatureConfig(fps=fps))
    band = int(np.argmin(np.abs(feats.band_freqs - 440.0)))
    vals = np.array([feats.frame(k).bands[band] for k in range(feats.n_frames)])

    # Fast attack: the tone band is driven high within the burst (frames 0..7).
    burst_peak = float(vals[:8].max())
    assert burst_peak > 0.5

    # After the analysis window fully clears the burst (~0.25 s + half window),
    # input is zero, so the smoothed value must decay monotonically.
    decay_start = 10  # t = 0.333 s at 30 fps
    decay = vals[decay_start:]
    assert np.all(np.diff(decay) <= 1e-6)
    # release_tau = 0.25 s: over ~2.6 s the value collapses toward zero.
    assert float(decay[-1]) < 0.05 * burst_peak


def test_smoothing_deterministic_recompute(burst_audio: AudioData) -> None:
    cfg = FeatureConfig(fps=30.0)
    a = compute_features(burst_audio, cfg)
    b = compute_features(burst_audio, cfg)
    for k in [0, 3, a.n_frames // 2, a.n_frames - 1]:
        fa, fb = a.frame(k), b.frame(k)
        assert np.array_equal(fa.bands, fb.bands)
        assert np.array_equal(fa.waveform, fb.waveform)
        assert fa.rms == fb.rms


# --- waveform_overview -------------------------------------------------------


def test_waveform_overview_shape_and_range(audio: AudioData) -> None:
    ov = waveform_overview(audio)
    assert ov.shape == (1000, 2)
    assert ov.dtype == np.float32
    assert np.all(ov[:, 0] <= ov[:, 1])
    assert float(ov.min()) >= -1.0
    assert float(ov.max()) <= 1.0
    assert float(np.abs(ov).max()) > 0.05  # real signal present


def test_waveform_overview_custom_n(audio: AudioData) -> None:
    ov = waveform_overview(audio, n=250)
    assert ov.shape == (250, 2)


def test_waveform_overview_localizes_energy(burst_audio: AudioData) -> None:
    ov = waveform_overview(burst_audio, n=300)
    # burst occupies the first 0.2/3.0 of the track -> first 20 buckets
    assert float(ov[:20, 1].max()) > 0.5
    # tail is pure silence
    assert np.all(ov[150:] == 0.0)


# --- filterbank --------------------------------------------------------------


def test_filterbank_rows_normalized_no_dead_bands() -> None:
    W, freqs = build_filterbank(48000, 4096, 64, 40.0, 16000.0)
    assert W.shape == (64, 4096 // 2 + 1)
    assert W.dtype == np.float32
    assert np.all(W >= 0.0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-5)
    assert freqs.shape == (64,)
    # DC bin never contributes (band 0 starts at 40 Hz > first bin at 11.72 Hz)
    assert np.all(W[:, 0] == 0.0)
