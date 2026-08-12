"""Decode tests against the real tests/assets/test.mp3 via real ffmpeg."""
from pathlib import Path

import numpy as np
import pytest

from visualizer.audio import AudioData, AudioDecodeError, decode_audio, probe_duration

ASSET = Path(__file__).parent / "assets" / "test.mp3"


@pytest.fixture(scope="module")
def audio() -> AudioData:
    return decode_audio(ASSET)


def test_decode_produces_48k_mono_float32(audio: AudioData) -> None:
    assert audio.sample_rate == 48000
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1


def test_decode_duration_about_12s(audio: AudioData) -> None:
    assert 11.5 <= audio.duration <= 12.5
    assert audio.duration == len(audio.samples) / 48000


def test_decode_preserves_original_path(audio: AudioData) -> None:
    assert audio.path == ASSET


def test_decode_removes_dc_offset(audio: AudioData) -> None:
    assert abs(float(audio.samples.mean())) < 1e-6


def test_decode_has_signal(audio: AudioData) -> None:
    rms = float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2)))
    assert rms > 1e-3


def test_decode_custom_sample_rate() -> None:
    audio = decode_audio(ASSET, sample_rate=24000)
    assert audio.sample_rate == 24000
    assert audio.duration == len(audio.samples) / 24000
    assert 11.5 <= audio.duration <= 12.5


def test_decode_missing_file_raises() -> None:
    with pytest.raises(AudioDecodeError):
        decode_audio(ASSET.parent / "does_not_exist.mp3")


def test_decode_invalid_file_raises(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"this is not audio data at all")
    with pytest.raises(AudioDecodeError):
        decode_audio(junk)


def test_probe_duration_matches_decode(audio: AudioData) -> None:
    probed = probe_duration(ASSET)
    assert abs(probed - audio.duration) < 0.3


def test_probe_duration_missing_file_raises() -> None:
    with pytest.raises(AudioDecodeError):
        probe_duration(ASSET.parent / "does_not_exist.mp3")
