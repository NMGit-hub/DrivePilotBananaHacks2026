"""Unit tests for the wake-word tooling logic (no data, GPU or models needed).

Run with:  .venv/Scripts/python -m pytest scripts/wakeword -q
"""

from __future__ import annotations

import numpy as np
import pytest
from augment_features import mix_at_snr, place_in_window, rms
from common import MULTI_SPEAKER_VOICES, SINGLE_SPEAKER_VOICES, SPEAKER_SPLIT_GROUP, speaker_splits
from evaluate import fired_frames, wilson


def test_speaker_splits_are_disjoint_complete_and_deterministic() -> None:
    splits = speaker_splits()
    assert splits == speaker_splits()
    for voice, n in MULTI_SPEAKER_VOICES.items():
        s = splits[voice]
        train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (train & val or train & test or val & test), voice
        assert train | val | test == set(range(n)), voice
        assert test, f"{voice} has no held-out test speakers"
    for voice, parent in SPEAKER_SPLIT_GROUP.items():
        assert splits[voice] == splits[parent]


def test_single_speaker_voices_and_same_person_voices_do_not_leak() -> None:
    def person(name: str) -> str:
        return name.split("-")[1]

    train = {person(v) for v in SINGLE_SPEAKER_VOICES["train"]}
    for split in ("val", "test"):
        assert not train & {person(v) for v in SINGLE_SPEAKER_VOICES[split]}
    assert not {person(v) for v in SINGLE_SPEAKER_VOICES["val"]} & {person(v) for v in SINGLE_SPEAKER_VOICES["test"]}


def test_fired_frames_threshold_debounce_and_patience() -> None:
    s = np.zeros(60, dtype=np.float32)
    s[[10, 11, 12, 40, 41]] = 0.9
    s[20] = 0.6
    assert fired_frames(s, 0.5, debounce_s=0.0).tolist() == [10, 11, 12, 20, 40, 41]
    # 2 s debounce = 25 frames: 11, 12 and 20 are suppressed after the fire at 10
    assert fired_frames(s, 0.5, debounce_s=2.0).tolist() == [10, 40]
    assert fired_frames(s, 0.5, patience=2, debounce_s=0.0).tolist() == [11, 12, 41]
    assert fired_frames(s, 0.95).tolist() == []


def test_wilson_interval() -> None:
    lo, hi = wilson(95, 100)
    assert lo < 0.95 < hi
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0


def test_place_in_window_right_aligns_with_jitter() -> None:
    rng = np.random.default_rng(0)
    x = np.ones(16000, dtype=np.float32)
    out, (start, end) = place_in_window(x, 32000, rng, keep_end=True)
    assert out.shape == (32000,)
    assert 32000 - 3200 <= end <= 32000
    assert end - start == 16000
    assert out[start:end].sum() == 16000 and out.sum() == 16000


def test_place_in_window_keeps_end_of_long_positive() -> None:
    rng = np.random.default_rng(1)
    x = np.arange(40000, dtype=np.float32)
    out, (start, end) = place_in_window(x, 32000, rng, keep_end=True)
    assert start == 0
    assert out[end - 1] == 39999  # last sample of the phrase is kept


@pytest.mark.parametrize("snr", [0.0, 5.0, 10.0, 20.0])
def test_mix_at_snr_defines_snr_on_the_speech_span(snr: float) -> None:
    rng = np.random.default_rng(2)
    speech = np.zeros(32000, dtype=np.float32)
    speech[10000:26000] = rng.normal(0, 0.1, 16000)
    noise = rng.normal(0, 0.3, 32000).astype(np.float32)
    mixed = mix_at_snr(speech, noise, snr, (10000, 26000))
    added = mixed - speech
    measured = 20 * np.log10(rms(speech[10000:26000]) / rms(added))
    assert measured == pytest.approx(snr, abs=1e-3)
