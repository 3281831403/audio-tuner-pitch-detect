"""Regression tests for the YIN tuner stubs in ``tuner/yin.py``.

The tests are deliberately implementation-facing: every test calls at least
one public stub, so the initial repository shows 15 red tests. Audio is always
represented as ``list[float]``; all signals are deterministic and generated
with ``math`` only.
"""

import math

import pytest

from tuner.yin import (
    TunerError,
    cents_offset,
    cumulative_mean,
    detect_batch,
    difference_fn,
    freq_from_lag,
    nearest_note,
    parabolic_interp,
    pick_lag,
)


SAMPLE_RATE = 48000
FRAME_SIZE = 2048
MAX_LAG = 400
THRESHOLD = 0.15


def _tone(freq, length=FRAME_SIZE, sample_rate=SAMPLE_RATE):
    """Return a deterministic sine tone as a list of samples."""
    return [
        math.sin(2.0 * math.pi * freq * index / sample_rate)
        for index in range(length)
    ]


def _silence(length=FRAME_SIZE):
    return [0.0] * length


def _config(**overrides):
    config = {
        "sample_rate": SAMPLE_RATE,
        "max_lag": MAX_LAG,
        "threshold": THRESHOLD,
        "target": 440.0,
    }
    config.update(overrides)
    return config


def _cents_between(actual, expected):
    return 1200.0 * math.log2(actual / expected)


# --- one focused test per defect -------------------------------------------


def test_difference_fn_normalizes_by_window_length():
    # max_lag=2 and len(frame)=4 mean W=2. For tau=1 the squared differences
    # are 1 and 1, so the window-normalized result is 1.0. An implementation
    # that omits the division returns 2.0 here.
    frame = [0.0, 1.0, 0.0, -1.0]
    diff = difference_fn(frame, 2)
    assert len(diff) == 2
    assert diff[0] == pytest.approx(1.0)
    assert diff[1] == pytest.approx(2.0)


def test_cumulative_mean_first_value_is_one():
    # diff[1] / mean(diff[0:2]) = 2.0 / 1.25 = 1.6.
    cmnd = cumulative_mean([0.5, 2.0])
    assert cmnd[0] == pytest.approx(1.0)
    assert cmnd[1] == pytest.approx(1.6)


def test_pick_lag_uses_threshold_not_first_valley():
    # Index 1 is the first local minimum but remains above the threshold.
    # The first sufficiently deep valley is index 3, i.e. sample lag 4.
    cmnd = [1.0, 0.50, 0.80, 0.05, 0.20, 0.30]
    assert pick_lag(cmnd, 0.10) == 4
    assert pick_lag([1.0, 0.40, 0.50, 0.60], 0.10) == -1


def test_parabolic_interp_refines_within_one_cent():
    true_lag = 109.37
    cmnd = [1.0] * 160
    for tau in range(100, 121):
        cmnd[tau - 1] = (tau - true_lag) ** 2
    refined = parabolic_interp(cmnd, 109)
    error_cents = 1200.0 * math.log2(refined / true_lag)
    assert abs(error_cents) <= 1.0


def test_freq_from_lag_preserves_fractional_lag():
    expected = SAMPLE_RATE / 109.5
    actual = freq_from_lag(109.5, SAMPLE_RATE)
    assert actual == pytest.approx(expected, rel=1e-12)
    assert abs(1200.0 * math.log2(actual / expected)) <= 1.0


def test_cents_offset_uses_base_two_log():
    assert cents_offset(880.0, 440.0) == pytest.approx(1200.0)
    assert cents_offset(660.0, 440.0) == pytest.approx(701.955, abs=1e-6)


def test_nearest_note_uses_a4_440():
    assert nearest_note(440.0) == ("A", 4)
    # With an A4=432 reference this frequency rounds up to A#4.
    assert nearest_note(445.0) == ("A", 4)


def test_silent_frame_returns_no_pitch():
    result = detect_batch([_silence()], _config())
    assert result["ok"] is True
    entry = result["results"][0]
    assert entry["freq"] == -1.0
    assert entry["lag"] == -1.0
    assert entry["note"] is None
    assert entry["octave"] is None


def test_detect_batch_isolates_frames_and_rolls_back():
    frames = [
        _tone(82.4069, length=4096),
        _tone(110.0, length=4096),
        _tone(146.8324, length=4096),
    ]
    config = _config(max_lag=700)
    result = detect_batch(frames, config)
    assert result["ok"] is True
    freqs = [entry["freq"] for entry in result["results"]]
    assert all(freq > 0.0 for freq in freqs)
    assert abs(_cents_between(freqs[0], 82.4069)) <= 1.0
    assert abs(_cents_between(freqs[1], 110.0)) <= 1.0
    assert abs(_cents_between(freqs[2], 146.8324)) <= 1.0
    assert [entry["note"] for entry in result["results"]] == ["E", "A", "D"]

    bad = _tone(220.0)
    bad[100] = float("nan")
    rollback = detect_batch([frames[0], bad], config)
    assert rollback["ok"] is False
    assert rollback["results"] == []
    assert rollback["error"]


def test_difference_fn_rejects_zero_max_lag():
    with pytest.raises(TunerError):
        difference_fn(_tone(440.0), 0)


# --- five integration/contract checks --------------------------------------


def test_440_hz_tone_is_a4():
    result = detect_batch([_tone(440.0)], _config())
    assert result["ok"] is True
    entry = result["results"][0]
    assert entry["note"] == "A"
    assert entry["octave"] == 4
    assert abs(_cents_between(entry["freq"], 440.0)) <= 1.0


def test_freq_from_lag_zero_raises_tuner_error():
    with pytest.raises(TunerError):
        freq_from_lag(0.0, SAMPLE_RATE)


def test_silence_returns_minus_one():
    diff = difference_fn(_silence(), MAX_LAG)
    cmnd = cumulative_mean(diff)
    assert pick_lag(cmnd, THRESHOLD) == -1


def test_same_pitch_has_zero_cents():
    assert abs(cents_offset(440.0, 440.0)) <= 1.0
    assert cents_offset(220.0, 220.0) == pytest.approx(0.0, abs=1e-12)


def test_batch_success_preserves_order():
    frames = [_tone(440.0), _tone(220.0), _tone(330.0)]
    result = detect_batch(frames, _config())
    assert result["ok"] is True
    assert [entry["index"] for entry in result["results"]] == [0, 1, 2]
    freqs = [entry["freq"] for entry in result["results"]]
    assert all(freq > 0.0 for freq in freqs)
    for actual, expected in zip(freqs, [440.0, 220.0, 330.0]):
        assert abs(_cents_between(actual, expected)) <= 1.0
