"""YIN pitch-detection primitives for a monophonic guitar tuner.

Conventions used throughout this module
---------------------------------------
* ``sample_rate`` is expressed in Hz.
* A frame is a non-empty ``list[float]`` of equally spaced audio samples in
  ``[-1.0, 1.0]``; ``len(frame)`` is the frame length in samples.
* Frequencies are expressed in Hz.
* Pitch error is measured in cents. One cent is ``1 / 1200`` of an octave, so
  a ratio ``r`` corresponds to ``1200 * log2(r)`` cents. The detection
  tolerance is one cent.
* Invalid input raises :class:`TunerError`.
"""

import math


class TunerError(Exception):
    """Raised when tuner input or configuration is invalid."""


def difference_fn(frame: list[float], max_lag: int) -> list[float]:
    """Compute the YIN difference function for ``frame``.

    Parameters
    ----------
    frame:
        Audio samples in ``[-1.0, 1.0]``. Its length must be greater than
        ``max_lag``.
    max_lag:
        Positive integer maximum lag in samples. Lags ``1 .. max_lag`` are
        evaluated.

    Returns
    -------
    list[float]
        Exactly ``max_lag`` values. Index ``tau - 1`` holds the mean squared
        difference for lag ``tau``::

            W = len(frame) - max_lag
            d[tau] = (1 / W) * sum(
                (frame[j] - frame[j + tau]) ** 2 for j in range(W)
            )

        Dividing by the window length ``W`` keeps the magnitude independent
        of frame length, so the same waveform at different lengths produces
        comparable differences.

    Raises
    ------
    TunerError
        If ``max_lag <= 0``, ``max_lag >= len(frame)``, the frame is empty,
        or any sample is not finite.
    """
    if not isinstance(frame, list) or len(frame) == 0:
        raise TunerError("frame must be a non-empty list of samples")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TunerError("max_lag must be a positive integer")
    if max_lag <= 0:
        raise TunerError("max_lag must be positive")
    if max_lag >= len(frame):
        raise TunerError("max_lag must be smaller than the frame length")
    for sample in frame:
        if not isinstance(sample, (int, float)) or not math.isfinite(sample):
            raise TunerError("every sample must be finite")

    window = len(frame) - max_lag
    diff = [0.0] * max_lag
    for tau in range(1, max_lag + 1):
        total = 0.0
        for j in range(window):
            delta = frame[j] - frame[j + tau]
            total += delta * delta
        # Window-length normalization keeps the magnitude independent of
        # frame length (omitting this division makes d[] grow with W).
        diff[tau - 1] = total / window
    return diff


def cumulative_mean(diff: list[float]) -> list[float]:
    """Convert raw differences to the cumulative mean normalized difference.

    ``diff`` is the output of :func:`difference_fn`, so ``diff[i]`` belongs to
    positive lag ``tau = i + 1``. The returned list has the same length and
    follows::

        out[0] = 1.0
        out[i] = diff[i] / mean(diff[0:i + 1])   for i >= 1

    If a running mean is zero (for example on a silent frame), that output is
    ``1.0`` instead of dividing by zero.

    Raises
    ------
    TunerError
        If ``diff`` is empty, contains a negative or non-finite value, or any
        result would be non-finite.
    """
    if not isinstance(diff, list) or len(diff) == 0:
        raise TunerError("diff must be a non-empty list")
    for value in diff:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TunerError("every difference must be finite")
        if value < 0.0:
            raise TunerError("differences must be non-negative")

    out = [0.0] * len(diff)
    # First entry is defined as 1.0 (tau=0 in the classic YIN paper),
    # never 0.0: a zero first term would poison every later ratio.
    out[0] = 1.0
    running_sum = diff[0]
    for i in range(1, len(diff)):
        running_sum += diff[i]
        running_mean = running_sum / (i + 1)
        if running_mean == 0.0:
            out[i] = 1.0
        else:
            value = diff[i] / running_mean
            if not math.isfinite(value):
                raise TunerError("cumulative mean result is not finite")
            out[i] = value
    return out


def pick_lag(cmnd: list[float], threshold: float) -> int:
    """Return the first sufficiently deep YIN valley as a sample lag.

    ``cmnd`` uses the indexing of :func:`cumulative_mean`: index ``i`` is
    sample lag ``i + 1``. Scan lags in increasing order, skipping every value
    at or above ``threshold``. At the first value strictly below ``threshold``,
    keep advancing while the next adjacent value is smaller, then return that
    lag. This prevents an early local minimum that is still above the
    threshold from being mistaken for the fundamental period.

    Returns ``-1`` when no lag is below ``threshold``, when ``cmnd`` has fewer
    than two entries, or when the selected valley is out of bounds.

    Raises
    ------
    TunerError
        If ``threshold`` is not in ``(0.0, 1.0)`` or ``cmnd`` contains a
        non-finite value.
    """
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise TunerError("threshold must be finite")
    if not (0.0 < threshold < 1.0):
        raise TunerError("threshold must lie strictly between 0.0 and 1.0")
    if not isinstance(cmnd, list) or len(cmnd) < 2:
        return -1
    for value in cmnd:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TunerError("every CMND value must be finite")

    n = len(cmnd)
    i = 0
    # Skip every value at or above the threshold instead of stopping at the
    # first local minimum, which can sit on an octave-up harmonic.
    while i < n and cmnd[i] >= threshold:
        i += 1
    if i >= n:
        return -1
    # Walk to the bottom of the valley while adjacent values keep dropping.
    while i + 1 < n and cmnd[i + 1] < cmnd[i]:
        i += 1
    lag = i + 1
    if lag < 1 or lag > n:
        return -1
    return lag


def parabolic_interp(cmnd: list[float], lag: int) -> float:
    """Refine an integer lag by fitting a parabola to three CMND values.

    ``lag`` is a sample lag with ``2 <= lag < len(cmnd)``. Because CMND index
    ``i`` corresponds to lag ``i + 1``, the three points are
    ``cmnd[lag - 2]`` (left), ``cmnd[lag - 1]`` (center), and ``cmnd[lag]``
    (right). The returned fractional lag is::

        delta = 0.5 * (left - right) / (left - 2 * center + right)
        refined = lag + delta

    Parabolic interpolation is required instead of linear interpolation: the
    refined frequency must be within one cent of the true frequency for a
    well-formed dip. If the denominator is zero or the refined value would
    leave the interval, return ``float(lag)``.

    Raises
    ------
    TunerError
        If ``lag`` is out of range or ``cmnd`` contains a non-finite value.
    """
    if not isinstance(cmnd, list):
        raise TunerError("cmnd must be a list")
    for value in cmnd:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TunerError("every CMND value must be finite")
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise TunerError("lag must be an integer sample lag")
    if not (2 <= lag < len(cmnd)):
        raise TunerError("lag must satisfy 2 <= lag < len(cmnd)")

    left = cmnd[lag - 2]
    center = cmnd[lag - 1]
    right = cmnd[lag]
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return float(lag)
    delta = 0.5 * (left - right) / denominator
    refined = lag + delta
    # Stay inside the bracket spanned by the three samples.
    if refined < lag - 1.0 or refined > lag + 1.0 or not math.isfinite(refined):
        return float(lag)
    return refined


def freq_from_lag(lag: float, sample_rate: float) -> float:
    """Convert a sample lag to a frequency in Hz.

    ``freq = sample_rate / lag``. ``lag`` is a (possibly fractional) count of
    samples and must be positive; fractional lag must not be truncated before
    the division. ``sample_rate`` is in Hz and must also be positive.

    Raises
    ------
    TunerError
        If ``lag <= 0`` (including ``lag == 0``), ``sample_rate <= 0``, or
        either argument is not finite.
    """
    if not isinstance(lag, (int, float)) or not math.isfinite(lag):
        raise TunerError("lag must be finite")
    if not isinstance(sample_rate, (int, float)) or not math.isfinite(sample_rate):
        raise TunerError("sample_rate must be finite")
    if lag <= 0:
        raise TunerError("lag must be positive")
    if sample_rate <= 0:
        raise TunerError("sample_rate must be positive")
    # Keep the fractional lag intact; truncating it biases the frequency up.
    return sample_rate / lag


def cents_offset(freq: float, target: float) -> float:
    """Return the pitch deviation from ``target`` in cents.

    The result is ``1200 * log2(freq / target)``. Both frequencies are in Hz
    and must be positive and finite. A result of zero means the two pitches
    are identical; comparisons use a one-cent tolerance.

    Raises
    ------
    TunerError
        If either frequency is non-positive or not finite.
    """
    if not isinstance(freq, (int, float)) or not math.isfinite(freq):
        raise TunerError("freq must be finite")
    if not isinstance(target, (int, float)) or not math.isfinite(target):
        raise TunerError("target must be finite")
    if freq <= 0.0 or target <= 0.0:
        raise TunerError("frequencies must be positive")
    # Base-two logarithm: 1200 * ln(ratio) (natural log) is off by a factor
    # of ln(2) and gives a ~0.7-cent error at the octave.
    return 1200.0 * math.log2(freq / target)


def nearest_note(freq: float) -> tuple[str, int]:
    """Return the nearest equal-tempered note as ``(name, octave)``.

    The reference is A4 = 440 Hz (MIDI note 69), with twelve equal-tempered
    semitones per octave. Names are ASCII spellings: ``C``, ``C#``, ``D``,
    ``D#``, ``E``, ``F``, ``F#``, ``G``, ``G#``, ``A``, ``A#``, ``B``.
    Octave numbers follow scientific pitch notation (A4 = 440 Hz). Selection
    rounds to the nearest semitone using the 440 Hz reference, so 445 Hz is
    A4 rather than A#4.

    Raises
    ------
    TunerError
        If ``freq`` is non-positive or not finite.
    """
    if not isinstance(freq, (int, float)) or not math.isfinite(freq):
        raise TunerError("freq must be finite")
    if freq <= 0.0:
        raise TunerError("freq must be positive")

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # MIDI note 69 is A4 at exactly 440 Hz -- not 432 Hz.
    semitones_from_a4 = 12.0 * math.log2(freq / 440.0)
    midi_note = int(round(69.0 + semitones_from_a4))
    name = names[midi_note % 12]
    # MIDI octave: note 60 is C4, so scientific octave = midi // 12 - 1.
    octave = midi_note // 12 - 1
    return name, octave


def _validate_config(config: dict) -> tuple[float, int, float, float]:
    """Validate a batch config and return its typed values."""
    if not isinstance(config, dict):
        raise TunerError("config must be a dictionary")
    required = ("sample_rate", "max_lag", "threshold", "target")
    for key in required:
        if key not in config:
            raise TunerError(f"missing config key: {key}")

    sample_rate = config["sample_rate"]
    max_lag = config["max_lag"]
    threshold = config["threshold"]
    target = config["target"]

    if not isinstance(sample_rate, (int, float)) or not math.isfinite(sample_rate):
        raise TunerError("sample_rate must be finite")
    if sample_rate <= 0:
        raise TunerError("sample_rate must be positive")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag <= 0:
        raise TunerError("max_lag must be a positive integer")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise TunerError("threshold must be finite")
    if not (0.0 < threshold < 1.0):
        raise TunerError("threshold must lie strictly between 0.0 and 1.0")
    if not isinstance(target, (int, float)) or not math.isfinite(target):
        raise TunerError("target must be finite")
    if target <= 0:
        raise TunerError("target must be positive")
    return float(sample_rate), max_lag, float(threshold), float(target)


def _detect_one(frame: list[float], sample_rate: float, max_lag: int,
                threshold: float, target: float) -> dict:
    """Run the full YIN pipeline on one frame using frame-local buffers."""
    if not isinstance(frame, list) or len(frame) == 0:
        raise TunerError("every frame must be a non-empty list of samples")
    if max_lag >= len(frame):
        raise TunerError("frame is too short for the configured max_lag")
    for sample in frame:
        if not isinstance(sample, (int, float)) or not math.isfinite(sample):
            raise TunerError("every sample must be finite")

    # A silent frame has no pitch: report -1 markers instead of inventing one.
    if all(sample == 0.0 for sample in frame):
        return {
            "freq": -1.0,
            "note": None,
            "octave": None,
            "cents": 0.0,
            "lag": -1.0,
        }

    # Each frame gets its own difference/CMND buffers so frames cannot
    # contaminate one another through shared state.
    diff = difference_fn(frame, max_lag)
    cmnd = cumulative_mean(diff)
    lag = pick_lag(cmnd, threshold)
    if lag == -1:
        return {
            "freq": -1.0,
            "note": None,
            "octave": None,
            "cents": 0.0,
            "lag": -1.0,
        }

    if 2 <= lag < len(cmnd):
        refined_lag = parabolic_interp(cmnd, lag)
    else:
        refined_lag = float(lag)
    freq = freq_from_lag(refined_lag, sample_rate)
    note, octave = nearest_note(freq)
    cents = cents_offset(freq, target)
    return {
        "freq": freq,
        "note": note,
        "octave": octave,
        "cents": cents,
        "lag": refined_lag,
    }


def detect_batch(frames: list[list[float]], config: dict) -> dict:
    """Detect the pitch of every frame atomically, preserving frame order.

    ``frames`` is a list of audio frames using the frame convention above.
    ``config`` contains:

    * ``sample_rate`` -- sample rate in Hz.
    * ``max_lag`` -- positive YIN search limit in samples.
    * ``threshold`` -- YIN CMND threshold in ``(0.0, 1.0)``.
    * ``target`` -- target frequency in Hz for the cents offset.

    On success the returned dictionary is::

        {
            "ok": True,
            "results": [
                {
                    "index": 0,
                    "freq": 440.0,       # Hz, or -1.0 for silence
                    "note": "A",         # None for silence
                    "octave": 4,         # None for silence
                    "cents": 0.0,        # offset from target; 0.0 for silence
                    "lag": 109.1,        # refined sample lag; -1.0 for silence
                },
                ...
            ],
            "error": None,
        }

    Results appear in input order and each result carries its input index.
    A silent frame (all samples zero) yields ``freq == -1.0``, ``lag == -1.0``,
    and ``note``/``octave`` set to ``None`` rather than an invented pitch.

    If any frame is invalid (not a non-empty finite list, too short for the
    configured lags, or otherwise failing a detector precondition) or if the
    configuration is invalid, the whole call must roll back: return
    ``{"ok": False, "results": [], "error": "<reason>"}``. Never return partial
    results from frames that happened to precede the bad frame. Per-frame
    difference/CMND buffers must be local to each frame so one frame cannot
    contaminate the next.

    Raises
    ------
    TunerError
        Only for errors outside per-frame data, if an implementation chooses
        to propagate configuration errors; the batch contract above uses the
        result dictionary for invalid frames and configuration.
    """
    if not isinstance(frames, list):
        return {"ok": False, "results": [], "error": "frames must be a list"}
    try:
        sample_rate, max_lag, threshold, target = _validate_config(config)
        # Validate every frame before computing anything, so a bad later
        # frame cannot leave partial results behind.
        for frame in frames:
            if not isinstance(frame, list) or len(frame) == 0:
                raise TunerError("every frame must be a non-empty list")
            if max_lag >= len(frame):
                raise TunerError(
                    "frame is too short for the configured max_lag"
                )
            for sample in frame:
                if not isinstance(sample, (int, float)) or not math.isfinite(
                    sample
                ):
                    raise TunerError("every sample must be finite")

        results = []
        for index, frame in enumerate(frames):
            entry = _detect_one(
                frame, sample_rate, max_lag, threshold, target
            )
            entry = {"index": index, **entry}
            results.append(entry)
    except TunerError as exc:
        # Full rollback: no partial results survive.
        return {"ok": False, "results": [], "error": str(exc)}

    return {"ok": True, "results": results, "error": None}
