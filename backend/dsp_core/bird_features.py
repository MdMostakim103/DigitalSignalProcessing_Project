"""Bird Sound Detector — traditional DSP only, no ML/AI.

Every "species" is represented by a small set of classical signal
descriptors (energy, zero-crossings, and three spectral-shape numbers).
A live 3-second clip is reduced to the same descriptors and compared to
each species' averaged descriptor set with a weighted distance. This is
the same family of technique as k-nearest-neighbour matching on hand
crafted features — no neural network, no trained model, no external AI
service.
"""

import numpy as np

from dsp_core.audio_fx import extract_loudest_window

# How much of a clip actually gets measured. Reference recordings and live
# mic clips both arrive at wildly different lengths (a Xeno-canto download
# might be 20s of habitat noise plus a 0.5s call; a live clip is a fixed
# RECORD_SECONDS window that may or may not contain the call in the same
# spot every time). Every clip is auto-trimmed to its own loudest window of
# this length before any feature is computed, so RMS/ZCR/spectral-shape are
# measured over comparable, call-dominated audio instead of being diluted
# by however much silence happened to surround the call. Replaces trimming
# reference files by eye, which produced inconsistent clip lengths (0.75s-
# 10.5s) and left silence/noise diluting the features either way.
TRIM_WINDOW_SECONDS = 2.0

# Feature order matters: it must be identical between reference building
# and live detection so the weight vector lines up correctly.
FEATURE_NAMES = (
    "rms",
    "zcr",
    "dominant_freq",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
)

# Relative importance of each feature in the distance calculation.
# Frequency-shape features carry more weight than loudness (rms), because
# loudness depends heavily on mic distance/gain and shouldn't be a strong
# vote in "which bird does this sound like".
FEATURE_WEIGHTS = np.array([0.5, 1.0, 1.5, 1.5, 1.0, 1.0])


def _rms(y: np.ndarray) -> float:
    """Root-mean-square amplitude: the classic loudness/energy measure,
    same idea as compute_short_time_energy in audio_fx.py but over the
    whole clip instead of per-frame."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))


def _zero_crossing_rate(y: np.ndarray) -> float:
    """Fraction of samples where the signal changes sign. A noisy, high
    pitched chirp crosses zero far more often per second than a low,
    smooth call — a cheap stand-in for 'how buzzy/tonal is this sound'."""
    if y.size < 2:
        return 0.0
    signs = np.sign(y)
    signs[signs == 0] = 1
    crossings = np.count_nonzero(np.diff(signs) != 0)
    return float(crossings) / float(y.size)


def _spectral_shape(y: np.ndarray, sr: int):
    """One Hann-windowed FFT, reused for four different descriptors —
    the same window/FFT pattern as detect_dominant_frequency() in
    audio_fx.py, so this stays consistent with the rest of the project."""
    if y.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    window = np.hanning(y.size)
    spectrum = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(y.size, d=1 / sr)

    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    # Dominant frequency: the single strongest bin.
    dominant_freq = float(freqs[int(np.argmax(spectrum))])

    # Spectral centroid: the "center of mass" of the spectrum — the
    # frequency-domain equivalent of an average, weighted by loudness at
    # each frequency. Higher centroid = brighter/higher-pitched sound.
    centroid = float(np.sum(freqs * spectrum) / total)

    # Spectral bandwidth: how spread out the energy is around the
    # centroid (its weighted standard deviation). A pure whistle has a
    # small bandwidth; a harsh, noisy call has a large one.
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))

    # Spectral rolloff: the frequency below which 85% of the total
    # spectral energy sits. Distinguishes calls that are mostly
    # low-frequency from calls with a lot of high-frequency content,
    # even when their centroid happens to be similar.
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff_idx = min(rolloff_idx, freqs.size - 1)
    rolloff = float(freqs[rolloff_idx])

    return dominant_freq, centroid, bandwidth, rolloff


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Reduce a raw audio clip to the fixed-length feature vector defined
    by FEATURE_NAMES. Called identically by reference building (on files in
    static/bird_samples/) and live detection (on the recorded mic clip) —
    both paths go through this one function, so trimming to the loudest
    window happens exactly once, in exactly one place, and can't drift out
    of sync between the two call sites."""
    y = np.asarray(y, dtype=np.float64)
    y = extract_loudest_window(y, sr, window_seconds=TRIM_WINDOW_SECONDS)
    rms = _rms(y)
    zcr = _zero_crossing_rate(y)
    dominant_freq, centroid, bandwidth, rolloff = _spectral_shape(y, sr)
    return np.array([rms, zcr, dominant_freq, centroid, bandwidth, rolloff], dtype=np.float64)


def normalize_features(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) across a set of feature vectors (rows), so every
    feature contributes to the distance on a comparable scale — otherwise
    dominant_freq (hundreds/thousands of Hz) would dwarf zcr (0-1) no
    matter what the weights say."""
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def weighted_distance(vec_a: np.ndarray, vec_b: np.ndarray, std: np.ndarray) -> float:
    """Weighted Euclidean distance between two (already feature-scaled)
    vectors. Each term is scaled by 1/std (z-score style) and then by the
    feature's importance weight before squaring and summing."""
    diff = (vec_a - vec_b) / std
    weighted = diff * FEATURE_WEIGHTS
    return float(np.sqrt(np.sum(weighted ** 2)))
