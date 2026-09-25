import numpy as np
from scipy.signal import fftconvolve, butter, cheby1, cheby2, ellip, bessel, sosfiltfilt, sosfreqz
import librosa

def amplify_volume(audio_array: np.ndarray,factor: float=4.0) -> np.ndarray:
    modified_audio = audio_array * factor

    print("Original: ",np.max(np.abs(audio_array)))
    print("Modified: ",np.max(np.abs(modified_audio)))
    return modified_audio

def apply_reverb(y: np.ndarray, sr: int) -> np.ndarray:

    ir_length = int(sr * 1.5)
    ir = np.zeros(ir_length)

    ir[0] = 1.0

    # Reflections
    # for simplicity ,, this can be replaced by a real impulse response,
    ir[int(0.2 * sr)] = 0.6
    ir[int(0.4 * sr)] = 0.35
    ir[int(0.7 * sr)] = 0.2
    ir[int(1.0 * sr)] = 0.1
    ir[int(1.3 * sr)] = 0.05

    y_reverb = fftconvolve(y, ir, mode="full")

    # Prevent clipping
    y_reverb = y_reverb / np.max(np.abs(y_reverb))

    return y_reverb


def apply_echo(y: np.ndarray, sr: int, delay_seconds: float = 0.28, decay: float = 0.55, repeats: int = 5) -> np.ndarray:
    """y[n] + decay*x[n-D] + decay^2*x[n-2D] + ... — several decaying,
    evenly-spaced repeats of the dry signal (a feedback delay line)."""
    delay_samples = max(1, int(sr * delay_seconds))

    out_len = len(y) + delay_samples * repeats
    out = np.zeros(out_len)
    out[:len(y)] += y

    gain = decay
    for r in range(1, repeats + 1):
        start = delay_samples * r
        out[start:start + len(y)] += gain * y
        gain *= decay

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak

    return out


def apply_delay(y: np.ndarray, sr: int, delay_seconds: float = 0.35, wet: float = 0.85) -> np.ndarray:
    """y[n] mixed with a single x[n-D] repeat — one distinct, undecayed
    repetition buffered and played back after a fixed interval."""
    delay_samples = max(1, int(sr * delay_seconds))

    out = np.zeros(len(y) + delay_samples)
    out[:len(y)] += y
    out[delay_samples:delay_samples + len(y)] += wet * y

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak

    return out


def apply_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The literal definition: reverse h, slide it across x, multiply and
    accumulate the overlap at every step. fftconvolve computes exactly that
    sum (just efficiently); it is not a different algorithm."""
    y = fftconvolve(x, h, mode="full")

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y


def apply_noise_reduction(y: np.ndarray, sr: int) -> np.ndarray:
    
    stft_matrix = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512
    )
    magnitude = np.abs(stft_matrix)
    noise_profile = np.percentile(magnitude, 60, axis=1)

    threshold = noise_profile[:, np.newaxis] * 2.0

    # Components below the threshold are treated as noise.
    mask = magnitude > threshold

    # suppressing the guessed noise
    gain = np.where(mask, 1.0, 0.01)

    cleaned_stft = stft_matrix * gain

    cleaned_audio = librosa.istft(
        cleaned_stft,
        hop_length=512,
        length=len(y)
    )

    max_value = np.max(np.abs(cleaned_audio))

    if max_value > 0:
        cleaned_audio = cleaned_audio / max_value

    return cleaned_audio

BAND_TYPES = ("lowpass", "highpass", "bandpass", "bandstop")
IIR_FAMILIES = ("butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel")


PASSBAND_RIPPLE_DB = 1.0
STOPBAND_ATTENUATION_DB = 40.0


def _normalized_edges(band_type: str, sr: int, cutoff: float, cutoff2: float = None):
    """Cutoff(s) as a fraction of Nyquist, clamped just inside (0, 1) since
    scipy rejects the edges."""
    nyq = sr / 2.0
    if band_type in ("bandpass", "bandstop"):
        low = min(max(cutoff, 1.0), nyq - 2.0) / nyq
        high = min(max(cutoff2 or (cutoff + 1000.0), 1.0), nyq - 1.0) / nyq
        if low >= high:
            high = min(0.999, low + 0.01)
        return [low, high]
    return min(0.999, max(0.001, cutoff / nyq))


def _design_coeffs(filter_family: str, band_type: str, sr: int, cutoff: float, cutoff2: float, order: int):
    """Designs are returned as second-order sections (SOS), never as
    transfer-function (b, a) coefficients.

    This is not a style preference. In (b, a) form the coefficients of a
    high-order or narrow-band IIR filter span a huge dynamic range, and the
    round-off cancellation is severe enough to destroy the result: a
    4th-order 40-60 Hz notch at 44.1 kHz produces all-NaN output, as does an
    8th-order 60 Hz low-pass. Cascaded second-order sections keep every
    stage well conditioned, so the same designs come out clean.
    """
    wn = _normalized_edges(band_type, sr, cutoff, cutoff2)
    if filter_family == "butterworth":
        return butter(order, wn, btype=band_type, output="sos")
    if filter_family == "chebyshev1":
        return cheby1(order, PASSBAND_RIPPLE_DB, wn, btype=band_type, output="sos")
    if filter_family == "chebyshev2":
        return cheby2(order, STOPBAND_ATTENUATION_DB, wn, btype=band_type, output="sos")
    if filter_family == "elliptic":
        return ellip(order, PASSBAND_RIPPLE_DB, STOPBAND_ATTENUATION_DB, wn, btype=band_type, output="sos")
    if filter_family == "bessel":
        return bessel(order, wn, btype=band_type, norm="phase", output="sos")
    raise ValueError(f"Unknown filter_family: {filter_family}")


def _ideal_mask(freqs: np.ndarray, band_type: str, cutoff: float, cutoff2: float) -> np.ndarray:
    """Boolean brick-wall mask for the ideal filter family. All four band
    types are just set membership on the frequency axis — no different in
    kind from lowpass/highpass, only in how many cutoffs they need."""
    if band_type == "lowpass":
        return freqs <= cutoff
    if band_type == "highpass":
        return freqs >= cutoff
    lo, hi = min(cutoff, cutoff2), max(cutoff, cutoff2)
    if band_type == "bandpass":
        return (freqs >= lo) & (freqs <= hi)
    if band_type == "bandstop":
        return (freqs < lo) | (freqs > hi)
    raise ValueError(f"Unknown band_type: {band_type}")


def apply_filter(
    y: np.ndarray,
    sr: int,
    filter_family: str = "butterworth",
    band_type: str = "lowpass",
    cutoff: float = 1000.0,
    cutoff2: float = 4000.0,
    order: int = 4,
) -> np.ndarray:
    """Two ideas from the same course unit, side by side:
    - Five IIR design families (Butterworth, Chebyshev I/II, Elliptic,
      Bessel), each a different trade-off between passband flatness,
      stopband attenuation, transition sharpness, and phase linearity —
      all built from the same order/cutoff inputs.
    - The "ideal" brick-wall filter, built by zeroing FFT bins outside the
      passband outright. Perfectly sharp in frequency, but that sharp edge
      causes ringing (Gibbs phenomenon) in time.
    """
    if filter_family in IIR_FAMILIES:
        order = max(1, min(10, int(order)))
        sos = _design_coeffs(filter_family, band_type, sr, cutoff, cutoff2, order)
        # sosfiltfilt runs the cascade forward and backward: zero phase
        # distortion, so the output waveform stays aligned with the input.
        y_out = sosfiltfilt(sos, y)
    elif filter_family == "ideal":
        spectrum = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), d=1 / sr)
        mask = _ideal_mask(freqs, band_type, cutoff, cutoff2)
        y_out = np.fft.irfft(spectrum * mask, n=len(y))
    else:
        raise ValueError(f"Unknown filter_family: {filter_family}")

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = y_out / peak

    return y_out


def compute_filter_frequency_response(
    filter_family: str,
    band_type: str,
    sr: int,
    cutoff: float = 1000.0,
    cutoff2: float = 4000.0,
    order: int = 4,
    n_points: int = 512,
):
    """The filter's own |H(f)| curve, independent of any audio — used both
    for the standalone live filter-response preview and as the 'multiply
    by' stage in the frequency-domain visualization."""
    max_freq = min(sr / 2.0, 20000.0)
    freqs = np.linspace(0, max_freq, n_points)

    if filter_family in IIR_FAMILIES:
        order = max(1, min(10, int(order)))
        sos = _design_coeffs(filter_family, band_type, sr, cutoff, cutoff2, order)
        _, h = sosfreqz(sos, worN=freqs, fs=sr)
        magnitude = np.abs(h)
    elif filter_family == "ideal":
        magnitude = _ideal_mask(freqs, band_type, cutoff, cutoff2).astype(float)
    else:
        raise ValueError(f"Unknown filter_family: {filter_family}")

    return freqs, magnitude


def compute_short_time_energy(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    """RMS energy per frame — the same short-time energy (STE) idea speech-
    activity detectors use to tell voiced/active regions from background
    silence, before anything as complex as pitch or spectral analysis."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    frame_len = min(frame_len, max(1, len(y)))

    if len(y) <= frame_len:
        n_frames = 1
    else:
        n_frames = 1 + (len(y) - frame_len) // hop_len

    energies = np.zeros(n_frames, dtype=np.float64)
    frame_times = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start:start + frame_len]
        energies[i] = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        frame_times[i] = (start + frame_len / 2) / sr

    return energies, frame_times, frame_len, hop_len


def compute_zero_crossing_rate(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Fraction of adjacent-sample sign flips per frame, using the SAME
    framing (frame_len, hop_len) as compute_short_time_energy so energy and
    ZCR line up frame-for-frame. Voiced speech is quasi-periodic around a low
    fundamental, so it crosses zero slowly (low ZCR); broadband noise crosses
    zero on almost every sample (high ZCR) even at the same loudness —
    that's the extra discrimination energy alone doesn't have.
    """
    if len(y) <= frame_len:
        n_frames = 1
    else:
        n_frames = 1 + (len(y) - frame_len) // hop_len

    zcr = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start:start + frame_len]
        if frame.size < 2:
            continue
        signs = np.sign(frame)
        signs[signs == 0] = 1.0
        zcr[i] = np.mean(np.abs(np.diff(signs)) > 0)

    return zcr


def _estimate_noise_floor(energies: np.ndarray, percentile: float = 20.0) -> float:
    """The Nth percentile of frame energies, not the max. A threshold set as
    a fraction of the LOUDEST frame is fragile — one loud transient (a door
    slam, a clipped syllable) rescales the threshold for the entire file. The
    bottom percentile of frames is far more likely to be background noise/
    silence in most recordings, and a single outlier there barely moves a
    percentile the way it dominates a max."""
    if energies.size == 0:
        return 0.0
    return float(np.percentile(energies, percentile))


def _estimate_signal_level(energies: np.ndarray, percentile: float = 95.0) -> float:
    """The high-percentile counterpart to _estimate_noise_floor: a robust
    stand-in for 'how loud does this recording normally get', used instead
    of the literal max(). A single 2ms transient can still double the true
    peak; a 95th-percentile level barely moves, because it takes many loud
    frames — not one — to shift a percentile. This is what actually closes
    the threshold's sensitivity to one-off spikes; anchoring the noise floor
    alone still leaves the top end fully exposed to whatever the single
    loudest frame happens to be."""
    if energies.size == 0:
        return 0.0
    return float(np.percentile(energies, percentile))


def _apply_min_duration(active: np.ndarray, hop_len: int, sr: int, min_speech_ms: float, min_silence_ms: float) -> np.ndarray:
    """Two clean-up passes over the boolean frame decisions: first bridge
    short silence gaps (so a single frame that dips below threshold in the
    middle of a word doesn't chop it in two), then discard speech islands
    that are too short to be a real syllable (spurious single-frame blips).
    This is what a per-frame threshold test alone can never provide — it has
    no notion of "this is happening in the middle of an ongoing word."
    """
    n = active.size
    if n == 0:
        return active
    hop_seconds = hop_len / sr
    min_speech_frames = max(1, int(round(min_speech_ms / 1000 / hop_seconds)))
    min_silence_frames = max(1, int(round(min_silence_ms / 1000 / hop_seconds)))

    out = active.copy()

    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            gap_len = j - i
            flanked = i > 0 and j < n
            if flanked and gap_len < min_silence_frames:
                out[i:j] = True
            i = j
        else:
            i += 1

    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_speech_frames:
                out[i:j] = False
            i = j
        else:
            i += 1

    return out


def _activity_gain_from_frames(active: np.ndarray, hop_len: int, total_len: int, fade_ms: float, sr: int) -> np.ndarray:
    """Frame decisions -> a per-sample gain envelope. Earlier this convolved
    with a RECTANGULAR (boxcar) kernel, which is a poor choice: a boxcar's
    frequency response has sidelobes and its output ramps linearly with a
    sharp kink at both ends, not a smooth curve. A Hann (raised-cosine)
    kernel is the standard fix — it has no sidelobes and produces a genuinely
    smooth S-shaped fade in and out, the same window shape used everywhere
    else in this app's own FFT analysis (Modules 2, 5, 6)."""
    gain = np.repeat(active.astype(np.float32), hop_len)
    if gain.size < total_len:
        gain = np.pad(gain, (0, total_len - gain.size), mode="edge")
    gain = gain[:total_len]

    fade_len = max(1, int(sr * fade_ms / 1000))
    if fade_len > 1 and gain.size:
        kernel = np.hanning(fade_len).astype(np.float32)
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
            gain = np.convolve(gain, kernel, mode="same")

    return gain


def apply_activity_gate(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    method: str = "energy",
    energy_threshold_ratio: float = 0.15,
    zcr_threshold_hz: float = 700.0,
    hysteresis_ratio: float = 0.5,
    min_speech_ms: float = 60.0,
    min_silence_ms: float = 60.0,
    fade_ms: float = 15.0,
):
    """Voice activity detection as a small classical pipeline, not a single
    threshold test:

      framing -> energy (+ ZCR) -> noise-floor-relative hysteresis -> minimum
      duration cleanup -> smooth gain -> gated audio

    Two things a bare "energy > threshold_ratio * peak" test gets wrong, both
    fixed here:

    1. A single loud transient sets the scale for the WHOLE file (the peak).
       The threshold is instead set relative to an estimated noise floor
       (a low percentile of frame energies) blended with the peak, so it
       tracks the actual quiet/loud contrast in the recording rather than
       one outlier sample.

    2. A single strict per-frame test rejects unvoiced speech (s, f, sh, and
       stop-consonant bursts) outright, because those are quiet AND
       high-ZCR — they would fail an energy+ZCR AND-test even though they
       are real speech, not noise. Fixed with HYSTERESIS: ZCR only gates
       *entering* the active state (so pure noise can't trigger a false
       onset), while *staying* active only requires clearing a lower exit
       threshold on energy. A trailing unvoiced consonant right after a
       voiced vowel — quieter, higher ZCR — keeps its already-active state
       instead of being cut off mid-word.

    method="energy" skips the ZCR gate entirely (the "simple" mode); both
    modes get the noise-floor-relative hysteresis, minimum-duration cleanup,
    and Hann-smoothed gain.
    """
    energies, frame_times, frame_len, hop_len = compute_short_time_energy(y, sr, frame_ms, hop_ms)

    noise_floor = _estimate_noise_floor(energies)
    peak = float(np.max(energies)) if energies.size else 0.0
    # The threshold is anchored to a robust signal level, not the literal
    # peak: a single loud transient can double the true max, but barely
    # shifts a 95th-percentile estimate. `peak` itself is still reported
    # (e.g. for display) as the real maximum.
    signal_level = _estimate_signal_level(energies)
    headroom = max(0.0, signal_level - noise_floor)
    enter_threshold = noise_floor + energy_threshold_ratio * headroom
    exit_threshold = noise_floor + energy_threshold_ratio * hysteresis_ratio * headroom

    use_zcr = method == "energy_zcr"
    zcr = compute_zero_crossing_rate(y, frame_len, hop_len) if use_zcr else None
    zcr_threshold_ratio = (zcr_threshold_hz * 2 / sr) if use_zcr else None

    n = energies.size
    active = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        if not state:
            enters = energies[i] > enter_threshold
            if use_zcr:
                enters = enters and zcr[i] < zcr_threshold_ratio
            state = bool(enters)
        else:
            state = bool(energies[i] >= exit_threshold)
        active[i] = state

    active = _apply_min_duration(active, hop_len, sr, min_speech_ms, min_silence_ms)

    gain = _activity_gain_from_frames(active, hop_len, len(y), fade_ms, sr)
    y_out = y * gain

    peak_out = np.max(np.abs(y_out))
    if peak_out > 0:
        y_out = y_out / peak_out

    return y_out, energies, zcr, frame_times, active, enter_threshold, exit_threshold, noise_floor, peak, zcr_threshold_ratio


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq: float) -> dict:
    """Standard equal-temperament conversion: MIDI note number from
    frequency (A4 = 440 Hz = MIDI 69), then note name/octave from that,
    plus how many cents sharp/flat the real peak is from the nearest note."""
    if freq <= 0:
        return {"note": "-", "octave": None, "cents": 0.0, "midi": None}

    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_round = int(round(midi))
    cents = float((midi - midi_round) * 100)
    name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    return {"note": name, "octave": octave, "cents": cents, "midi": midi_round}


def detect_dominant_frequency(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0):
    """A Hann-windowed FFT magnitude spectrum, searched only inside
    [fmin, fmax] so DC offset and inaudible sub-bass don't win the peak by
    accident — the same idea a pitch detector uses before anything as
    complex as autocorrelation or cepstral analysis."""
    window = np.hanning(len(y)) if len(y) > 1 else np.ones_like(y)
    spectrum = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)

    nyq = sr / 2.0
    fmin = max(1.0, min(fmin, nyq - 2.0))
    fmax = max(fmin + 1.0, min(fmax, nyq - 1.0))
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask) or not np.any(spectrum[mask] > 0):
        return 0.0, spectrum, freqs

    candidates = spectrum[mask]
    peak_freq = float(freqs[mask][int(np.argmax(candidates))])
    return peak_freq, spectrum, freqs


def synthesize_tone(freq: float, duration_seconds: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """A pure sine at the detected frequency, same length as the input —
    the audible answer to 'is this really the dominant pitch?' A short
    fade in/out avoids a click at the start/end."""
    n_samples = max(1, int(sr * duration_seconds))
    t = np.arange(n_samples) / sr
    tone = amplitude * np.sin(2 * np.pi * max(freq, 0.0) * t)

    fade_len = min(n_samples // 20, int(sr * 0.02))
    if fade_len > 1:
        fade = np.linspace(0.0, 1.0, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]

    return tone.astype(np.float64)


MORPH_MODES = ("pitch", "stretch", "robot", "whisper")


def apply_voice_morph(
    y: np.ndarray,
    sr: int,
    morph_mode: str = "pitch",
    n_steps: float = 4.0,
    rate: float = 1.5,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Four phase-vocoder experiments that separate what the magnitude
    spectrum carries from what the phase carries:

    - pitch    : shift pitch, keep duration. Time-stretch with the phase
                 vocoder (which advances phase correctly per frame), then
                 resample back — so only the pitch moves.
    - stretch  : change duration, keep pitch. The same phase vocoder, but
                 without the resampling step.
    - robot    : throw the phase away (set every bin's phase to zero). The
                 magnitudes are untouched, yet the voice turns into a flat
                 monotone buzz — that difference *is* the phase.
    - whisper  : randomize the phase instead. Same magnitudes again, but the
                 result is breathy and unvoiced.

    Returns (y_out, modified_stft). modified_stft is the spectrum this
    function actually wrote for robot/whisper, and None for pitch/stretch.
    It matters for the visualization: re-analyzing the reconstructed audio
    would show phase that overlap-add put back, not the phase the morph
    applied, so robot would misleadingly appear to still have phase.
    """
    modified_stft = None

    if morph_mode == "pitch":
        y_out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(n_steps))
    elif morph_mode == "stretch":
        y_out = librosa.effects.time_stretch(y=y, rate=max(0.25, float(rate)))
    elif morph_mode in ("robot", "whisper"):
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        if morph_mode == "robot":
            # Zero phase everywhere: magnitudes survive, phase does not.
            modified_stft = magnitude.astype(np.complex64)
        else:
            rng = np.random.default_rng(0)
            random_phase = np.exp(2j * np.pi * rng.random(magnitude.shape))
            modified_stft = (magnitude * random_phase).astype(np.complex64)
        y_out = librosa.istft(modified_stft, hop_length=hop_length, length=len(y))
    else:
        raise ValueError(f"Unknown morph_mode: {morph_mode}")

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = y_out / peak

    return y_out, modified_stft


def apply_equalizer(
    y: np.ndarray,
    sr: int,
    low_level: float = 5.0,
    mid_level: float = 5.0,
    high_level: float = 5.0
) -> np.ndarray:

    low_db = (low_level - 5) * (12 / 5)
    mid_db = (mid_level - 5) * (12 / 5)
    high_db = (high_level - 5) * (12 / 5)

    # Convert dB gain to amplitude multiplier
    low_gain = 10 ** (low_db / 20)
    mid_gain = 10 ** (mid_db / 20)
    high_gain = 10 ** (high_db / 20)

    Y_freq = np.fft.rfft(y)

    freqs = np.fft.rfftfreq(
        len(y),
        d=1 / sr
    )

    gain_curve = np.ones_like(freqs)

    low_end = 250
    low_transition = 500

    mid_transition_start = 2000
    mid_transition_end = 4000

    # LOW region
    low_mask = freqs <= low_end
    gain_curve[low_mask] = low_gain

    # LOW -> MID transition
    low_mid_mask = (
        (freqs > low_end) &
        (freqs < low_transition)
    )

    gain_curve[low_mid_mask] = np.interp(
        freqs[low_mid_mask],
        [low_end, low_transition],
        [low_gain, mid_gain]
    )

    # MID region
    mid_mask = (
        (freqs >= low_transition) &
        (freqs <= mid_transition_start)
    )

    gain_curve[mid_mask] = mid_gain

    # MID -> HIGH transition
    mid_high_mask = (
        (freqs > mid_transition_start) &
        (freqs < mid_transition_end)
    )

    gain_curve[mid_high_mask] = np.interp(
        freqs[mid_high_mask],
        [mid_transition_start, mid_transition_end],
        [mid_gain, high_gain]
    )

    # HIGH region
    high_mask = freqs >= mid_transition_end
    gain_curve[high_mask] = high_gain

    Y_freq *= gain_curve


    y_eq = np.fft.irfft(
        Y_freq,
        n=len(y)
    )
    
    max_value = np.max(np.abs(y_eq))

    if max_value > 0:
        y_eq = y_eq / max_value

    return y_eq


def signal_level_stats(y: np.ndarray) -> dict:
    """Peak, RMS and dBFS for a signal — the same three numbers every level
    meter in the app shows, factored out so the Studio chain can report them
    per stage without duplicating the arithmetic."""
    if y is None or y.size == 0:
        return {"peak": 0.0, "rms": 0.0, "db": float("-inf")}

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    db = 20 * np.log10(max(rms, 1e-6))
    return {"peak": peak, "rms": rms, "db": float(db)}



