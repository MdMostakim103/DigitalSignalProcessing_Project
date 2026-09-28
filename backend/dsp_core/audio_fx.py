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


def extract_loudest_window(
    y: np.ndarray,
    sr: int,
    window_seconds: float = 2.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Slide a window_seconds-long window across the clip and keep only the
    single stretch with the most short-time energy, discarding the rest.
    Used by the Bird Sound Detector to auto-trim both reference recordings
    and live mic clips down to (most likely) just the call, so every clip
    feeding feature extraction is measured the same length and dominated by
    signal instead of whatever silence/handling-noise happened to surround
    the call.
    """
    window_samples = int(sr * window_seconds)
    if y.size <= window_samples:
        return y

    energies, _, _, hop_len = compute_short_time_energy(y, sr, frame_ms, hop_ms)
    window_frames = max(1, int(round(window_seconds * 1000 / hop_ms)))
    if energies.size <= window_frames:
        return y

    # Sliding sum of per-frame energy over window_frames-wide spans, via a
    # cumulative-sum difference — the discrete equivalent of sliding the
    # window continuously across overlapping frames.
    cumulative = np.concatenate(([0.0], np.cumsum(energies)))
    window_sums = cumulative[window_frames:] - cumulative[:-window_frames]
    best_start_frame = int(np.argmax(window_sums))

    start_sample = best_start_frame * hop_len
    end_sample = min(start_sample + window_samples, y.size)
    start_sample = max(0, end_sample - window_samples)
    return y[start_sample:end_sample]


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

# Named sci-fi "robot voice" presets for morph_mode == "robot". Every preset
# is a chain of DSP primitives already used elsewhere in this module (pitch
# shift, filtering, EQ, reverb) plus two new ones below (ring modulation,
# soft-clip distortion) — not audio cloning and not built from any actual
# recorded reference voice, just the same category of technique classic
# sci-fi sound design uses to build a "robot"/"radio"/"masked" character.
ROBOT_PRESETS = ("classic", "vader", "droid", "trooper", "cylon")


def apply_ring_modulation(y: np.ndarray, sr: int, carrier_freq: float = 30.0, mix: float = 1.0) -> np.ndarray:
    """Multiply the signal by a carrier sine wave. Amplitude-modulating a
    voice like this creates sum/difference sidebands around every frequency
    component the voice already has — new frequencies that weren't there
    before — which is what gives a ring-modulated voice its metallic,
    buzzing character (unlike pitch-shifting, which only moves existing
    frequencies around, or filtering, which only removes them).

    mix blends dry/wet (0 = untouched, 1 = fully ring-modulated) so the
    effect can be layered in underneath other processing instead of always
    completely replacing the signal.
    """
    t = np.arange(len(y)) / sr
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    ringed = y * carrier
    mix = min(1.0, max(0.0, mix))
    return y * (1 - mix) + ringed * mix


def apply_soft_clip(y: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """tanh soft-clip distortion. Pushes the loudest parts of the waveform
    toward a smooth ceiling instead of the harsh flat-top a hard clip
    produces, adding the odd-harmonic "grit" a voice-through-a-small-speaker
    effect needs without collapsing into pure noise. Dividing by tanh(drive)
    keeps quiet passages close to unity gain regardless of how hard the
    loud parts are being driven into the curve.
    """
    drive = max(1.0, float(drive))
    return np.tanh(y * drive) / np.tanh(drive)


def apply_robot_voice(
    y: np.ndarray,
    sr: int,
    preset: str = "classic",
    n_fft: int = 2048,
    hop_length: int = 512,
):
    """Five "robot voice" presets, all built from the same foundation as the
    original robot morph — every bin's phase zeroed, magnitude kept — with a
    different chain of coloring effects layered on top of that base for
    each named character:

    - classic : the zero-phase base alone. Flat, textureless monotone buzz
                (this is exactly what morph_mode="robot" always did).
    - vader   : pitched down ~5 semitones, lowpassed so it sounds muffled
                behind a mask, soft-clipped for grit, then a touch of
                reverb for a helmet-chamber resonance.
    - droid   : pitched up slightly with a bright EQ tilt and a faint
                high-frequency ring modulation, for a prim, fussy, faintly
                metallic protocol-droid tone.
    - trooper : squeezed through a narrow "radio" bandpass, buzzed with
                ring modulation, and dusted with broadband noise for
                helmet-comms static.
    - cylon   : the zero-phase base with a heavier ring modulation layered
                on top — a harsher, more overtly mechanical monotone than
                "classic" alone.

    Returns (y_out, modified_stft). modified_stft is only populated for
    "classic" — the one preset whose output really is just that zero-phase
    spectrum, unmodified afterward. Every other preset keeps processing y_out
    in the time domain after that point (pitch shift, filtering, ...), so
    there is no single "the" spectrum left to show for those — the
    visualizer falls back to re-analyzing y_out directly instead.
    """
    if preset not in ROBOT_PRESETS:
        raise ValueError(f"Unknown robot preset: {preset}")

    # Stage 1 — always: the shared zero-phase "robot" foundation.
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    zero_phase_stft = np.abs(stft).astype(np.complex64)
    y_out = librosa.istft(zero_phase_stft, hop_length=hop_length, length=len(y))

    # Stage 2 — preset-specific character layered on top of that base.
    if preset == "classic":
        pass
    elif preset == "vader":
        y_out = librosa.effects.pitch_shift(y=y_out, sr=sr, n_steps=-5.0)
        y_out = apply_filter(y_out, sr, filter_family="butterworth", band_type="lowpass", cutoff=3000.0, order=4)
        y_out = apply_soft_clip(y_out, drive=2.5)
        y_out = apply_reverb(y_out, sr)
    elif preset == "droid":
        y_out = librosa.effects.pitch_shift(y=y_out, sr=sr, n_steps=2.5)
        y_out = apply_equalizer(y_out, sr, low_level=4.0, mid_level=6.0, high_level=8.0)
        y_out = apply_ring_modulation(y_out, sr, carrier_freq=90.0, mix=0.15)
    elif preset == "trooper":
        y_out = apply_filter(y_out, sr, filter_family="butterworth", band_type="bandpass", cutoff=300.0, cutoff2=3000.0, order=4)
        y_out = apply_ring_modulation(y_out, sr, carrier_freq=40.0, mix=0.28)
        # Static sized relative to the signal's own RMS (~24 dB SNR), not a
        # fixed absolute level — a flat 0.02 std swamped this preset's quiet
        # passages once bandpass+ring-mod had already knocked its RMS down
        # to ~0.07, giving an ~11 dB SNR that read as pure noise over voice.
        rms = float(np.sqrt(np.mean(y_out ** 2))) if y_out.size else 0.0
        noise_std = rms * 0.06
        rng = np.random.default_rng(1)
        y_out = y_out + rng.normal(0.0, noise_std, size=len(y_out))
    elif preset == "cylon":
        y_out = apply_ring_modulation(y_out, sr, carrier_freq=45.0, mix=0.5)

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = y_out / peak

    modified_stft = zero_phase_stft if preset == "classic" else None
    return y_out, modified_stft


def apply_voice_morph(
    y: np.ndarray,
    sr: int,
    morph_mode: str = "pitch",
    n_steps: float = 4.0,
    rate: float = 1.5,
    n_fft: int = 2048,
    hop_length: int = 512,
    robot_preset: str = "classic",
) -> np.ndarray:
    """Four phase-vocoder experiments that separate what the magnitude
    spectrum carries from what the phase carries:

    - pitch    : shift pitch, keep duration. Time-stretch with the phase
                 vocoder (which advances phase correctly per frame), then
                 resample back — so only the pitch moves.
    - stretch  : change duration, keep pitch. The same phase vocoder, but
                 without the resampling step.
    - robot    : throw the phase away (set every bin's phase to zero), then
                 layer on a named character preset (see apply_robot_voice) —
                 "classic" reproduces the original flat monotone buzz
                 exactly; the other presets add pitch/filter/ring-mod/
                 distortion/reverb coloring on top of that same base.
    - whisper  : randomize the phase instead. Same magnitudes again, but the
                 result is breathy and unvoiced.

    Returns (y_out, modified_stft). modified_stft is the spectrum this
    function actually wrote for robot/whisper, and None for pitch/stretch
    (and for every robot preset except "classic" — see apply_robot_voice).
    It matters for the visualization: re-analyzing the reconstructed audio
    would show phase that overlap-add put back, not the phase the morph
    applied, so robot would misleadingly appear to still have phase.
    """
    modified_stft = None

    if morph_mode == "pitch":
        y_out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(n_steps))
    elif morph_mode == "stretch":
        y_out = librosa.effects.time_stretch(y=y, rate=max(0.25, float(rate)))
    elif morph_mode == "robot":
        y_out, modified_stft = apply_robot_voice(y, sr, preset=robot_preset, n_fft=n_fft, hop_length=hop_length)
    elif morph_mode == "whisper":
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
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


def choose_spectrogram_hop_length(n_samples: int, sr: int, n_fft: int = 2048, max_frames: int = 400) -> int:
    """Pick the hop length a spectrogram preview and a mask application both
    use, from nothing but (n_samples, sr) — so the two requests land on the
    identical time-frame grid without either one having to send the other
    its hop_length. Starts at 512 (matches every other STFT effect in this
    module) and doubles until the frame count is display-sized; never goes
    below 512 so short clips keep fine time resolution."""
    hop = 512
    while (n_samples // hop) + 1 > max_frames:
        hop *= 2
    return hop


def apply_spectral_mask(
    y: np.ndarray,
    sr: int,
    regions: list,
    n_fft: int = 2048,
    hop_length: int = None,
    feather_freq_bins: float = 3.0,
    feather_time_frames: float = 2.0,
):
    """'Paint on the spectrogram' — X'(k,m) = M(k,m)X(k,m).

    Each region is {freqMin, freqMax, timeMin, timeMax, action}, action
    being "keep" or "erase", in real Hz/seconds so it's independent of
    whatever bin grid the preview happened to show. If any region is a
    "keep", everything outside every "keep" region starts silent (an
    isolating mask); otherwise everything starts audible and only "erase"
    regions are cut (a subtractive mask). Later regions in the list win
    where they overlap earlier ones, matching paint order.

    The mask is built as flat 0/1 rectangles, then Gaussian-blurred across
    both axes before multiplying. A hard rectangular mask creates sharp
    discontinuities in the STFT that the inverse transform hears as
    ringing/musical noise; blurring first feathers every edge by a few
    bins/frames so the cut is smooth instead of a wall.
    """
    hop_length = hop_length or choose_spectrogram_hop_length(len(y), sr, n_fft)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

    has_keep = any(r.get("action") == "keep" for r in regions)
    mask = np.full(stft.shape, 0.0 if has_keep else 1.0, dtype=np.float32)

    for region in regions:
        f_lo, f_hi = sorted((float(region["freqMin"]), float(region["freqMax"])))
        t_lo, t_hi = sorted((float(region["timeMin"]), float(region["timeMax"])))
        f_sel = (freqs >= f_lo) & (freqs <= f_hi)
        t_sel = (frame_times >= t_lo) & (frame_times <= t_hi)
        mask[np.ix_(f_sel, t_sel)] = 1.0 if region.get("action") == "keep" else 0.0

    from scipy.ndimage import gaussian_filter
    mask = np.clip(gaussian_filter(mask, sigma=(feather_freq_bins, feather_time_frames)), 0.0, 1.0)

    y_out = librosa.istft(stft * mask, hop_length=hop_length, length=len(y))

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = y_out / peak

    return y_out, hop_length


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


def signal_stats(y: np.ndarray) -> dict:
    """Peak, RMS and clipping percentage for a signal — the quick before/
    after health check a gain change needs. clip_percent is the percentage
    of samples whose absolute value exceeds 1.0 (full scale for a
    normalized float signal), i.e. samples that would clip on playback or
    when written out as fixed-point PCM."""
    if y is None or y.size == 0:
        return {"peak": 0.0, "rms": 0.0, "clip_percent": 0.0}

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    clip_percent = float(np.mean(np.abs(y) > 1.0) * 100)
    return {"peak": peak, "rms": rms, "clip_percent": clip_percent}


def quantize_bitdepth(y: np.ndarray, bits: int) -> np.ndarray:
    """Uniform requantization to `bits`-bit resolution: round every sample
    to the nearest of 2**bits evenly spaced levels across [-1, 1] — the
    same reduced-resolution effect a low-bit-depth ADC/DAC has on a signal,
    audible as added quantization noise/"crunch" as bits drops."""
    bits = max(1, min(16, int(bits)))
    levels = 2 ** bits
    step = 2.0 / (levels - 1) if levels > 1 else 2.0

    y_clipped = np.clip(y, -1.0, 1.0)
    y_quantized = np.round(y_clipped / step) * step
    return np.clip(y_quantized, -1.0, 1.0)


def downsample_hold(y: np.ndarray, factor: int) -> np.ndarray:
    """Sample-and-hold at a lower effective rate: keep every `factor`-th
    sample and repeat it to fill the gap, instead of interpolating. This is
    exactly what a zero-order-hold DAC does when fed a lower sample rate,
    and it's what makes aliasing audible — the output is the same length
    as the input (each held sample just repeats), so it stays directly
    comparable to the original sample-for-sample."""
    factor = max(1, int(factor))
    if factor == 1 or y.size == 0:
        return y.copy()

    n = y.size
    hold_index = (np.arange(n) // factor) * factor
    hold_index = np.minimum(hold_index, n - 1)
    return y[hold_index]