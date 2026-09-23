import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_comparison_plot(
    y_input: np.ndarray,
    y_output: np.ndarray,
    sr: int,
    filename: str,
    start_sample: int = None,
    end_sample: int = None,
) -> Path:
    if start_sample is not None and end_sample is not None:
        y_in = y_input[start_sample:end_sample]
        y_out = y_output[start_sample:end_sample]
    else:
        y_in = y_input
        y_out = y_output

    time_input = np.linspace(0, len(y_in) / sr, num=len(y_in))
    time_output = np.linspace(0, len(y_out) / sr, num=len(y_out))

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs[0, 0].plot(time_input, y_in, color='blue', alpha=0.7)
    axs[0, 0].set_title('Time Domain: Original Audio')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)

    axs[0, 1].plot(time_output, y_out, color='orange', alpha=0.7)
    axs[0, 1].set_title('Time Domain: Modified Audio')
    axs[0, 1].grid(True)

    d_in = np.abs(np.fft.rfft(y_in))
    freqs_in = np.fft.rfftfreq(len(y_in), 1 / sr)
    axs[1, 0].plot(freqs_in, d_in, color='blue', alpha=0.7)
    axs[1, 0].set_title('Frequency Domain: Original Audio')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_xlim(0, min(10000, sr / 2))
    axs[1, 0].grid(True)

    d_out = np.abs(np.fft.rfft(y_out))
    freqs_out = np.fft.rfftfreq(len(y_out), 1 / sr)
    axs[1, 1].plot(freqs_out, d_out, color='orange', alpha=0.7)
    axs[1, 1].set_title('Frequency Domain: Modified Audio')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_xlim(0, min(10000, sr / 2))
    axs[1, 1].grid(True)

    plt.tight_layout()
    plots_dir = Path('static/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    image_path = plots_dir / f'plot_{filename}.png'
    plt.savefig(image_path)
    plt.close(fig)
    return image_path


def _downsample_time(signal: np.ndarray, points: int = 180) -> list[float]:
    """Create a compact, smooth display representation without changing the audio."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return []

    count = min(points, signal.size)
    edges = np.linspace(0, signal.size, count + 1, dtype=np.int64)
    values = np.zeros(count, dtype=np.float32)
    for i in range(count):
        chunk = signal[edges[i]:max(edges[i] + 1, edges[i + 1])]
        if not chunk.size:
            continue
        # Keep the strongest excursion's *size* in each chunk, not the mean —
        # averaging a fast-oscillating signal lets positive and negative
        # samples cancel out, which produced a noisy, misleading curve
        # instead of the signal's actual loudness envelope. The *sign* comes
        # from the chunk's overall trend (its mean), not from the single
        # loudest sample: 16-bit PCM can store -0.5 exactly but not +0.5
        # (it lands at ~0.49997), so picking the sign off one sample makes a
        # perfectly symmetric wave render as entirely negative.
        peak_abs = float(np.max(np.abs(chunk)))
        values[i] = -peak_abs if float(np.mean(chunk)) < 0 else peak_abs

    # No smoothing here on purpose. A triangular blur (used previously) mixes
    # each point with its neighbours — harmless for slow-varying loudness,
    # but when a chunk's dominant sign alternates from one point to the next
    # (common once the display's point spacing is close to the audio's own
    # period), averaging +peak next to -peak cancels most of the amplitude
    # right back out, the same cancellation problem this function exists to
    # avoid. Returning the per-chunk peaks directly keeps the true amplitude.
    return values.astype(float).tolist()


def _bin_magnitude(frequencies: np.ndarray, magnitude: np.ndarray, max_frequency: float, bins: int = 56) -> dict:
    """Bucket an arbitrary (frequencies, magnitude) curve into a fixed number
    of display bars, max-pooled so narrow peaks stay visible when squeezed
    into a small chart."""
    valid = frequencies <= max_frequency
    magnitude = magnitude[valid]

    count = min(bins, magnitude.size)
    if count == 0:
        return {"magnitude": [], "maxFrequency": max_frequency, "displayMax": 1}

    edges = np.linspace(0, magnitude.size, count + 1, dtype=np.int64)
    values = np.zeros(count, dtype=np.float32)
    for i in range(count):
        chunk = magnitude[edges[i]:max(edges[i] + 1, edges[i + 1])]
        values[i] = float(np.max(chunk)) if chunk.size else 0.0

    return {
        "magnitude": values.astype(float).tolist(),
        "maxFrequency": max_frequency,
        "displayMax": float(np.max(values)) if values.size else 1.0,
    }


def _spectrum(signal: np.ndarray, sr: int, bins: int = 56) -> dict:
    """Return compact FFT magnitude bars suitable for the frequency-domain UI."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return {"magnitude": [], "maxFrequency": 0, "displayMax": 1}

    # A Hann window reduces leakage so the visual peaks are easier to read.
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window)) / max(1, signal.size)
    frequencies = np.fft.rfftfreq(signal.size, d=1 / sr)

    max_frequency = min(float(sr / 2), 20000.0)
    return _bin_magnitude(frequencies, spectrum, max_frequency, bins)


def build_activity_data(
    energies: np.ndarray,
    frame_times: np.ndarray,
    active: np.ndarray,
    threshold: float,
    peak: float,
    points: int = 180,
) -> dict:
    """Bucket per-frame short-time energy + active/quiet flags into a fixed
    number of display points — same idea as _downsample_time, but carrying
    the boolean activity flag alongside the energy curve so the frontend can
    highlight active vs quiet regions directly."""
    energies = np.asarray(energies, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    frame_times = np.asarray(frame_times, dtype=np.float32)
    n = energies.size

    if n == 0:
        return {"energy": [], "active": [], "time": [], "threshold": 0.0, "peak": 0.0}

    count = min(points, n)
    edges = np.linspace(0, n, count + 1, dtype=np.int64)
    energy_ds = np.zeros(count, dtype=np.float32)
    active_ds = np.zeros(count, dtype=bool)
    time_ds = np.zeros(count, dtype=np.float32)
    for i in range(count):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk_e = energies[lo:hi]
        chunk_a = active[lo:hi]
        energy_ds[i] = float(np.mean(chunk_e)) if chunk_e.size else 0.0
        active_ds[i] = bool(np.any(chunk_a)) if chunk_a.size else False
        time_ds[i] = float(frame_times[min(lo, n - 1)])

    return {
        "energy": energy_ds.tolist(),
        "active": active_ds.tolist(),
        "time": time_ds.tolist(),
        "threshold": float(threshold),
        "peak": float(peak),
    }


def build_pitch_data(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    sr: int,
    peak_freq: float,
    note: dict,
    bins: int = 56,
) -> dict:
    """Bin the full-resolution FFT magnitude spectrum into display bars
    (same shape as an audio spectrum) and record which bar the detected
    peak falls into, so the frontend can highlight that exact bar during
    the 'find peak' animation phase."""
    max_frequency = min(float(sr / 2), 20000.0)
    bars = _bin_magnitude(freqs, spectrum, max_frequency, bins)

    count = len(bars["magnitude"])
    peak_bin_index = 0
    if count and max_frequency > 0:
        peak_bin_index = int(min(count - 1, max(0, round((peak_freq / max_frequency) * (count - 1)))))

    return {
        **bars,
        "peakFrequency": float(peak_freq),
        "peakBinIndex": peak_bin_index,
        "note": note,
    }


def _frame_from_stft(stft: np.ndarray, sr: int, n_fft: int = 2048, bins: int = 48):
    """Magnitude AND phase of one representative STFT frame — the highest
    energy frame, so the numbers describe real content rather than silence.

    Phase is what Module 6 is about, so it is carried alongside the
    magnitude instead of being discarded the way _spectrum() does. Within
    each display bucket we report the phase of the *strongest* bin, since
    the phase of a near-zero bin is numerically meaningless.
    """
    empty = ([0.0] * bins, [0.0] * bins, min(float(sr / 2), 8000.0))
    if stft is None or stft.size == 0 or stft.shape[1] == 0:
        return empty

    frame_index = int(np.argmax(np.abs(stft).sum(axis=0)))
    column = stft[:, frame_index]
    magnitude = np.abs(column)
    phase = np.angle(column)

    frequencies = np.fft.rfftfreq(n_fft, d=1 / sr)
    max_frequency = min(float(sr / 2), 8000.0)
    valid = frequencies <= max_frequency
    magnitude = magnitude[valid]
    phase = phase[valid]

    count = min(bins, magnitude.size)
    if count == 0:
        return empty

    edges = np.linspace(0, magnitude.size, count + 1, dtype=np.int64)
    mag_ds = np.zeros(count, dtype=np.float32)
    phase_ds = np.zeros(count, dtype=np.float32)
    for i in range(count):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk_mag = magnitude[lo:hi]
        if chunk_mag.size == 0:
            continue
        strongest = int(np.argmax(chunk_mag))
        mag_ds[i] = float(chunk_mag[strongest])
        phase_ds[i] = float(phase[lo:hi][strongest])

    return mag_ds.astype(float).tolist(), phase_ds.astype(float).tolist(), max_frequency


def _frame_magnitude_phase(signal: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, bins: int = 48):
    """Analyze a signal, then pull its representative frame."""
    import librosa

    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < n_fft:
        return ([0.0] * bins, [0.0] * bins, min(float(sr / 2), 8000.0))

    return _frame_from_stft(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length), sr, n_fft, bins)


def build_morph_data(
    y_input: np.ndarray,
    y_output: np.ndarray,
    sr: int,
    morph_mode: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    bins: int = 48,
    output_stft: np.ndarray = None,
) -> dict:
    """Magnitude + phase of a representative frame, before and after — the
    side-by-side that shows a robot/whisper morph leaving |X(f)| alone while
    rewriting the phase, and a pitch shift moving the magnitude bins.

    output_stft, when given, is the spectrum the morph actually wrote. It is
    used instead of re-analyzing the reconstructed audio, because overlap-add
    resynthesis puts phase back: a robot morph really did zero every phase,
    but re-analyzing its output would not show that.
    """
    in_mag, in_phase, max_frequency = _frame_magnitude_phase(y_input, sr, n_fft, hop_length, bins)
    if output_stft is not None:
        out_mag, out_phase, _ = _frame_from_stft(output_stft, sr, n_fft, bins)
    else:
        out_mag, out_phase, _ = _frame_magnitude_phase(y_output, sr, n_fft, hop_length, bins)

    display_max = max(max(in_mag, default=0.0), max(out_mag, default=0.0), 1e-9)

    return {
        "mode": morph_mode,
        "maxFrequency": max_frequency,
        "displayMax": float(display_max),
        "magnitude": {"input": in_mag, "output": out_mag},
        "phase": {"input": in_phase, "output": out_phase},
    }


def filter_response_bars(freqs: np.ndarray, magnitude: np.ndarray, sr: int, bins: int = 56) -> dict:
    """Bin a filter's own |H(f)| curve (from compute_filter_frequency_response)
    into the same bar shape as an audio spectrum, so it can share an axis
    with the input/output spectra in the frequency-domain visualization."""
    max_frequency = min(float(sr / 2), 20000.0)
    return _bin_magnitude(np.asarray(freqs), np.asarray(magnitude), max_frequency, bins)


def build_bird_data(y: np.ndarray, sr: int, detection: dict) -> dict:
    """Time waveform + spectrum bars for the recorded clip, reusing the
    same _downsample_time/_spectrum building blocks every other module
    uses, plus the detection result so the frontend has one payload to
    render from."""
    return {
        "waveform": _downsample_time(y),
        "spectrum": _spectrum(y, sr),
        "detection": detection,
    }


def build_visualization_data(
    y_input: np.ndarray,
    y_output: np.ndarray,
    sr: int,
    y_ir: np.ndarray = None,
    filter_response: dict = None,
) -> dict:
    """Data used by React for the animated time/frequency visualizations.

    y_ir, when given (convolution mode), is the impulse response h[n] the
    output was actually convolved with — its spectrum lets the frequency
    view show Y(f) = X(f) . H(f) with real backend-computed bars.

    filter_response, when given (filter mode), is the pre-binned |H(f)| bars
    from filter_response_bars() — the filter's own frequency response, shown
    the same way the impulse response is for convolution.
    """
    input_spectrum = _spectrum(y_input, sr)
    output_spectrum = _spectrum(y_output, sr)
    ir_spectrum = _spectrum(y_ir, sr) if y_ir is not None else None

    # Audio spectra share one scale so input/output heights are comparable.
    #
    # A filter response must NOT join that scale. It is a dimensionless gain
    # peaking at 1.0, while FFT magnitudes of real audio are ~1e-3, so folding
    # it in raised shared_max by ~250x and flattened both audio panels onto
    # the chart's minimum bar height — input and output rendered identically
    # no matter what the filter did.
    shared_max = max(
        input_spectrum.get('displayMax', 0.0),
        output_spectrum.get('displayMax', 0.0),
        ir_spectrum.get('displayMax', 0.0) if ir_spectrum else 0.0,
        1e-12,
    )
    input_spectrum['displayMax'] = shared_max
    output_spectrum['displayMax'] = shared_max
    if ir_spectrum:
        ir_spectrum['displayMax'] = shared_max

    frequency = {
        'input': input_spectrum,
        'output': output_spectrum,
    }
    if ir_spectrum:
        frequency['impulseResponse'] = ir_spectrum
    if filter_response:
        frequency['filterResponse'] = filter_response

    return {
        'time': {
            'input': _downsample_time(y_input),
            'output': _downsample_time(y_output),
        },
        'frequency': frequency,
    }
