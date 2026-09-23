from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import librosa
import soundfile as sf

from dsp_core.audio_fx import (
    amplify_volume, apply_reverb, apply_echo, apply_delay,
    apply_convolution, apply_noise_reduction, apply_equalizer,
    apply_filter, compute_filter_frequency_response,
    apply_activity_gate,
    detect_dominant_frequency, freq_to_note, synthesize_tone,
    apply_voice_morph,
)
from dsp_core.visualizer import (
    generate_comparison_plot, build_visualization_data, filter_response_bars,
    build_activity_data, build_pitch_data, build_morph_data,
)

router = APIRouter()

STATIC_DIR = Path("static")

@router.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    effect: str = Form(...),
    value: float = Form(2.0),
    eq_low: float = Form(5.0),
    eq_mid: float = Form(5.0),
    eq_high: float = Form(5.0),
    ir_file: Optional[UploadFile] = File(None),
    filter_family: str = Form("butterworth"),
    band_type: str = Form("lowpass"),
    cutoff: float = Form(1000.0),
    cutoff2: float = Form(4000.0),
    order: int = Form(4),
    threshold_ratio: float = Form(0.15),
    pitch_fmin: float = Form(50.0),
    pitch_fmax: float = Form(2000.0),
    morph_mode: str = Form("pitch"),
    n_steps: float = Form(4.0),
    rate: float = Form(1.5),
):

    # 1. Save to uploads/
    input_path = Path("static/uploads") / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    y, sr = librosa.load(input_path, sr=None)

    # 2. Route the math based on the frontend selection
    y_ir = None
    filter_response_data = None
    activity_data = None
    pitch_data = None
    morph_data = None
    if effect == "convolution":
        if ir_file is None:
            raise HTTPException(
                status_code=400,
                detail="Convolution needs an impulse response file (ir_file).",
            )
        ir_path = Path("static/uploads") / f"ir_{ir_file.filename}"
        with open(ir_path, "wb") as f:
            f.write(await ir_file.read())
        # Resample the IR onto the input's sample rate so the two line up
        # sample-for-sample before they're convolved.
        y_ir, _ = librosa.load(ir_path, sr=sr)
        y_modified = apply_convolution(y, y_ir)
    elif effect == "echo":
        y_modified = apply_echo(y, sr)
    elif effect == "delay":
        y_modified = apply_delay(y, sr)
    elif effect == "reverb":
        y_modified = apply_reverb(y, sr)
    elif effect == "amplify":
        y_modified = amplify_volume(y, value)
    elif effect == "noise":
        y_modified = apply_noise_reduction(y, sr)
    elif effect == "equalizer":
        y_modified = apply_equalizer(y, sr, eq_low, eq_mid, eq_high)
    elif effect == "filter":
        y_modified = apply_filter(y, sr, filter_family=filter_family, band_type=band_type, cutoff=cutoff, cutoff2=cutoff2, order=order)
        resp_freqs, resp_mag = compute_filter_frequency_response(filter_family, band_type, sr, cutoff, cutoff2, order)
        filter_response_data = filter_response_bars(resp_freqs, resp_mag, sr)
    elif effect == "activity":
        y_modified, energies, frame_times, active, threshold, peak = apply_activity_gate(y, sr, threshold_ratio=threshold_ratio)
        activity_data = build_activity_data(energies, frame_times, active, threshold, peak)
    elif effect == "pitch":
        peak_freq, spectrum, freqs = detect_dominant_frequency(y, sr, fmin=pitch_fmin, fmax=pitch_fmax)
        note = freq_to_note(peak_freq)
        y_modified = synthesize_tone(peak_freq, len(y) / sr, sr)
        pitch_data = build_pitch_data(freqs, spectrum, sr, peak_freq, note)
    elif effect == "morph":
        y_modified, morph_stft = apply_voice_morph(y, sr, morph_mode=morph_mode, n_steps=n_steps, rate=rate)
        morph_data = build_morph_data(y, y_modified, sr, morph_mode, output_stft=morph_stft)
    else:
        y_modified = amplify_volume(y, 1)

    # 3. Add the effect name to the plot filename to bust the browser cache!
    plot_filename = f"{effect}_{file.filename}"
    generate_comparison_plot(y, y_modified, sr, plot_filename)

    # 4. Save to processed/
    output_filename = f"modified_{effect}_{file.filename}"
    output_path = Path("static/processed") / output_filename
    sf.write(output_path, y_modified, sr)

    input_duration = librosa.get_duration(y=y, sr=sr)
    processed_duration = librosa.get_duration(y=y_modified, sr=sr)
    visualization = build_visualization_data(y, y_modified, sr, y_ir=y_ir, filter_response=filter_response_data)

    # 5. Return the newly updated URLs sent to React
    return {
        "filename": file.filename,
        "input_duration_seconds": round(input_duration, 2),
        "processed_duration_seconds": round(processed_duration, 2),
        "status": f"Audio processed with '{effect}' and graphed successfully!",
        "plot_url": f"http://127.0.0.1:8000/static/plots/plot_{plot_filename}.png",
        "audio_url": f"http://127.0.0.1:8000/static/processed/{output_filename}",
        "visualization": visualization,
        "activity": activity_data,
        "pitch": pitch_data,
        "morph": morph_data,
    }


@router.get("/filter-response")
def filter_response(
    filter_family: str = "butterworth",
    band_type: str = "lowpass",
    cutoff: float = 1000.0,
    cutoff2: float = 4000.0,
    order: int = 4,
    sr: int = 44100,
):
    """Just the filter's own |H(f)| curve, with no audio involved — lets the
    frontend show a live 'what does this filter look like' graph the moment
    the user changes a slider, before they ever run anything."""
    freqs, magnitude = compute_filter_frequency_response(filter_family, band_type, sr, cutoff, cutoff2, order, n_points=300)
    return {
        "frequencies": freqs.tolist(),
        "magnitude": magnitude.tolist(),
        "maxFrequency": min(sr / 2.0, 20000.0),
    }
