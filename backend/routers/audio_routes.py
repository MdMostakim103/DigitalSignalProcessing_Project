from typing import Optional, List
import io
import json
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

from dsp_core.audio_fx import (
    amplify_volume, apply_reverb, apply_echo, apply_delay,
    apply_convolution, apply_noise_reduction, apply_equalizer,
    apply_filter, compute_filter_frequency_response,
    apply_activity_gate,
    detect_dominant_frequency, freq_to_note, synthesize_tone,
    apply_voice_morph, signal_level_stats,
)
from dsp_core.visualizer import (
    generate_comparison_plot, build_visualization_data, filter_response_bars,
    build_activity_data, build_pitch_data, build_morph_data, build_bird_data,
    build_stage_data,
)
from dsp_core.bird_detector import classify_bird_sound

router = APIRouter()

STATIC_DIR = Path("static")

# Operations the Studio chain builder is allowed to compose. Pitch Detection
# is deliberately excluded: it doesn't transform the incoming signal, it
# analyzes it for a dominant frequency and replaces it outright with a
# synthesized tone, so it doesn't compose with the other steps the way an
# effect does. Bird detection is a bonus-module classifier, not an effect.
CHAINABLE_OPERATIONS = (
    "amplify", "filter", "convolution", "echo", "delay",
    "reverb", "equalizer", "noise", "activity", "morph",
)

MAX_CHAIN_STEPS = 25

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
    activity_method: str = Form("energy"),
    zcr_threshold_hz: float = Form(700.0),
    pitch_fmin: float = Form(50.0),
    pitch_fmax: float = Form(2000.0),
    morph_mode: str = Form("pitch"),
    n_steps: float = Form(4.0),
    rate: float = Form(1.5),
    delay_ms: float = Form(280.0),
    decay: float = Form(0.55),
    repeats: int = Form(5),
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
        y_modified = apply_echo(y, sr, delay_seconds=delay_ms / 1000.0, decay=decay, repeats=repeats)
    elif effect == "delay":
        y_modified = apply_delay(y, sr, delay_seconds=delay_ms / 1000.0, wet=decay)
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
        # zcr_threshold_hz is converted to a ratio INSIDE apply_activity_gate,
        # against the REAL sr librosa just loaded — not client-side, where the
        # browser's AudioContext may have silently resampled the decoded
        # buffer to a different rate than the file's actual native one.
        y_modified, energies, zcr, frame_times, active, enter_thr, exit_thr, noise_floor, peak, zcr_thr = apply_activity_gate(
            y, sr,
            method=activity_method,
            energy_threshold_ratio=threshold_ratio,
            zcr_threshold_hz=zcr_threshold_hz,
        )
        activity_data = build_activity_data(
            energies, frame_times, active, enter_thr, peak,
            zcr=zcr, zcr_threshold=zcr_thr,
            exit_threshold=exit_thr, noise_floor=noise_floor,
        )
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


def _run_chain_step(op_type: str, y: np.ndarray, sr: int, params: dict, ir_wave: Optional[np.ndarray]):
    """Dispatch one chain step to the same dsp_core function the standalone
    module page for that effect uses. Returns (y_out, extra) where extra
    carries anything beyond the plain waveform (a filter's |H(f)| curve, an
    impulse response's own waveform/spectrum) for that step's accordion."""
    extra = {}

    if op_type == "amplify":
        y_out = amplify_volume(y, float(params.get("value", 4.0)))

    elif op_type == "filter":
        filter_family = params.get("filter_family", "butterworth")
        band_type = params.get("band_type", "lowpass")
        cutoff = float(params.get("cutoff", 1000.0))
        cutoff2 = float(params.get("cutoff2", 4000.0))
        order = int(params.get("order", 4))
        y_out = apply_filter(y, sr, filter_family=filter_family, band_type=band_type, cutoff=cutoff, cutoff2=cutoff2, order=order)
        resp_freqs, resp_mag = compute_filter_frequency_response(filter_family, band_type, sr, cutoff, cutoff2, order)
        extra["filterResponse"] = filter_response_bars(resp_freqs, resp_mag, sr)

    elif op_type == "convolution":
        if ir_wave is None:
            raise HTTPException(status_code=400, detail="Convolution step is missing its impulse response file.")
        y_out = apply_convolution(y, ir_wave)
        extra["impulseResponse"] = build_stage_data(ir_wave, sr)

    elif op_type == "echo":
        y_out = apply_echo(
            y, sr,
            delay_seconds=float(params.get("delay_ms", 280.0)) / 1000.0,
            decay=float(params.get("decay", 0.55)),
            repeats=int(params.get("repeats", 5)),
        )

    elif op_type == "delay":
        y_out = apply_delay(
            y, sr,
            delay_seconds=float(params.get("delay_ms", 350.0)) / 1000.0,
            wet=float(params.get("wet", 0.85)),
        )

    elif op_type == "reverb":
        y_out = apply_reverb(y, sr)

    elif op_type == "equalizer":
        y_out = apply_equalizer(
            y, sr,
            float(params.get("low", 5.0)),
            float(params.get("mid", 5.0)),
            float(params.get("high", 5.0)),
        )

    elif op_type == "noise":
        y_out = apply_noise_reduction(y, sr)

    elif op_type == "activity":
        y_out, energies, zcr, frame_times, active, enter_thr, exit_thr, noise_floor, peak, zcr_thr = apply_activity_gate(
            y, sr,
            method=params.get("activity_method", "energy"),
            energy_threshold_ratio=float(params.get("threshold_ratio", 0.15)),
            zcr_threshold_hz=float(params.get("zcr_threshold_hz", 700.0)),
        )
        extra["activity"] = build_activity_data(
            energies, frame_times, active, enter_thr, peak,
            zcr=zcr, zcr_threshold=zcr_thr,
            exit_threshold=exit_thr, noise_floor=noise_floor,
        )

    elif op_type == "morph":
        y_out, _ = apply_voice_morph(
            y, sr,
            morph_mode=params.get("morph_mode", "pitch"),
            n_steps=float(params.get("n_steps", 4.0)),
            rate=float(params.get("rate", 1.5)),
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown or non-chainable operation: {op_type}")

    return y_out, extra


@router.post("/process-chain")
async def process_chain(
    file: UploadFile = File(...),
    chain: str = Form(...),
    ir_files: List[UploadFile] = File(default=[]),
):
    """Apply an ordered list of effects to one uploaded file, each step's
    output feeding the next — the Studio's multi-operation chain. `chain` is
    a JSON string: [{"type": "filter", "params": {...}}, ...]. Convolution
    steps consume impulse-response files from `ir_files` in the order those
    steps appear in the chain (a chain with two convolution steps needs two
    files there, in order).
    """
    try:
        steps = json.loads(chain)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`chain` must be valid JSON.")

    if not isinstance(steps, list) or not steps:
        raise HTTPException(status_code=400, detail="`chain` must be a non-empty list of steps.")
    if len(steps) > MAX_CHAIN_STEPS:
        raise HTTPException(status_code=400, detail=f"A chain can have at most {MAX_CHAIN_STEPS} steps.")

    for step in steps:
        if not isinstance(step, dict) or step.get("type") not in CHAINABLE_OPERATIONS:
            raise HTTPException(status_code=400, detail=f"Invalid or non-chainable step: {step}")

    input_path = Path("static/uploads") / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    y, sr = librosa.load(input_path, sr=None)

    ir_cursor = 0
    y_current = y
    stage_results = []

    for index, step in enumerate(steps):
        op_type = step["type"]
        params = step.get("params", {})

        ir_wave = None
        if op_type == "convolution":
            if ir_cursor >= len(ir_files):
                raise HTTPException(
                    status_code=400,
                    detail=f"Step {index + 1} is a convolution step but no matching impulse response file was uploaded.",
                )
            ir_upload = ir_files[ir_cursor]
            ir_cursor += 1
            ir_path = Path("static/uploads") / f"chain_ir_{index}_{ir_upload.filename}"
            with open(ir_path, "wb") as f:
                f.write(await ir_upload.read())
            ir_wave, _ = librosa.load(ir_path, sr=sr)

        start = time.perf_counter()
        y_current, extra = _run_chain_step(op_type, y_current, sr, params, ir_wave)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        stage_results.append({
            "index": index,
            "type": op_type,
            "params": params,
            "elapsed_ms": round(elapsed_ms, 2),
            **build_stage_data(y_current, sr),
            **extra,
        })

    output_filename = f"chain_{'-'.join(s['type'] for s in steps)}_{file.filename}"
    output_path = Path("static/processed") / output_filename
    sf.write(output_path, y_current, sr)

    return {
        "status": f"Chain of {len(steps)} operation(s) processed successfully!",
        "filename": file.filename,
        "input": build_stage_data(y, sr),
        "stages": stage_results,
        "output": {
            "audio_url": f"http://127.0.0.1:8000/static/processed/{output_filename}",
            "stats": signal_level_stats(y_current),
            **build_stage_data(y_current, sr),
        },
    }


@router.post("/detect-bird")
async def detect_bird(file: UploadFile = File(...)):
    """Bird Sound Detector — Record 3s -> Identify.

    Deliberately does NOT write the microphone clip to static/uploads (or
    anywhere else on disk): the bytes are decoded straight out of memory
    with librosa, classified, and then discarded when this request ends.
    This is the only route in the app that behaves this way; every other
    /process-audio effect still persists its input/output because those
    are uploaded files the user is choosing to process, not live mic audio.

    Trimming: this route passes the full decoded clip to classify_bird_sound
    unmodified — the auto-trim to the loudest ~2s window (audio_fx.
    extract_loudest_window) happens inside bird_features.extract_features,
    which both this live path and scripts/generate_bird_references.py call.
    Trimming there instead of here guarantees the live clip and every
    reference recording are trimmed by the identical function with the
    identical window, and it can't quietly drift out of sync between the
    two call sites the way two separate copies of the same logic could.
    """
    raw_bytes = await file.read()

    try:
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the recorded audio clip.")

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Recorded clip was empty.")

    detection = classify_bird_sound(y, sr)
    visualization = build_bird_data(y, sr, detection)

    return {
        "status": "Bird sound analyzed.",
        "detection": detection,
        "visualization": visualization,
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
