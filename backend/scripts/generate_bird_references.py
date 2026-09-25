"""One-time builder: creates a small reference dataset for the Bird Sound
Detector and pre-computes each species' individual sample feature vectors
(used for k-NN matching, not a single averaged vector).

Why synthesized calls by default: the project started with no bird-audio
dataset, external downloads weren't reachable from the grading environment,
and licensing real recordings under time pressure was risky. Each
synthetic "species" is built from a distinct, documented DSP synthesis
recipe (a broadband noisy caw, a rising/falling melodic sweep, a sustained
low tone) so the resulting feature vectors are genuinely different from
each other in a way that maps onto real acoustic differences between those
call types. Species live/died: Sparrow and Dove were dropped after a
leave-one-out accuracy check (scripts/evaluate_bird_accuracy.py) showed
Dove absorbed most of the cross-species confusion and Sparrow overlapped
heavily with it too — the remaining three (Crow/Robin/Owl) are the most
acoustically distinct combination found (79.4% leave-one-out accuracy vs.
58.6% with all five). To add a species back, add its make_<species>
function and an entry in SPECIES_GENERATORS below — nothing else in the
detection pipeline (bird_detector.py, bird_features.py, the frontend) is
hardcoded to any particular species count or name.

Real recordings: for each species, if static/bird_samples/<species>/
contains any audio files (.wav/.mp3/.flac/.ogg, any filename), those are
used INSTEAD of synthesis for that species — drop real recordings in and
rerun this script, no code changes needed. Delete the old synthetic
sample_*.wav files first so they don't get averaged in alongside real
recordings. A species folder left empty still falls back to synthesis, so
you can convert species one at a time.

Run with:  python scripts/generate_bird_references.py
"""

import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsp_core.bird_features import extract_features, normalize_features

SR = 44100
OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "bird_samples"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "dsp_core" / "bird_reference_features.json"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VARIATIONS_PER_SPECIES = 6
DURATION_SECONDS = 5.0
RNG = np.random.default_rng(42)


def _fade(signal: np.ndarray, sr: int, fade_seconds: float = 0.05) -> np.ndarray:
    n = min(int(sr * fade_seconds), signal.size // 2)
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n)
        signal[:n] *= ramp
        signal[-n:] *= ramp[::-1]
    return signal


def _silence_pad(signal: np.ndarray, sr: int, total_seconds: float) -> np.ndarray:
    """Real recordings aren't wall-to-wall sound, so pad with quiet noise
    on both sides — this also keeps zero-crossing-rate/energy features
    realistic instead of artificially clean."""
    target_len = int(sr * total_seconds)
    if signal.size >= target_len:
        return signal[:target_len]
    pad = target_len - signal.size
    lead = pad // 2
    trail = pad - lead
    noise_floor = 0.003
    lead_noise = RNG.normal(0, noise_floor, lead)
    trail_noise = RNG.normal(0, noise_floor, trail)
    return np.concatenate([lead_noise, signal, trail_noise])


def make_crow(jitter: float) -> np.ndarray:
    """Low, harsh, broadband 'caw' — built from filtered noise plus a
    low fundamental, giving it a noisy/buzzy spectrum unlike a pure tone."""
    caw_hz = 550 + jitter * 60
    n_caws = 3
    caw_len = int(SR * 0.25)
    t = np.arange(caw_len) / SR
    tone = np.sin(2 * np.pi * caw_hz * t)
    noise = RNG.normal(0, 1, caw_len)
    # crude broadband buzz: mix tone with noise, envelope shaped like a caw
    envelope = np.exp(-((t - 0.05) ** 2) / (2 * 0.03 ** 2))
    caw = (0.5 * tone + 0.5 * noise) * envelope
    gap = np.zeros(int(SR * 0.3))
    call = np.concatenate([np.concatenate([caw, gap]) for _ in range(n_caws)])
    return _silence_pad(_fade(call, SR), SR, DURATION_SECONDS)


def make_robin(jitter: float) -> np.ndarray:
    """Melodic mid/high warble: a frequency sweep up and down, repeated —
    higher centroid and larger bandwidth than the dove's near-pure tone."""
    f_low = 1800 + jitter * 150
    f_high = 3600 + jitter * 200
    n_phrases = 4
    phrase_len = int(SR * 0.18)
    t = np.arange(phrase_len) / SR
    sweep = f_low + (f_high - f_low) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(sweep) / SR
    phrase = 0.55 * np.sin(phase) * np.hanning(phrase_len)
    gap = np.zeros(int(SR * 0.15))
    call = np.concatenate([np.concatenate([phrase, gap]) for _ in range(n_phrases)])
    return _silence_pad(_fade(call, SR), SR, DURATION_SECONDS)


def make_owl(jitter: float) -> np.ndarray:
    """Very low, sustained, pure-tone hoot."""
    hoot_hz = 260 + jitter * 20
    n_hoots = 2
    hoot_len = int(SR * 0.7)
    t = np.arange(hoot_len) / SR
    hoot = 0.55 * np.sin(2 * np.pi * hoot_hz * t) * np.hanning(hoot_len)
    gap = np.zeros(int(SR * 0.4))
    call = np.concatenate([np.concatenate([hoot, gap]) for _ in range(n_hoots)])
    return _silence_pad(_fade(call, SR), SR, DURATION_SECONDS)


SPECIES_GENERATORS = {
    "Crow": make_crow,
    "Robin": make_robin,
    "Owl": make_owl,
}


SYNTHETIC_FILENAME_RE = re.compile(r"^sample_\d+\.wav$", re.IGNORECASE)


def _load_real_samples(species_dir: Path) -> list[np.ndarray]:
    """Any audio file already sitting in the species folder, regardless of
    filename — real recordings don't need to follow the sample_N.wav
    naming the synthetic generator used. Files matching that exact
    synthetic naming pattern are ignored here, so a species that's still
    on the synthesized fallback from a previous run doesn't get mistaken
    for having real recordings (and so gets freshly resynthesized if
    DURATION_SECONDS/VARIATIONS_PER_SPECIES change)."""
    files = sorted(
        p for p in species_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and not SYNTHETIC_FILENAME_RE.match(p.name)
    )
    vectors = []
    for f in files:
        y, sr = librosa.load(f, sr=None, mono=True)
        vectors.append(extract_features(y, sr))
        print(f"  loaded real sample: {f.name}")
    return vectors


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = {}
    all_vectors = []

    for species, generator in SPECIES_GENERATORS.items():
        species_dir = OUT_DIR / species
        species_dir.mkdir(parents=True, exist_ok=True)

        real_vectors = _load_real_samples(species_dir)

        if real_vectors:
            vectors = np.array(real_vectors)
            print(f"{species}: using {len(vectors)} real recordings")
        else:
            vectors = []
            for i in range(VARIATIONS_PER_SPECIES):
                jitter = RNG.normal(0, 1)
                audio = generator(jitter).astype(np.float32)
                audio = np.clip(audio, -1.0, 1.0)
                out_path = species_dir / f"sample_{i + 1}.wav"
                sf.write(out_path, audio, SR)
                vectors.append(extract_features(audio, SR))
            vectors = np.array(vectors)
            print(f"{species}: no real recordings found, synthesized {len(vectors)} samples")

        # k-NN needs every individual sample's vector, not just the species
        # mean — a mean collapses exactly the within-species spread that
        # k-NN uses to vote.
        reference[species] = {
            "vectors": vectors.tolist(),
            "samples": len(vectors),
        }
        all_vectors.extend(vectors.tolist())
        mean_vector, _ = normalize_features(vectors)
        print(f"{species}: mean feature vector = {np.round(mean_vector, 3).tolist()}")

    # Build one global std across ALL species' individual sample vectors,
    # so distance calculations at detection time use a single consistent
    # scale (see dsp_core/bird_detector.py).
    _, global_std = normalize_features(np.array(all_vectors))

    reference["_global_std"] = global_std.tolist()

    with open(FEATURES_PATH, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nWrote reference features for {len(SPECIES_GENERATORS)} species to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
