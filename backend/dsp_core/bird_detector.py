"""Traditional-DSP bird species matcher.

Loads the pre-computed reference feature vectors (built once by
scripts/generate_bird_references.py) and classifies a live clip with
k-nearest-neighbours: compute the weighted distance from the live clip's
feature vector to every individual reference sample (not a species
average), take the k closest samples, and let them vote on the species.
Still no machine learning, no training step at request time, and no
external service call — k-NN is a lookup + vote, not a trained model.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from dsp_core.bird_features import extract_features, weighted_distance

REFERENCE_PATH = Path(__file__).resolve().parent / "bird_reference_features.json"

# How many nearest reference samples get a vote. Must stay below the
# smallest per-species sample count or that species can never win a
# majority; 5 leaves headroom for the current 6-samples-per-species set
# while still outvoting a single noisy neighbour.
K_NEIGHBORS = 5

# Calibrated empirically (see scripts/generate_bird_references.py output):
# a genuine match's nearest neighbour lands under ~1.1, unrelated audio
# (noise, silence, an unrepresented sound) lands above ~3. 2.0 sits in the
# gap with margin on both sides. Re-check this once real recordings replace
# the synthetic reference set — individual-sample distances run smaller
# than the old distance-to-mean did, so the gap may shift.
UNKNOWN_DISTANCE_THRESHOLD = 2.0


def _load_reference():
    with open(REFERENCE_PATH) as f:
        data = json.load(f)
    global_std = np.array(data.pop("_global_std"))
    samples = {
        species: np.array(entry["vectors"]) for species, entry in data.items()
    }
    return samples, global_std


# Loaded once at import time — this is a small, static lookup table, not a
# trained model, and re-reading it per request would be wasted I/O.
_SPECIES_SAMPLES, _GLOBAL_STD = _load_reference()


def classify_bird_sound(y: np.ndarray, sr: int) -> dict:
    """Extract features from a clip and report the k-NN majority species,
    honestly hedged: a numeric confidence and, below the threshold, an
    explicit 'Unknown' result rather than a forced guess."""
    vector = extract_features(y, sr)

    # Distance from the live clip to every individual reference sample,
    # across all species, so the k nearest can come from any mix of
    # species rather than being pre-grouped by class.
    neighbors = []
    for species, sample_vectors in _SPECIES_SAMPLES.items():
        for sample_vector in sample_vectors:
            d = weighted_distance(vector, sample_vector, _GLOBAL_STD)
            neighbors.append((d, species))
    neighbors.sort(key=lambda pair: pair[0])

    k = min(K_NEIGHBORS, len(neighbors))
    nearest = neighbors[:k]
    nearest_distance = nearest[0][0]

    votes = Counter(species for _, species in nearest)
    top_count = max(votes.values())
    # Ties broken by whichever tied species has the smaller total distance
    # summed over just its votes among the k neighbours.
    tied = [s for s, c in votes.items() if c == top_count]
    if len(tied) == 1:
        best_species = tied[0]
    else:
        tied_totals = {s: sum(d for d, sp in nearest if sp == s) for s in tied}
        best_species = min(tied_totals, key=tied_totals.get)

    # Per-species minimum distance among ALL reference samples (not just
    # the k that won the vote) — lets the UI show how close every species
    # got, same shape as before k-NN.
    best_per_species = {}
    for d, species in neighbors:
        if species not in best_per_species or d < best_per_species[species]:
            best_per_species[species] = d

    # Display convenience, not a probability: blends how decisively the
    # k neighbours agreed with how close the single nearest match was.
    vote_fraction = top_count / k
    proximity = max(0.0, 1.0 - nearest_distance / (UNKNOWN_DISTANCE_THRESHOLD * 2))
    confidence = min(99.0, 100.0 * vote_fraction * proximity)

    is_confident = nearest_distance <= UNKNOWN_DISTANCE_THRESHOLD

    return {
        "species": best_species if is_confident else "Unknown",
        "isConfident": is_confident,
        "confidence": round(confidence, 1),
        "bestDistance": round(float(nearest_distance), 3),
        "threshold": UNKNOWN_DISTANCE_THRESHOLD,
        "kNeighbors": k,
        "votes": dict(votes),
        "allDistances": {s: round(float(d), 3) for s, d in best_per_species.items()},
        "featureVector": {
            name: round(float(val), 4)
            for name, val in zip(
                ("rms", "zcr", "dominantFreq", "spectralCentroid", "spectralBandwidth", "spectralRolloff"),
                vector,
            )
        },
    }
