"""Leave-one-out accuracy check for the Bird Sound Detector's k-NN
classifier.

For every reference sample, run the exact same k-NN vote
(dsp_core.bird_detector.knn_classify) used at request time, but with that
one sample's own vector removed from its species' reference set first — so
a sample never gets to vote for itself, which would make every species
look artificially separable. This is what "accuracy" means for this
project: not a held-out test set (there isn't one to spare), but "if this
recording didn't exist yet, would the rest of the reference set correctly
identify it?"

Run with:  python scripts/evaluate_bird_accuracy.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsp_core.bird_detector import K_NEIGHBORS, _load_reference, knn_classify


def _leave_one_out_samples(species_samples: dict, species: str, index: int) -> dict:
    """species_samples with sample `index` of `species` removed — everyone
    else's data stays untouched."""
    out = {}
    for s, vectors in species_samples.items():
        out[s] = np.delete(vectors, index, axis=0) if s == species else vectors
    return out


def main():
    species_samples, global_std = _load_reference()

    rows = []  # (true_species, predicted_species, nearest_distance, correct)
    for species, vectors in species_samples.items():
        for i, vector in enumerate(vectors):
            pool = _leave_one_out_samples(species_samples, species, i)
            result = knn_classify(vector, pool, global_std, k=K_NEIGHBORS)
            predicted = result["rawSpecies"]
            rows.append((species, predicted, result["bestDistance"], predicted == species))

    total = len(rows)
    correct_rows = [r for r in rows if r[3]]
    incorrect_rows = [r for r in rows if not r[3]]
    accuracy = len(correct_rows) / total if total else 0.0

    print(f"Leave-one-out accuracy: {len(correct_rows)}/{total} = {accuracy * 100:.1f}%\n")

    print("Per-species accuracy:")
    for species in species_samples:
        species_rows = [r for r in rows if r[0] == species]
        species_correct = [r for r in species_rows if r[3]]
        acc = len(species_correct) / len(species_rows) if species_rows else 0.0
        print(f"  {species:8s}: {len(species_correct)}/{len(species_rows)} = {acc * 100:.1f}%")

    print("\nConfusion (true -> predicted, wrong guesses only):")
    for species in species_samples:
        wrong = [r for r in incorrect_rows if r[0] == species]
        if wrong:
            for true_sp, pred_sp, dist, _ in wrong:
                print(f"  {true_sp} -> {pred_sp}  (nearest-neighbour distance {dist:.3f})")

    correct_distances = np.array([r[2] for r in correct_rows]) if correct_rows else np.array([])
    incorrect_distances = np.array([r[2] for r in incorrect_rows]) if incorrect_rows else np.array([])

    print("\nNearest-neighbour distance, correct guesses:")
    if correct_distances.size:
        print(f"  min={correct_distances.min():.3f}  median={np.median(correct_distances):.3f}  "
              f"90th pct={np.percentile(correct_distances, 90):.3f}  max={correct_distances.max():.3f}")
    else:
        print("  (none)")

    print("Nearest-neighbour distance, incorrect guesses:")
    if incorrect_distances.size:
        print(f"  min={incorrect_distances.min():.3f}  10th pct={np.percentile(incorrect_distances, 10):.3f}  "
              f"median={np.median(incorrect_distances):.3f}  max={incorrect_distances.max():.3f}")
    else:
        print("  (none)")

    # A threshold only decides "Unknown" vs. trusting the top-1 species —
    # it cannot fix a wrong top-1 guess. It's useful only if correct and
    # incorrect guesses tend to sit at different distances; report whether
    # that's actually true here rather than assuming it.
    if correct_distances.size and incorrect_distances.size:
        correct_p90 = np.percentile(correct_distances, 90)
        incorrect_p10 = np.percentile(incorrect_distances, 10)
        print(f"\ncorrect 90th pct ({correct_p90:.3f}) vs incorrect 10th pct ({incorrect_p10:.3f}):")
        if correct_p90 < incorrect_p10:
            suggested = (correct_p90 + incorrect_p10) / 2
            print(f"  clean gap — suggested UNKNOWN_DISTANCE_THRESHOLD ~= {suggested:.2f}")
        else:
            print("  overlapping — no distance threshold cleanly separates correct from incorrect "
                  "guesses here; a threshold can still reject the worst outliers but can't fix "
                  "species-vs-species confusion by itself.")


if __name__ == "__main__":
    main()
