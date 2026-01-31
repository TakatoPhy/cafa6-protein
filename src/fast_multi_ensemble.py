"""
Fast Multi-Model Ensemble for CAFA6
Uses streaming/chunked processing for large files.

Target: 0.387 → 0.40+ (Gold Medal: 0.408)
"""
import polars as pl
from pathlib import Path
import sys
from collections import defaultdict
import gc

# Paths
BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Core high-scoring submissions
CORE_SUBMISSIONS = [
    ('sota_27jan', NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv', 1.0),
    ('esm2_0386', NOTEBOOKS_DIR / 'esm2_0386_output' / 'submission.tsv', 1.0),
    ('cafa_ensemble_378', NOTEBOOKS_DIR / 'cafa_ensemble_378_output' / 'submission.tsv', 0.9),
]

# Additional submissions for extended ensemble
EXTRA_SUBMISSIONS = [
    ('ktdk', NOTEBOOKS_DIR / 'ktdk_output' / 'submission.tsv', 0.8),
    ('goa_propagation', NOTEBOOKS_DIR / 'goa_propagation_output' / 'submission.tsv', 0.7),
]


def load_as_dict(path: Path, name: str) -> dict:
    """Load submission as dictionary for memory efficiency."""
    print(f"Loading {name}...", flush=True)

    scores = defaultdict(dict)
    count = 0

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                scores[protein][go_term] = max(scores[protein].get(go_term, 0), score)
                count += 1

            if count % 10_000_000 == 0:
                print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins", flush=True)
    return scores


def weighted_merge(submissions: list, output_path: Path, top_k: int = 500):
    """Merge submissions with weighted average."""
    print("\n=== Weighted Merge ===")

    # Load all submissions
    all_scores = []
    weights = []

    for name, path, weight in submissions:
        if not path.exists():
            print(f"  Skipping {name} (not found)")
            continue
        scores = load_as_dict(path, name)
        all_scores.append(scores)
        weights.append(weight)
        gc.collect()

    if not all_scores:
        print("No submissions found!")
        return

    # Get all unique proteins
    all_proteins = set()
    for scores in all_scores:
        all_proteins.update(scores.keys())

    print(f"\nTotal unique proteins: {len(all_proteins):,}")

    # Merge scores
    print("Merging predictions...")
    total_weight = sum(weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with open(output_path, 'w') as out:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 10000 == 0:
                print(f"  Processing protein {i:,}/{len(all_proteins):,}...", flush=True)

            # Collect all GO terms for this protein
            all_terms = set()
            for scores in all_scores:
                if protein in scores:
                    all_terms.update(scores[protein].keys())

            # Compute weighted average for each term
            term_scores = {}
            for term in all_terms:
                weighted_sum = 0.0
                weight_sum = 0.0

                for scores, weight in zip(all_scores, weights):
                    if protein in scores and term in scores[protein]:
                        weighted_sum += scores[protein][term] * weight
                        weight_sum += weight

                if weight_sum > 0:
                    term_scores[term] = weighted_sum / weight_sum

            # Keep top-K
            sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:top_k]

            for term, score in sorted_terms:
                if score >= 0.001:
                    out.write(f"{protein}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("CAFA6 Fast Multi-Model Ensemble")
    print("=" * 60)

    # Select submissions based on command line
    if len(sys.argv) > 1 and sys.argv[1] == 'extended':
        submissions = CORE_SUBMISSIONS + EXTRA_SUBMISSIONS
        output_name = 'ensemble_5models.tsv'
    else:
        submissions = CORE_SUBMISSIONS
        output_name = 'ensemble_3models.tsv'

    print(f"\nSubmissions: {[s[0] for s in submissions]}")

    output_path = OUTPUT_DIR / output_name
    weighted_merge(submissions, output_path, top_k=500)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
