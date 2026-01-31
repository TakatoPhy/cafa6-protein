"""
Optimize ensemble weights for 2-model merge (SOTA + ESM2).
Generate multiple weighted combinations for submission.
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Best 2 models
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'


def load_submission(path: Path, name: str) -> dict:
    """Load submission as dict[protein][go_term] = score."""
    print(f"Loading {name}...")

    scores = defaultdict(dict)
    count = 0

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])

                if not go_term.startswith('GO:'):
                    continue

                # Keep max score per (protein, go_term)
                if go_term not in scores[protein] or scores[protein][go_term] < score:
                    scores[protein][go_term] = score

                count += 1
                if count % 10_000_000 == 0:
                    print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins")
    return scores


def weighted_merge(sota_scores: dict, esm2_scores: dict, sota_weight: float):
    """Merge with weighted average."""
    esm2_weight = 1.0 - sota_weight

    all_proteins = set(sota_scores.keys()) | set(esm2_scores.keys())

    result = []
    for protein in all_proteins:
        all_terms = set()
        if protein in sota_scores:
            all_terms.update(sota_scores[protein].keys())
        if protein in esm2_scores:
            all_terms.update(esm2_scores[protein].keys())

        for term in all_terms:
            s_sota = sota_scores.get(protein, {}).get(term, 0)
            s_esm2 = esm2_scores.get(protein, {}).get(term, 0)

            # Weighted average (only for present scores)
            if s_sota > 0 and s_esm2 > 0:
                score = s_sota * sota_weight + s_esm2 * esm2_weight
            elif s_sota > 0:
                score = s_sota
            else:
                score = s_esm2

            result.append((protein, term, min(score, 1.0)))

    return result


def main():
    # Load both submissions
    sota_scores = load_submission(SOTA_PATH, "SOTA 27Jan")
    esm2_scores = load_submission(ESM2_PATH, "ESM2 3785")

    # Try different weights
    weights_to_try = [0.3, 0.4, 0.5, 0.6, 0.7]

    if len(sys.argv) > 1:
        weights_to_try = [float(sys.argv[1])]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sota_weight in weights_to_try:
        print(f"\n=== SOTA weight: {sota_weight}, ESM2 weight: {1-sota_weight:.1f} ===")

        result = weighted_merge(sota_scores, esm2_scores, sota_weight)

        # Sort by protein, then by score descending
        result.sort(key=lambda x: (x[0], -x[2]))

        output_path = OUTPUT_DIR / f'weighted_sota{int(sota_weight*100)}_esm2{int((1-sota_weight)*100)}.tsv'

        with open(output_path, 'w') as f:
            for protein, term, score in result:
                f.write(f"{protein}\t{term}\t{score:.6f}\n")

        print(f"Saved: {output_path}")
        print(f"Rows: {len(result):,}")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
