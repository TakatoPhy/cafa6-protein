"""
Geometric mean ensemble.
More robust to outliers than arithmetic mean.
"""
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Best 2 models
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'


def load_submission(path: Path, name: str) -> dict:
    """Load submission as dict."""
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
                if go_term not in scores[protein] or scores[protein][go_term] < score:
                    scores[protein][go_term] = score
                count += 1
                if count % 10_000_000 == 0:
                    print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins")
    return scores


def geometric_mean_ensemble():
    """Create geometric mean ensemble."""
    print("\n=== Geometric Mean Ensemble ===\n")

    sota = load_submission(SOTA_PATH, "SOTA 27Jan")
    esm2 = load_submission(ESM2_PATH, "ESM2 3785")

    all_proteins = set(sota.keys()) | set(esm2.keys())
    print(f"\nTotal proteins: {len(all_proteins):,}")
    print("Computing geometric mean...")

    output_path = OUTPUT_DIR / 'geometric_mean.tsv'
    row_count = 0

    with open(output_path, 'w') as f:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 50000 == 0:
                print(f"  Protein {i:,}/{len(all_proteins):,}...", flush=True)

            all_terms = set()
            if protein in sota:
                all_terms.update(sota[protein].keys())
            if protein in esm2:
                all_terms.update(esm2[protein].keys())

            for term in all_terms:
                s1 = sota.get(protein, {}).get(term, 0)
                s2 = esm2.get(protein, {}).get(term, 0)

                # Geometric mean (handle zeros)
                if s1 > 0 and s2 > 0:
                    score = np.sqrt(s1 * s2)
                elif s1 > 0:
                    score = s1
                elif s2 > 0:
                    score = s2
                else:
                    continue

                if score >= 0.001:
                    f.write(f"{protein}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    geometric_mean_ensemble()
