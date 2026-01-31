"""
Simple merge of submissions WITHOUT filtering by test set.
Just weighted average and top-K per protein.
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Submissions with weights
SUBMISSIONS = [
    ('sota_27jan', NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv', 1.0),
    ('esm2_0386', NOTEBOOKS_DIR / 'esm2_0386_output' / 'submission.tsv', 1.0),
    ('cafa_ensemble_378', NOTEBOOKS_DIR / 'cafa_ensemble_378_output' / 'submission.tsv', 0.9),
]


def load_submission(path: Path, name: str, weight: float) -> dict:
    """Load submission as dictionary."""
    print(f"Loading {name} (weight={weight})...")

    scores = defaultdict(dict)
    count = 0

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])

                # Skip invalid GO terms
                if not go_term.startswith('GO:'):
                    continue

                # Keep max score
                if go_term not in scores[protein] or scores[protein][go_term] < score:
                    scores[protein][go_term] = score

                count += 1
                if count % 10_000_000 == 0:
                    print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins")
    return scores


def merge_submissions(top_k: int = 200):
    """Merge submissions with weighted average."""
    print("\n=== Simple Merge ===")

    all_scores = []
    weights = []

    for name, path, weight in SUBMISSIONS:
        if not path.exists():
            print(f"  Skipping {name} (not found)")
            continue
        scores = load_submission(path, name, weight)
        all_scores.append(scores)
        weights.append(weight)

    if not all_scores:
        print("No submissions found!")
        return

    # Get all proteins
    all_proteins = set()
    for scores in all_scores:
        all_proteins.update(scores.keys())

    print(f"\nTotal proteins: {len(all_proteins):,}")
    print("Merging...")

    output_path = OUTPUT_DIR / f'merged_simple_top{top_k}.tsv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0

    with open(output_path, 'w') as out:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 50000 == 0:
                print(f"  Protein {i:,}/{len(all_proteins):,}...", flush=True)

            # Collect all GO terms
            all_terms = set()
            for scores in all_scores:
                if protein in scores:
                    all_terms.update(scores[protein].keys())

            # Weighted average
            term_scores = {}
            for term in all_terms:
                weighted_sum = 0.0
                weight_sum = 0.0

                for scores, weight in zip(all_scores, weights):
                    if protein in scores and term in scores[protein]:
                        weighted_sum += scores[protein][term] * weight
                        weight_sum += weight

                if weight_sum > 0:
                    term_scores[term] = min(weighted_sum / weight_sum, 1.0)

            # Top-K
            sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:top_k]

            for term, score in sorted_terms:
                if score >= 0.001:
                    out.write(f"{protein}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    merge_submissions(top_k)
