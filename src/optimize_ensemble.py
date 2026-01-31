"""
Optimize ensemble submission for CAFA6.
1. Filter to test proteins only
2. Clip scores to [0, 1]
3. Apply Top-K filtering
4. Reduce file size for Kaggle submission
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
TEST_FASTA = BASE_DIR / 'data' / 'Test' / 'testsuperset.fasta'


def load_test_ids():
    """Load test protein IDs from FASTA."""
    print("Loading test protein IDs...")
    ids = set()
    with open(TEST_FASTA, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip().split()[0]
                if '|' in header:
                    header = header.split('|')[1]
                ids.add(header)
    print(f"  Test proteins: {len(ids):,}")
    return ids


def optimize_submission(input_path: Path, output_path: Path, test_ids: set, top_k: int = 300):
    """Optimize a submission file."""
    print(f"\nOptimizing {input_path.name}...")

    # Load and filter
    protein_scores = defaultdict(dict)
    total_rows = 0
    filtered_rows = 0

    with open(input_path, 'r') as f:
        for line in f:
            total_rows += 1
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])

                # Filter to test proteins
                if protein not in test_ids:
                    continue

                # Filter invalid GO terms
                if not go_term.startswith('GO:'):
                    continue

                # Clip score to [0, 1]
                score = min(max(score, 0.0), 1.0)

                # Keep max score per protein+term
                if go_term not in protein_scores[protein] or protein_scores[protein][go_term] < score:
                    protein_scores[protein][go_term] = score
                    filtered_rows += 1

            if total_rows % 10_000_000 == 0:
                print(f"  Read {total_rows:,} rows...", flush=True)

    print(f"  Total rows: {total_rows:,}")
    print(f"  Proteins in test set: {len(protein_scores):,}")

    # Apply top-K and save
    print(f"Applying Top-{top_k} and saving...")
    output_rows = 0

    with open(output_path, 'w') as out:
        for protein in sorted(protein_scores.keys()):
            terms = protein_scores[protein]
            # Sort by score descending, take top-K
            sorted_terms = sorted(terms.items(), key=lambda x: -x[1])[:top_k]

            for go_term, score in sorted_terms:
                if score >= 0.001:  # Minimum threshold
                    out.write(f"{protein}\t{go_term}\t{score:.6f}\n")
                    output_rows += 1

    print(f"  Output rows: {output_rows:,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_ensemble.py <input.tsv> [top_k]")
        print("Example: python optimize_ensemble.py submissions/ensemble_3models.tsv 300")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_name = input_path.stem + f"_top{top_k}.tsv"
    output_path = input_path.parent / output_name

    print("=" * 60)
    print("CAFA6 Submission Optimizer")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Top-K: {top_k}")

    test_ids = load_test_ids()
    optimize_submission(input_path, output_path, test_ids, top_k)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
