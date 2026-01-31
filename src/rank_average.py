"""
Rank-based averaging for ensemble.
Instead of averaging raw scores, convert to ranks and average ranks.
This handles different score distributions better.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Best 2 models
SUBMISSIONS = [
    ('SOTA', NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'),
    ('ESM2', NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'),
]


def load_and_rank(path: Path, name: str) -> dict:
    """Load submission and convert scores to ranks per protein."""
    print(f"Loading {name}...")

    # Load data
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

    # Convert to ranks per protein
    print(f"  Converting to ranks...")
    ranks = {}
    for protein, term_scores in scores.items():
        # Sort by score descending
        sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
        n = len(sorted_terms)
        # Assign percentile ranks (0 = best, 1 = worst)
        ranks[protein] = {term: i / n for i, (term, _) in enumerate(sorted_terms)}

    return scores, ranks


def rank_average():
    """Create rank-averaged ensemble."""
    print("\n=== Rank Average Ensemble ===\n")

    all_scores = []
    all_ranks = []

    for name, path in SUBMISSIONS:
        scores, ranks = load_and_rank(path, name)
        all_scores.append(scores)
        all_ranks.append(ranks)

    # Get all proteins
    all_proteins = set()
    for scores in all_scores:
        all_proteins.update(scores.keys())

    print(f"\nTotal proteins: {len(all_proteins):,}")
    print("Merging with rank average...")

    output_path = OUTPUT_DIR / 'rank_average.tsv'
    row_count = 0

    with open(output_path, 'w') as f:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 50000 == 0:
                print(f"  Protein {i:,}/{len(all_proteins):,}...", flush=True)

            # Collect all GO terms
            all_terms = set()
            for scores in all_scores:
                if protein in scores:
                    all_terms.update(scores[protein].keys())

            # Calculate average rank and original score
            term_data = []
            for term in all_terms:
                rank_sum = 0
                rank_count = 0
                score_sum = 0
                score_count = 0

                for scores, ranks in zip(all_scores, all_ranks):
                    if protein in scores and term in scores[protein]:
                        rank_sum += ranks[protein][term]
                        rank_count += 1
                        score_sum += scores[protein][term]
                        score_count += 1

                if rank_count > 0:
                    avg_rank = rank_sum / rank_count
                    avg_score = score_sum / score_count
                    term_data.append((term, avg_rank, avg_score))

            # Sort by average rank (lower = better)
            term_data.sort(key=lambda x: x[1])

            # Output with original score (but sorted by rank)
            for term, avg_rank, avg_score in term_data:
                if avg_score >= 0.001:
                    f.write(f"{protein}\t{term}\t{avg_score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    rank_average()
