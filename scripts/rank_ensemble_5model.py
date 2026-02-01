"""
Multi-model Rank Normalization Ensemble.

Instead of averaging raw scores (which can have different distributions),
convert scores to percentile ranks within each protein, then average ranks.

This handles the case where one model outputs scores in [0, 1] while
another outputs in [0.3, 0.7] - both get normalized to [0, 1] ranks.

Usage:
    python scripts/rank_ensemble_5model.py output.tsv [model1.tsv model2.tsv ...]
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).parent.parent

# Default models to ensemble
DEFAULT_MODELS = [
    ('current_best', BASE_DIR / 'submission.tsv'),
    ('cafa_opt', BASE_DIR / 'data' / 'processed' / 'cafa_optimization' / 'submission.tsv'),
    ('goa', BASE_DIR / 'data' / 'goa_predictions' / 'goa_submission.tsv'),
]


def load_and_rank(path: Path, name: str) -> tuple:
    """
    Load submission and convert scores to ranks per protein.

    Returns:
        scores: dict[protein][term] -> original score
        ranks: dict[protein][term] -> percentile rank (0 = best, 1 = worst)
    """
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
                # Keep max score for each (protein, term) pair
                if go_term not in scores[protein] or scores[protein][go_term] < score:
                    scores[protein][go_term] = score
                count += 1
                if count % 10_000_000 == 0:
                    print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins")

    # Convert to percentile ranks per protein
    print(f"  Converting to ranks...")
    ranks = {}
    for protein, term_scores in scores.items():
        # Sort by score descending
        sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
        n = len(sorted_terms)
        if n > 0:
            # Assign percentile ranks (0 = best, 1 = worst)
            ranks[protein] = {term: i / max(n - 1, 1) for i, (term, _) in enumerate(sorted_terms)}
        else:
            ranks[protein] = {}

    return scores, ranks


def rank_ensemble(models: list, output_path: Path, min_score: float = 0.001, use_rank_output: bool = False):
    """
    Create rank-averaged ensemble from multiple models.

    Args:
        models: List of (name, path) tuples
        output_path: Where to save the result
        min_score: Minimum score threshold for output
        use_rank_output: If True, output averaged ranks; if False, output averaged original scores
    """
    print(f"\n=== {len(models)}-Model Rank Ensemble ===\n")

    all_scores = []
    all_ranks = []
    model_names = []

    for name, path in models:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        scores, ranks = load_and_rank(path, name)
        all_scores.append(scores)
        all_ranks.append(ranks)
        model_names.append(name)

    if len(all_scores) < 2:
        print("Error: Need at least 2 models")
        return

    print(f"\nUsing {len(all_scores)} models: {', '.join(model_names)}")

    # Get all proteins from all models
    all_proteins = set()
    for scores in all_scores:
        all_proteins.update(scores.keys())

    print(f"Total proteins: {len(all_proteins):,}")
    print("Merging with rank average...")

    row_count = 0
    with open(output_path, 'w') as f:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 50000 == 0:
                print(f"  Protein {i:,}/{len(all_proteins):,}...", flush=True)

            # Collect all GO terms from all models for this protein
            all_terms = set()
            for scores in all_scores:
                if protein in scores:
                    all_terms.update(scores[protein].keys())

            # Calculate average rank and score for each term
            term_data = []
            for term in all_terms:
                rank_values = []
                score_values = []

                for scores, ranks in zip(all_scores, all_ranks):
                    if protein in scores and term in scores[protein]:
                        rank_values.append(ranks[protein][term])
                        score_values.append(scores[protein][term])

                if rank_values:
                    avg_rank = np.mean(rank_values)
                    avg_score = np.mean(score_values)
                    # Weight by number of models that have this prediction
                    coverage = len(rank_values) / len(all_scores)
                    term_data.append((term, avg_rank, avg_score, coverage))

            # Sort by average rank (lower = better)
            term_data.sort(key=lambda x: x[1])

            # Output
            for term, avg_rank, avg_score, coverage in term_data:
                # Use the original score for output, but sorted by rank
                output_score = avg_score
                if output_score >= min_score:
                    f.write(f"{protein}\t{term}\t{output_score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python rank_ensemble_5model.py output.tsv [model1.tsv model2.tsv ...]")
        print("\nDefault models:")
        for name, path in DEFAULT_MODELS:
            print(f"  {name}: {path}")
        sys.exit(1)

    output_path = Path(sys.argv[1])

    # Use provided models or defaults
    if len(sys.argv) > 2:
        models = [(f"model_{i}", Path(p)) for i, p in enumerate(sys.argv[2:])]
    else:
        models = DEFAULT_MODELS

    # Filter to existing files
    models = [(name, path) for name, path in models if path.exists()]

    if len(models) < 2:
        print("Error: Need at least 2 model files")
        sys.exit(1)

    rank_ensemble(models, output_path)


if __name__ == '__main__':
    main()
