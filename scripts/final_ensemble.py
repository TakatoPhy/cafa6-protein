"""
Final Ensemble Script for CAFA6 Submission.

Combines all available model predictions with:
1. Rank normalization (handles different score distributions)
2. Weighted averaging (optimized weights)
3. GO hierarchy propagation (ensures parent scores >= child scores)

Usage:
    python scripts/final_ensemble.py [--weights w1,w2,w3,...] [--propagate]
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from parse_go import parse_go_obo

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'submissions'
GO_OBO_PATH = DATA_DIR / 'Train' / 'go-basic.obo'

# All available model predictions
AVAILABLE_MODELS = {
    'current_best': BASE_DIR / 'submission.tsv',
    'cafa_opt': DATA_DIR / 'processed' / 'cafa_optimization' / 'submission.tsv',
    'goa': DATA_DIR / 'goa_predictions' / 'goa_submission.tsv',
    'protboost_4500': OUTPUT_DIR / 'protboost_4500_esm2_650M.tsv',
    'gcn_stacking': OUTPUT_DIR / 'gcn_stacking.tsv',
    'stacking_mlp': OUTPUT_DIR / 'stacking_mlp.tsv',
    'rank_ensemble': OUTPUT_DIR / 'rank_ensemble.tsv',
}


def load_submission(path: Path, name: str) -> tuple:
    """Load submission and convert to scores and ranks."""
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

    # Convert to ranks per protein
    ranks = {}
    for protein, term_scores in scores.items():
        sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
        n = max(len(sorted_terms) - 1, 1)
        ranks[protein] = {term: i / n for i, (term, _) in enumerate(sorted_terms)}

    return scores, ranks


def propagate_max(protein_scores: dict, term_to_parents: dict) -> dict:
    """Propagate max scores up the GO hierarchy."""
    # Get all terms including ancestors
    all_terms = set(protein_scores.keys())
    for term in list(all_terms):
        for parent in term_to_parents.get(term, []):
            all_terms.add(parent)

    result = dict(protein_scores)

    # Add missing ancestors with 0 score
    for term in all_terms:
        if term not in result:
            result[term] = 0.0

    # Propagate (iterate until convergence)
    changed = True
    max_iter = 100
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        for child in list(result.keys()):
            child_score = result[child]
            for parent in term_to_parents.get(child, []):
                if parent in result and result[parent] < child_score:
                    result[parent] = child_score
                    changed = True

    return result


def final_ensemble(model_paths: list, model_names: list, weights: list = None,
                   apply_propagation: bool = True, min_score: float = 0.001) -> Path:
    """Create final ensemble submission."""
    print(f"\n=== Final Ensemble ===\n")

    n_models = len(model_paths)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"Models: {', '.join(model_names)}")
    print(f"Weights: {', '.join(f'{w:.3f}' for w in weights)}")

    # Load all models
    all_scores = []
    all_ranks = []

    for name, path in zip(model_names, model_paths):
        scores, ranks = load_submission(path, name)
        all_scores.append(scores)
        all_ranks.append(ranks)

    # Load GO ontology for propagation
    if apply_propagation:
        print("\nLoading GO ontology...")
        term_to_parents, _, _ = parse_go_obo(GO_OBO_PATH)

    # Get all proteins
    all_proteins = set()
    for scores in all_scores:
        all_proteins.update(scores.keys())

    print(f"\nTotal proteins: {len(all_proteins):,}")
    print("Creating ensemble...")

    output_path = OUTPUT_DIR / 'final_ensemble.tsv'
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

            # Weighted average using rank-normalized scores
            protein_scores = {}
            for term in all_terms:
                weighted_sum = 0
                weight_sum = 0

                for scores, ranks, weight in zip(all_scores, all_ranks, weights):
                    if protein in scores and term in scores[protein]:
                        # Use (1 - rank) as normalized score (higher = better)
                        rank = ranks[protein][term]
                        normalized = 1 - rank
                        weighted_sum += normalized * weight
                        weight_sum += weight

                if weight_sum > 0:
                    # Final score combines normalized rank with original scores
                    avg_rank_score = weighted_sum / weight_sum

                    # Also compute weighted average of original scores
                    orig_sum = 0
                    orig_weight = 0
                    for scores, weight in zip(all_scores, weights):
                        if protein in scores and term in scores[protein]:
                            orig_sum += scores[protein][term] * weight
                            orig_weight += weight

                    avg_orig = orig_sum / orig_weight if orig_weight > 0 else 0

                    # Blend: 50% rank-based, 50% original
                    protein_scores[term] = 0.5 * avg_rank_score + 0.5 * avg_orig

            # Apply GO propagation
            if apply_propagation:
                protein_scores = propagate_max(protein_scores, term_to_parents)

            # Write output
            for term, score in sorted(protein_scores.items(), key=lambda x: -x[1]):
                if score >= min_score:
                    f.write(f"{protein}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Final Ensemble')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated weights (e.g., "0.4,0.3,0.3")')
    parser.add_argument('--no-propagate', action='store_true',
                        help='Skip GO hierarchy propagation')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names to use')
    parser.add_argument('--min-score', type=float, default=0.001,
                        help='Minimum score threshold')
    args = parser.parse_args()

    # Determine which models to use
    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
    else:
        # Use all available models
        model_names = list(AVAILABLE_MODELS.keys())

    # Filter to existing files
    existing_models = []
    existing_paths = []
    for name in model_names:
        if name in AVAILABLE_MODELS and AVAILABLE_MODELS[name].exists():
            existing_models.append(name)
            existing_paths.append(AVAILABLE_MODELS[name])
        else:
            print(f"Warning: {name} not found, skipping")

    if len(existing_models) < 2:
        print("Error: Need at least 2 model predictions")
        print("\nAvailable models:")
        for name, path in AVAILABLE_MODELS.items():
            status = "OK" if path.exists() else "NOT FOUND"
            print(f"  {name}: {status}")
        sys.exit(1)

    # Parse weights
    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        if len(weights) != len(existing_models):
            print(f"Error: Number of weights ({len(weights)}) must match number of models ({len(existing_models)})")
            sys.exit(1)

    # Create ensemble
    final_ensemble(
        existing_paths, existing_models, weights,
        apply_propagation=not args.no_propagate,
        min_score=args.min_score
    )

    print("\n=== Submission Commands ===")
    print("ssh taka@100.75.229.83")
    print("cd ~/cafa6-protein/submissions")
    print("kaggle competitions submit -c cafa-6-protein-function-prediction \\")
    print("  -f final_ensemble.tsv.gz -m 'Final ensemble with GO propagation'")


if __name__ == '__main__':
    main()
