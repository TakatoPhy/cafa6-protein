"""
CAFA6 Easy/Hard Separation Strategy
Based on Japan AI Cup 1st Place Solution

Strategy:
- Easy samples (low disagreement): Standard blend (0.9:0.1)
- Hard samples (high disagreement): Max ensemble
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

SOTA_PATH = SUBMISSIONS_DIR / 'merged_sota_esm2.tsv'
TAXON_PATH = SUBMISSIONS_DIR / 'protboost_taxon_esm2_650M.tsv'


def load_predictions(path: Path):
    """Load predictions into dict."""
    predictions = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                predictions[protein][go_term] = score
    return dict(predictions)


def calculate_stats(predictions: dict):
    """Calculate stats for validation."""
    total = 0.0
    count = 0
    high_score_count = 0

    for protein, terms in predictions.items():
        for term, score in terms.items():
            total += score
            count += 1
            if score >= 0.9:
                high_score_count += 1

    mean = total / count if count > 0 else 0
    return mean, high_score_count, count


def predict_lb(mean):
    """Predict LB using validated formula."""
    return -0.622 * mean + 0.580


def create_easy_hard_blend(sota: dict, taxon: dict, hard_percentile: float = 85, easy_alpha: float = 0.9):
    """
    Create Easy/Hard blend.

    Args:
        sota: SOTA predictions dict
        taxon: Taxon predictions dict
        hard_percentile: Percentile threshold for hard samples (default 85 = top 15% are hard)
        easy_alpha: SOTA weight for easy samples (default 0.9)
    """
    # First pass: calculate disagreement for each (protein, term) pair
    disagreements = []
    pairs = []

    for protein in sota.keys():
        sota_terms = sota.get(protein, {})
        taxon_terms = taxon.get(protein, {})

        for term in sota_terms.keys():
            sota_score = sota_terms.get(term, 0.0)
            taxon_score = taxon_terms.get(term, 0.0)

            # Disagreement = absolute difference
            disagreement = abs(sota_score - taxon_score)
            disagreements.append(disagreement)
            pairs.append((protein, term, sota_score, taxon_score))

    disagreements = np.array(disagreements)

    # Calculate threshold
    threshold = np.percentile(disagreements, hard_percentile)
    hard_mask = disagreements > threshold

    n_hard = hard_mask.sum()
    n_easy = len(hard_mask) - n_hard

    print(f"   Disagreement threshold (p{hard_percentile}): {threshold:.4f}")
    print(f"   Easy samples: {n_easy:,} ({100*n_easy/len(pairs):.1f}%)")
    print(f"   Hard samples: {n_hard:,} ({100*n_hard/len(pairs):.1f}%)")

    # Create blended predictions
    blended = defaultdict(dict)

    for i, (protein, term, sota_score, taxon_score) in enumerate(pairs):
        if hard_mask[i]:
            # Hard: Max ensemble
            blended_score = max(sota_score, taxon_score)
        else:
            # Easy: Weighted blend
            blended_score = easy_alpha * sota_score + (1 - easy_alpha) * taxon_score

        blended[protein][term] = blended_score

    return dict(blended), n_easy, n_hard


def save_predictions(predictions: dict, output_path: Path):
    """Save predictions to TSV file."""
    with open(output_path, 'w') as f:
        for protein in sorted(predictions.keys()):
            for term, score in sorted(predictions[protein].items()):
                f.write(f"{protein}\t{term}\t{score:.6f}\n")


def main():
    print("=" * 60)
    print("CAFA6 Easy/Hard Separation Strategy")
    print("Based on Japan AI Cup 1st Place Solution")
    print("=" * 60)

    # Load predictions
    print(f"\n1. Loading SOTA predictions...")
    sota = load_predictions(SOTA_PATH)
    print(f"   Loaded {len(sota):,} proteins")

    print(f"\n2. Loading Taxon predictions...")
    taxon = load_predictions(TAXON_PATH)
    print(f"   Loaded {len(taxon):,} proteins")

    # Test different configurations
    configs = [
        (85, 0.90, "easy90_hard_max_p85"),  # Top 15% hard
        (80, 0.90, "easy90_hard_max_p80"),  # Top 20% hard
        (90, 0.90, "easy90_hard_max_p90"),  # Top 10% hard
        (85, 0.95, "easy95_hard_max_p85"),  # More conservative easy blend
    ]

    results = []

    for hard_percentile, easy_alpha, name in configs:
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"  Easy blend: {easy_alpha:.2f}:{1-easy_alpha:.2f}")
        print(f"  Hard: Max ensemble (top {100-hard_percentile}%)")
        print("=" * 60)

        print(f"\n3. Creating Easy/Hard blend...")
        blended, n_easy, n_hard = create_easy_hard_blend(
            sota, taxon,
            hard_percentile=hard_percentile,
            easy_alpha=easy_alpha
        )

        # Calculate stats
        print(f"\n4. Calculating stats...")
        mean, high_scores, total = calculate_stats(blended)
        pred_lb = predict_lb(mean)

        print(f"   Mean: {mean:.4f}")
        print(f"   High scores (>=0.9): {high_scores:,}")
        print(f"   Total predictions: {total:,}")
        print(f"   Predicted LB: {pred_lb:.3f}")

        results.append({
            'name': name,
            'mean': mean,
            'high_scores': high_scores,
            'pred_lb': pred_lb,
            'n_easy': n_easy,
            'n_hard': n_hard
        })

        # Safety check
        if high_scores < 1_000_000:
            print(f"   ❌ DANGER: High scores < 1M - skipping save")
            continue
        elif high_scores < 4_000_000:
            print(f"   ⚠️  WARNING: High scores < 4M")
        else:
            print(f"   ✅ SAFE: High scores >= 4M")

        # Save
        output_path = SUBMISSIONS_DIR / f"blend_{name}.tsv"
        print(f"\n5. Saving to {output_path.name}...")
        save_predictions(blended, output_path)
        print(f"   Done!")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    # Get original stats for comparison
    sota_mean, sota_high, _ = calculate_stats(sota)
    sota_pred = predict_lb(sota_mean)
    print(f"\nOriginal SOTA+ESM2:")
    print(f"   Mean: {sota_mean:.4f}, High scores: {sota_high:,}, Pred LB: {sota_pred:.3f}")

    print(f"\nEasy/Hard Blends:")
    print(f"{'Name':<30} {'Mean':>7} {'>=0.9':>12} {'Pred LB':>8} {'Δ':>7}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['pred_lb'], reverse=True):
        delta = r['pred_lb'] - sota_pred
        safe = "✅" if r['high_scores'] >= 4_000_000 else ("⚠️" if r['high_scores'] >= 1_000_000 else "❌")
        print(f"{safe} {r['name']:<28} {r['mean']:>7.4f} {r['high_scores']:>12,} {r['pred_lb']:>8.3f} {delta:>+7.3f}")


if __name__ == '__main__':
    main()
