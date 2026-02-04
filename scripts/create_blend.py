"""
Create blended submission files for CAFA6.
Blends SOTA+ESM2 with Taxon predictions.
"""
import argparse
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


def blend_predictions(sota: dict, taxon: dict, alpha: float):
    """Blend predictions with alpha weight for SOTA."""
    blended = defaultdict(dict)

    # Get all proteins from SOTA (primary)
    all_proteins = set(sota.keys())

    for protein in all_proteins:
        sota_terms = sota.get(protein, {})
        taxon_terms = taxon.get(protein, {})

        # All terms from SOTA
        all_terms = set(sota_terms.keys())

        for term in all_terms:
            sota_score = sota_terms.get(term, 0.0)
            taxon_score = taxon_terms.get(term, 0.0)

            # Weighted blend
            blended_score = alpha * sota_score + (1 - alpha) * taxon_score
            blended[protein][term] = blended_score

    return dict(blended)


def save_predictions(predictions: dict, output_path: Path):
    """Save predictions to TSV file."""
    with open(output_path, 'w') as f:
        for protein in sorted(predictions.keys()):
            for term, score in sorted(predictions[protein].items()):
                f.write(f"{protein}\t{term}\t{score:.6f}\n")


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


def main():
    parser = argparse.ArgumentParser(description='Create blended submission')
    parser.add_argument('--ratio', type=float, required=True,
                        help='SOTA weight (e.g., 0.95 for 95:5 blend)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto-generated)')
    args = parser.parse_args()

    alpha = args.ratio

    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        ratio_str = f"{int(alpha*100)}_{int((1-alpha)*100)}"
        output_path = SUBMISSIONS_DIR / f"blend_sota{int(alpha*100)}_taxon{int((1-alpha)*100)}.tsv"

    print(f"=" * 60)
    print(f"Creating blend: SOTA*{alpha:.2f} + Taxon*{1-alpha:.2f}")
    print(f"=" * 60)

    # Load predictions
    print(f"\n1. Loading SOTA predictions from {SOTA_PATH.name}...")
    sota = load_predictions(SOTA_PATH)
    print(f"   Loaded {len(sota):,} proteins")

    print(f"\n2. Loading Taxon predictions from {TAXON_PATH.name}...")
    taxon = load_predictions(TAXON_PATH)
    print(f"   Loaded {len(taxon):,} proteins")

    # Blend
    print(f"\n3. Blending predictions...")
    blended = blend_predictions(sota, taxon, alpha)
    print(f"   Blended {len(blended):,} proteins")

    # Calculate stats
    print(f"\n4. Calculating stats...")
    mean, high_scores, total = calculate_stats(blended)
    pred_lb = predict_lb(mean)

    print(f"   Mean: {mean:.4f}")
    print(f"   High scores (>=0.9): {high_scores:,}")
    print(f"   Total predictions: {total:,}")
    print(f"   Predicted LB: {pred_lb:.3f}")

    # Safety check
    print(f"\n5. Safety check...")
    if high_scores < 1_000_000:
        print(f"   ❌ DANGER: High scores < 1M ({high_scores:,})")
        print(f"   This will likely result in LB ~0.14x")
        print(f"   ABORTING - file not saved")
        sys.exit(1)
    elif high_scores < 4_000_000:
        print(f"   ⚠️  WARNING: High scores < 4M ({high_scores:,})")
        print(f"   Proceed with caution")
    else:
        print(f"   ✅ SAFE: High scores = {high_scores:,} (>= 4M)")

    # Save
    print(f"\n6. Saving to {output_path}...")
    save_predictions(blended, output_path)
    print(f"   Done!")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Output: {output_path}")
    print(f"Mean: {mean:.4f}")
    print(f"High scores: {high_scores:,}")
    print(f"Predicted LB: {pred_lb:.3f}")

    # Compare with original
    print(f"\nCompared to original SOTA+ESM2:")
    sota_mean, sota_high, _ = calculate_stats(sota)
    sota_pred = predict_lb(sota_mean)
    print(f"   Original mean: {sota_mean:.4f}, high scores: {sota_high:,}, pred LB: {sota_pred:.3f}")
    print(f"   Blended mean:  {mean:.4f}, high scores: {high_scores:,}, pred LB: {pred_lb:.3f}")
    print(f"   Δ pred LB: {pred_lb - sota_pred:+.3f}")


if __name__ == '__main__':
    main()
