#!/usr/bin/env python3
"""
安全なブレンドスクリプト

0.364事件の教訓:
- 和集合でなく交差集合を使う
- ベースラインの(protein, term)ペアのみを対象
- スコア分布が大きく変わらないことを確認

Usage:
    python scripts/safe_blend.py output.tsv model1.tsv model2.tsv [weights]
    python scripts/safe_blend.py output.tsv model1.tsv model2.tsv 0.7,0.3
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).parent.parent
BASELINE_PATH = BASE_DIR / 'submission.tsv'


def load_predictions(path: Path) -> dict:
    """Load predictions as dict[protein][term] -> score"""
    scores = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, term, score = parts[0], parts[1], float(parts[2])
                if term.startswith('GO:'):
                    scores[protein][term] = score
    return scores


def safe_blend(model_paths: list, output_path: Path, weights: list = None,
               use_baseline_keys: bool = True):
    """
    安全なブレンド: ベースラインの(protein, term)ペアのみを対象

    Args:
        model_paths: モデルファイルのパスリスト
        output_path: 出力パス
        weights: 重みリスト（Noneなら均等）
        use_baseline_keys: Trueならベースラインの(protein, term)のみ対象
    """
    print(f"\n=== Safe Blend ===\n")

    # Load baseline for keys
    print("Loading baseline for key filtering...")
    baseline = load_predictions(BASELINE_PATH)
    baseline_pairs = set()
    for protein, terms in baseline.items():
        for term in terms:
            baseline_pairs.add((protein, term))
    print(f"  Baseline pairs: {len(baseline_pairs):,}")

    # Load models
    models = []
    for path in model_paths:
        print(f"Loading {path.name}...")
        scores = load_predictions(path)
        total = sum(len(t) for t in scores.values())
        print(f"  {total:,} predictions, {len(scores):,} proteins")
        models.append(scores)

    # Set weights
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    print(f"\nWeights: {weights}")

    # Blend (baseline keys only)
    print("\nBlending (baseline keys only)...")
    result = {}
    included = 0
    excluded = 0

    for protein, term in baseline_pairs:
        scores_for_blend = []
        weights_for_blend = []

        for model, weight in zip(models, weights):
            if protein in model and term in model[protein]:
                scores_for_blend.append(model[protein][term])
                weights_for_blend.append(weight)

        if scores_for_blend:
            # Weighted average (renormalize weights)
            total_weight = sum(weights_for_blend)
            blended = sum(s * w for s, w in zip(scores_for_blend, weights_for_blend)) / total_weight

            if protein not in result:
                result[protein] = {}
            result[protein][term] = blended
            included += 1
        else:
            excluded += 1

    print(f"  Included: {included:,}")
    print(f"  Excluded (no predictions): {excluded:,}")

    # Write output
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        for protein in sorted(result.keys()):
            for term, score in sorted(result[protein].items(), key=lambda x: -x[1]):
                f.write(f"{protein}\t{term}\t{score:.6f}\n")

    # Stats
    all_scores = [s for terms in result.values() for s in terms.values()]
    print(f"\nOutput stats:")
    print(f"  Rows: {len(all_scores):,}")
    print(f"  Proteins: {len(result):,}")
    print(f"  Score mean: {np.mean(all_scores):.3f}")
    print(f"  Score std: {np.std(all_scores):.3f}")

    # Compare with baseline
    baseline_scores = [s for terms in baseline.values() for s in terms.values()]
    print(f"\nBaseline comparison:")
    print(f"  Baseline mean: {np.mean(baseline_scores):.3f}, Blend mean: {np.mean(all_scores):.3f}")
    print(f"  Difference: {abs(np.mean(all_scores) - np.mean(baseline_scores)):.3f}")

    if abs(np.mean(all_scores) - np.mean(baseline_scores)) > 0.1:
        print("\n⚠️ WARNING: Score distribution significantly changed!")
    else:
        print("\n✅ Score distribution looks reasonable")

    return output_path


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/safe_blend.py output.tsv model1.tsv model2.tsv [weights]")
        print("Example: python scripts/safe_blend.py blend.tsv submission.tsv nb153.tsv 0.7,0.3")
        sys.exit(1)

    output_path = Path(sys.argv[1])
    model_paths = [Path(p) for p in sys.argv[2:-1] if not ',' in p]

    # Check for weights in last arg
    weights = None
    if ',' in sys.argv[-1]:
        weights = [float(w) for w in sys.argv[-1].split(',')]
    else:
        model_paths.append(Path(sys.argv[-1]))

    # Validate
    for path in model_paths:
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)

    safe_blend(model_paths, output_path, weights)


if __name__ == '__main__':
    main()
