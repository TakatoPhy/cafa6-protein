#!/usr/bin/env python3
"""
複数の提出ファイルをマージするスクリプト

Usage:
    # 2ファイルを平均でマージ
    python scripts/merge_submissions.py file1.tsv file2.tsv -o merged.tsv

    # 重み付きマージ
    python scripts/merge_submissions.py file1.tsv file2.tsv -w 0.6 0.4 -o merged.tsv

    # 3ファイル以上
    python scripts/merge_submissions.py f1.tsv f2.tsv f3.tsv -w 0.5 0.3 0.2 -o merged.tsv
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).parent.parent


def load_submission(path: Path) -> dict:
    """Load submission as {(protein, term): score}."""
    scores = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % 5000000 == 0 and i > 0:
                print(f"  {i:,} rows...")
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if go_term.startswith('GO:'):
                    scores[(protein, go_term)] = score
            except ValueError:
                continue
    return scores


def merge_submissions(files: list[Path], weights: list[float], output: Path):
    """Merge multiple submissions with weighted average."""
    print(f"\n{'='*60}")
    print(f"Merge {len(files)} files")
    print(f"{'='*60}\n")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    print(f"Weights (normalized): {weights}")

    # Load all files
    all_scores = []
    all_keys = set()

    for i, (file, weight) in enumerate(zip(files, weights)):
        print(f"\nLoading {file.name} (weight={weight:.3f})...")
        scores = load_submission(file)
        all_scores.append(scores)
        all_keys.update(scores.keys())
        print(f"  Loaded {len(scores):,} pairs")

    print(f"\nTotal unique (protein, term) pairs: {len(all_keys):,}")

    # Merge with weighted average
    print(f"\nMerging...")
    merged = {}

    for key in all_keys:
        total_weight = 0
        weighted_sum = 0

        for scores, weight in zip(all_scores, weights):
            if key in scores:
                weighted_sum += scores[key] * weight
                total_weight += weight

        if total_weight > 0:
            merged[key] = weighted_sum / total_weight

    print(f"  Merged pairs: {len(merged):,}")

    # Save
    print(f"\nSaving to {output}...")
    with open(output, 'w') as f:
        for (protein, term), score in sorted(merged.items()):
            f.write(f"{protein}\t{term}\t{score:.6f}\n")

    print(f"  Done!")

    # Stats
    scores_array = np.array(list(merged.values()))
    print(f"\n統計:")
    print(f"  行数: {len(merged):,}")
    print(f"  mean: {np.mean(scores_array):.4f}")
    print(f"  >=0.9: {np.sum(scores_array >= 0.9):,}")

    # Predict LB
    pred_lb = -0.6221 * np.mean(scores_array) + 0.5798
    print(f"\n予測LB: {pred_lb:.3f}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Merge submission files')
    parser.add_argument('files', nargs='+', help='Input files')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument('-w', '--weights', nargs='+', type=float, help='Weights for each file')

    args = parser.parse_args()

    files = [Path(f) for f in args.files]

    # Default to equal weights
    if args.weights:
        weights = args.weights
        if len(weights) != len(files):
            print(f"Error: {len(weights)} weights for {len(files)} files")
            sys.exit(1)
    else:
        weights = [1.0] * len(files)

    output = Path(args.output)

    merge_submissions(files, weights, output)


if __name__ == '__main__':
    main()
