"""
Multi-Model Ensemble for CAFA6
Merge multiple high-scoring submissions to improve final score.

Target: 0.387 → 0.40+ (Gold Medal: 0.408)
"""
import polars as pl
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse

# Paths
BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Submission files with their known/estimated scores
SUBMISSIONS = {
    'sota_27jan': {
        'path': NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv',
        'score': 0.386,
        'weight': 1.0,
    },
    'esm2_0386': {
        'path': NOTEBOOKS_DIR / 'esm2_0386_output' / 'submission.tsv',
        'score': 0.386,
        'weight': 1.0,
    },
    'cafa_ensemble_378': {
        'path': NOTEBOOKS_DIR / 'cafa_ensemble_378_output' / 'submission.tsv',
        'score': 0.378,
        'weight': 0.8,
    },
    'ktdk': {
        'path': NOTEBOOKS_DIR / 'ktdk_output' / 'submission.tsv',
        'score': None,  # Unknown, but high votes
        'weight': 0.9,
    },
    'cafa_tuning': {
        'path': NOTEBOOKS_DIR / 'cafa_tuning_output' / 'submission.tsv',
        'score': None,
        'weight': 0.7,
    },
    'goa_propagation': {
        'path': NOTEBOOKS_DIR / 'goa_propagation_output' / 'submission.tsv',
        'score': None,
        'weight': 0.7,
    },
    'cafa6_submission': {
        'path': NOTEBOOKS_DIR / 'cafa6_submission' / 'submission.tsv',
        'score': None,
        'weight': 0.6,
    },
}


def load_submission(path: Path, name: str) -> pl.DataFrame:
    """Load a submission TSV file."""
    print(f"  Loading {name}...")
    df = pl.read_csv(
        path,
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    )
    print(f"    Rows: {len(df):,}, Proteins: {df['protein'].n_unique():,}")
    return df


def simple_average(submissions: list[str]) -> pl.DataFrame:
    """Simple average of all submissions."""
    print("\n=== Simple Average Ensemble ===")

    all_dfs = []
    for name in submissions:
        if name not in SUBMISSIONS:
            print(f"  Warning: {name} not found, skipping")
            continue
        info = SUBMISSIONS[name]
        if not info['path'].exists():
            print(f"  Warning: {info['path']} not found, skipping")
            continue

        df = load_submission(info['path'], name)
        df = df.with_columns(pl.lit(name).alias('source'))
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid submissions found")

    print(f"\nMerging {len(all_dfs)} submissions...")
    combined = pl.concat(all_dfs)

    # Group by protein + go_term, average scores
    merged = combined.group_by(['protein', 'go_term']).agg([
        pl.col('score').mean().alias('avg_score'),
        pl.col('source').n_unique().alias('n_sources'),
    ])

    print(f"  Unique predictions: {len(merged):,}")
    print(f"  Multi-source predictions: {len(merged.filter(pl.col('n_sources') > 1)):,}")

    return merged.select(['protein', 'go_term', 'avg_score']).rename({'avg_score': 'score'})


def weighted_average(submissions: list[str], custom_weights: dict = None) -> pl.DataFrame:
    """Weighted average of submissions based on their known scores."""
    print("\n=== Weighted Average Ensemble ===")

    all_dfs = []
    weights = {}

    for name in submissions:
        if name not in SUBMISSIONS:
            print(f"  Warning: {name} not found, skipping")
            continue
        info = SUBMISSIONS[name]
        if not info['path'].exists():
            print(f"  Warning: {info['path']} not found, skipping")
            continue

        df = load_submission(info['path'], name)
        df = df.with_columns(pl.lit(name).alias('source'))
        all_dfs.append(df)

        # Use custom weight if provided, else use default
        w = custom_weights.get(name, info['weight']) if custom_weights else info['weight']
        weights[name] = w
        print(f"    Weight: {w:.2f}")

    if not all_dfs:
        raise ValueError("No valid submissions found")

    print(f"\nMerging {len(all_dfs)} submissions with weights...")
    print(f"  Weights: {weights}")

    combined = pl.concat(all_dfs)

    # Add weight column
    combined = combined.with_columns(
        pl.col('source').replace(weights).alias('weight')
    )

    # Weighted average
    merged = combined.group_by(['protein', 'go_term']).agg([
        (pl.col('score') * pl.col('weight')).sum().alias('weighted_sum'),
        pl.col('weight').sum().alias('weight_sum'),
        pl.col('source').n_unique().alias('n_sources'),
    ])

    merged = merged.with_columns(
        (pl.col('weighted_sum') / pl.col('weight_sum')).alias('score')
    )

    print(f"  Unique predictions: {len(merged):,}")

    return merged.select(['protein', 'go_term', 'score'])


def rank_average(submissions: list[str]) -> pl.DataFrame:
    """Rank-based averaging (more robust to outliers)."""
    print("\n=== Rank Average Ensemble ===")

    all_dfs = []
    for name in submissions:
        if name not in SUBMISSIONS:
            continue
        info = SUBMISSIONS[name]
        if not info['path'].exists():
            continue

        df = load_submission(info['path'], name)

        # Convert scores to ranks per protein
        df = df.with_columns([
            pl.col('score').rank(method='average', descending=True).over('protein').alias('rank'),
            pl.lit(name).alias('source')
        ])
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid submissions found")

    print(f"\nMerging {len(all_dfs)} submissions by rank...")
    combined = pl.concat(all_dfs)

    # Average ranks, then convert back to scores (lower rank = higher score)
    merged = combined.group_by(['protein', 'go_term']).agg([
        pl.col('rank').mean().alias('avg_rank'),
        pl.col('score').mean().alias('avg_score'),
        pl.col('source').n_unique().alias('n_sources'),
    ])

    # Normalize: lower rank -> higher score
    max_rank = merged['avg_rank'].max()
    merged = merged.with_columns(
        (1.0 - pl.col('avg_rank') / max_rank).alias('rank_score')
    )

    # Blend rank score with average score
    merged = merged.with_columns(
        (0.5 * pl.col('rank_score') + 0.5 * pl.col('avg_score')).alias('score')
    )

    print(f"  Unique predictions: {len(merged):,}")

    return merged.select(['protein', 'go_term', 'score'])


def filter_and_save(df: pl.DataFrame, output_path: Path, top_k: int = 500, min_score: float = 0.001):
    """Filter predictions and save submission."""
    print(f"\nFiltering predictions...")

    # Filter by minimum score
    df = df.filter(pl.col('score') >= min_score)

    # Keep top-K per protein
    df = df.sort(['protein', 'score'], descending=[False, True])
    df = df.group_by('protein').head(top_k)

    # Clip scores to [0, 1]
    df = df.with_columns(
        pl.col('score').clip(0.0, 1.0)
    )

    # Sort for consistent output
    df = df.sort(['protein', 'score'], descending=[False, True])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path, separator='\t', include_header=False)

    print(f"  Saved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Proteins: {df['protein'].n_unique():,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return df


def main():
    parser = argparse.ArgumentParser(description='Multi-model ensemble for CAFA6')
    parser.add_argument('--method', choices=['simple', 'weighted', 'rank'], default='weighted',
                        help='Ensemble method')
    parser.add_argument('--submissions', nargs='+',
                        default=['sota_27jan', 'esm2_0386', 'cafa_ensemble_378'],
                        help='Submissions to ensemble')
    parser.add_argument('--top-k', type=int, default=500, help='Top-K predictions per protein')
    parser.add_argument('--output', type=str, default='submission_ensemble.tsv', help='Output filename')
    args = parser.parse_args()

    print("=" * 60)
    print("CAFA6 Multi-Model Ensemble")
    print("=" * 60)
    print(f"\nMethod: {args.method}")
    print(f"Submissions: {args.submissions}")

    # Run ensemble
    if args.method == 'simple':
        result = simple_average(args.submissions)
    elif args.method == 'weighted':
        result = weighted_average(args.submissions)
    elif args.method == 'rank':
        result = rank_average(args.submissions)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save
    output_path = OUTPUT_DIR / args.output
    filter_and_save(result, output_path, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
