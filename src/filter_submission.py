#!/usr/bin/env python3
"""
Filter submission to meet Kaggle file size limit (< 100MB).
Uses Top-K per protein with score threshold.
"""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
INPUT_PATH = DATA_DIR / 'processed' / 'cafa_ensemble_378' / 'submission.tsv'
OUTPUT_PATH = DATA_DIR.parent / 'submission_filtered.tsv'

# Parameters to tune
TOP_K = 15  # Max predictions per protein (reduced for size)
MIN_SCORE = 0.10  # Minimum score threshold
TARGET_SIZE_MB = 95  # Target file size


def main():
    print("=" * 60)
    print("Filter Submission for Kaggle")
    print("=" * 60)

    # Load submission
    print(f"\nLoading {INPUT_PATH}...")
    df = pl.read_csv(
        INPUT_PATH,
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    )
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique proteins: {df.select('protein').unique().height:,}")

    # Filter by score threshold
    print(f"\nFiltering by score >= {MIN_SCORE}...")
    df = df.filter(pl.col('score') >= MIN_SCORE)
    print(f"  Rows after filter: {len(df):,}")

    # Top-K per protein
    print(f"\nSelecting top {TOP_K} per protein...")
    df = df.sort(['protein', 'score'], descending=[False, True])
    df = df.group_by('protein', maintain_order=True).head(TOP_K)
    print(f"  Rows after top-k: {len(df):,}")

    # Round scores for smaller file
    df = df.with_columns(pl.col('score').round(4))

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df.write_csv(OUTPUT_PATH, separator='\t', include_header=False)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"  Rows: {len(df):,}")
    print(f"  Size: {size_mb:.1f} MB")

    if size_mb > 100:
        print(f"\n  WARNING: File too large ({size_mb:.1f} MB > 100 MB)")
        print(f"  Try reducing TOP_K or increasing MIN_SCORE")
    else:
        print(f"\n  OK: File size within limit")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
