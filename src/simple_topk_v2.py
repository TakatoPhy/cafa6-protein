"""
Simple Top-K extraction with stricter filtering
Target: < 100MB submission file
"""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
GOA_PATH = DATA_DIR / 'goa_predictions' / 'goa_submission.tsv'
PROTT5_PATH = DATA_DIR / 'goa_predictions' / 'prott5_interpro_predictions.tsv'

# Stricter parameters for smaller file
TOP_K = 200  # Reduced from 270
MIN_SCORE = 0.05  # Increased from 0.001


def main():
    print("=" * 60)
    print("Top-K Extraction v2 (Stricter Filtering)")
    print("=" * 60)

    # Load test IDs
    test_fasta = DATA_DIR / 'Test' / 'testsuperset.fasta'
    test_ids = set()
    with open(test_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip().split()[0]
                if '|' in header:
                    header = header.split('|')[1]
                test_ids.add(header)
    print(f"Test proteins: {len(test_ids):,}")

    # Load GOA
    print("\nLoading GOA predictions...")
    goa = pl.scan_csv(
        GOA_PATH,
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    ).filter(
        pl.col('protein').is_in(list(test_ids))
    ).filter(
        pl.col('score') >= MIN_SCORE
    ).collect()
    print(f"  GOA rows: {len(goa):,}")

    # Load ProtT5
    print("\nLoading ProtT5 predictions...")
    prott5 = pl.scan_csv(
        PROTT5_PATH,
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    ).filter(
        pl.col('protein').is_in(list(test_ids))
    ).filter(
        pl.col('score') >= MIN_SCORE
    ).collect()
    print(f"  ProtT5 rows: {len(prott5):,}")

    # Combine with weighted average (GOA 60%, ProtT5 40%)
    print("\nMerging with weighted average...")
    goa = goa.with_columns(pl.lit('goa').alias('source'))
    prott5 = prott5.with_columns(pl.lit('prott5').alias('source'))

    combined = pl.concat([goa, prott5])

    # Group and merge
    merged = combined.group_by(['protein', 'go_term']).agg([
        (pl.col('score').filter(pl.col('source') == 'goa').max().fill_null(0) * 0.6 +
         pl.col('score').filter(pl.col('source') == 'prott5').max().fill_null(0) * 0.4).alias('weighted_score'),
        pl.col('score').max().alias('max_score'),
        pl.col('source').n_unique().alias('n_sources')
    ]).with_columns(
        pl.when(pl.col('n_sources') == 2)
        .then(pl.col('weighted_score'))
        .otherwise(pl.col('max_score'))
        .alias('final_score')
    ).select(['protein', 'go_term', 'final_score']).rename({'final_score': 'score'})

    print(f"  Merged rows: {len(merged):,}")

    # Sort and take top-K per protein
    print("\nApplying top-K filtering...")
    result = merged.sort(['protein', 'score'], descending=[False, True]).group_by(
        'protein', maintain_order=True
    ).head(TOP_K)

    print(f"  Final rows: {len(result):,}")

    # Round scores
    result = result.with_columns(
        pl.col('score').round(6)
    )

    # Save
    output_path = DATA_DIR.parent / 'submission_v2.tsv'
    result.write_csv(output_path, separator='\t', include_header=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(result):,}")
    print(f"  Size: {size_mb:.1f} MB")

    # Check unique proteins
    unique_proteins = result.select('protein').unique().height
    print(f"  Unique proteins: {unique_proteins:,}")
    print(f"{'=' * 60}")

    if size_mb > 100:
        print(f"\nWARNING: File > 100MB. May need stricter filtering.")


if __name__ == '__main__':
    main()
