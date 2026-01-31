"""
Top-K extraction v3 - Target < 100MB
"""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
GOA_PATH = DATA_DIR / 'goa_predictions' / 'goa_submission.tsv'
PROTT5_PATH = DATA_DIR / 'goa_predictions' / 'prott5_interpro_predictions.tsv'

# Very strict parameters
TOP_K = 150
MIN_SCORE = 0.15


def main():
    print("=" * 60)
    print(f"Top-K Extraction v3 (TOP_K={TOP_K}, MIN_SCORE={MIN_SCORE})")
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

    # Combine (GOA 55%, ProtT5 45%)
    print("\nMerging...")
    goa = goa.with_columns(pl.lit('goa').alias('source'))
    prott5 = prott5.with_columns(pl.lit('prott5').alias('source'))

    combined = pl.concat([goa, prott5])

    merged = combined.group_by(['protein', 'go_term']).agg([
        (pl.col('score').filter(pl.col('source') == 'goa').max().fill_null(0) * 0.55 +
         pl.col('score').filter(pl.col('source') == 'prott5').max().fill_null(0) * 0.45).alias('weighted'),
        pl.col('score').max().alias('max_score'),
        pl.col('source').n_unique().alias('n')
    ]).with_columns(
        pl.when(pl.col('n') == 2)
        .then(pl.col('weighted'))
        .otherwise(pl.col('max_score'))
        .alias('score')
    ).select(['protein', 'go_term', 'score'])

    print(f"  Merged rows: {len(merged):,}")

    # Top-K
    result = merged.sort(['protein', 'score'], descending=[False, True]).group_by(
        'protein', maintain_order=True
    ).head(TOP_K).with_columns(pl.col('score').round(4))

    print(f"  Final rows: {len(result):,}")

    # Save
    output_path = DATA_DIR.parent / 'submission_v3.tsv'
    result.write_csv(output_path, separator='\t', include_header=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(result):,}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Unique proteins: {result.select('protein').unique().height:,}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
