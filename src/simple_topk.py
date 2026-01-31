"""
Simple Top-K extraction from GOA submission
Assumes propagation is already applied in the source file
"""
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
GOA_PATH = DATA_DIR / 'goa_predictions' / 'goa_submission.tsv'

TOP_K = 270
MIN_SCORE = 0.001


def main():
    print("=" * 60)
    print("Simple Top-K Extraction")
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

    # Load GOA with lazy evaluation
    print("\nLoading GOA predictions...")
    df = pl.scan_csv(
        GOA_PATH,
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    ).filter(
        pl.col('protein').is_in(list(test_ids))
    ).filter(
        pl.col('score') >= MIN_SCORE
    ).collect()

    print(f"  Rows after filtering: {len(df):,}")

    # Sort and take top-K per protein
    print("\nApplying top-K filtering...")
    result = df.sort(['protein', 'score'], descending=[False, True]).group_by(
        'protein', maintain_order=True
    ).head(TOP_K)

    print(f"  Final rows: {len(result):,}")

    # Save
    output_path = DATA_DIR.parent / 'submission_goa_topk.tsv'
    result.write_csv(output_path, separator='\t', include_header=False)

    print(f"\n{'=' * 60}")
    print(f"Saved: {output_path}")
    print(f"  Rows: {len(result):,}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
