"""
Simple blend: Concatenate and let pandas merge.
Much more memory efficient.
"""
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

DT_PATH = SUBMISSIONS_DIR / 'dual_tower.tsv'
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'


def blend(dt_weight: float = 0.3):
    print(f"=== Simple Blend (DT weight={dt_weight}) ===\n")

    bl_weight = 1.0 - dt_weight

    # Read in chunks and compute weighted scores
    print("Processing Dual-Tower (chunked)...")
    dt_chunks = []
    for i, chunk in enumerate(pd.read_csv(DT_PATH, sep='\t', header=None,
                                           names=['protein', 'go', 'score'],
                                           chunksize=10_000_000)):
        chunk['score'] = chunk['score'] * dt_weight
        chunk['source'] = 'dt'
        dt_chunks.append(chunk[['protein', 'go', 'score']])
        print(f"  Chunk {i+1}: {len(chunk):,} rows", flush=True)

    print("Concatenating DT chunks...")
    dt = pd.concat(dt_chunks, ignore_index=True)
    print(f"  DT total: {len(dt):,} rows")
    del dt_chunks

    print("\nProcessing SOTA...")
    sota = pd.read_csv(SOTA_PATH, sep='\t', header=None, names=['protein', 'go', 'score'])
    sota = sota[sota['go'].str.startswith('GO:')]
    sota['score'] = sota['score'] * bl_weight / 2
    print(f"  SOTA: {len(sota):,} rows")

    print("Processing ESM2...")
    esm2 = pd.read_csv(ESM2_PATH, sep='\t', header=None, names=['protein', 'go', 'score'])
    esm2 = esm2[esm2['go'].str.startswith('GO:')]
    esm2['score'] = esm2['score'] * bl_weight / 2
    print(f"  ESM2: {len(esm2):,} rows")

    print("\nMerging all...")
    combined = pd.concat([dt, sota, esm2], ignore_index=True)
    del dt, sota, esm2
    print(f"  Combined: {len(combined):,} rows")

    print("Grouping by (protein, go) and summing scores...")
    merged = combined.groupby(['protein', 'go'], as_index=False)['score'].sum()
    del combined
    print(f"  Merged: {len(merged):,} rows")

    # Clip scores to [0, 1]
    merged['score'] = merged['score'].clip(0, 1)

    # Filter low scores
    merged = merged[merged['score'] >= 0.001]
    print(f"  After filtering: {len(merged):,} rows")

    # Save
    output_path = SUBMISSIONS_DIR / f'blend_dual_tower_{int(dt_weight*100)}.tsv'
    merged.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    blend(weight)
