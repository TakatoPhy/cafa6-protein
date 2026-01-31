"""
Memory-efficient Blend: Dual-Tower + Public Notebooks.
Uses streaming approach to handle large files.
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

# Input files
DUAL_TOWER_PATH = SUBMISSIONS_DIR / 'dual_tower.tsv'
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'


def load_baseline(sota_path: Path, esm2_path: Path) -> dict:
    """Load SOTA + ESM2 as baseline (smaller files, fit in memory)."""
    print("Loading baseline (SOTA + ESM2)...")

    baseline = defaultdict(dict)

    # Load SOTA
    count = 0
    with open(sota_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if go_term.startswith('GO:'):
                    if go_term not in baseline[protein]:
                        baseline[protein][go_term] = []
                    baseline[protein][go_term].append(score)
                    count += 1
                    if count % 10_000_000 == 0:
                        print(f"  SOTA: {count:,} rows...", flush=True)
    print(f"  SOTA: {count:,} rows")

    # Load ESM2
    count = 0
    with open(esm2_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if go_term.startswith('GO:'):
                    if go_term not in baseline[protein]:
                        baseline[protein][go_term] = []
                    baseline[protein][go_term].append(score)
                    count += 1
                    if count % 10_000_000 == 0:
                        print(f"  ESM2: {count:,} rows...", flush=True)
    print(f"  ESM2: {count:,} rows")

    # Compute average
    print("  Computing baseline average...")
    for protein in baseline:
        for term in baseline[protein]:
            baseline[protein][term] = sum(baseline[protein][term]) / len(baseline[protein][term])

    print(f"  Baseline: {len(baseline):,} proteins")
    return baseline


def stream_blend(dual_tower_weight: float = 0.3):
    """Stream dual tower and blend with preloaded baseline."""
    print(f"\n=== Stream Blend Dual-Tower (weight={dual_tower_weight}) ===\n")

    # Load baseline first
    baseline = load_baseline(SOTA_PATH, ESM2_PATH)
    baseline_weight = 1.0 - dual_tower_weight

    # Get all baseline proteins
    baseline_proteins = set(baseline.keys())

    # Stream dual tower and blend
    print(f"\nStreaming Dual-Tower and blending...")

    output_path = SUBMISSIONS_DIR / f'blend_dual_tower_{int(dual_tower_weight*100)}.tsv'

    seen_pairs = set()  # Track (protein, term) pairs we've written
    row_count = 0
    dt_count = 0

    with open(output_path, 'w') as out_f:
        # First pass: stream dual tower and blend
        with open(DUAL_TOWER_PATH, 'r') as in_f:
            for line in in_f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    protein, go_term, dt_score = parts[0], parts[1], float(parts[2])

                    if not go_term.startswith('GO:'):
                        continue

                    dt_count += 1
                    if dt_count % 50_000_000 == 0:
                        print(f"  DT: {dt_count:,} rows, written: {row_count:,}...", flush=True)

                    # Get baseline score
                    bl_score = baseline.get(protein, {}).get(go_term, 0)

                    # Blend
                    if bl_score > 0:
                        score = dt_score * dual_tower_weight + bl_score * baseline_weight
                    else:
                        score = dt_score

                    if score >= 0.001:
                        out_f.write(f"{protein}\t{go_term}\t{score:.6f}\n")
                        seen_pairs.add((protein, go_term))
                        row_count += 1

        print(f"  DT pass done: {dt_count:,} rows, written: {row_count:,}")

        # Second pass: write baseline entries not in dual tower
        print("  Writing remaining baseline entries...")
        for protein in baseline:
            for term, bl_score in baseline[protein].items():
                if (protein, term) not in seen_pairs:
                    if bl_score >= 0.001:
                        out_f.write(f"{protein}\t{term}\t{bl_score:.6f}\n")
                        row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    stream_blend(weight)
