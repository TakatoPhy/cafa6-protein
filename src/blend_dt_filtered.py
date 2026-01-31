"""
Blend with filtering: Only keep high confidence predictions.
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

DT_PATH = SUBMISSIONS_DIR / 'dual_tower.tsv'
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'

# Only keep DT predictions above this threshold
DT_THRESHOLD = 0.5


def blend(dt_weight: float = 0.3):
    print(f"=== Filtered Blend (DT weight={dt_weight}, threshold={DT_THRESHOLD}) ===\n")

    bl_weight = 1.0 - dt_weight

    # Load baseline
    print("Loading baseline...")
    baseline = defaultdict(dict)

    count = 0
    for path, name in [(SOTA_PATH, 'SOTA'), (ESM2_PATH, 'ESM2')]:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    protein, go_term, score = parts[0], parts[1], float(parts[2])
                    if go_term.startswith('GO:'):
                        if go_term not in baseline[protein]:
                            baseline[protein][go_term] = []
                        baseline[protein][go_term].append(score)
                        count += 1
        print(f"  {name}: loaded", flush=True)

    # Average baseline
    for protein in baseline:
        for term in baseline[protein]:
            baseline[protein][term] = sum(baseline[protein][term]) / len(baseline[protein][term])

    print(f"  Baseline: {len(baseline):,} proteins\n")

    # Stream DT and blend
    print(f"Streaming Dual-Tower (keeping >= {DT_THRESHOLD})...")

    output_path = SUBMISSIONS_DIR / f'blend_dual_tower_{int(dt_weight*100)}.tsv'
    blended = defaultdict(dict)

    dt_kept = 0
    dt_total = 0
    with open(DT_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, dt_score = parts[0], parts[1], float(parts[2])

                if not go_term.startswith('GO:'):
                    continue

                dt_total += 1

                # Only keep high confidence DT predictions
                if dt_score >= DT_THRESHOLD:
                    bl_score = baseline.get(protein, {}).get(go_term, 0)

                    if bl_score > 0:
                        score = dt_score * dt_weight + bl_score * bl_weight
                    else:
                        score = dt_score * dt_weight

                    if go_term not in blended[protein] or blended[protein][go_term] < score:
                        blended[protein][go_term] = score

                    dt_kept += 1

                if dt_total % 50_000_000 == 0:
                    print(f"  {dt_total:,} rows, kept {dt_kept:,}...", flush=True)

    print(f"  DT total: {dt_total:,}, kept: {dt_kept:,} ({dt_kept/dt_total*100:.1f}%)\n")

    # Add baseline entries not in blended
    print("Adding remaining baseline entries...")
    for protein in baseline:
        for term, bl_score in baseline[protein].items():
            if term not in blended.get(protein, {}):
                blended[protein][term] = bl_score * bl_weight

    # Write output
    print("Writing output...")
    row_count = 0
    with open(output_path, 'w') as f:
        for protein in sorted(blended.keys()):
            for term, score in blended[protein].items():
                if score >= 0.001:
                    f.write(f"{protein}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    blend(weight)
