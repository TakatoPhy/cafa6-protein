"""
Blend ProtBoost predictions with existing best submission.
Strategy: Use ProtBoost for GO terms it covers, fill the rest from public notebooks.
"""
from pathlib import Path
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / 'notebooks' / 'external'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

# Input files
PROTBOOST_PATH = SUBMISSIONS_DIR / 'protboost_simple.tsv'
SOTA_PATH = NOTEBOOKS_DIR / 'sota_27jan_output' / 'submission.tsv'
ESM2_PATH = NOTEBOOKS_DIR / 'esm2_3785_output' / 'submission.tsv'


def load_submission(path: Path, name: str) -> dict:
    """Load submission as dict[protein][go_term] = score."""
    print(f"Loading {name}...")

    scores = defaultdict(dict)
    count = 0

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])

                if not go_term.startswith('GO:'):
                    continue

                if go_term not in scores[protein] or scores[protein][go_term] < score:
                    scores[protein][go_term] = score

                count += 1
                if count % 10_000_000 == 0:
                    print(f"  {count:,} rows...", flush=True)

    print(f"  Loaded {count:,} rows, {len(scores):,} proteins")
    return scores


def blend_submissions(protboost_weight: float = 0.3):
    """Blend ProtBoost with SOTA+ESM2 average."""
    print(f"\n=== Blend ProtBoost (weight={protboost_weight}) ===\n")

    # Load all submissions
    protboost = load_submission(PROTBOOST_PATH, "ProtBoost")
    sota = load_submission(SOTA_PATH, "SOTA 27Jan")
    esm2 = load_submission(ESM2_PATH, "ESM2 3785")

    # Get ProtBoost GO terms
    protboost_terms = set()
    for protein_scores in protboost.values():
        protboost_terms.update(protein_scores.keys())
    print(f"\nProtBoost covers {len(protboost_terms)} GO terms")

    # Get all proteins
    all_proteins = set(protboost.keys()) | set(sota.keys()) | set(esm2.keys())
    print(f"Total proteins: {len(all_proteins):,}")

    # Blend
    output_path = SUBMISSIONS_DIR / f'blend_protboost_{int(protboost_weight*100)}.tsv'

    row_count = 0
    with open(output_path, 'w') as f:
        for i, protein in enumerate(sorted(all_proteins)):
            if i % 50000 == 0:
                print(f"  Processing protein {i:,}/{len(all_proteins):,}...", flush=True)

            # Get all GO terms for this protein from all sources
            all_terms = set()
            if protein in protboost:
                all_terms.update(protboost[protein].keys())
            if protein in sota:
                all_terms.update(sota[protein].keys())
            if protein in esm2:
                all_terms.update(esm2[protein].keys())

            for term in all_terms:
                # Get scores from each source
                pb_score = protboost.get(protein, {}).get(term, 0)
                sota_score = sota.get(protein, {}).get(term, 0)
                esm2_score = esm2.get(protein, {}).get(term, 0)

                # SOTA + ESM2 average
                baseline_scores = [s for s in [sota_score, esm2_score] if s > 0]
                if baseline_scores:
                    baseline = sum(baseline_scores) / len(baseline_scores)
                else:
                    baseline = 0

                # Blend strategy:
                # - If ProtBoost has prediction: weighted blend
                # - Otherwise: use baseline
                if pb_score > 0 and baseline > 0:
                    score = pb_score * protboost_weight + baseline * (1 - protboost_weight)
                elif pb_score > 0:
                    score = pb_score
                else:
                    score = baseline

                if score >= 0.001:
                    f.write(f"{protein}\t{term}\t{min(score, 1.0):.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    blend_submissions(weight)
