"""
CAFA6 Ensemble: GOA + ProtT5 with Advanced Propagation
Target: 0.37+ score (based on top public notebooks)
"""
import os
import heapq
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
COMPETITION_DATA = DATA_DIR
GOA_DATA = DATA_DIR / 'goa_predictions'

GO_OBO_PATH = COMPETITION_DATA / 'Train' / 'go-basic.obo'
GOA_PREDICTIONS = GOA_DATA / 'goa_submission.tsv'
PROTT5_PREDICTIONS = GOA_DATA / 'prott5_interpro_predictions.tsv'

# Weights
WEIGHT_GOA = 0.60
WEIGHT_PROTT5 = 0.40

# GO Roots
GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}

# Hyperparameters
NEG_PROP_ALPHA = 0.70
SCALING_POWER = 0.80
MAX_SCORE = 0.93
TOP_K = 270
MIN_SCORE = 0.001


def load_test_ids():
    """Load test protein IDs from FASTA"""
    test_fasta = COMPETITION_DATA / 'Test' / 'testsuperset.fasta'
    if not test_fasta.exists():
        # Try alternative names
        test_dir = COMPETITION_DATA / 'Test'
        for f in test_dir.iterdir():
            if f.suffix in ['.fasta', '.fa']:
                test_fasta = f
                break

    if not test_fasta.exists():
        print("Test FASTA not found, loading all proteins")
        return None

    ids = set()
    with open(test_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip()
                token = header.split()[0]
                # Handle UniProt format: sp|A0A0C5B5G6|MOTSC_HUMAN
                if '|' in token:
                    parts = token.split('|')
                    if len(parts) >= 2:
                        token = parts[1]
                ids.add(token)

    print(f"Loaded {len(ids):,} test protein IDs")
    return ids


def parse_go_ontology(obo_path):
    """Parse GO ontology and build ancestor map"""
    print("Parsing GO ontology...")
    term_parents = defaultdict(set)

    with open(obo_path, 'r') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('id: '):
                current_id = line.split('id: ')[1].strip()
            elif line.startswith('is_a: ') and current_id:
                parent = line.split()[1].strip()
                term_parents[current_id].add(parent)
            elif line.startswith('relationship: part_of ') and current_id:
                parts = line.split()
                if len(parts) >= 3:
                    parent = parts[2].strip()
                    term_parents[current_id].add(parent)

    # Build ancestor cache
    ancestors_map = {}

    def get_ancestors(term):
        if term in ancestors_map:
            return ancestors_map[term]
        parents = term_parents.get(term, set())
        all_anc = set(parents)
        for p in parents:
            all_anc |= get_ancestors(p)
        ancestors_map[term] = all_anc
        return all_anc

    # Pre-compute all ancestors
    for term in tqdm(list(term_parents.keys()), desc="Building ancestors"):
        get_ancestors(term)

    print(f"Cached {len(ancestors_map):,} GO terms")
    return ancestors_map


def load_predictions(filepath, allowed_proteins=None, desc="Loading"):
    """Load predictions from TSV file"""
    preds = defaultdict(dict)

    with open(filepath, 'r') as f:
        for line in tqdm(f, desc=desc):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            protein = parts[0].strip()
            # Handle UniProt format
            if '|' in protein:
                parts_id = protein.split('|')
                if len(parts_id) >= 2:
                    protein = parts_id[1]

            if allowed_proteins is not None and protein not in allowed_proteins:
                continue

            go_term = parts[1].strip()
            try:
                score = float(parts[2])
            except ValueError:
                continue

            # Keep max score if duplicate
            if go_term in preds[protein]:
                preds[protein][go_term] = max(preds[protein][go_term], score)
            else:
                preds[protein][go_term] = score

    return dict(preds)


def merge_predictions(goa_preds, prott5_preds):
    """Merge GOA and ProtT5 predictions with weighted average"""
    print("Merging predictions...")
    all_proteins = set(goa_preds.keys()) | set(prott5_preds.keys())

    merged = {}
    for pid in tqdm(all_proteins, desc="Merging"):
        a = goa_preds.get(pid, {})
        b = prott5_preds.get(pid, {})

        if not a and not b:
            continue

        terms = set(a.keys()) | set(b.keys())
        merged[pid] = {}

        for t in terms:
            s1 = a.get(t, 0.0)
            s2 = b.get(t, 0.0)

            if s1 > 0.0 and s2 > 0.0:
                merged[pid][t] = WEIGHT_GOA * s1 + WEIGHT_PROTT5 * s2
            else:
                merged[pid][t] = s1 if s1 > 0.0 else s2

    return merged


def positive_propagation(base_scores, ancestors_map):
    """Propagate scores to ancestor terms"""
    upd = dict(base_scores)

    for term, score in base_scores.items():
        if term in GO_ROOTS:
            continue
        for anc in ancestors_map.get(term, ()):
            prev = upd.get(anc)
            if prev is None or score > prev:
                upd[anc] = score

    # Roots always 1.0
    for root in GO_ROOTS:
        upd[root] = 1.0

    return upd


def negative_propagation(pos_scores, ancestors_map, alpha=0.70):
    """Adjust child scores based on parent scores"""
    if alpha <= 0.0:
        return dict(pos_scores)

    out = dict(pos_scores)

    for term, score in pos_scores.items():
        if term in GO_ROOTS:
            continue

        anc = ancestors_map.get(term, ())
        if not anc:
            continue

        # Find minimum ancestor score
        min_anc = None
        for a in anc:
            v = pos_scores.get(a)
            if v is not None:
                if min_anc is None or v < min_anc:
                    min_anc = v

        # Adjust if child > min parent
        if min_anc is not None and min_anc < score:
            out[term] = alpha * min_anc + (1.0 - alpha) * score

    # Roots always 1.0
    for root in GO_ROOTS:
        out[root] = 1.0

    return out


def power_scaling(scores, power=0.80, max_score=0.93):
    """Apply power scaling to scores"""
    out = dict(scores)

    non_root = [s for t, s in out.items() if t not in GO_ROOTS]
    if not non_root:
        for r in GO_ROOTS:
            out[r] = 1.0
        return out

    mx = max(non_root)
    if mx <= 0.0 or mx >= max_score:
        for r in GO_ROOTS:
            out[r] = 1.0
        return out

    inv = 1.0 / mx
    for t in list(out.keys()):
        if t in GO_ROOTS:
            continue
        val = (out[t] * inv) ** power * max_score
        out[t] = min(1.0, val)

    for r in GO_ROOTS:
        out[r] = 1.0

    return out


def topk_filter(scores, k=270):
    """Keep only top K predictions"""
    if k <= 0:
        return []
    return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


def main():
    print("=" * 60)
    print("CAFA6 Ensemble: GOA + ProtT5")
    print("=" * 60)

    # Load test IDs
    test_ids = load_test_ids()

    # Parse GO ontology
    ancestors_map = parse_go_ontology(GO_OBO_PATH)

    # Load predictions
    print("\nLoading GOA predictions...")
    goa_preds = load_predictions(GOA_PREDICTIONS, test_ids, "GOA")
    print(f"  GOA proteins: {len(goa_preds):,}")

    print("\nLoading ProtT5 predictions...")
    prott5_preds = load_predictions(PROTT5_PREDICTIONS, test_ids, "ProtT5")
    print(f"  ProtT5 proteins: {len(prott5_preds):,}")

    # Merge
    merged = merge_predictions(goa_preds, prott5_preds)
    print(f"  Merged proteins: {len(merged):,}")

    # Free memory
    del goa_preds, prott5_preds

    # Process each protein
    print("\nProcessing proteins...")
    output_lines = []

    for pid in tqdm(merged, desc="Processing"):
        base = merged[pid]
        if not base:
            continue

        # 1. Positive propagation
        pos = positive_propagation(base, ancestors_map)

        # 2. Negative propagation
        neg = negative_propagation(pos, ancestors_map, NEG_PROP_ALPHA)

        # 3. Power scaling
        scaled = power_scaling(neg, SCALING_POWER, MAX_SCORE)

        # 4. Top-K filtering
        top_terms = topk_filter(scaled, TOP_K)

        # 5. Write predictions
        for go_term, score in top_terms:
            if score >= MIN_SCORE:
                output_lines.append(f"{pid}\t{go_term}\t{score:.6f}")

    # Save submission
    output_path = DATA_DIR.parent / 'submission_ensemble.tsv'
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\n{'=' * 60}")
    print(f"Submission saved: {output_path}")
    print(f"  Total predictions: {len(output_lines):,}")
    print(f"  Unique proteins: {len(merged):,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
