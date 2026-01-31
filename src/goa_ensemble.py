#!/usr/bin/env python3
"""
GOA + ProtT5 Ensemble with GO Propagation
Target: Maximize F-max score while keeping file size < 100MB
"""
import os
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import heapq

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
COMPETITION_DATA = DATA_DIR
GOA_PATH = DATA_DIR / 'goa_dataset' / 'goa_submission.tsv'
PROTT5_PATH = DATA_DIR / 'goa_dataset' / 'prott5_interpro_predictions.tsv'
GO_OBO_PATH = COMPETITION_DATA / 'Train' / 'go-basic.obo'
OUTPUT_PATH = DATA_DIR.parent / 'submission.tsv'

# Weights
WEIGHT_GOA = 0.55
WEIGHT_PROTT5 = 0.45

# GO Roots
GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}

# Parameters
TOP_K = 18  # Balanced for file size
MIN_SCORE = 0.10  # Higher threshold
POWER_SCALE = 0.8
MAX_SCORE = 0.93


def load_test_ids():
    """Load test protein IDs"""
    test_fasta = COMPETITION_DATA / 'Test' / 'testsuperset.fasta'
    ids = set()
    with open(test_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip().split()[0]
                if '|' in header:
                    header = header.split('|')[1]
                ids.add(header)
    print(f"Loaded {len(ids):,} test proteins")
    return ids


def parse_go_ontology():
    """Parse GO ontology and build ancestor map"""
    print("Parsing GO ontology...")
    term_parents = defaultdict(set)

    with open(GO_OBO_PATH, 'r') as f:
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

    for term in tqdm(list(term_parents.keys()), desc="Building ancestors"):
        get_ancestors(term)

    print(f"Cached {len(ancestors_map):,} GO terms")
    return ancestors_map


def load_predictions(filepath, allowed_proteins, desc="Loading"):
    """Load predictions from TSV file"""
    preds = defaultdict(dict)

    with open(filepath, 'r') as f:
        for line in tqdm(f, desc=desc):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            protein = parts[0].strip()
            if '|' in protein:
                protein = protein.split('|')[1]

            if protein not in allowed_proteins:
                continue

            go_term = parts[1].strip()
            try:
                score = float(parts[2])
            except ValueError:
                continue

            if go_term in preds[protein]:
                preds[protein][go_term] = max(preds[protein][go_term], score)
            else:
                preds[protein][go_term] = score

    return dict(preds)


def merge_predictions(goa_preds, prott5_preds):
    """Merge GOA and ProtT5 with weighted average"""
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


def topk_filter(scores, k):
    """Keep only top K predictions"""
    return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


def main():
    print("=" * 60)
    print("GOA + ProtT5 Ensemble")
    print(f"TOP_K={TOP_K}, MIN_SCORE={MIN_SCORE}")
    print("=" * 60)

    # Load test IDs
    test_ids = load_test_ids()

    # Parse GO ontology
    ancestors_map = parse_go_ontology()

    # Load predictions
    print("\nLoading GOA predictions...")
    goa_preds = load_predictions(GOA_PATH, test_ids, "GOA")
    print(f"  GOA proteins: {len(goa_preds):,}")

    print("\nLoading ProtT5 predictions...")
    prott5_preds = load_predictions(PROTT5_PATH, test_ids, "ProtT5")
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

        # 2. Power scaling
        scaled = power_scaling(pos, POWER_SCALE, MAX_SCORE)

        # 3. Top-K filtering
        top_terms = topk_filter(scaled, TOP_K)

        # 4. Write predictions
        for go_term, score in top_terms:
            if score >= MIN_SCORE:
                output_lines.append(f"{pid}\t{go_term}\t{score:.4f}")

    # Save submission
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(output_lines))

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"Submission saved: {OUTPUT_PATH}")
    print(f"  Total predictions: {len(output_lines):,}")
    print(f"  Unique proteins: {len(merged):,}")
    print(f"  File size: {size_mb:.1f} MB")

    if size_mb > 100:
        print(f"  WARNING: File too large! Adjust TOP_K or MIN_SCORE")
    else:
        print(f"  OK: File size within limit")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
