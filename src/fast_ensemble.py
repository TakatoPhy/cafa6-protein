"""
Fast Ensemble using Polars (10x faster than pandas)
Target: 0.37+ score
"""
import polars as pl
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
GOA_DATA = DATA_DIR / 'goa_predictions'
GO_OBO_PATH = DATA_DIR / 'Train' / 'go-basic.obo'

# Config
WEIGHT_GOA = 0.60
WEIGHT_PROTT5 = 0.40
GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}
TOP_K = 270
MIN_SCORE = 0.001


def load_test_ids():
    """Load test protein IDs"""
    test_fasta = DATA_DIR / 'Test' / 'testsuperset.fasta'
    if not test_fasta.exists():
        for f in (DATA_DIR / 'Test').iterdir():
            if f.suffix in ['.fasta', '.fa']:
                test_fasta = f
                break

    if not test_fasta.exists():
        return None

    ids = set()
    with open(test_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip().split()[0]
                if '|' in header:
                    header = header.split('|')[1]
                ids.add(header)
    print(f"Test proteins: {len(ids):,}")
    return ids


def parse_go_ontology():
    """Parse GO ontology"""
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
                    term_parents[current_id].add(parts[2].strip())

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

    for term in term_parents.keys():
        get_ancestors(term)

    print(f"  GO terms: {len(ancestors_map):,}")
    return ancestors_map


def main():
    print("=" * 60)
    print("CAFA6 Fast Ensemble (Polars)")
    print("=" * 60)

    test_ids = load_test_ids()
    ancestors_map = parse_go_ontology()

    # Load GOA predictions with polars (much faster)
    print("\nLoading GOA predictions with Polars...")
    goa_df = pl.read_csv(
        GOA_DATA / 'goa_submission.tsv',
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    )
    print(f"  GOA rows: {len(goa_df):,}")

    # Filter to test proteins
    if test_ids:
        goa_df = goa_df.filter(pl.col('protein').is_in(list(test_ids)))
        print(f"  After filtering: {len(goa_df):,}")

    print("\nLoading ProtT5 predictions...")
    prott5_df = pl.read_csv(
        GOA_DATA / 'prott5_interpro_predictions.tsv',
        separator='\t',
        has_header=False,
        new_columns=['protein', 'go_term', 'score']
    )
    print(f"  ProtT5 rows: {len(prott5_df):,}")

    if test_ids:
        prott5_df = prott5_df.filter(pl.col('protein').is_in(list(test_ids)))
        print(f"  After filtering: {len(prott5_df):,}")

    # Merge with weighted average
    print("\nMerging predictions...")

    # Add source column
    goa_df = goa_df.with_columns(pl.lit('goa').alias('source'))
    prott5_df = prott5_df.with_columns(pl.lit('prott5').alias('source'))

    # Combine
    combined = pl.concat([goa_df, prott5_df])

    # Group by protein + go_term and compute weighted average
    merged = combined.group_by(['protein', 'go_term']).agg([
        (pl.col('score').filter(pl.col('source') == 'goa').max().fill_null(0) * WEIGHT_GOA +
         pl.col('score').filter(pl.col('source') == 'prott5').max().fill_null(0) * WEIGHT_PROTT5).alias('score_weighted'),
        pl.col('score').max().alias('score_max'),
        pl.col('source').n_unique().alias('n_sources')
    ])

    # Use weighted when both sources, else use max
    merged = merged.with_columns(
        pl.when(pl.col('n_sources') == 2)
        .then(pl.col('score_weighted'))
        .otherwise(pl.col('score_max'))
        .alias('final_score')
    )

    print(f"  Merged rows: {len(merged):,}")

    # Process by protein for propagation and top-k
    print("\nApplying propagation and top-K...")
    proteins = merged.select('protein').unique().to_series().to_list()
    print(f"  Proteins to process: {len(proteins):,}")

    output_rows = []
    for protein in tqdm(proteins, desc="Processing"):
        protein_df = merged.filter(pl.col('protein') == protein)
        scores = dict(zip(
            protein_df['go_term'].to_list(),
            protein_df['final_score'].to_list()
        ))

        # Positive propagation
        propagated = dict(scores)
        for term, score in scores.items():
            if term in GO_ROOTS:
                continue
            for anc in ancestors_map.get(term, ()):
                if anc not in propagated or propagated[anc] < score:
                    propagated[anc] = score

        # Roots = 1.0
        for root in GO_ROOTS:
            propagated[root] = 1.0

        # Negative propagation (alpha=0.7)
        alpha = 0.7
        final_scores = dict(propagated)
        for term, score in propagated.items():
            if term in GO_ROOTS:
                continue
            anc = ancestors_map.get(term, ())
            if not anc:
                continue
            min_anc = min([propagated.get(a, 1.0) for a in anc if a in propagated], default=1.0)
            if min_anc < score:
                final_scores[term] = alpha * min_anc + (1.0 - alpha) * score

        for root in GO_ROOTS:
            final_scores[root] = 1.0

        # Top-K
        sorted_terms = sorted(final_scores.items(), key=lambda x: -x[1])[:TOP_K]

        for go_term, score in sorted_terms:
            if score >= MIN_SCORE:
                output_rows.append((protein, go_term, score))

    # Save
    output_df = pl.DataFrame({
        'protein': [r[0] for r in output_rows],
        'go_term': [r[1] for r in output_rows],
        'score': [r[2] for r in output_rows]
    })

    output_path = DATA_DIR.parent / 'submission_fast.tsv'
    output_df.write_csv(output_path, separator='\t', include_header=False)

    print(f"\n{'=' * 60}")
    print(f"Submission saved: {output_path}")
    print(f"  Total predictions: {len(output_df):,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
