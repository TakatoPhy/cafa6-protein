"""
Optimized GO hierarchy expansion.
Avoids file size explosion by filtering low-confidence ancestors.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / 'data'

def parse_go_obo(filepath):
    """Parse GO OBO file and extract is_a relationships."""
    term_to_parents = defaultdict(list)

    current_term = None
    is_obsolete = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                current_term = None
                is_obsolete = False

            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]

            elif line.startswith('is_a: GO:'):
                parent = line.split('is_a: ')[1].split(' !')[0]
                if current_term and not is_obsolete:
                    term_to_parents[current_term].append(parent)

            elif line == 'is_obsolete: true':
                is_obsolete = True

    return term_to_parents

def get_all_ancestors(term, term_to_parents, cache=None):
    """Get all ancestors of a term."""
    if cache is None:
        cache = {}

    if term in cache:
        return cache[term]

    ancestors = set()
    for parent in term_to_parents.get(term, []):
        ancestors.add(parent)
        ancestors.update(get_all_ancestors(parent, term_to_parents, cache))

    cache[term] = ancestors
    return ancestors

def expand_submission_optimized(input_path, output_path, term_to_parents,
                                 min_conf=0.03, max_terms_per_protein=100):
    """
    Expand submission with GO ancestors.

    Optimizations:
    - min_conf: Minimum confidence for ancestors (default 0.03)
    - max_terms_per_protein: Max terms per protein to avoid explosion
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, sep='\t', header=None,
                     names=['EntryID', 'term', 'confidence'])

    print(f"Original predictions: {len(df)}")

    # Build ancestor cache
    cache = {}

    # Group by protein
    protein_predictions = defaultdict(dict)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
        protein_predictions[row['EntryID']][row['term']] = row['confidence']

    # Expand with ancestors
    expanded_rows = []
    for protein_id, terms_conf in tqdm(protein_predictions.items(), desc="Expanding"):
        all_terms = {}

        for term, conf in terms_conf.items():
            # Add original term
            if term not in all_terms or conf > all_terms[term]:
                all_terms[term] = conf

            # Add ancestors (inherit confidence from child)
            ancestors = get_all_ancestors(term, term_to_parents, cache)
            for anc in ancestors:
                # Only add if confidence is high enough
                if conf >= min_conf:
                    if anc not in all_terms or conf > all_terms[anc]:
                        all_terms[anc] = conf

        # Sort by confidence and limit
        sorted_terms = sorted(all_terms.items(), key=lambda x: -x[1])
        if len(sorted_terms) > max_terms_per_protein:
            sorted_terms = sorted_terms[:max_terms_per_protein]

        for term, conf in sorted_terms:
            expanded_rows.append({
                'EntryID': protein_id,
                'term': term,
                'confidence': round(conf, 3)
            })

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df.sort_values(['EntryID', 'confidence'], ascending=[True, False])

    print(f"Expanded predictions: {len(expanded_df)}")

    expanded_df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Saved to {output_path}")

    # Check file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")

    return expanded_df

def main():
    print("Parsing GO ontology...")
    term_to_parents = parse_go_obo(DATA_DIR / 'Train' / 'go-basic.obo')
    print(f"Terms with parents: {len(term_to_parents)}")

    # Expand the ESM2 submission
    input_path = DATA_DIR.parent / 'submission.tsv'
    output_path = DATA_DIR.parent / 'submission_expanded.tsv'

    expand_submission_optimized(
        input_path,
        output_path,
        term_to_parents,
        min_conf=0.03,  # Filter very low confidence ancestors
        max_terms_per_protein=150  # Limit to top 150 terms per protein
    )

if __name__ == '__main__':
    main()
