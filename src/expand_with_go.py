"""
Expand predictions using GO hierarchy.
If we predict a child term, also predict all its ancestors.
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

def expand_submission(input_path, output_path, term_to_parents):
    """Expand submission by adding ancestor terms."""
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, sep='\t', header=None, names=['EntryID', 'term', 'confidence'])

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

            # Add ancestors with same confidence
            ancestors = get_all_ancestors(term, term_to_parents, cache)
            for anc in ancestors:
                if anc not in all_terms or conf > all_terms[anc]:
                    all_terms[anc] = conf

        for term, conf in all_terms.items():
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

    return expanded_df

def main():
    print("Parsing GO ontology...")
    term_to_parents = parse_go_obo(DATA_DIR / 'Train' / 'go-basic.obo')
    print(f"Terms with parents: {len(term_to_parents)}")

    # Use the original submission
    input_path = DATA_DIR.parent / 'submission_final.tsv'
    output_path = DATA_DIR.parent / 'submission_expanded.tsv'

    if not input_path.exists():
        input_path = DATA_DIR.parent / 'submission.tsv'

    expand_submission(input_path, output_path, term_to_parents)

if __name__ == '__main__':
    main()
