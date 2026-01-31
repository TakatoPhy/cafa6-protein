"""
CAFA6 Submission Optimization (Fast Version)
Based on antonoof/cafa-optimization approach
"""
import os
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from collections import defaultdict

# Paths
OBO_FILE = "data/Train/go-basic.obo"
GOA_FILE = "data/goa_predictions/goa_submission.tsv"
PROTT5_FILE = "data/goa_predictions/prott5_interpro_predictions.tsv"
OUTPUT_FILE = "submission_optimized.tsv"

# Weights
WEIGHT_GOA = 0.60
WEIGHT_PROTT5 = 0.40
SCORE_THRESHOLD = 0.0013


def create_ontology_graph(obo_path):
    """Parse GO ontology and create graph with extended relationships."""
    print("Loading GO ontology...")
    ontology_graph = nx.DiGraph()
    current_id = None

    with open(obo_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: "):
                current_id = line.split("id: ", 1)[1].strip()
                ontology_graph.add_node(current_id)
            elif line.startswith("is_a: "):
                parent_id = line.split()[1].strip()
                if current_id:
                    ontology_graph.add_edge(current_id, parent_id)
            elif line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] in {"part_of", "regulates", "positively_regulates", "negatively_regulates"}:
                    parent_id = parts[2].strip()
                    if current_id:
                        ontology_graph.add_edge(current_id, parent_id)

    if not nx.is_directed_acyclic_graph(ontology_graph):
        for cycle in nx.simple_cycles(ontology_graph):
            ontology_graph.remove_edge(cycle[0], cycle[1])

    topological_order = list(nx.topological_sort(ontology_graph))
    child_to_parents = {node: list(ontology_graph.successors(node)) for node in ontology_graph.nodes()}
    print(f"  Loaded {len(ontology_graph.nodes())} GO terms")
    return topological_order, child_to_parents, set(ontology_graph.nodes())


def main():
    print("=" * 60)
    print("CAFA6 Submission Optimization (Fast)")
    print("=" * 60)

    # Load ontology
    topo_order, parent_map, valid_terms = create_ontology_graph(OBO_FILE)

    # Load predictions efficiently
    print("Loading GOA...")
    goa_df = pd.read_csv(GOA_FILE, sep='\t', header=None, names=['protein_id', 'go_term', 'score'])
    print(f"  {len(goa_df):,} rows")

    print("Loading ProtT5...")
    prott5_df = pd.read_csv(PROTT5_FILE, sep='\t', header=None, names=['protein_id', 'go_term', 'score'])
    print(f"  {len(prott5_df):,} rows")

    # Get test proteins (from ProtT5)
    test_proteins = set(prott5_df['protein_id'].unique())
    print(f"Test proteins: {len(test_proteins):,}")

    # Filter GOA to test proteins
    goa_df = goa_df[goa_df['protein_id'].isin(test_proteins)]
    print(f"GOA filtered: {len(goa_df):,} rows")

    # Create pivot tables for fast lookup
    print("Building lookup tables...")
    goa_df['key'] = goa_df['protein_id'] + '|' + goa_df['go_term']
    prott5_df['key'] = prott5_df['protein_id'] + '|' + prott5_df['go_term']

    goa_scores = dict(zip(goa_df['key'], goa_df['score']))
    prott5_scores = dict(zip(prott5_df['key'], prott5_df['score']))

    # All unique keys
    all_keys = set(goa_scores.keys()) | set(prott5_scores.keys())
    print(f"Unique protein-term pairs: {len(all_keys):,}")

    # Blend scores
    print("Blending scores...")
    blended = {}
    for key in tqdm(all_keys, desc="  Blending"):
        g = goa_scores.get(key, 0)
        p = prott5_scores.get(key, 0)

        if g > 0 and p > 0:
            blended[key] = WEIGHT_GOA * g + WEIGHT_PROTT5 * p
        elif g > 0:
            blended[key] = g
        else:
            blended[key] = p * 0.85

    # Convert to DataFrame
    print("Converting to DataFrame...")
    rows = [(k.split('|')[0], k.split('|')[1], v) for k, v in blended.items()]
    blended_df = pd.DataFrame(rows, columns=['protein_id', 'go_term', 'score'])
    print(f"  Blended: {len(blended_df):,} rows")

    # Filter to valid GO terms
    blended_df = blended_df[blended_df['go_term'].isin(valid_terms)]
    print(f"  Valid GO terms: {len(blended_df):,} rows")

    # Propagate scores
    print("Propagating scores...")
    grouped = blended_df.groupby('protein_id')
    results = []

    for protein_id, group in tqdm(grouped, total=len(grouped), desc="  Propagating"):
        term_scores = dict(zip(group['go_term'], group['score']))

        # Propagate to parents
        for child_term in topo_order:
            if child_term not in term_scores:
                continue
            child_score = term_scores[child_term]
            for parent_term in parent_map.get(child_term, []):
                current = term_scores.get(parent_term, 0.0)
                if child_score > current:
                    term_scores[parent_term] = child_score

        # Filter and collect
        for term, score in term_scores.items():
            score = max(0.0, min(1.0, score))
            if score >= SCORE_THRESHOLD:
                results.append((protein_id, term, round(score, 4)))

    # Create final DataFrame
    final_df = pd.DataFrame(results, columns=['protein_id', 'go_term', 'score'])
    final_df.drop_duplicates(subset=['protein_id', 'go_term'], inplace=True)

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False)

    # Stats
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Proteins: {final_df['protein_id'].nunique():,}")
    print(f"  Total predictions: {len(final_df):,}")
    print(f"  Avg terms/protein: {len(final_df) / final_df['protein_id'].nunique():.1f}")
    print(f"  Score range: {final_df['score'].min():.4f} - {final_df['score'].max():.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
