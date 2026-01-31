"""
CAFA6 Submission Optimization v2 - Much Faster
Key optimization: Only iterate through GO terms the protein has, not all 48k terms
"""
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
    """Parse GO ontology and create ancestor lookup."""
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

    # Remove cycles if any
    if not nx.is_directed_acyclic_graph(ontology_graph):
        for cycle in nx.simple_cycles(ontology_graph):
            ontology_graph.remove_edge(cycle[0], cycle[1])

    # Pre-compute all ancestors for each term (much faster than computing on-the-fly)
    print("Pre-computing ancestors...")
    ancestors = {}
    for node in tqdm(ontology_graph.nodes(), desc="  Ancestors"):
        ancestors[node] = set(nx.ancestors(ontology_graph, node))

    print(f"  Loaded {len(ontology_graph.nodes())} GO terms")
    return ancestors, set(ontology_graph.nodes())


def main():
    print("=" * 60)
    print("CAFA6 Submission Optimization v2 (Fast)")
    print("=" * 60)

    # Load ontology
    ancestors, valid_terms = create_ontology_graph(OBO_FILE)

    # Load predictions
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

    # Build lookup dicts
    print("Building lookup tables...")
    goa_dict = defaultdict(dict)
    for pid, go, score in tqdm(zip(goa_df['protein_id'], goa_df['go_term'], goa_df['score']),
                                total=len(goa_df), desc="  GOA"):
        goa_dict[pid][go] = score

    prott5_dict = defaultdict(dict)
    for pid, go, score in tqdm(zip(prott5_df['protein_id'], prott5_df['go_term'], prott5_df['score']),
                                total=len(prott5_df), desc="  ProtT5"):
        prott5_dict[pid][go] = score

    # Process each protein
    print("Processing proteins...")
    results = []

    for pid in tqdm(test_proteins, desc="  Proteins"):
        goa = goa_dict.get(pid, {})
        prott5 = prott5_dict.get(pid, {})
        all_terms = set(goa.keys()) | set(prott5.keys())

        # Blend scores
        term_scores = {}
        for go in all_terms:
            g = goa.get(go, 0)
            p = prott5.get(go, 0)

            if g > 0 and p > 0:
                term_scores[go] = WEIGHT_GOA * g + WEIGHT_PROTT5 * p
            elif g > 0:
                term_scores[go] = g
            else:
                term_scores[go] = p * 0.85

        # Propagate to ancestors (only for terms we have)
        propagated = dict(term_scores)
        for term, score in term_scores.items():
            if term not in ancestors:
                continue
            for ancestor in ancestors[term]:
                if ancestor in valid_terms:
                    current = propagated.get(ancestor, 0.0)
                    if score > current:
                        propagated[ancestor] = score

        # Filter and collect
        for term, score in propagated.items():
            score = max(0.0, min(1.0, score))
            if score >= SCORE_THRESHOLD:
                results.append((pid, term, round(score, 4)))

    # Create final DataFrame
    print("Creating output...")
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
