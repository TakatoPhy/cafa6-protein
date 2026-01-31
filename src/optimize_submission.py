"""
CAFA6 Submission Optimization
Based on antonoof/cafa-optimization approach:
1. Blend GOA (60%) + ProtT5 (40%)
2. GO hierarchy propagation with extended relationships
3. Threshold filtering at 0.0013
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
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"File not found: {obo_path}")

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
                # Extended relationships (from cafa-optimization)
                if len(parts) >= 3 and parts[1] in {"part_of", "regulates", "positively_regulates", "negatively_regulates"}:
                    parent_id = parts[2].strip()
                    if current_id:
                        ontology_graph.add_edge(current_id, parent_id)

    # Remove cycles if any
    if not nx.is_directed_acyclic_graph(ontology_graph):
        for cycle in nx.simple_cycles(ontology_graph):
            ontology_graph.remove_edge(cycle[0], cycle[1])

    topological_order = list(nx.topological_sort(ontology_graph))
    child_to_parents = {node: list(ontology_graph.successors(node)) for node in ontology_graph.nodes()}

    print(f"  Loaded {len(ontology_graph.nodes())} GO terms")
    return topological_order, child_to_parents, set(ontology_graph.nodes())


def load_predictions(filepath, name):
    """Load prediction file."""
    print(f"Loading {name}...")
    df = pd.read_csv(filepath, sep='\t', header=None, names=['protein_id', 'go_term', 'score'])
    print(f"  {len(df):,} rows, {df['protein_id'].nunique():,} proteins")
    return df


def blend_predictions(goa_df, prott5_df):
    """Blend GOA and ProtT5 predictions with 60/40 weighting."""
    print(f"Blending predictions (GOA {WEIGHT_GOA:.0%} + ProtT5 {WEIGHT_PROTT5:.0%})...")

    # Convert to dict for faster lookup
    goa_dict = defaultdict(dict)
    for _, row in tqdm(goa_df.iterrows(), total=len(goa_df), desc="  Building GOA dict"):
        goa_dict[row['protein_id']][row['go_term']] = row['score']

    prott5_dict = defaultdict(dict)
    for _, row in tqdm(prott5_df.iterrows(), total=len(prott5_df), desc="  Building ProtT5 dict"):
        prott5_dict[row['protein_id']][row['go_term']] = row['score']

    # Get all proteins (use ProtT5 proteins as test set)
    all_proteins = set(prott5_dict.keys())
    print(f"  {len(all_proteins):,} test proteins")

    # Blend
    blended = defaultdict(dict)
    for pid in tqdm(all_proteins, desc="  Blending"):
        goa = goa_dict.get(pid, {})
        prott5 = prott5_dict.get(pid, {})
        all_terms = set(goa.keys()) | set(prott5.keys())

        for go in all_terms:
            g = goa.get(go, 0)
            p = prott5.get(go, 0)

            if g > 0 and p > 0:
                # Both have predictions - weighted blend
                blended[pid][go] = WEIGHT_GOA * g + WEIGHT_PROTT5 * p
            elif g > 0:
                # GOA only
                blended[pid][go] = g
            else:
                # ProtT5 only - slight discount (from 100 experiments)
                blended[pid][go] = p * 0.85

    # Convert to DataFrame
    rows = []
    for pid, terms in blended.items():
        for go, score in terms.items():
            rows.append((pid, go, score))

    df = pd.DataFrame(rows, columns=['protein_id', 'go_term', 'score'])
    print(f"  Blended: {len(df):,} rows")
    return df


def propagate_scores(predictions_df, topological_order, child_to_parents, valid_go_terms):
    """Propagate scores through GO hierarchy."""
    print("Propagating scores through GO hierarchy...")

    # Filter to valid GO terms
    mask = predictions_df['go_term'].isin(valid_go_terms)
    filtered_df = predictions_df[mask].copy()
    print(f"  Valid GO terms: {len(filtered_df):,} rows")

    grouped_by_protein = filtered_df.groupby('protein_id')
    results = []

    for protein_id, protein_group in tqdm(grouped_by_protein, total=grouped_by_protein.ngroups, desc="  Propagating"):
        term_scores = dict(zip(protein_group['go_term'], protein_group['score']))

        # Propagate to parents
        relevant_terms = set(term_scores.keys()) & set(topological_order)
        for child_term in topological_order:
            if child_term not in relevant_terms:
                continue
            child_score = term_scores[child_term]
            for parent_term in child_to_parents.get(child_term, []):
                current_parent_score = term_scores.get(parent_term, 0.0)
                if child_score > current_parent_score:
                    term_scores[parent_term] = child_score
                    relevant_terms.add(parent_term)

        # Filter and clamp
        for term, score in term_scores.items():
            clamped_score = max(0.0, min(1.0, score))
            if clamped_score >= SCORE_THRESHOLD:
                results.append((protein_id, term, round(clamped_score, 4)))

    df = pd.DataFrame(results, columns=['protein_id', 'go_term', 'score'])
    df.drop_duplicates(subset=['protein_id', 'go_term'], inplace=True)
    print(f"  Final: {len(df):,} rows")
    return df


def main():
    print("=" * 60)
    print("CAFA6 Submission Optimization")
    print("=" * 60)

    # Load ontology
    topo_order, parent_map, valid_terms = create_ontology_graph(OBO_FILE)

    # Load predictions
    goa_df = load_predictions(GOA_FILE, "GOA")
    prott5_df = load_predictions(PROTT5_FILE, "ProtT5")

    # Blend
    blended_df = blend_predictions(goa_df, prott5_df)

    # Propagate
    final_df = propagate_scores(blended_df, topo_order, parent_map, valid_terms)

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False)
    print("Done!")

    # Stats
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Proteins: {final_df['protein_id'].nunique():,}")
    print(f"  Total predictions: {len(final_df):,}")
    print(f"  Avg terms/protein: {len(final_df) / final_df['protein_id'].nunique():.1f}")
    print(f"  Score range: {final_df['score'].min():.4f} - {final_df['score'].max():.4f}")


if __name__ == "__main__":
    main()
