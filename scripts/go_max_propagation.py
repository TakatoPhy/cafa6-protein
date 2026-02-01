"""
GO Hierarchy Max Propagation.

Ensures child scores >= parent scores by propagating maximum values up the GO hierarchy.
This is based on the true-path rule: if a protein has a specific function,
it must also have all the parent functions.

Usage:
    python scripts/go_max_propagation.py input.tsv output.tsv
"""
import sys
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from parse_go import parse_go_obo

BASE_DIR = Path(__file__).parent.parent
GO_OBO_PATH = BASE_DIR / 'data' / 'Train' / 'go-basic.obo'


def build_child_to_parents(term_to_parents: dict) -> dict:
    """Build child -> parents mapping."""
    return term_to_parents


def build_parent_to_children(term_to_parents: dict) -> dict:
    """Build parent -> children mapping."""
    parent_to_children = defaultdict(list)
    for child, parents in term_to_parents.items():
        for parent in parents:
            parent_to_children[parent].append(child)
    return parent_to_children


def topological_sort(term_to_parents: dict, all_terms: set) -> list:
    """Return terms in topological order (children before parents)."""
    # Build parent-to-children for traversal
    parent_to_children = build_parent_to_children(term_to_parents)

    # Count in-degrees (number of children for each term)
    in_degree = defaultdict(int)
    for term in all_terms:
        in_degree[term] = 0

    for child, parents in term_to_parents.items():
        if child in all_terms:
            for parent in parents:
                if parent in all_terms:
                    in_degree[parent] += 1

    # Start with leaves (terms with no children in our set)
    queue = [term for term in all_terms if in_degree[term] == 0]
    result = []

    while queue:
        term = queue.pop(0)
        result.append(term)

        for parent in term_to_parents.get(term, []):
            if parent in all_terms:
                in_degree[parent] -= 1
                if in_degree[parent] == 0:
                    queue.append(parent)

    return result


def propagate_max(protein_scores: dict, term_to_parents: dict) -> dict:
    """
    Propagate max scores up the GO hierarchy.

    For each protein, ensure that parent term scores are >= max of child scores.
    """
    # Get all terms for this protein
    all_terms = set(protein_scores.keys())

    # Add ancestors that might need scores
    ancestors_to_add = set()
    for term in list(all_terms):
        for parent in term_to_parents.get(term, []):
            if parent not in all_terms:
                ancestors_to_add.add(parent)
    all_terms.update(ancestors_to_add)

    # Initialize scores for missing ancestors
    result = dict(protein_scores)
    for term in ancestors_to_add:
        result[term] = 0.0

    # Topological sort: process children before parents
    sorted_terms = topological_sort(term_to_parents, all_terms)

    # Propagate max scores
    for term in sorted_terms:
        term_score = result.get(term, 0.0)
        for parent in term_to_parents.get(term, []):
            if parent in result:
                # Parent score should be at least as high as child
                result[parent] = max(result[parent], term_score)

    return result


def process_submission(input_path: Path, output_path: Path, min_score: float = 0.001):
    """Process submission file with GO max propagation."""
    print(f"\n=== GO Max Propagation ===\n")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Parse GO ontology
    print("Parsing GO ontology...")
    term_to_parents, term_to_name, _ = parse_go_obo(GO_OBO_PATH)
    print(f"  Loaded {len(term_to_parents)} terms with parent relationships")

    # Load submission
    print(f"Loading submission...")
    protein_scores = defaultdict(dict)
    row_count = 0

    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if not go_term.startswith('GO:'):
                    continue
                if go_term not in protein_scores[protein] or protein_scores[protein][go_term] < score:
                    protein_scores[protein][go_term] = score
                row_count += 1
                if row_count % 10_000_000 == 0:
                    print(f"  {row_count:,} rows...", flush=True)

    print(f"  Loaded {row_count:,} rows, {len(protein_scores):,} proteins")

    # Apply propagation
    print("Applying max propagation...")
    output_rows = 0
    propagated_count = 0

    with open(output_path, 'w') as f:
        for i, (protein, scores) in enumerate(protein_scores.items()):
            if i % 50000 == 0:
                print(f"  Protein {i:,}/{len(protein_scores):,}...", flush=True)

            # Propagate
            propagated = propagate_max(scores, term_to_parents)

            # Count propagations
            for term, new_score in propagated.items():
                old_score = scores.get(term, 0.0)
                if new_score > old_score:
                    propagated_count += 1

            # Write output
            for term, score in sorted(propagated.items(), key=lambda x: -x[1]):
                if score >= min_score:
                    f.write(f"{protein}\t{term}\t{score:.6f}\n")
                    output_rows += 1

    print(f"\nDone!")
    print(f"  Output rows: {output_rows:,}")
    print(f"  Scores propagated: {propagated_count:,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    if len(sys.argv) < 3:
        print("Usage: python go_max_propagation.py input.tsv output.tsv [min_score]")
        print("\nExample:")
        print("  python scripts/go_max_propagation.py submission.tsv submission_propagated.tsv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    min_score = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    process_submission(input_path, output_path, min_score)


if __name__ == '__main__':
    main()
