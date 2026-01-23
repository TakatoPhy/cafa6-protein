"""
Parse GO ontology and extract parent-child relationships.
"""
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / 'data'

def parse_go_obo(filepath):
    """Parse GO OBO file and extract is_a relationships."""
    term_to_parents = defaultdict(list)
    term_to_name = {}
    term_to_namespace = {}

    current_term = None
    current_name = None
    current_namespace = None
    is_obsolete = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                if current_term and not is_obsolete:
                    term_to_name[current_term] = current_name
                    term_to_namespace[current_term] = current_namespace
                current_term = None
                current_name = None
                current_namespace = None
                is_obsolete = False

            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]

            elif line.startswith('name: '):
                current_name = line.split('name: ')[1]

            elif line.startswith('namespace: '):
                current_namespace = line.split('namespace: ')[1]

            elif line.startswith('is_a: GO:'):
                parent = line.split('is_a: ')[1].split(' !')[0]
                if current_term:
                    term_to_parents[current_term].append(parent)

            elif line == 'is_obsolete: true':
                is_obsolete = True

    # Handle last term
    if current_term and not is_obsolete:
        term_to_name[current_term] = current_name
        term_to_namespace[current_term] = current_namespace

    return term_to_parents, term_to_name, term_to_namespace

def get_all_ancestors(term, term_to_parents, cache=None):
    """Get all ancestors of a term (including itself)."""
    if cache is None:
        cache = {}

    if term in cache:
        return cache[term]

    ancestors = {term}
    for parent in term_to_parents.get(term, []):
        ancestors.update(get_all_ancestors(parent, term_to_parents, cache))

    cache[term] = ancestors
    return ancestors

def main():
    print("Parsing GO ontology...")
    term_to_parents, term_to_name, term_to_namespace = parse_go_obo(DATA_DIR / 'Train' / 'go-basic.obo')

    print(f"Total terms: {len(term_to_name)}")
    print(f"Terms with parents: {len(term_to_parents)}")

    # Example
    example_term = 'GO:0000001'
    ancestors = get_all_ancestors(example_term, term_to_parents)
    print(f"\nExample: {example_term} ({term_to_name.get(example_term, 'unknown')})")
    print(f"Ancestors: {ancestors}")

    # Count by namespace
    ns_counts = defaultdict(int)
    for term, ns in term_to_namespace.items():
        ns_counts[ns] += 1
    print(f"\nBy namespace:")
    for ns, count in ns_counts.items():
        print(f"  {ns}: {count}")

    return term_to_parents, term_to_name, term_to_namespace

if __name__ == '__main__':
    main()
