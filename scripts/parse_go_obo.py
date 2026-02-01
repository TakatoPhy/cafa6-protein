#!/usr/bin/env python3
"""
Parse GO-basic.obo file and extract parent-child edges.
Output format: child_term\tparent_term
"""

import re
from pathlib import Path


def parse_go_obo(obo_path: str, output_path: str) -> dict:
    """
    Parse OBO file and extract is_a and part_of relationships.

    Returns stats about the parsing.
    """
    edges = []
    current_term = None
    is_obsolete = False

    # Pattern to extract GO term from is_a line: "is_a: GO:0048308 ! organelle inheritance"
    is_a_pattern = re.compile(r'^is_a:\s+(GO:\d+)')
    # Pattern for relationship: "relationship: part_of GO:0005829 ! cytosol"
    part_of_pattern = re.compile(r'^relationship:\s+part_of\s+(GO:\d+)')

    stats = {
        'total_terms': 0,
        'obsolete_terms': 0,
        'is_a_edges': 0,
        'part_of_edges': 0,
        'namespaces': {'biological_process': 0, 'molecular_function': 0, 'cellular_component': 0}
    }

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                # Save previous term's edges
                current_term = None
                is_obsolete = False

            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]
                stats['total_terms'] += 1

            elif line == 'is_obsolete: true':
                is_obsolete = True
                stats['obsolete_terms'] += 1

            elif line.startswith('namespace:'):
                ns = line.split('namespace: ')[1]
                if ns in stats['namespaces']:
                    stats['namespaces'][ns] += 1

            elif current_term and not is_obsolete:
                # Extract is_a relationships
                match = is_a_pattern.match(line)
                if match:
                    parent = match.group(1)
                    edges.append((current_term, parent))
                    stats['is_a_edges'] += 1
                    continue

                # Extract part_of relationships
                match = part_of_pattern.match(line)
                if match:
                    parent = match.group(1)
                    edges.append((current_term, parent))
                    stats['part_of_edges'] += 1

    # Write edges to file
    with open(output_path, 'w') as f:
        for child, parent in edges:
            f.write(f"{child}\t{parent}\n")

    stats['total_edges'] = len(edges)
    return stats


def main():
    base_dir = Path(__file__).parent.parent
    obo_path = base_dir / 'data' / 'Train' / 'go-basic.obo'
    output_path = base_dir / 'data' / 'go_edges.txt'

    print(f"Parsing: {obo_path}")
    print(f"Output: {output_path}")

    stats = parse_go_obo(str(obo_path), str(output_path))

    print("\n=== GO Graph Statistics ===")
    print(f"Total terms: {stats['total_terms']}")
    print(f"Obsolete terms: {stats['obsolete_terms']}")
    print(f"Active terms: {stats['total_terms'] - stats['obsolete_terms']}")
    print(f"\nNamespaces:")
    for ns, count in stats['namespaces'].items():
        print(f"  {ns}: {count}")
    print(f"\nEdges:")
    print(f"  is_a edges: {stats['is_a_edges']}")
    print(f"  part_of edges: {stats['part_of_edges']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
