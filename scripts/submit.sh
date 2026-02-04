#!/bin/bash
# Usage: ./scripts/submit.sh <blend_file> <message>
# Example: ./scripts/submit.sh blend_sota95_taxon5.tsv "SOTA*0.95 + Taxon*0.05"

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/submit.sh <blend_file> <message>"
    echo "Available files:"
    ls -la submissions/blend_sota*.tsv
    exit 1
fi

BLEND_FILE="submissions/$1"
MESSAGE="${2:-blend submission}"

if [ ! -f "$BLEND_FILE" ]; then
    echo "Error: $BLEND_FILE not found"
    exit 1
fi

echo "Copying $BLEND_FILE to submission.tsv..."
cp "$BLEND_FILE" submission.tsv

echo "Submitting..."
kaggle competitions submit cafa-6-protein-function-prediction \
    -f submission.tsv \
    -m "$MESSAGE"

echo "Done!"
