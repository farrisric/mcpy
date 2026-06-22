#!/usr/bin/env bash
# Compile the JOSS paper to paper.pdf using the official openjournals/inara
# image (same pipeline JOSS runs). Output: paper/paper.pdf
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
  --volume "$PAPER_DIR":/data \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara:latest -o pdf paper.md

echo "Wrote $PAPER_DIR/paper.pdf"
