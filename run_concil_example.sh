#!/usr/bin/env bash
# Run CONCIL from repository root. Usage:
#   ./run_concil_example.sh
# Or: bash run_concil_example.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$REPO_ROOT"

# Single CONCIL run: CUB, 8 stages, default hyperparameters
python src/experiments/CONCIL_1114.py \
  -dataset cub \
  -num_stages 8 \
  -buffer_size 25000 \
  -saved_dir results/concil_cub_example

echo "Done. Results in results/concil_cub_example"
