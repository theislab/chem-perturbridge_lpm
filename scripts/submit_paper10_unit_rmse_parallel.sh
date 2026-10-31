#!/bin/bash
# Thin wrapper: submit the CPU orchestrator (task list is built on a compute node).
#
#   sbatch scripts/submit_paper10_unit_rmse_parallel.sbatch
#   # or:
#   bash scripts/submit_paper10_unit_rmse_parallel.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs

EXTRA_EXPORT=ALL
for var in DATASETS MAX_SEEDS SEEDS ARRAY_THROTTLE PYTHON LIST_PYTHON OUTPUT_DIR PARTS_DIR TASK_LIST LONG_TSV; do
  if [[ -n "${!var:-}" ]]; then
    EXTRA_EXPORT+=",${var}"
  fi
done

echo "Submitting CPU orchestrator (task list + GPU array + merge) ..."
exec sbatch --export="${EXTRA_EXPORT}" scripts/submit_paper10_unit_rmse_parallel.sbatch
