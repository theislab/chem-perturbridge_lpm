#!/bin/bash
#SBATCH -J paper10_unit_submit
#SBATCH -o logs/paper10_unit_submit.%j.out
#SBATCH -e logs/paper10_unit_submit.%j.err
#SBATCH -t 00:20:00
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU orchestrator (not login node):
#   1) write task list
#   2) sbatch GPU array (≤15 concurrent)
#   3) sbatch merge afterok
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch scripts/submit_paper10_unit_rmse_parallel.sbatch
#
# Optional:
#   MAX_SEEDS=1 DATASETS="op3" sbatch --export=ALL,MAX_SEEDS,DATASETS scripts/submit_paper10_unit_rmse_parallel.sbatch
#   ARRAY_THROTTLE=15 sbatch --export=ALL,ARRAY_THROTTLE scripts/submit_paper10_unit_rmse_parallel.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs results/paper10_bootstrap/unit_rmse_parts

# Scoring jobs need the training venv; task-list only needs pandas.
PYTHON="${PYTHON:-./lpm_training_venv/bin/python}"
LIST_PYTHON="${LIST_PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper10_bootstrap}"
PARTS_DIR="${PARTS_DIR:-${OUTPUT_DIR}/unit_rmse_parts}"
TASK_LIST="${TASK_LIST:-${OUTPUT_DIR}/unit_rmse_score_tasks.tsv}"
LONG_TSV="${LONG_TSV:-results/lpm_paper10_results_current_check_long.tsv}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-15}"

echo "[submit] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[submit] writing task list on compute node with ${LIST_PYTHON} ..."

LIST_CMD=(
  "${LIST_PYTHON}" scripts/paper10_compound_win_rates.py
  --long-tsv "${LONG_TSV}"
  --output-dir "${OUTPUT_DIR}"
  --task-list "${TASK_LIST}"
  --write-task-list
)
if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DS=( ${DATASETS} )
  LIST_CMD+=(--datasets "${DS[@]}")
fi
if [[ -n "${MAX_SEEDS:-}" ]]; then
  LIST_CMD+=(--max-seeds "${MAX_SEEDS}")
fi
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SD=( ${SEEDS} )
  LIST_CMD+=(--seeds "${SD[@]}")
fi

echo "[submit] cmd: ${LIST_CMD[*]}"
"${LIST_CMD[@]}"

N_TASKS="$(awk 'NR>1' "${TASK_LIST}" | wc -l | tr -d ' ')"
if [[ "${N_TASKS}" -lt 1 ]]; then
  echo "No tasks in ${TASK_LIST}" >&2
  exit 1
fi
LAST=$((N_TASKS - 1))
echo "[submit] ${N_TASKS} tasks → array 0-${LAST}%${ARRAY_THROTTLE}"

export OUTPUT_DIR PARTS_DIR TASK_LIST LONG_TSV PYTHON
ARRAY_JOB="$(sbatch --parsable --export=ALL --array="0-${LAST}%${ARRAY_THROTTLE}" scripts/paper10_unit_rmse_dose_time.sbatch)"
echo "[submit] array job ${ARRAY_JOB}"

MERGE_JOB="$(sbatch --parsable --export=ALL --dependency="afterok:${ARRAY_JOB}" scripts/paper10_unit_rmse_merge.sbatch)"
echo "[submit] merge job ${MERGE_JOB} (afterok:${ARRAY_JOB})"
echo "[submit] when merge finishes, run Chem-style cells in notebooks_work/paper10_bootstrap.ipynb (RUN_SCORING=False)"
