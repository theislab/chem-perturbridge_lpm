#!/bin/bash
#SBATCH -J paper10_unit_rmse
#SBATCH -o logs/paper10_unit_rmse.%A_%a.out
#SBATCH -e logs/paper10_unit_rmse.%A_%a.err
#SBATCH -t 04:00:00
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --nice=10000
#SBATCH --exclude=gpusrv29
#SBATCH --array=0-219%15

# Parallel precompute: one GPU array task per (dataset, family, seed) checkpoint.
# Writes part TSVs; merge afterwards before notebook bootstrap.
#
# Recommended submit (writes task list, then array + merge dependency):
#   bash scripts/submit_paper10_unit_rmse_parallel.sh
#
# Manual:
#   mkdir -p logs results/paper10_bootstrap/unit_rmse_parts
#   ./lpm_training_venv/bin/python scripts/paper10_compound_win_rates.py --write-task-list
#   sbatch scripts/paper10_unit_rmse_dose_time.sbatch
#   sbatch --dependency=afterok:<array_jobid> scripts/paper10_unit_rmse_merge.sbatch
#
# Optional:
#   MAX_SEEDS=1 bash scripts/submit_paper10_unit_rmse_parallel.sh
#   DATASETS="op3 sciplex" bash scripts/submit_paper10_unit_rmse_parallel.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs results/paper10_bootstrap/unit_rmse_parts

PYTHON="${PYTHON:-./lpm_training_venv/bin/python}"
SHARD_ROOT="${SHARD_ROOT:-/lustre/groups/ml01/workspace/olga.novitskaia/lpm_style/.plib_cache/plibdata_multiout/lpm_multiout_all_data_plus_tahoe_novartis_op3_molholdout_h100_2x_bs4096_200epoch_lustre}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper10_bootstrap}"
PARTS_DIR="${PARTS_DIR:-${OUTPUT_DIR}/unit_rmse_parts}"
TASK_LIST="${TASK_LIST:-${OUTPUT_DIR}/unit_rmse_score_tasks.tsv}"
LONG_TSV="${LONG_TSV:-results/lpm_paper10_results_current_check_long.tsv}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"

echo "[paper10_unit_rmse] job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_JOB_ID:-none}:${TASK_INDEX} host=$(hostname)"
echo "[paper10_unit_rmse] task_list=${TASK_LIST}"
echo "[paper10_unit_rmse] parts_dir=${PARTS_DIR}"
nvidia-smi || true

if [[ ! -f "${TASK_LIST}" ]]; then
  echo "Missing ${TASK_LIST}. Run: bash scripts/submit_paper10_unit_rmse_parallel.sh" >&2
  exit 2
fi

srun "${PYTHON}" scripts/paper10_compound_win_rates.py \
  --shard-root "${SHARD_ROOT}" \
  --long-tsv "${LONG_TSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --parts-dir "${PARTS_DIR}" \
  --task-list "${TASK_LIST}" \
  --task-index "${TASK_INDEX}" \
  --skip-existing-parts

echo "[paper10_unit_rmse] task ${TASK_INDEX} done"
