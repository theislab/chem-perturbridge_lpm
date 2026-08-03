#!/bin/bash
#SBATCH -J paper10_unit_merge
#SBATCH -o logs/paper10_unit_merge.%j.out
#SBATCH -e logs/paper10_unit_merge.%j.err
#SBATCH -t 01:00:00
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Merge parallel unit-RMSE part TSVs into the long table used by paper10_bootstrap.ipynb.
#
#   sbatch --dependency=afterok:<array_jobid> scripts/paper10_unit_rmse_merge.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

PYTHON="${PYTHON:-./lpm_training_venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper10_bootstrap}"
PARTS_DIR="${PARTS_DIR:-${OUTPUT_DIR}/unit_rmse_parts}"
OUTPUT_NAME="${OUTPUT_NAME:-paper10_unit_rmse_long_scratch_vs_ft_morgan_learned_fixmol_mol_ctx_dose_time.tsv}"

echo "[paper10_unit_merge] job=${SLURM_JOB_ID:-local} parts_dir=${PARTS_DIR}"
"${PYTHON}" scripts/paper10_compound_win_rates.py \
  --output-dir "${OUTPUT_DIR}" \
  --parts-dir "${PARTS_DIR}" \
  --output-name "${OUTPUT_NAME}" \
  --merge-parts

echo "[paper10_unit_merge] done → ${OUTPUT_DIR}/${OUTPUT_NAME}"
