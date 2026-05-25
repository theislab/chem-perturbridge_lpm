#!/bin/bash
#SBATCH -J p10_source_eval
#SBATCH -o logs/lpm_paper10_source_eval.%j.out
#SBATCH -e logs/lpm_paper10_source_eval.%j.err
#SBATCH -t 06:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --constraint=h100_80gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p logs .tmp
export HOME=${HOME:-/home/icb/olga.novitskaia}
export TMPDIR="${SLURM_TMPDIR:-/tmp/lpm_p10_eval_${USER}_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export PERTURB_GYM_FLOAT32_MATMUL_PRECISION="${PERTURB_GYM_FLOAT32_MATMUL_PRECISION:-high}"

./lpm_training_venv/bin/python scripts/evaluate_lpm_paper10_source_checkpoints.py "$@"
