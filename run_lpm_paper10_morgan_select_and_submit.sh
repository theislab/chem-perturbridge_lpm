#!/bin/bash
#SBATCH -J p10_morgan_select
#SBATCH -o logs/lpm_paper10_morgan_select.%j.out
#SBATCH -e logs/lpm_paper10_morgan_select.%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p logs .tmp
export TMPDIR="${SLURM_TMPDIR:-$(pwd)/.tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

./lpm_training_venv/bin/python scripts/select_and_submit_lpm_paper10_morgan_finetune_jobs.py "$@"
