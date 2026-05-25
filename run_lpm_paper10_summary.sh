#!/bin/bash
#SBATCH -J p10_summary
#SBATCH -o logs/lpm_paper10_summary.%j.out
#SBATCH -e logs/lpm_paper10_summary.%j.err
#SBATCH -t 00:30:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p logs .tmp
export TMPDIR="${SLURM_TMPDIR:-$(pwd)/.tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/matplotlib}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

./lpm_training_venv/bin/python scripts/summarize_lpm_paper10_results.py
