#!/bin/bash
#SBATCH -J write_narrowed_shards
#SBATCH -o logs/write_narrowed_shards.%j.out
#SBATCH -e logs/write_narrowed_shards.%j.err
#SBATCH -t 12:00:00
#SBATCH --qos=cpu_normal
#SBATCH --partition=cpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# ============================================================================
# Slurm launcher: write L1000-narrowed plibdata shards.
#
# Usage (from repo root):
#   mkdir -p logs
#   sbatch scripts/run_write_narrowed_plibdata_shards.sh
#   sbatch scripts/run_write_narrowed_plibdata_shards.sh --build-annot
#
# Extra args after the script name are forwarded to
# scripts/write_narrowed_plibdata_shards.py
# ============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

PYTHON="${PYTHON:-./lpm_training_venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "[write_narrowed_shards] host=$(hostname) cwd=$(pwd)"
echo "[write_narrowed_shards] python=${PYTHON}"
echo "[write_narrowed_shards] args: $*"

"${PYTHON}" scripts/write_narrowed_plibdata_shards.py "$@"

echo "[write_narrowed_shards] done"
