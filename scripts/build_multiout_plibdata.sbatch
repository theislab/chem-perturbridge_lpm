#!/bin/bash
#SBATCH -J lpm_multiout_data
#SBATCH -o logs/lpm_multiout_data.%A_%a.out
#SBATCH -e logs/lpm_multiout_data.%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-159%16

# Build the multi-output, sample-level cache for the exact source list used by
# perturb_gym/configs/collection/lpm_modified_all_data_1epoch_lustre.yaml.
#
# Submit:
#   mkdir -p logs
#   sbatch scripts/build_multiout_plibdata.sbatch
#
# Optional overrides:
#   OUTPUT_ROOT=/path/to/output sbatch scripts/build_multiout_plibdata.sbatch
#   TARGET_VALUES_PER_SHARD=16000000 sbatch scripts/build_multiout_plibdata.sbatch
#   OVERWRITE=1 sbatch scripts/build_multiout_plibdata.sbatch
#
# If this cluster requires explicit CPU partition/QOS directives, add them
# above. They are intentionally left unset here because the training launcher's
# gpu_p/gpu_normal directives would be wasteful for this CPU-only conversion.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CONFIG="${CONFIG:-perturb_gym/configs/collection/lpm_modified_all_data_1epoch_lustre.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.plib_cache/plibdata_multiout/lpm_modified_all_data_1epoch_lustre}"
PYTHON="${PYTHON:-${SLURM_SUBMIT_DIR:-$(pwd)}/lpm_training_venv/bin/python}"
TARGET_VALUES_PER_SHARD="${TARGET_VALUES_PER_SHARD:-8000000}"
VALUE_DTYPE="${VALUE_DTYPE:-float32}"
FLOAT_FEATURE_DTYPE="${FLOAT_FEATURE_DTYPE:-float32}"
OVERWRITE="${OVERWRITE:-0}"
INCLUDE_SYMBOLS="${INCLUDE_SYMBOLS:-0}"

echo "[build_multiout_plibdata.sbatch] job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[build_multiout_plibdata.sbatch] config=${CONFIG}"
echo "[build_multiout_plibdata.sbatch] output_root=${OUTPUT_ROOT}"
echo "[build_multiout_plibdata.sbatch] python=${PYTHON}"
echo "[build_multiout_plibdata.sbatch] cpus=${SLURM_CPUS_PER_TASK:-8}"

SOURCE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA_ARGS=(--skip-existing)
if [[ "${OVERWRITE}" == "1" ]]; then
    EXTRA_ARGS=(--overwrite)
fi
if [[ "${INCLUDE_SYMBOLS}" == "1" ]]; then
    EXTRA_ARGS+=(--include-symbols)
fi

"${PYTHON}" scripts/build_multiout_plibdata.py \
    --config "${CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --source-index "${SOURCE_INDEX}" \
    --target-values-per-shard "${TARGET_VALUES_PER_SHARD}" \
    --value-dtype "${VALUE_DTYPE}" \
    --float-feature-dtype "${FLOAT_FEATURE_DTYPE}" \
    "${EXTRA_ARGS[@]}"
