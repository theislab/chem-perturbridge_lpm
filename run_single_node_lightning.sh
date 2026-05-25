#!/bin/bash
#SBATCH -J perturb_lib_lpm_local
#SBATCH -o logs/perturb_lib_local.%j.out
#SBATCH -e logs/perturb_lib_local.%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --constraint=h100_80gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

CONFIG_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config|--config-id)
            CONFIG_ID="$2"
            shift 2
            ;;
        --config=*|--config-id=*)
            CONFIG_ID="${1#*=}"
            shift
            ;;
        *)
            echo "[run_single_node_lightning.sh] unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_ID" ]]; then
    echo "[run_single_node_lightning.sh] --config is required" >&2
    exit 1
fi

CONFIG_FILE="perturb_gym/configs/collection/${CONFIG_ID}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[run_single_node_lightning.sh] config '$CONFIG_FILE' not found" >&2
    exit 1
fi

echo "[run_single_node_lightning.sh] CONFIG_ID=$CONFIG_ID ($CONFIG_FILE)"
echo "[run_single_node_lightning.sh] Slurm job id: ${SLURM_JOB_ID:-local}"
echo "[run_single_node_lightning.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export HOME=${HOME:-/home/icb/olga.novitskaia}
export TMPDIR=${SLURM_TMPDIR:-${HOME}/tmp}
mkdir -p "$TMPDIR" logs
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [[ -z "${MASTER_PORT:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        export MASTER_PORT="$((20000 + (SLURM_JOB_ID % 30000)))"
    else
        export MASTER_PORT=29500
    fi
fi
echo "[run_single_node_lightning.sh] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
export PERTURB_GYM_FORCE_LIGHTNING_ENV=1
export PERTURB_GYM_RESULTS_PRE_CLEANED=1
export DATA_PREP_JOBS="${DATA_PREP_JOBS:-4}"

venv_path="${SLURM_SUBMIT_DIR:-$(pwd)}/lpm_training_venv"
if [[ ! -d "$venv_path" ]]; then
    echo "[run_single_node_lightning.sh] conda env not found at '$venv_path'" >&2
    exit 1
fi

unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
set +u
eval "$(conda shell.bash hook)"
conda activate "$venv_path"
export PATH="$venv_path/bin:$PATH"
set -u

if [[ "${PREPARE_MULTI_OUTPUT_DATA:-1}" == "1" ]]; then
    echo "[run_single_node_lightning.sh] Preparing multi-output data for $CONFIG_FILE"
    python scripts/prepare_multiout_plibdata.py \
        --config "$CONFIG_FILE" \
        --jobs "$DATA_PREP_JOBS"
fi

results_dir="${SLURM_SUBMIT_DIR}/.plib_cache/results/${CONFIG_ID}"
if [[ "${CLEAN_RESULTS:-1}" == "1" && -d "$results_dir" ]]; then
    echo "[run_single_node_lightning.sh] Removing stale $results_dir"
    rm -rf "$results_dir"
fi

python -m perturb_gym.training train_from_config_file --config_file_id_or_path="$CONFIG_ID"
