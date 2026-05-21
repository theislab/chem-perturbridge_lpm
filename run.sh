#!/bin/bash
#SBATCH -J perturb_lib_lpm
#SBATCH -o logs/perturb_lib.%j.out
#SBATCH -e logs/perturb_lib.%j.err
#SBATCH -t 24:00:00
# NOTE: the directory in -o/-e above is parsed by Slurm before this script runs,
# so it has to be a literal string. Keep it in sync with LOG_DIR below.
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100_20gb
#SBATCH --exclude=gpusrv34,gpusrv53,gpusrv62

# ============================================================================
# Slurm launcher for multi-node DDP LPM training.
#
# Usage:
#   sbatch run.sh                                       # → lpm_modified_all_data.yaml (default)
#   sbatch run.sh -c lpm_modified_l1000_data            # short flag
#   sbatch run.sh --config lpm_modified_all_data        # long flag
#   sbatch run.sh --config=lpm_modified_all_data        # long flag, equals form
#   bash   run.sh --help                                # show usage and exit
#
# Env-var overrides (prepend to sbatch):
#   CLEAN_RESULTS=0   sbatch run.sh ...                 # skip pre-clean of results dir
#   NCCL_DEBUG=WARN   sbatch run.sh ...                 # quieten NCCL logs (default INFO)
# ============================================================================

set -euo pipefail

# Anchor to the directory the user submitted from. Under sbatch, $0 may point
# at a Slurm spool copy of the script, so dirname "$0" is unreliable; prefer
# SLURM_SUBMIT_DIR (set by sbatch) and fall back to dirname for direct shell.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly CONFIGS_DIR="perturb_gym/configs/collection"
readonly DEFAULT_CONFIG_ID="lpm_modified_all_data"
# Slurm log dir. MUST match the path in the #SBATCH -o/-e directives at the top
# of this file -- those are parsed before the shell runs and can't reference
# this variable. The directory must exist *before* sbatch is invoked, otherwise
# Slurm silently drops stdout/stderr.
readonly LOG_DIR="logs"

# Cluster firewall blocks most peer-to-peer ports between gpusrv* nodes.
# The first port reachable from every rank to the master becomes MASTER_PORT.
readonly PORT_CANDIDATES=(
    29500 12355 23456 8888 7000 6000 5000 9000 10000
    30000 35000 40000 45000 50000 55000 60000 4000 3000 2000
    100 200 300 400 500 600 700 800 900 1000 1100 1200
)

# Globals populated by the component functions and exported to children.
CONFIG_ID="$DEFAULT_CONFIG_ID"
CONFIG_FILE=""
MASTER_NODE_SHORT=""
MASTER_ADDR=""
MASTER_PORT=""
NCCL_IFACE=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[run.sh] $*"; }
warn() { echo "[run.sh] WARN: $*" >&2; }
die()  { echo "[run.sh] ERROR: $*" >&2; exit 1; }

# Print available config ids (one per line, indented).
list_configs() {
    ls -1 "${CONFIGS_DIR}"/*.yaml 2>/dev/null \
        | sed 's|.*/||;s|\.yaml$||;s|^|  |'
}

usage() {
    cat <<EOF
Usage: sbatch run.sh [options]

Options:
  -c, --config <id>     YAML stem under ${CONFIGS_DIR}/ (default: ${DEFAULT_CONFIG_ID})
  -h, --help            Show this help and exit

Env-var overrides:
  CLEAN_RESULTS=0       Skip pre-clean of .plib_cache/results/<config_id>/
  NCCL_DEBUG=WARN       Quieten NCCL logs (default INFO)

Available configs:
EOF
    list_configs
}

# ---------------------------------------------------------------------------
# Stage 1: argument parsing + config validation
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--config|--config-id)
                [[ $# -ge 2 ]] || die "'$1' requires a value."
                CONFIG_ID="$2"
                shift 2
                ;;
            --config=*|--config-id=*)
                CONFIG_ID="${1#*=}"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown argument '$1'. Run 'bash run.sh --help' for usage."
                ;;
        esac
    done
}

validate_config() {
    CONFIG_FILE="${CONFIGS_DIR}/${CONFIG_ID}.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        warn "config '$CONFIG_FILE' not found."
        echo "[run.sh] Available config ids:" >&2
        list_configs >&2
        exit 1
    fi
    log "CONFIG_ID=$CONFIG_ID  ($CONFIG_FILE)"
}

# ---------------------------------------------------------------------------
# Stage 2: environment
# ---------------------------------------------------------------------------
setup_env() {
    export HOME=${HOME:-/home/icb/olga.novitskaia}
    export TMPDIR=${SLURM_TMPDIR:-${HOME}/tmp}
    mkdir -p "${TMPDIR}"
    # Defensive: Slurm already opened the log file at job start, but if someone
    # blew away logs/ between submissions we want the next sbatch to succeed.
    mkdir -p "${LOG_DIR}"
}

# Print a small header so the log file self-identifies (job id, config, log path).
print_log_header() {
    local jid="${SLURM_JOB_ID:-local}"
    log "================================================================"
    log "Slurm job id : ${jid}"
    log "Config       : ${CONFIG_ID}"
    log "Submit dir   : ${SLURM_SUBMIT_DIR:-$(pwd)}"
    log "Log files    : ${LOG_DIR}/perturb_lib.${jid}.{out,err}"
    log "================================================================"
}

# ---------------------------------------------------------------------------
# Stage 3: DDP rendezvous — master IP + open TCP port
# ---------------------------------------------------------------------------
resolve_master_addr() {
    local master_node
    master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_NODE_SHORT="${master_node%%.*}"

    MASTER_ADDR=$(getent hosts "$master_node"        2>/dev/null | awk '{print $1}' | head -n 1)
    [[ -z "$MASTER_ADDR" ]] && \
    MASTER_ADDR=$(getent hosts "$MASTER_NODE_SHORT" 2>/dev/null | awk '{print $1}' | head -n 1)
    [[ -n "$MASTER_ADDR" ]] || die "could not resolve $MASTER_NODE_SHORT to an IP."

    export MASTER_ADDR
    log "MASTER_NODE=$MASTER_NODE_SHORT  MASTER_ADDR=$MASTER_ADDR  NODES=$SLURM_JOB_NODELIST"
}

# Probe a single (ip, port) pair: spin a short-lived listener on the master node
# and try to connect from every rank. Returns 0 if every rank reaches it.
probe_port() {
    local ip="$1" port="$2"
    local expected_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
    local probe_dir ready_file fail_file
    log "-- Probing $ip:$port --"

    probe_dir=$(mktemp -d "${SLURM_SUBMIT_DIR}/.port_probe.${SLURM_JOB_ID:-local}.${port}.XXXXXX")
    ready_file="${probe_dir}/ready"
    fail_file="${probe_dir}/fail"

    # Listener on master.
    srun --cpu-bind=none --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" -w "$MASTER_NODE_SHORT" \
        python3 -c "
import socket, sys, time

port = int(sys.argv[1])
expected = int(sys.argv[2])
ready_file = sys.argv[3]
fail_file = sys.argv[4]

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', port))
except OSError as e:
    with open(fail_file, 'w') as f:
        f.write(f'LISTENER_BIND_FAIL: {e}\n')
    raise
s.listen(max(128, expected * 2))
with open(ready_file, 'w') as f:
    f.write('ready\n')
s.settimeout(1.0)
accepted = 0
end = time.time() + 30
while time.time() < end and accepted < expected:
    try:
        c, _ = s.accept(); c.close()
        accepted += 1
    except socket.timeout:
        pass
if accepted < expected:
    with open(fail_file, 'w') as f:
        f.write(f'accepted {accepted}/{expected} connections\n')
    raise SystemExit(99)
" "$port" "$expected_nodes" "$ready_file" "$fail_file" >"${probe_dir}/listener.out" 2>"${probe_dir}/listener.err" &
    local pid=$!

    local listener_ready=0
    for _ in $(seq 1 50); do
        if [[ -f "$ready_file" ]]; then
            listener_ready=1
            break
        fi
        if [[ -f "$fail_file" ]]; then
            warn "Listener could not bind $ip:$port: $(cat "$fail_file")"
            wait "$pid" 2>/dev/null || true
            rm -rf "$probe_dir"
            return 98
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            warn "Listener exited before becoming ready on $ip:$port"
            rm -rf "$probe_dir"
            return 98
        fi
        sleep 0.1
    done

    if [[ "$listener_ready" != "1" ]]; then
        warn "Listener did not become ready on $ip:$port"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        rm -rf "$probe_dir"
        return 98
    fi

    # Connect probe from every rank.
    local rc=0
    srun --cpu-bind=none --ntasks-per-node=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" python3 -c "
import socket, sys, os
host, port = '$ip', $port
hostname = os.uname().nodename.split('.')[0]
try:
    s = socket.socket(); s.settimeout(3); s.connect((host, port)); s.close()
    print(f'[{hostname}] OK -> {host}:{port}')
except Exception as e:
    print(f'[{hostname}] UNREACHABLE -> {host}:{port}  ({e})')
    sys.exit(113)
" || rc=$?

    if [[ "$rc" -eq 0 ]]; then
        wait "$pid" 2>/dev/null || rc=$?
        if [[ "$rc" -ne 0 && -f "$fail_file" ]]; then
            warn "Listener probe failed on $ip:$port: $(cat "$fail_file")"
        fi
    else
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    if [[ "$rc" -eq 0 ]]; then
        srun --cpu-bind=none --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" -w "$MASTER_NODE_SHORT" \
            python3 -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', int(sys.argv[1])))
s.close()
" "$port" >/dev/null 2>&1 || rc=$?
        if [[ "$rc" -ne 0 ]]; then
            warn "Port $ip:$port was reachable, but could not be rebound after the probe."
        fi
    fi

    rm -rf "$probe_dir"
    return $rc
}

find_master_port() {
    for port in "${PORT_CANDIDATES[@]}"; do
        if probe_port "$MASTER_ADDR" "$port"; then
            MASTER_PORT="$port"
            log "Selected MASTER_PORT=$MASTER_PORT (all nodes reachable on $MASTER_ADDR:$port)."
            export MASTER_PORT
            return 0
        fi
        log "Port $port blocked or already in use; trying next candidate."
    done

    warn "none of the candidate ports passed the all-peers probe."
    echo "[run.sh] Tried: ${PORT_CANDIDATES[*]}" >&2
    echo "[run.sh] Next step: ask cluster admins for the allowed peer-to-peer TCP" >&2
    echo "[run.sh] port range, or switch to a shared-FS rendezvous (init_method=file://)." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Stage 4: NCCL / Gloo — pin to the interface that owns MASTER_ADDR
# ---------------------------------------------------------------------------
pin_nccl_iface() {
    NCCL_IFACE=$(srun --cpu-bind=none --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" \
                  -w "$MASTER_NODE_SHORT" \
                  bash -c "ip -4 -o addr show | awk -v ip='$MASTER_ADDR' '\$4 ~ \"^\"ip\"/\" {print \$2; exit}'" \
                  | tr -d '[:space:]')

    if [[ -z "$NCCL_IFACE" ]]; then
        warn "could not resolve interface owning $MASTER_ADDR on $MASTER_NODE_SHORT;"
        warn "      falling back to NCCL/Gloo SOCKET_IFNAME=^lo,docker."
        export GLOO_SOCKET_IFNAME="^lo,docker"
        export NCCL_SOCKET_IFNAME="^lo,docker"
    else
        log "$MASTER_ADDR is on interface '$NCCL_IFACE' on $MASTER_NODE_SHORT; pinning NCCL/Gloo to it."
        export GLOO_SOCKET_IFNAME="$NCCL_IFACE"
        export NCCL_SOCKET_IFNAME="$NCCL_IFACE"
    fi
    log "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
}

# ---------------------------------------------------------------------------
# Stage 5: Python environment
# ---------------------------------------------------------------------------
activate_env() {
    # Project-local conda env. We resolve to an absolute path so the activation
    # is unaffected by any later `cd`s (e.g. inside srun child shells).
    local venv_path="${SLURM_SUBMIT_DIR:-$(pwd)}/lpm_training_venv"

    if [[ ! -d "$venv_path" ]]; then
        die "conda env not found at '$venv_path'. Create it first:
        cd \"${SLURM_SUBMIT_DIR:-.}\"
        mamba env create --prefix ./lpm_training_venv --file env.yml
        mamba activate ./lpm_training_venv
        poetry install"
    fi

    log "Activating conda env at $venv_path"
    # Avoid stale virtualenv state from the submit shell taking precedence over
    # the project-local conda prefix on PATH.
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
    # `set +u` because conda's shell hook references unbound vars.
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$venv_path"
    export PATH="$venv_path/bin:$PATH"
    set -u
}

# ---------------------------------------------------------------------------
# Stage 6: pre-clean previous results for THIS config
# ---------------------------------------------------------------------------
clean_results() {
    local clean=${CLEAN_RESULTS:-1}
    local results_dir="${SLURM_SUBMIT_DIR}/.plib_cache/results/${CONFIG_ID}"
    if [[ "$clean" == "1" && -d "$results_dir" ]]; then
        log "Removing stale $results_dir"
        rm -rf "$results_dir"
    fi
}

# ---------------------------------------------------------------------------
# Stage 7: launch training
# ---------------------------------------------------------------------------
launch_training() {
    log "Launching training for config_id=${CONFIG_ID}"
    export PERTURB_GYM_RESULTS_PRE_CLEANED=1
    srun --cpu-bind=none --chdir="${SLURM_SUBMIT_DIR}" \
        python -m perturb_gym.training train_from_config_file \
        --config_file_id_or_path="${CONFIG_ID}"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    validate_config
    setup_env
    print_log_header
    resolve_master_addr
    find_master_port
    pin_nccl_iface
    activate_env
    clean_results
    launch_training
}

main "$@"
