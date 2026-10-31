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
# supergpu04/07 are not configured in the cluster, so a contiguous supergpu[02-35]
# range makes sbatch fail with "Invalid node name specified". List only the
# supergpus that actually exist in gpu_p.
# gpusrv[26,29,41,46,59,72] are excluded on the strength of a single failure
# (job 40054914): all 16 nodes passed the all-peers probe on the selected
# 10.233.0.109:61001, then these six never connected to the rank-0 TCP store, so
# init_process_group died with "10/16 clients joined" after the full 30-minute
# timeout. Since the probe had just verified that exact pair from every node, the
# fault may be transient rather than a property of these six -- drop them from
# this list if a later job fails the same way on different nodes.
#SBATCH --exclude=gpusrv34,gpusrv53,gpusrv62,supergpu[02-03,05,08-09,14-32],gpusrv[26,29,41,46,59,72]

# ============================================================================
# Slurm launcher for multi-node DDP LPM training.
#
# Usage:
#   sbatch run.sh                                       # → lpm_modified_all_data.yaml (default)
#   sbatch run.sh -c lpm_modified_l1000_data
#   sbatch run.sh -c lpm_modified_all_data_l1000_narrowed
#   bash   run.sh --help
#
# Env-var overrides (prepend to sbatch):
#   CLEAN_RESULTS=0   sbatch run.sh ...
#   NCCL_DEBUG=WARN   sbatch run.sh ...
#   PERTURB_LIB_CACHE_DIR=...  sbatch run.sh ...
#   NCCL_IB_DISABLE=1 sbatch run.sh ...   # opt out of IB; IB is used by default
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
readonly DEFAULT_CACHE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/.plib_cache"
# Configs whose results/checkpoints should land next to their on-disk shards
# unless the user overrides -d / PERTURB_LIB_CACHE_DIR.
readonly NARROWED_CONFIG_ID="lpm_modified_all_data_l1000_narrowed"
readonly NARROWED_CACHE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/.plib_cache_paper_narrowed"
# Slurm log dir. MUST match the path in the #SBATCH -o/-e directives at the top
# of this file -- those are parsed before the shell runs and can't reference
# this variable. The directory must exist *before* sbatch is invoked, otherwise
# Slurm silently drops stdout/stderr.
readonly LOG_DIR="logs"

# The compute-node firewall rejects inbound TCP outside 60001-63000 (measured by
# scripts/probe_ddp_ports.sh; everything else returns icmp-host-prohibited).
# Candidates are drawn from 61001-63000 rather than 60001-60999 because the
# latter overlaps ip_local_port_range (32768-60999), so the kernel could hand the
# port to another process as an ephemeral port before we bind it. Ports observed
# to be in use are skipped: 60432 60606 61737 61765 62046 62059.
readonly PORT_CANDIDATES=(
    61001 61101 61201 61301 61401 61501 61601 61801 61901
    62001 62101 62201 62301 62401 62501 62601 62701 62801 62901
)

# NCCL's bootstrap sockets ask the kernel for a port, and the kernel draws from
# ip_local_port_range (32768-60999), which is almost entirely blocked. NCCL has
# no env var to constrain those ports, so scripts/port_shim.c interposes on
# bind() to place them inside the allowlist instead.
readonly SHIM_PORT_RANGE="61001-63000"
readonly SHIM_SRC="scripts/port_shim.c"
readonly SHIM_SO=".tmp/port_shim.so"

# torch 2.6 bundles NCCL 2.21.5, which segfaults during InfiniBand device
# discovery against the CUDA 13 driver the nodes now run (job 39941547). 2.27.7
# handles it, and is preloaded rather than installed into the venv so torch's
# pinned dependency stays intact. Provision with:
#   lpm_training_venv/bin/pip install --target .nccl/2.27.7 nvidia-nccl-cu12==2.27.7
readonly NCCL_LIB=".nccl/2.27.7/nvidia/nccl/lib/libnccl.so.2"

# Globals populated by the component functions and exported to children.
CONFIG_ID="$DEFAULT_CONFIG_ID"
CONFIG_FILE=""
CACHE_DIR="$DEFAULT_CACHE_DIR"
CACHE_DIR_EXPLICIT=0
MASTER_NODE_SHORT=""
MASTER_ADDR=""
MASTER_PORT=""
MASTER_IP_CANDIDATES=()
PROBE_ERRKIND=""
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
  -c, --config <id>         YAML stem under ${CONFIGS_DIR}/ (default: ${DEFAULT_CONFIG_ID})
  -d, --cache-dir <path>    perturb_lib cache dir; exported as PERTURB_LIB_CACHE_DIR
                            (default: .plib_cache, or .plib_cache_paper_narrowed
                            when -c ${NARROWED_CONFIG_ID})
  -h, --help                Show this help and exit

Env-var overrides:
  CLEAN_RESULTS=0           Skip pre-clean of <cache-dir>/results/<config_id>/
  NCCL_DEBUG=WARN           Quieten NCCL logs (default INFO)
  PERTURB_LIB_CACHE_DIR     Alternative to --cache-dir; -d wins if both are set
  NCCL_IB_DISABLE=1         Fall back to Ethernet. InfiniBand is used by
                            default and needs .nccl/2.27.7 (see NCCL_LIB below)
  NCCL_IB_HCA=<dev>         Override the pinned HCA (default: mlx5_0)
  PROBE_ALL_PORTS=1         Sweep every candidate port even on IPs that look
                            unreachable (default: stop after the first port)
  PORT_SHIM=0               Do not confine kernel-chosen TCP ports to the
                            firewall allowlist (default: 1 = shim enabled)
  PORT_SHIM_VERBOSE=1       Log every port the shim rewrites

Available configs:
EOF
    list_configs
}

# ---------------------------------------------------------------------------
# Stage 1: argument parsing + config validation
# ---------------------------------------------------------------------------
parse_args() {
    # If PERTURB_LIB_CACHE_DIR was exported by the caller, seed CACHE_DIR with it
    # so an unflagged invocation honors the env var. An explicit -d/--cache-dir
    # still wins because it overwrites CACHE_DIR below.
    if [[ -n "${PERTURB_LIB_CACHE_DIR:-}" ]]; then
        CACHE_DIR="$PERTURB_LIB_CACHE_DIR"
        CACHE_DIR_EXPLICIT=1
    fi

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
            -d|--cache-dir|--cache_dir)
                [[ $# -ge 2 ]] || die "'$1' requires a value."
                CACHE_DIR="$2"
                CACHE_DIR_EXPLICIT=1
                shift 2
                ;;
            --cache-dir=*|--cache_dir=*)
                CACHE_DIR="${1#*=}"
                CACHE_DIR_EXPLICIT=1
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

    # Auto-route the L1000-narrowed config to its dedicated cache unless the
    # user already picked a cache via -d or PERTURB_LIB_CACHE_DIR.
    if [[ "$CACHE_DIR_EXPLICIT" -eq 0 && "$CONFIG_ID" == "$NARROWED_CONFIG_ID" ]]; then
        CACHE_DIR="$NARROWED_CACHE_DIR"
    fi
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

    # Hand the cache dir to perturb_lib BEFORE Python imports it. The guard in
    # perturb_lib/environment.py only honors PERTURB_LIB_CACHE_DIR if it is set
    # at import time, so this export has to happen before launch_training.
    # Resolve relative paths against the submit dir so srun children agree.
    if [[ "${CACHE_DIR}" != /* ]]; then
        CACHE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/${CACHE_DIR}"
    fi
    mkdir -p "${CACHE_DIR}"
    export PERTURB_LIB_CACHE_DIR="${CACHE_DIR}"
}

# Print a small header so the log file self-identifies (job id, config, log path).
print_log_header() {
    local jid="${SLURM_JOB_ID:-local}"
    log "================================================================"
    log "Slurm job id : ${jid}"
    log "Config       : ${CONFIG_ID}"
    log "Cache dir    : ${CACHE_DIR}"
    log "Submit dir   : ${SLURM_SUBMIT_DIR:-$(pwd)}"
    log "Log files    : ${LOG_DIR}/perturb_lib.${jid}.{out,err}"
    log "================================================================"
}

# ---------------------------------------------------------------------------
# Stage 3: DDP rendezvous — pick a master IP + port every rank can reach
# ---------------------------------------------------------------------------
resolve_master_node() {
    local master_node
    master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_NODE_SHORT="${master_node%%.*}"
    log "MASTER_NODE=$MASTER_NODE_SHORT  NODES=$SLURM_JOB_NODELIST"
}

# Record every rank's IPv4 interfaces in the log. When the rendezvous fails this
# is the data needed to tell which networks the nodes actually share, so capture
# it unconditionally rather than only on error.
dump_node_interfaces() {
    log "-- IPv4 interfaces per node --"
    srun --ntasks-per-node=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" \
        bash -c 'printf "[%s] " "$(hostname -s)"; ip -4 -o addr show | awk "{printf \"%s=%s \", \$2, \$4}"; echo' \
        2>/dev/null || warn "could not enumerate interfaces."
}

# Candidate master IPs, most likely first. DNS is deliberately last: getent can
# return an address on a network the compute nodes cannot route to (the cluster
# has done exactly this), so the master's own interface list is more trustworthy.
collect_master_ips() {
    local raw dns_ip iface ip
    raw=$(srun --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" -w "$MASTER_NODE_SHORT" \
            bash -c "ip -4 -o addr show | awk '{split(\$4,a,\"/\"); print \$2\" \"a[1]}'" \
            2>/dev/null) || raw=""

    local preferred=() others=()
    while read -r iface ip; do
        [[ -z "${iface:-}" || -z "${ip:-}" ]] && continue
        # Skip container/virtual bridges: they are node-local by construction.
        case "$iface" in
            lo|docker*|veth*|br-*|virbr*|cni*|flannel*|tunl*|kube*|cali*) continue ;;
        esac
        # Loopback and link-local addresses can never serve as a rendezvous.
        case "$ip" in
            127.*|169.254.*) continue ;;
        esac
        case "$iface" in
            bond*|ib*|eth*|en*) preferred+=("$ip") ;;
            *)                  others+=("$ip") ;;
        esac
    done <<< "$raw"

    dns_ip=$(getent hosts "$MASTER_NODE_SHORT" 2>/dev/null | awk '{print $1}' | head -n 1) || dns_ip=""

    MASTER_IP_CANDIDATES=()
    local cand
    for cand in "${preferred[@]:-}" "${others[@]:-}" "$dns_ip"; do
        [[ -z "$cand" ]] && continue
        # de-duplicate
        local seen=0 existing
        for existing in "${MASTER_IP_CANDIDATES[@]:-}"; do
            [[ "$existing" == "$cand" ]] && seen=1 && break
        done
        [[ "$seen" -eq 0 ]] && MASTER_IP_CANDIDATES+=("$cand")
    done

    [[ ${#MASTER_IP_CANDIDATES[@]} -gt 0 ]] || \
        die "could not find any candidate IP for $MASTER_NODE_SHORT."
    log "Candidate master IPs: ${MASTER_IP_CANDIDATES[*]}"
}

# Probe a single (ip, port) pair: spin a short-lived listener on the master node
# and try to connect from every rank. Returns 0 if every rank reaches it, and
# sets PROBE_ERRKIND so the caller can tell an unroutable address (no point
# trying more ports) apart from a merely blocked one.
probe_pair() {
    local ip="$1" port="$2"
    log "-- Probing $ip:$port --"
    PROBE_ERRKIND=""

    # Listener on master.
    srun --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" -w "$MASTER_NODE_SHORT" \
        python3 -c "
import socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', $port))
except OSError as e:
    print(f'LISTENER_BIND_FAIL: {e}'); raise
s.listen(128)
s.settimeout(1.0)
end = time.time() + 20
while time.time() < end:
    try:
        c, _ = s.accept(); c.close()
    except socket.timeout:
        pass
" >/dev/null 2>&1 &
    local pid=$!
    sleep 2

    # Connect probe from every rank.
    local rc=0 out=""
    out=$(srun --ntasks-per-node=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" python3 -c "
import socket, sys, os
host, port = '$ip', $port
hostname = os.uname().nodename.split('.')[0]
try:
    s = socket.socket(); s.settimeout(3); s.connect((host, port)); s.close()
    print(f'[{hostname}] OK -> {host}:{port}')
except Exception as e:
    print(f'[{hostname}] UNREACHABLE -> {host}:{port}  ({e})')
    sys.exit(113)
" 2>&1) || rc=$?
    [[ -n "$out" ]] && echo "$out"

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    if [[ $rc -eq 0 ]]; then
        PROBE_ERRKIND="ok"
    elif grep -q -e 'No route to host' -e 'Network is unreachable' <<< "$out"; then
        PROBE_ERRKIND="noroute"
    elif grep -q 'Connection refused' <<< "$out"; then
        PROBE_ERRKIND="refused"
    elif grep -q -e 'timed out' -e 'timeout' <<< "$out"; then
        PROBE_ERRKIND="filtered"
    else
        PROBE_ERRKIND="other"
    fi
    return $rc
}

# Search (master IP x port) for a pair every rank can reach. An unroutable IP
# fails identically on every port, so stop after the first probe for that IP
# instead of burning ~5s per port on all 31 candidates.
find_rendezvous() {
    local ip port saw_noroute=0 saw_block=0
    for ip in "${MASTER_IP_CANDIDATES[@]}"; do
        for port in "${PORT_CANDIDATES[@]}"; do
            if probe_pair "$ip" "$port"; then
                MASTER_ADDR="$ip"
                MASTER_PORT="$port"
                export MASTER_ADDR MASTER_PORT
                log "Selected MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT (all nodes reachable)."
                return 0
            fi
            if [[ "$PROBE_ERRKIND" == "noroute" ]]; then
                saw_noroute=1
                if [[ "${PROBE_ALL_PORTS:-0}" != "1" ]]; then
                    log "$ip is unreachable (no route to host); skipping its remaining ports."
                    log "Set PROBE_ALL_PORTS=1 to sweep every port anyway (e.g. if only some are rejected)."
                    break
                fi
            fi
            saw_block=1
            log "Port $port on $ip blocked or already in use; trying next candidate."
        done
    done

    warn "no (master IP, port) pair passed the all-peers probe."
    echo "[run.sh] IPs tried  : ${MASTER_IP_CANDIDATES[*]}" >&2
    if [[ "$saw_noroute" -eq 1 && "${PROBE_ALL_PORTS:-0}" != "1" ]]; then
        echo "[run.sh] Ports tried: ${PORT_CANDIDATES[0]} only on unreachable IPs (rest skipped)" >&2
    else
        echo "[run.sh] Ports tried: ${PORT_CANDIDATES[*]}" >&2
    fi
    if [[ "$saw_noroute" -eq 1 ]]; then
        echo "[run.sh] Diagnosis: EHOSTUNREACH ('no route to host') on an address that IS" >&2
        echo "[run.sh] on the peers' own subnet points at a host firewall rejecting inbound" >&2
        echo "[run.sh] TCP with icmp-host-prohibited, not a missing route. Ask the admins" >&2
        echo "[run.sh] which network/ports are sanctioned for NCCL and torch.distributed;" >&2
        echo "[run.sh] see the per-node interface dump above for the shared subnets." >&2
    elif [[ "$saw_block" -eq 1 ]]; then
        echo "[run.sh] Diagnosis: connections were refused/filtered rather than unroutable," >&2
        echo "[run.sh] so routing works and this is a port policy. Ask the admins for the" >&2
        echo "[run.sh] allowed peer-to-peer TCP port range and add it to PORT_CANDIDATES." >&2
    fi
    echo "[run.sh] Workaround: run on a single node via run_single_node.sh (set" >&2
    echo "[run.sh] num_nodes: 1 in the config) -- steps/epoch is unchanged by world size." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Stage 4: NCCL / Gloo — pin to the interface that owns MASTER_ADDR
# ---------------------------------------------------------------------------
pin_nccl_iface() {
    NCCL_IFACE=$(srun --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" \
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

    # IB carries the gradients as RDMA, which never touches the IP stack and so
    # never meets the firewall; only NCCL's out-of-band bootstrap stays on bond0.
    # Verified working on 4 nodes in job 39941914, once the two blockers below
    # (stale NCCL, 8 MB memlock cap) were dealt with.
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    log "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"

    if [[ "$NCCL_IB_DISABLE" == "0" ]]; then
        [[ -f "$NCCL_LIB" ]] || die "IB needs a newer NCCL than torch bundles, missing at $NCCL_LIB.
        Provision it with:
          lpm_training_venv/bin/pip install --target .nccl/2.27.7 nvidia-nccl-cu12==2.27.7
        or run with NCCL_IB_DISABLE=1 to fall back to Ethernet."
        export LD_PRELOAD="${SLURM_SUBMIT_DIR}/${NCCL_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
        log "Preloading NCCL from $NCCL_LIB"

        # Prefix match, so both mlx5_0 and mlx5_1 qualify: every node carries the
        # pair, but which one has the live port varies (gpusrv52 runs on mlx5_1,
        # hence its ib1 interface). Naming a single device makes NCCL report
        # "No device found" on the odd node out, which then falls back to sockets
        # and breaks the communicator with an opaque internal error (job
        # 39941920). The qedr RoCE cards are excluded by not matching.
        export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
        log "NCCL_IB_HCA=$NCCL_IB_HCA"
    fi

    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
}

# ---------------------------------------------------------------------------
# Stage 4b: constrain kernel-chosen TCP ports to the firewall's allowlist
# ---------------------------------------------------------------------------
enable_port_shim() {
    if [[ "${PORT_SHIM:-1}" != "1" ]]; then
        log "Port shim disabled (PORT_SHIM=${PORT_SHIM:-1}); NCCL bootstrap may hang."
        return 0
    fi

    mkdir -p "$(dirname "$SHIM_SO")"
    if [[ ! -f "$SHIM_SO" || "$SHIM_SRC" -nt "$SHIM_SO" ]]; then
        log "Building $SHIM_SO from $SHIM_SRC"
        gcc -O2 -fPIC -shared -o "$SHIM_SO" "$SHIM_SRC" -ldl \
            || die "failed to build $SHIM_SO; rerun with PORT_SHIM=0 to skip it."
    fi

    export PORT_SHIM_RANGE="$SHIM_PORT_RANGE"
    export LD_PRELOAD="${SLURM_SUBMIT_DIR}/${SHIM_SO}${LD_PRELOAD:+:$LD_PRELOAD}"
    log "Port shim active: bind(0) confined to $PORT_SHIM_RANGE"
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
    # `set +u` because mamba's shell hook references unbound vars.
    set +u
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$venv_path"
    set -u
}

# ---------------------------------------------------------------------------
# Stage 6: pre-clean previous results for THIS config
# ---------------------------------------------------------------------------
clean_results() {
    local clean=${CLEAN_RESULTS:-1}
    local results_dir="${CACHE_DIR}/results/${CONFIG_ID}"
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
    # IB memory registration fails with "Cannot allocate memory" under the 8 MB
    # soft memlock limit the nodes default to. The hard limit is unlimited, so
    # raise it inside the task rather than depending on Slurm's limit
    # propagation, which varies with cluster configuration.
    srun --chdir="${SLURM_SUBMIT_DIR}" bash -c '
        ulimit -l unlimited 2>/dev/null || true
        exec poetry run python -m perturb_gym.training train_from_config_file \
            --config_file_id_or_path="$1"
    ' _ "${CONFIG_ID}"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    validate_config
    setup_env
    print_log_header
    resolve_master_node
    dump_node_interfaces
    collect_master_ips
    find_rendezvous
    pin_nccl_iface
    enable_port_shim
    activate_env
    clean_results
    launch_training
}

main "$@"
