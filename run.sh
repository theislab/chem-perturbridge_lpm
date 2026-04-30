#!/bin/bash
#SBATCH -J perturb_lib_lpm
#SBATCH -o perturb_lib.%j.out
#SBATCH -e perturb_lib.%j.err
#SBATCH -t 24:00:00
#SBATCH --qos=gpu_normal
#SBATCH --partition=gpu_p
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=gpusrv34,gpusrv53,gpusrv62

set -euo pipefail

cd "$(dirname "$0")"

export HOME=${HOME:-/home/icb/olga.novitskaia}
export TMPDIR=${SLURM_TMPDIR:-${HOME}/tmp}
mkdir -p ${TMPDIR}

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_NODE_SHORT=$(echo "$MASTER_NODE" | cut -d. -f1)

MASTER_ADDR=$(getent hosts "$MASTER_NODE" 2>/dev/null | awk '{print $1}' | head -n 1)
[[ -z "$MASTER_ADDR" ]] && MASTER_ADDR=$(getent hosts "$MASTER_NODE_SHORT" 2>/dev/null | awk '{print $1}' | head -n 1)
if [[ -z "$MASTER_ADDR" ]]; then
  echo "[run.sh] ERROR: could not resolve $MASTER_NODE_SHORT to an IP." >&2
  exit 1
fi
export MASTER_ADDR
echo "[run.sh] MASTER_NODE=$MASTER_NODE_SHORT  MASTER_ADDR=$MASTER_ADDR  NODES=$SLURM_JOB_NODELIST"

PORT_CANDIDATES=(29500 12355 23456 8888 7000 6000 5000 9000 10000 \
                 30000 35000 40000 45000 50000 55000 60000 4000 3000 2000
                 100 200 300 400 500 600 700 800 900 1000 1100 1200)

probe_addr_port() {
  local ip="$1" port="$2"
  echo "[run.sh] -- Probing $ip:$port --"
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
  local rc=0
  srun --ntasks-per-node=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" python3 -c "
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
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  return $rc
}

MASTER_PORT=""
for port in "${PORT_CANDIDATES[@]}"; do
  if probe_addr_port "$MASTER_ADDR" "$port"; then
    MASTER_PORT="$port"
    echo "[run.sh] Selected MASTER_PORT=$MASTER_PORT (all nodes reachable on $MASTER_ADDR:$port)."
    break
  else
    echo "[run.sh] Port $port blocked or already in use; trying next candidate."
  fi
done

if [[ -z "$MASTER_PORT" ]]; then
  echo "[run.sh] ERROR: none of the candidate ports passed the all-peers probe." >&2
  echo "[run.sh] Tried: ${PORT_CANDIDATES[*]}" >&2
  echo "[run.sh] Next step: ask cluster admins for the allowed peer-to-peer TCP" >&2
  echo "[run.sh] port range, or switch to a shared-FS rendezvous (init_method=file://)." >&2
  exit 1
fi
export MASTER_PORT

NCCL_IFACE=$(srun --nodes=1 --ntasks=1 --overlap --chdir="${SLURM_SUBMIT_DIR}" \
              -w "$MASTER_NODE_SHORT" \
              bash -c "ip -4 -o addr show | awk -v ip='$MASTER_ADDR' '\$4 ~ \"^\"ip\"/\" {print \$2; exit}'" \
              | tr -d '[:space:]')

if [[ -z "$NCCL_IFACE" ]]; then
  echo "[run.sh] WARN: could not resolve interface owning $MASTER_ADDR on $MASTER_NODE_SHORT;" >&2
  echo "[run.sh]       falling back to NCCL/Gloo SOCKET_IFNAME=^lo,docker." >&2
  export GLOO_SOCKET_IFNAME="^lo,docker"
  export NCCL_SOCKET_IFNAME="^lo,docker"
else
  echo "[run.sh] $MASTER_ADDR is on interface '$NCCL_IFACE' on $MASTER_NODE_SHORT; pinning NCCL/Gloo to it."
  export GLOO_SOCKET_IFNAME="$NCCL_IFACE"
  export NCCL_SOCKET_IFNAME="$NCCL_IFACE"
fi
echo "[run.sh] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME (OOB bootstrap on bond0)"

export NCCL_DEBUG=INFO

set +u
eval "$(mamba shell hook --shell bash)"
mamba activate ~/deg_venv
set -u

CLEAN_RESULTS=${CLEAN_RESULTS:-1}
RESULTS_DIR="${SLURM_SUBMIT_DIR}/.plib_cache/results/lpm_modified"
if [[ "$CLEAN_RESULTS" == "1" && -d "$RESULTS_DIR" ]]; then
  echo "[run.sh] Removing stale $RESULTS_DIR"
  rm -rf "$RESULTS_DIR"
fi

srun --chdir="${SLURM_SUBMIT_DIR}" \
  poetry run python -m perturb_gym.training train_from_config_file \
  --config_file_id_or_path=lpm_modified
