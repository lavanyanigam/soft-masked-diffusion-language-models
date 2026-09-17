#!/bin/bash
# ===========================================================================
# slurm_train.sbatch — submit a single-node multi-GPU slerp/topk finetune.
#
# Targets ONE node with N GPUs (single-node DDP — what the soft-mask gate
# covers). Lightning spawns one process per GPU itself, so we use --ntasks=1
# and call plain `python` (via the run_*.sh launcher); do NOT wrap it in srun
# with ntasks-per-node>1 or Lightning's SLURM integration will fight the
# launcher over the world size.
#
# EDIT THE #SBATCH LINES marked TODO for your cluster, then submit:
#   sbatch verda/slurm_train.sbatch                       # slerp (default)
#   sbatch --export=ALL,ALG=topk verda/slurm_train.sbatch # top-k
#   sbatch --export=ALL,ALG=slerp,FIXED_LAMBDA=0.5,SEED=2 verda/slurm_train.sbatch
#
# Change the GPU count by editing --gres=gpu:N below (the launcher reads the
# count from SLURM's CUDA_VISIBLE_DEVICES — do NOT also set NUM_GPUS).
# ===========================================================================
#SBATCH --job-name=sm-mdlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                 # take the node; the job pins a matched subset below
#SBATCH --ntasks=1                   # one task; Lightning spawns the per-GPU procs
#SBATCH --cpus-per-task=16           # TODO: ~4 dataloader workers/GPU
#SBATCH --mem=128G                   # TODO: cluster-dependent
#SBATCH --time=168:00:00             # TODO: walltime
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>     # TODO: uncomment + set
##SBATCH --account=<ACCOUNT>         # TODO: uncomment + set if your cluster needs it

set -euo pipefail

# ── User config (override via --export=ALL,VAR=...) ────────────────────────
ALG="${ALG:-slerp}"                    # slerp | slerp_euclid | topk | lerp_renorm
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-/opt/miniforge3}"

# Persistent storage visible from the compute node.
BASE_CKPT="${BASE_CKPT:-$HOME/mdlm/soft-masked-diffusion-language-models/mdlm.ckpt}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/sm_data/owt_cache}"
export OUT_ROOT="${OUT_ROOT:-$HOME/sm_outputs}"

# Run hyperparameters — all exported so run_slerp.sh / run_topk.sh inherit them.
export SEED="${SEED:-3}"
export MAX_STEPS="${MAX_STEPS:--1}"           # -1 = unlimited
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
export FIXED_LAMBDA="${FIXED_LAMBDA:--1.0}"   # >=0 pins lambda; -1.0 = learned schedule

# W&B — online since the lab boxes have internet access. WANDB_API_KEY should be
# passed through --export=ALL,WANDB_API_KEY=... when the node's saved netrc
# belongs to a different account.
export WANDB_MODE="${WANDB_MODE:-online}"

# NCCL. vitallab2 (driver 580 / CUDA 13, NCCL 2.21.5) fails its first collective
# with "unhandled cuda error ... invalid argument" unless P2P is off. Disabling
# P2P costs almost nothing on a single node at this batch size. Set
# NCCL_P2P_DISABLE=0 to restore it. Ignored entirely on a 1-GPU job.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ── Activate the conda env if there is one ─────────────────────────────────
# Not every box has one: on vitallab2 the dependencies are installed into the
# user site-packages of the system python, and there is no sm-env to activate.
# Missing conda is therefore a note, not a failure.
if [[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
    echo "[job] activated conda env '${ENV_NAME}'"
  else
    echo "[job] conda env '${ENV_NAME}' not found — using the current python"
  fi
else
  echo "[job] no conda at ${CONDA_HOME} — using the current python"
fi

# ── Run from language/ (using directory where sbatch command was run) ──────
cd "$SLURM_SUBMIT_DIR"

echo "[job] node=$(hostname)  ALG=${ALG}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Compute capabilities of the ALLOCATED devices. Two different architectures in
# one job is the second known cause of the NCCL "invalid argument" failure --
# a Blackwell card has no kernels in a CUDA 12.4 build -- and nvidia-smi above
# does not show it.
python - <<'PYEOF' || true
import torch
print(f"[job] torch {torch.__version__}  cuda {torch.version.cuda}  visible {torch.cuda.device_count()}")
caps = set()
for i in range(torch.cuda.device_count()):
    c = torch.cuda.get_device_capability(i)
    caps.add(c)
    print(f"[job]   gpu{i} {torch.cuda.get_device_name(i)} sm_{c[0]}{c[1]}")
if len(caps) > 1:
    print("[job] WARNING: mixed GPU architectures in this allocation — NCCL will likely fail.")
    print("[job]          request matched cards (--gres=gpu:a6000:N) or use --gres=gpu:1.")
PYEOF

# ── Pin to a homogeneous set of GPUs ───────────────────────────────────────
# Where the cluster's gres is untyped (vitallab2 advertises a bare "gpu:4"),
# Slurm cannot be asked for matched cards, and an allocation mixing
# architectures kills NCCL. So take the whole node and choose here.
#
# Selection: the most common GPU model in the allocation, then the largest
# count N that keeps the global batch divisible -- 512 / (N * 16) must be a
# whole number, so N is one of 4, 2, 1. Override with GPUS_WANTED.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || [[ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 1 ]]; then
  PIN="$(python - <<'PYEOF'
import subprocess, collections, os
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    raise SystemExit
rows = [r.split(",", 1) for r in out.splitlines() if r.strip()]
by = collections.defaultdict(list)
for i, name in rows:
    by[name.strip()].append(i.strip())
# Largest homogeneous group; ties broken by model name for determinism.
model, idxs = max(by.items(), key=lambda kv: (len(kv[1]), kv[0]))
want = os.environ.get("GPUS_WANTED")
n = int(want) if want else next((c for c in (4, 2, 1) if c <= len(idxs)), 1)
n = min(n, len(idxs))
print(",".join(idxs[:n]))
PYEOF
)"
  if [[ -n "$PIN" ]]; then
    export CUDA_VISIBLE_DEVICES="$PIN"
    echo "[job] pinned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (matched GPUs only)"
  fi
fi

# The launchers read the GPU count from CUDA_VISIBLE_DEVICES, so we deliberately
# do NOT pass NUM_GPUS.
case "$ALG" in
  slerp)        LAUNCHER="verda/run_slerp.sh" ;;
  slerp_euclid) LAUNCHER="verda/run_slerp_euclid.sh" ;;
  topk)         LAUNCHER="verda/run_topk.sh" ;;
  lerp_renorm)  LAUNCHER="verda/run_lerp_renorm.sh" ;;
  *) echo "[job] ALG must be 'slerp', 'slerp_euclid', 'topk', or 'lerp_renorm', got '${ALG}'"; exit 1 ;;
esac

echo "[job] launching ${LAUNCHER}"
BASE_CKPT="$BASE_CKPT" bash "$LAUNCHER"

echo "[job] done. Outputs under ${OUT_ROOT}"
[[ "$WANDB_MODE" == "offline" ]] && \
  echo "[job] W&B was offline — sync later:  wandb sync ${OUT_ROOT}/*/wandb/offline-run-*"
