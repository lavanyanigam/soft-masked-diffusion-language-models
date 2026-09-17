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
#SBATCH --gres=gpu:2                 # TODO: GPUs per node (also the DDP world size)
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

# W&B — online since vitallab1 has internet access.
export WANDB_MODE="${WANDB_MODE:-online}"

# ── Activate the env built by setup_cluster.sh ─────────────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found — run verda/setup_cluster.sh first."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from language/ (using directory where sbatch command was run) ──────
cd "$SLURM_SUBMIT_DIR"

echo "[job] node=$(hostname)  ALG=${ALG}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# The launchers read the GPU count from CUDA_VISIBLE_DEVICES (set by SLURM via
# --gres), so we deliberately do NOT pass NUM_GPUS.
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
