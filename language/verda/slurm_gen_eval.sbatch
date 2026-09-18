#!/bin/bash
# ===========================================================================
# slurm_gen_eval.sbatch — one NFE budget of unconditional generation +
# Gen PPL + MAUVE, on ONE GPU.
#
# Budgets are independent, so submit one job per budget and let them run in
# parallel. Generation needs far less memory than training (batch 1, no
# gradients, ~200 MB of logits), so these coexist with other users' jobs on a
# shared card in a way the training runs could not.
#
#   sbatch --export=ALL,T=64,CKPT=/path/0-6999.ckpt  verda/slurm_gen_eval.sbatch
#   sbatch --export=ALL,T=128,CKPT=/path/0-6999.ckpt verda/slurm_gen_eval.sbatch
# ===========================================================================
#SBATCH --job-name=gen-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>

set -euo pipefail

ALG="${ALG:-slerp_euclid_mean}"
CKPT="${CKPT:-}"
T="${T:-128}"
SEED="${SEED:-1}"
BATCH="${BATCH:-1}"
N_SAMPLES="${N_SAMPLES:-5000}"
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/.conda}"

export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/mdlm/data/owt_cache}"
export OUT_DIR="${OUT_DIR:-$HOME/sm_outputs/generations}"

if [[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" 2>/dev/null && echo "[job] conda env '${ENV_NAME}'" \
    || echo "[job] conda env '${ENV_NAME}' unavailable — using current python"
elif [[ -f /opt/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME" 2>/dev/null || true
else
  echo "[job] no conda found — using current python"
fi

cd "$SLURM_SUBMIT_DIR"

echo "[job] node=$(hostname)  T=${T}  ALG=${ALG}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALG="$ALG" CKPT="$CKPT" T="$T" SEED="$SEED" BATCH="$BATCH" N_SAMPLES="$N_SAMPLES" \
DATA_CACHE_DIR="$DATA_CACHE_DIR" OUT_DIR="$OUT_DIR" \
  bash verda/run_gen_eval.sh

echo "[job] done. Result JSON under ${OUT_DIR}"
