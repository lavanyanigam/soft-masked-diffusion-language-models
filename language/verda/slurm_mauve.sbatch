#!/bin/bash
# ===========================================================================
# slurm_mauve.sbatch — general-purpose MAUVE evaluation on SLURM (single GPU).
#
# Generates NUM_SAMPLES samples from a checkpoint and computes the MAUVE score
# against human OpenWebText references, using main.py's mode=sample_eval path.
#
# Supports vanilla MDLM and soft-masked MDLM (with optional remdm samplers),
# plus auto-downloading the checkpoint from HuggingFace if it's not local.
#
# EDIT the #SBATCH lines marked TODO for your cluster, then submit.
#
# Examples:
#   # Vanilla MDLM, 5000 samples, base ckpt already on disk
#   sbatch --export=ALL,ALGO=mdlm,SAMPLER=mdlm,CHECKPOINT=mdlm_owt.ckpt \
#          verda/slurm_mauve.sbatch
#
#   # Soft-masked finetune from the HF repo (auto-downloaded)
#   sbatch --export=ALL,ALGO=mdlm_sm,SAMPLER=mdlm,CHECKPOINT=slerp_seed3_lambda_learnt_step5000.ckpt,HF_REPO=lavanyanigam/soft-masking-checkpoints \
#          verda/slurm_mauve.sbatch
#
#   # Soft-masked + remdm-loop
#   sbatch --export=ALL,ALGO=mdlm_sm,SAMPLER=remdm-loop,ETA=0.02,T_ON=0.55,T_OFF=0.05,ALPHA_ON=0.9,CHECKPOINT=slerp_seed3_lambda_learnt_step5000.ckpt,HF_REPO=lavanyanigam/soft-masking-checkpoints \
#          verda/slurm_mauve.sbatch
#
#   # Quick smoke test (250 samples)
#   sbatch --export=ALL,ALGO=mdlm,CHECKPOINT=mdlm_owt.ckpt,NUM_SAMPLES=250,TIME=02:00:00 \
#          verda/slurm_mauve.sbatch
#
# ===========================================================================
#SBATCH --job-name=mauve-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                 # single GPU — vanilla MDLM is single-process
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # ~8 dataloader workers is plenty for batch=1
#SBATCH --mem=64G                    # TODO: cluster-dependent
#SBATCH --time=08:00:00              # TODO: 5000 samples @ seq_len=1024 takes ~5-7h; override with TIME for smoke tests
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<PARTITION>     # TODO: uncomment + set
##SBATCH --account=<ACCOUNT>         # TODO: uncomment + set if your cluster needs it

set -euo pipefail

# ── Required args ───────────────────────────────────────────────────────────
ALGO="${ALGO:-mdlm}"                 # mdlm | mdlm_sm
CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT=<filename.ckpt> via --export=ALL,CHECKPOINT=...}"

# ── Optional args (sensible defaults match the rest of the repo) ──────────
SAMPLER="${SAMPLER:-mdlm}"           # mdlm | remdm-loop | remdm-cap
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"
P_NUCLEUS="${P_NUCLEUS:-0.9}"
SEED="${SEED:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
SEQ_LEN="${SEQ_LEN:-1024}"
PERPLEXITY_BATCH_SIZE="${PERPLEXITY_BATCH_SIZE:-1}"

# remdm-loop / remdm-cap parameters (ignored when SAMPLER=mdlm)
ETA="${ETA:-0.02}"
T_ON="${T_ON:-0.55}"
T_OFF="${T_OFF:-0.05}"
ALPHA_ON="${ALPHA_ON:-0.9}"

# transparency head (only used when ALGO=mdlm_sm)
TRANSPARENCY_ALG="${TRANSPARENCY_ALG:-slerp_sm}"  # slerp_sm | topk_sm
FIXED_LAMBDA="${FIXED_LAMBDA:--1.0}"               # <0 keeps learned lambda; else pin to [0,1]

# Optional HuggingFace source — auto-downloads CHECKPOINT into CKPT_DIR if missing
HF_REPO="${HF_REPO:-}"                              # e.g. lavanyanigam/soft-masking-checkpoints
HF_REVISION="${HF_REVISION:-main}"
HF_TOKEN="${HF_TOKEN:-}"

# ── Paths / env ────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-sm-env}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

# Default checkpoint directory — overridden by HF_AUTO_DIR if downloading
CKPT_DIR="${CKPT_DIR:-$HOME/checkpoints}"
BASE_CKPT="${CKPT_DIR}/${CHECKPOINT}"

export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/sm_data/owt_cache}"
export OUT_DIR="${OUT_DIR:-$HOME/sm_outputs/mauve/${ALGO}_${SAMPLER}_${CHECKPOINT%.ckpt}_seed${SEED}}"

# W&B stays offline — sync from the login node later if you want.
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── Validate ───────────────────────────────────────────────────────────────
case "$ALGO" in
  mdlm|mdlm_sm) ;;
  *) echo "[job] ALGO must be 'mdlm' or 'mdlm_sm', got '${ALGO}'"; exit 1 ;;
esac
case "$SAMPLER" in
  mdlm|remdm-loop|remdm-cap) ;;
  *) echo "[job] SAMPLER must be 'mdlm', 'remdm-loop', or 'remdm-cap'; got '${SAMPLER}'"; exit 1 ;;
esac

# ── Activate env (built by setup_cluster.sh) ───────────────────────────────
if [[ ! -d "$CONDA_HOME" ]]; then
  echo "[job] CONDA_HOME=$CONDA_HOME not found — run verda/setup_cluster.sh first."; exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── Run from language/ ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

mkdir -p "$CKPT_DIR" "$OUT_DIR"

echo "[job] node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[job] ALGO=$ALGO  SAMPLER=$SAMPLER  CHECKPOINT=$CHECKPOINT"
echo "[job] NUM_SAMPLES=$NUM_SAMPLES  STEPS=$SAMPLING_STEPS  topp=$P_NUCLEUS  seed=$SEED  seq_len=$SEQ_LEN"
echo "[job] BASE_CKPT=$BASE_CKPT"
echo "[job] DATA_CACHE_DIR=$DATA_CACHE_DIR"
echo "[job] OUT_DIR=$OUT_DIR"
[[ -n "$HF_REPO" ]] && echo "[job] HF_REPO=$HF_REPO  HF_REVISION=$HF_REVISION"

# ── Optional: download checkpoint from HuggingFace if missing locally ─────
if [[ ! -f "$BASE_CKPT" ]]; then
  if [[ -z "$HF_REPO" ]]; then
    echo "[job] ERROR: $BASE_CKPT not found and HF_REPO not set."
    echo "[job] Either upload the checkpoint to $CKPT_DIR or set HF_REPO=<user/repo>."
    exit 1
  fi
  echo "[hf] Missing locally; downloading $CHECKPOINT from ${HF_REPO}@${HF_REVISION} into $CKPT_DIR ..."
  python - <<PY
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    revision="${HF_REVISION}",
    allow_patterns=["${CHECKPOINT}"],
    local_dir="${CKPT_DIR}",
    token=os.environ.get("HF_TOKEN") or "${HF_TOKEN}" or None,
)
PY
fi

# Confirm we have the file now
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[job] ERROR: $BASE_CKPT still not found after download attempt."
  exit 1
fi

# ── Drop GPUs this build cannot use ────────────────────────────────────────
# vitallab2 advertises an untyped "gpu:4" gres, so --gres=gpu:1 can hand back
# the RTX PRO 5000 Blackwell, which has no kernels in this CUDA 12.4 wheel.
# Test what was allocated and keep only what works. A healthy allocation is
# left exactly as Slurm gave it.
PINPY="$(mktemp)"
cat > "$PINPY" <<'PYEOF'
import os
try:
    import torch
except Exception:
    raise SystemExit
vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
phys = [x for x in vis.split(",") if x != ""] or [str(i) for i in range(torch.cuda.device_count())]
if len(phys) != torch.cuda.device_count():
    raise SystemExit
ok = []
for i, pid in enumerate(phys):
    try:
        torch.zeros(1, device="cuda:%d" % i)
        ok.append(pid)
    except Exception:
        pass
if ok and len(ok) != len(phys):
    print(ok[0])          # generation is single-GPU; one working card is enough
PYEOF
PIN="$(python "$PINPY" 2>/dev/null || true)"
rm -f "$PINPY"
if [[ -n "$PIN" ]]; then
  export CUDA_VISIBLE_DEVICES="$PIN"
  echo "[job] dropped unusable GPU(s); using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

GENERATED_SEQS_PATH="${OUT_DIR}/${ALGO}_${SAMPLER}_T-${SAMPLING_STEPS}_topp-${P_NUCLEUS}_eta-${ETA}_ton-${T_ON}_toff-${T_OFF}_alphaon-${ALPHA_ON}.json"

# ── Build the python command (mode=sample_eval runs generation + MAUVE) ───
PY_ARGS=(
  -u -m main
  "data.cache_dir=${DATA_CACHE_DIR}"
  "mode=sample_eval"
  "loader.batch_size=1"
  "loader.eval_batch_size=1"
  "eval.perplexity_batch_size=${PERPLEXITY_BATCH_SIZE}"
  "algo=${ALGO}"
  "model.length=${SEQ_LEN}"
  "eval.checkpoint_path=${BASE_CKPT}"
  "+wandb.offline=true"
  "hydra.run.dir=${PWD}"
  "sampling.steps=${SAMPLING_STEPS}"
  "sampling.num_sample_batches=${NUM_SAMPLES}"
  "sampling.p_nucleus=${P_NUCLEUS}"
  "sampling.sampler=${SAMPLER}"
  "sampling.generated_seqs_path=${GENERATED_SEQS_PATH}"
  "seed=${SEED}"
)

# remdm params (forwarded regardless of sampler; main.py only uses them when needed)
PY_ARGS+=(
  "sampling.eta=${ETA}"
  "sampling.t_on=${T_ON}"
  "sampling.t_off=${T_OFF}"
  "sampling.alpha_on=${ALPHA_ON}"
)

# soft-masking transparency head (forwarded regardless of algo; main.py only uses them for mdlm_sm)
if [[ "$ALGO" == "mdlm_sm" ]]; then
  PY_ARGS+=(
    "algo.transparency_alg=${TRANSPARENCY_ALG}"
    "algo.fixed_lambda=${FIXED_LAMBDA}"
  )
fi

echo "[job] launching: python ${PY_ARGS[*]}"
python "${PY_ARGS[@]}"

echo "[job] done. MAUVE result JSON: ${GENERATED_SEQS_PATH}"
[[ "$WANDB_MODE" == "offline" ]] && \
  echo "[job] W&B was offline — sync later:  wandb sync ${OUT_DIR}/wandb/offline-run-*"