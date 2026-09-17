#!/bin/bash
#
# verda/run_slerp_euclid.sh — single-node multi-GPU launcher for the
# `slerp_euclid_mean` reviewer ablation: the SLERP blend and norm restoration of
# `slerp_sm`, but with the normalised Euclidean (LERP) mean of the top-k
# embeddings in place of the Frechet mean.
#
# This is a byte-for-byte copy of verda/run_slerp.sh apart from
# `transparency_alg` and the run/group names. Keep it that way — the ablation is
# only controlled if every other hyperparameter is identical.
#
# DDP-safety: the soft-mask gate (algo.py MDLM_SM.nll) decides the branch
# identically on every rank (step-seeded Bernoulli + globally all-reduced
# t-band), so all ranks stay in sync. find_unused_parameters=True is required
# because tran_head params receive no grad on non-soft-mask steps (on ALL ranks).
#
# USAGE (from language/, env activated):
#   BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp_euclid.sh
#   NUM_GPUS=4 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp_euclid.sh
#   SEED=1 MAX_STEPS=7000 BASE_CKPT=/data/mdlm_owt.ckpt bash verda/run_slerp_euclid.sh
#
# KNOBS (env vars, defaults match the Modal runner):

set -euo pipefail

# --- Establish strict defaults (identical to verda/run_slerp.sh) ------------
NUM_GPUS="${NUM_GPUS:-}"                       # default: all visible GPUs
SEED="${SEED:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"
MODEL="${MODEL:-small}"
DATA="${DATA:-openwebtext-split}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-./data_cache/owt_cache}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SLERP_N_ITER="${SLERP_N_ITER:-3}"
MIXINPUTS_K="${MIXINPUTS_K:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-3}"
FIXED_LAMBDA="${FIXED_LAMBDA:--1.0}"           # >=0 pins lambda (must be in [0,1])
BASE_CKPT="${BASE_CKPT:-}"                      # REQUIRED: finetune-from checkpoint
OUT_ROOT="${OUT_ROOT:-./outputs}"

# Run from language/ (this script lives in language/verda/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LANG_DIR"

# --- Resolve GPU count + pin CUDA_VISIBLE_DEVICES ---------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE_N="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
  if [[ -n "$NUM_GPUS" && "$NUM_GPUS" != "$VISIBLE_N" ]]; then
    echo "[preflight] CUDA_VISIBLE_DEVICES exposes ${VISIBLE_N} GPU(s) but NUM_GPUS=${NUM_GPUS}."
    echo "            Unset one (CUDA_VISIBLE_DEVICES wins for specific GPUs)."; exit 1
  fi
  NUM_GPUS="$VISIBLE_N"
  echo "[preflight] honoring CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPU[s])"
else
  TOTAL_N="$(python -c 'import torch; print(torch.cuda.device_count())')"
  if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS="$TOTAL_N"
    echo "[preflight] NUM_GPUS not set — default to all visible GPUs (${NUM_GPUS})"
  fi
  if [[ "$NUM_GPUS" -gt "$TOTAL_N" ]]; then
    echo "[preflight] requested NUM_GPUS=${NUM_GPUS} but only ${TOTAL_N} GPU(s) exist."; exit 1
  fi
  # Pin to the first N GPUs so device_count() returns exactly NUM_GPUS
  GPUS_SEQ="$(seq -s, 0 $((NUM_GPUS - 1)))"
  export CUDA_VISIBLE_DEVICES="$GPUS_SEQ"
  echo "[preflight] setting CUDA_VISIBLE_DEVICES=${GPUS_SEQ} to restrict device_count()"
fi

if [[ -z "$BASE_CKPT" ]]; then
  echo "[preflight] BASE_CKPT env var is required (path to base .ckpt file)."; exit 1
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[preflight] checkpoint not found: ${BASE_CKPT}"; exit 1
fi

# Assert batch size parameters are divisible
if [[ $(( GLOBAL_BATCH % (NUM_GPUS * PER_GPU_BATCH) )) -ne 0 ]]; then
  echo "[preflight] ERROR: global_batch (${GLOBAL_BATCH}) must be divisible by num_gpus * batch_size (${NUM_GPUS} * ${PER_GPU_BATCH} = $((NUM_GPUS * PER_GPU_BATCH)))"
  exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (NUM_GPUS * PER_GPU_BATCH) ))

# Validate FIXED_LAMBDA
if ! python -c "
v = float('${FIXED_LAMBDA}')
if v >= 0.0 and not 0.0 <= v <= 1.0:
    import sys; sys.stderr.write('ERROR: FIXED_LAMBDA must lie in [0, 1], got ' + str(v) + '\n'); sys.exit(1)
"; then
  exit 1
fi

# Prepare Output Directories
GROUP="slerp_euclid_mg_${NUM_GPUS}gpu_${MODEL}_${DATA}_seed${SEED}"
RUN_NAME="slerp-euclid-mg-${NUM_GPUS}gpu-${MODEL}-${DATA}-seed${SEED}"
OUT_DIR="${OUT_ROOT}/${GROUP}"
mkdir -p "$OUT_DIR"

echo "[config] alg=slerp_euclid_mean  GPUs=${NUM_GPUS}  per-GPU=${PER_GPU_BATCH}  accum=${ACCUM}  global=${GLOBAL_BATCH}"
echo "[config] seed=${SEED}  steps=${MAX_STEPS}  ckpt=${BASE_CKPT}"
echo "[config] out_dir=${OUT_DIR}"

USE_FL=0
if python -c "import sys; sys.exit(0 if float('${FIXED_LAMBDA}') >= 0.0 else 1)" 2>/dev/null; then
  USE_FL=1
fi

ARGS=(
  algo=mdlm_sm
  algo.tran_head.transparency_alg=slerp_euclid_mean
  algo.tran_head.slerp_n_iter="$SLERP_N_ITER"
  algo.tran_head.mixinputs_k="$MIXINPUTS_K"
  algo.tran_head.init_scale=0.5
  algo.tran_head.init_centre=-4
  model="$MODEL"
  data="$DATA"
  data.cache_dir="$DATA_CACHE_DIR"
  seed="$SEED"
  loader.global_batch_size="$GLOBAL_BATCH"
  loader.eval_global_batch_size="$GLOBAL_BATCH"
  loader.batch_size="$PER_GPU_BATCH"
  loader.eval_batch_size="$PER_GPU_BATCH"
  loader.num_workers="$NUM_WORKERS"
  strategy=ddp
  strategy.find_unused_parameters=True
  trainer.devices="$NUM_GPUS"
  trainer.num_nodes=1
  trainer.max_steps="$MAX_STEPS"
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=0
  trainer.log_every_n_steps="$LOG_EVERY"
  optim.lr=3e-5
  optim.tran_head_lr=0.01
  optim.sm_prob=0.8
  optim.sm_t_min=0.2
  optim.sm_t_max=0.8
  lr_scheduler.num_warmup_steps=200
  sampling.predictor=sm
  eval.compute_generative_perplexity=False
  eval.generate_samples=False
  training.finetune_path="$BASE_CKPT"
  checkpointing.resume_from_ckpt=false
  callbacks.checkpoint_every_n_steps.every_n_train_steps="$CHECKPOINT_EVERY"
  wandb.project="finetune-main"
  wandb.entity="slerp-on-smdlm"
  wandb.group="$GROUP"
  wandb.name="$RUN_NAME"
  ++hydra.run.dir="$OUT_DIR"
  ++checkpointing.save_dir="$OUT_DIR"
)
[[ "$USE_FL" == "1" ]] && ARGS+=( algo.tran_head.fixed_lambda="$FIXED_LAMBDA" )

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
set -x
python -u -m main "${ARGS[@]}"
