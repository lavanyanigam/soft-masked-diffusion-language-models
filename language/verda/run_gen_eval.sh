#!/bin/bash
#
# verda/run_gen_eval.sh — unconditional generation + Gen PPL + MAUVE for one
# NFE budget, on one GPU.
#
# Reproduces the protocol behind Table 1/6: 5000 unconditional samples at
# length 1024, nucleus p=0.9, the `mdlm` sampler, scored with gpt2-large and
# MAUVE against the cached OpenWebText references. One invocation per budget;
# run several at once on separate GPUs.
#
# USAGE (from language/)
#   CKPT=/path/to/7000.ckpt T=256 bash verda/run_gen_eval.sh
#   ALG=slerp_sm CKPT=... T=128 bash verda/run_gen_eval.sh
#
# KNOBS
#   ALG        transparency_alg of the CHECKPOINT (default slerp_euclid_mean)
#   CKPT       required: the checkpoint to sample from
#   T          NFE budget / sampling.steps (default 256)
#   SEED       default 1
#   BATCH      per-step batch (default 1 -- matches the published runs exactly)
#   N_SAMPLES  total samples (default 5000)
#   OUT_DIR    where the result JSON lands
#   DATA_CACHE_DIR, P_NUCLEUS

set -euo pipefail

ALG="${ALG:-slerp_euclid_mean}"
CKPT="${CKPT:-}"
T="${T:-256}"
SEED="${SEED:-1}"
BATCH="${BATCH:-1}"
N_SAMPLES="${N_SAMPLES:-5000}"
P_NUCLEUS="${P_NUCLEUS:-0.9}"
MIXINPUTS_K="${MIXINPUTS_K:-3}"
SLERP_N_ITER="${SLERP_N_ITER:-3}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/sm_data/owt_cache}"
OUT_DIR="${OUT_DIR:-$HOME/sm_outputs/generations}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

if [[ -z "$CKPT" ]]; then echo "[preflight] CKPT is required."; exit 1; fi
if [[ ! -f "$CKPT" ]]; then echo "[preflight] CKPT not found: $CKPT"; exit 1; fi
if [[ ! -d "$DATA_CACHE_DIR" ]]; then
  echo "[preflight] DATA_CACHE_DIR not found: $DATA_CACHE_DIR"; exit 1; fi

if (( N_SAMPLES % BATCH != 0 )); then
  echo "[preflight] N_SAMPLES=${N_SAMPLES} must be divisible by BATCH=${BATCH}."; exit 1
fi
NUM_BATCHES=$(( N_SAMPLES / BATCH ))

mkdir -p "$OUT_DIR"
TAG="${ALG}_T-${T}_topp-${P_NUCLEUS}_seed${SEED}"
OUT_JSON="${OUT_DIR}/${TAG}.json"

echo "[gen] alg=${ALG}  T=${T}  seed=${SEED}  samples=${N_SAMPLES} (batch ${BATCH} x ${NUM_BATCHES})"
echo "[gen] ckpt=${CKPT}"
echo "[gen] out =${OUT_JSON}"
echo "[gen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# NOTE: sampling.predictor is deliberately left at its ddpm_cache default.
# Setting it to "sm" lands in trainer_base.py's self.sampler and routes
# generation down _analytic_update, which applies NO feedback at all -- the
# training launchers set it, and copying them here would silently evaluate a
# model with its feedback pathway switched off.
set -x
python -u -m main \
  mode=sample_eval \
  algo=mdlm_sm \
  algo.tran_head.transparency_alg="$ALG" \
  algo.tran_head.mixinputs_k="$MIXINPUTS_K" \
  algo.tran_head.slerp_n_iter="$SLERP_N_ITER" \
  eval.checkpoint_path="$CKPT" \
  model.length=1024 \
  sampling.steps="$T" \
  sampling.num_sample_batches="$NUM_BATCHES" \
  sampling.p_nucleus="$P_NUCLEUS" \
  sampling.sampler=mdlm \
  loader.batch_size="$BATCH" \
  loader.eval_batch_size="$BATCH" \
  eval.perplexity_batch_size="$BATCH" \
  seed="$SEED" \
  data.cache_dir="$DATA_CACHE_DIR" \
  sampling.generated_seqs_path="$OUT_JSON" \
  +wandb.offline=true \
  hydra.run.dir="${PWD}"
