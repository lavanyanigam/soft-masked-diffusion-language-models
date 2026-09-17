"""
Modal runner for the SLERP vs top-k soft-masking A/B experiment.

Setup (one-time, local):
  pip install modal
  modal setup                              # browser auth
  modal secret create wandb-secret WANDB_API_KEY=<your-key>

Upload the base MDLM checkpoint:
  modal volume put mdlm-checkpoints /local/path/mdlm_owt.ckpt mdlm_owt.ckpt

Run the A/B (sequential, one A100):
  cd language/
  modal run run_modal.py

Run in parallel (two A100s simultaneously):
  modal run run_modal.py --parallel

Run only one arm:
  modal run run_modal.py --alg topk
  modal run run_modal.py --alg slerp

Recovery variant (freeze backbone for first 1000 steps):
  modal run run_modal.py --freeze-until 1000

Download outputs:
  modal volume get mdlm-outputs <remote-path> <local-path>
"""

import os
import sys

import modal

# ── volumes (persistent across runs) ─────────────────────────────────────────
ckpt_volume = modal.Volume.from_name("mdlm-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("mdlm-data", create_if_missing=True)
out_volume = modal.Volume.from_name("mdlm-outputs", create_if_missing=True)

CKPT_DIR = "/vol/checkpoints"
DATA_DIR = "/vol/data"
OUT_DIR = "/vol/outputs"

# ── container image ───────────────────────────────────────────────────────────
# debian_slim + PyTorch cu124 wheels: torch bundles its own CUDA runtime so we
# don't need a heavy nvidia/cuda devel base.  flash_attn 2.7.4.post1 ships
# prebuilt PyPI wheels for torch 2.5.x / CUDA 12.4 / Python 3.12, so the
# ~15-minute compilation step is completely skipped.
#
# Version deltas vs requirements.txt (which targets torch 2.3.1):
#   torch 2.3.1  → 2.5.1   (API-stable for this use-case)
#   torchvision 0.18.1 → 0.20.1
#   torchaudio 2.3.1   → 2.5.1
#   triton 2.2.0       → 3.1.0  (required by torch 2.5.1)
image = (
    modal.Image.debian_slim(python_version="3.12").apt_install(
        "git", "curl", "patch", "git-lfs"
    )
    # Step 1: install ALL non-torch deps first.  Pin numpy<2 because the older
    # ecosystem here (transformers 4.38, datasets 2.15) is not numpy-2 ready,
    # and a transitive resolver may otherwise pull numpy 2.x.
    .pip_install(
        "numpy<2",
        "datasets==2.15.0",
        "einops==0.7.0",
        "fsspec==2023.10.0",  # match datasets 2.15 constraint
        "h5py==3.10.0",
        "hydra-core==1.3.2",
        "lightning==2.2.1",
        "omegaconf==2.3.0",
        "packaging==23.2",
        "pandas==2.2.1",
        "rich==13.7.1",
        "scikit-learn==1.4.0",
        "transformers==4.38.2",
        "torchmetrics==1.6.1",
        "wandb",
        "timm",
        "huggingface-hub",
        "hf_transfer",
        "mauve-text",  # imported eagerly by main.py even when unused
    )
    # Step 2: install torch LAST with cu124 index — overrides any CPU torch
    # that lightning/transformers may have pulled in transitively.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "triton==3.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Step 3: flash_attn wheel — binds to the cu124 torch installed above.
    # --no-deps is critical: the wheel's metadata declares a loose torch dep
    # that pip otherwise uses to downgrade torch to 2.4.1 (breaks ABI).
    # Verified at https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
    .run_commands(
        "pip install --no-deps "
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    )
)

# ── local code copy ──────────────────────────────────────────────────────────
# Bake the entire language/ directory into the image at build time.
# DUO files (models/dit.py etc.) are downloaded by setup.sh at runtime if absent.
# `copy=True` makes the files part of the image layer (cached, fast warm starts).
LANGUAGE_DIR = "/workspace/language"
image = image.add_local_dir(".", remote_path=LANGUAGE_DIR, copy=True)

app = modal.App("slerp-vs-topk-finetune")


# ── training function ─────────────────────────────────────────────────────────


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A100"),
    timeout=24 * 3600,  # 24 h — 7k steps at global batch 512 on one A100
    volumes={
        CKPT_DIR: ckpt_volume,
        DATA_DIR: data_volume,
        OUT_DIR: out_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    transparency_alg: str,
    base_ckpt_filename: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    freeze_until: int = 0,
    global_batch_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    val_check_interval: int = 200,
    checkpoint_every_n_steps: int = 100,
    freeze_transparency_head: bool = False,
    fixed_lambda: float = -1.0,
    log_every_n_steps: int = 3,
):
    import shutil
    import subprocess

    # ── preflight ────────────────────────────────────────────────────────────
    base_ckpt = f"{CKPT_DIR}/{base_ckpt_filename}"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at {base_ckpt}.\n"
            f"Upload it with:\n"
            f"  modal volume put mdlm-checkpoints /local/path/mdlm.ckpt {
                base_ckpt_filename
            }"
        )

    # The mounted source at LANGUAGE_DIR is read-only — setup.sh would fail
    # when curl tries to write models/dit.py.  Copy to a writable workdir.
    work_dir = "/root/language"
    if not os.path.exists(work_dir):
        shutil.copytree(LANGUAGE_DIR, work_dir)
    os.chdir(work_dir)

    # ── setup.sh: pull DUO files + apply patches (skipped if already present) ─
    if not os.path.exists(f"{work_dir}/models/dit.py"):
        print("[setup] DUO files missing — running setup.sh ...")
        subprocess.run(["bash", "setup.sh"], check=True, cwd=work_dir)
    else:
        print("[setup] DUO files present — skipping setup.sh")

    # ── validate cached dataset dirs ──────────────────────────────────────────
    # A previous crashed run can leave a .dat directory that exists but has
    # incomplete Arrow files. fsspec_exists() returns True, load_from_disk()
    # then raises FileNotFoundError. Detect and remove these before training.
    import datasets as hf_datasets

    owt_cache = f"{DATA_DIR}/owt_cache"
    for dat_name in [
        f"{data.replace('-split', '-train')}_train_bs1024_wrapped.dat",
        f"{data.replace('-split', '-valid')}_validation_bs1024_wrapped.dat",
    ]:
        dat_path = os.path.join(owt_cache, dat_name)
        if os.path.exists(dat_path):
            try:
                hf_datasets.load_from_disk(dat_path)
                print(f"[cache] Valid: {dat_path}")
            except Exception as e:
                raise RuntimeError(f"Cache corrupted at {dat_path}: {e}. Aborting to prevent re-tokenization costs.")
        else:
            raise FileNotFoundError(f"Cache directory not found at {dat_path}. Aborting to prevent re-tokenization costs.")

    # ── build the hydra command ───────────────────────────────────────────────
    if transparency_alg == "mixinputs_with_topk":
        alg_tag = "topk"
    elif transparency_alg == "slerp_sm":
        alg_tag = "slerp"
    elif transparency_alg == "slerp_euclid_mean":
        alg_tag = "slerp_euclid"
    elif transparency_alg == "lerp_renorm":
        alg_tag = "lerp_renorm"
    else:
        alg_tag = transparency_alg
    group = f"slerp_vs_topk_v2_{model}_{data}_seed{seed}"
    run_name = f"{alg_tag}-ft-v2-{model}-{data}-seed{seed}"
    out_dir = f"{OUT_DIR}/{group}/{alg_tag}"

    if global_batch_size % batch_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"batch_size ({batch_size}) for this single-GPU launcher."
        )
    if fixed_lambda >= 0.0 and not 0.0 <= fixed_lambda <= 1.0:
        raise ValueError(
            f"fixed_lambda must lie in [0, 1] when set, got {fixed_lambda}"
        )
    accumulate_grad_batches = global_batch_size // batch_size

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "main",
        "algo=mdlm_sm",
        f"model={model}",
        f"data={data}",
        f"data.cache_dir={DATA_DIR}/owt_cache",
        f"seed={seed}",
        f"loader.global_batch_size={global_batch_size}",
        f"loader.eval_global_batch_size={global_batch_size}",
        f"loader.batch_size={batch_size}",
        f"loader.eval_batch_size={batch_size}",
        # The default affinity-derived value can deadlock in Modal containers;
        # a small fixed worker pool is much safer and keeps the GPU fed.
        f"loader.num_workers={num_workers}",
        # bypass ${device_count:} dynamic resolver — returns 0 before CUDA init
        "trainer.devices=1",
        f"trainer.max_steps={max_steps}",
        # Skip the extra validation warmup pass; on OWT it adds a lot of wall time
        # before the first real training step.
        "trainer.num_sanity_val_steps=0",
        # Disable mid-training validation entirely: no validation loop runs
        # between training steps (limit_val_batches=0 turns the val loop off, so
        # val_check_interval is irrelevant).
        "trainer.limit_val_batches=0",
        # Show progress sooner. With grad accumulation each logged "step" is expensive.
        f"trainer.log_every_n_steps={log_every_n_steps}",
        "optim.lr=3e-5",
        "optim.tran_head_lr=0.01",
        "optim.sm_prob=0.8",
        # Soft-masking time band (original soft-masking paper): t in [0.2, 0.8].
        "optim.sm_t_min=0.2",
        "optim.sm_t_max=0.8",
        "lr_scheduler.num_warmup_steps=200",
        "sampling.predictor=sm",
        "strategy.find_unused_parameters=True",
        "eval.compute_generative_perplexity=False",
        # sample generation OOMs on A100-40GB (6 GB for [batch,seq,vocab] Gumbel tensor)
        "eval.generate_samples=False",
        "algo.tran_head.mixinputs_k=3",
        "algo.tran_head.init_scale=0.5",
        "algo.tran_head.init_centre=-4",
        f"algo.tran_head.learnable={str(not freeze_transparency_head).lower()}",
        f"training.finetune_path={base_ckpt}",
        "checkpointing.resume_from_ckpt=false",
        f"callbacks.checkpoint_every_n_steps.every_n_train_steps={checkpoint_every_n_steps}",
        f"wandb.group={group}",
        f"algo.tran_head.transparency_alg={transparency_alg}",
        f"wandb.name={run_name}",
        f"++hydra.run.dir={out_dir}",
        f"++checkpointing.save_dir={out_dir}",
    ]

    if transparency_alg in ("slerp_sm", "slerp_euclid_mean"):
        cmd.append(f"algo.tran_head.slerp_n_iter={slerp_n_iter}")
    if fixed_lambda >= 0.0:
        cmd.append(f"algo.tran_head.fixed_lambda={fixed_lambda}")

    if freeze_until > 0:
        cmd += [
            "+callbacks/freeze_backbone=freeze_backbone",
            f"callbacks.freeze_backbone.freeze_until_step={freeze_until}",
        ]

    print(f"[train] Starting: {run_name}")
    print(f"[train] Output dir: {out_dir}")
    print(
        "[train] Effective schedule: "
        f"global_batch_size={global_batch_size}, batch_size={batch_size}, "
        f"accumulate_grad_batches={accumulate_grad_batches}, "
        f"mid-training validation disabled (limit_val_batches=0)"
    )
    print(f"[train] Checkpoint cadence: every {checkpoint_every_n_steps} global steps")
    print(f"[train] Transparency head learnable: {not freeze_transparency_head}")
    if fixed_lambda >= 0.0:
        print(f"[train] Fixed lambda override: {fixed_lambda}")
    # expandable_segments avoids fragmentation OOMs: PyTorch reserves ~21 GB of
    # freed memory in small chunks; without this flag it can't satisfy a
    # contiguous 6 GB alloc even though total free > 6 GB.
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        subprocess.run(cmd, check=True, cwd=work_dir, env=env)
    finally:
        # data first — persists the tokenized .dat files so next run skips
        # the 30-min OWT download+tokenize phase.
        data_volume.commit()
        # then outputs — checkpoints and logs.
        out_volume.commit()
    print(f"[train] Done. Outputs saved to volume at {out_dir}")


# ── local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    alg: str = "both",  # "topk" | "slerp" | "both"
    base_ckpt: str = "mdlm_owt.ckpt",
    seed: int = 1,
    max_steps: int = 5000,
    model: str = "small",
    data: str = "openwebtext-split",
    slerp_n_iter: int = 3,
    freeze_until: int = 0,  # >0 enables FreezeBackboneCallback
    parallel: bool = False,  # True = two A100s simultaneously
    global_batch_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    val_check_interval: int = 200,
    checkpoint_every_n_steps: int = 100,
    freeze_transparency_head: bool = False,
    fixed_lambda: float = -1.0,
    log_every_n_steps: int = 10,
):
    kwargs = dict(
        base_ckpt_filename=base_ckpt,
        seed=seed,
        max_steps=max_steps,
        model=model,
        data=data,
        slerp_n_iter=slerp_n_iter,
        freeze_until=freeze_until,
        global_batch_size=global_batch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_check_interval=val_check_interval,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        freeze_transparency_head=freeze_transparency_head,
        fixed_lambda=fixed_lambda,
        log_every_n_steps=log_every_n_steps,
    )

    algs = []
    if alg in ("topk", "both", "all"):
        algs.append("mixinputs_with_topk")
    if alg in ("slerp", "both", "all"):
        algs.append("slerp_sm")
    if alg in ("lerp_renorm", "renorm", "all"):
        algs.append("lerp_renorm")
    if alg in ("slerp_euclid", "euclid", "all"):
        algs.append("slerp_euclid_mean")
    if not algs:
        raise ValueError(
            f"--alg must be 'topk', 'slerp', 'slerp_euclid', 'lerp_renorm', "
            f"'both', or 'all'; got '{alg}'"
        )

    if parallel and len(algs) > 1:
        # Spawn both on separate A100s; wait for both to finish
        print("[local] Launching both arms in parallel on separate A100s ...")
        handles = [train.spawn(a, **kwargs) for a in algs]
        for h in handles:
            h.get()
    else:
        for a in algs:
            train.remote(a, **kwargs)

    print("\n[local] All runs complete.")
    print(f"[local] Download outputs:  modal volume get mdlm-outputs {
            OUT_DIR
        }/<group> ./local_outputs/")
