#!/usr/bin/env python3
"""Does a high confidence weight amplify errors?

Reviewer B's hypothesis: S-SM sustains a larger lambda than LERP, so where the
backbone is confidently wrong the feedback injects that wrong token harder, and
the error spreads to neighbouring positions.

This tests it directly, on real text rather than unconditional samples:

  1. take OpenWebText validation sequences and mask ~50% of the tokens,
  2. denoise with a checkpoint, recording per-position lambda and the top-1
     prediction at every step,
  3. flag positions where lambda is high AND the top-1 prediction is wrong,
     scored against the token that was actually there,
  4. report, per arm: how often that happens, the final accuracy at those
     positions, and the final accuracy at their neighbours within +/-5.

If the hypothesis holds, S-SM should flag more often, and -- the part that
matters -- neighbours of flagged positions should be measurably worse than
masked positions far from any flag. A higher flag rate on its own is expected
from a larger lambda and proves nothing by itself.

USAGE (from language/)
  python -m analysis.lambda_error_trace \\
      --ckpt /path/to/0-6999.ckpt \\
      --alg slerp_sm \\
      --tag slerp \\
      --n-seqs 2 --steps 64                      # smoke test, ~2 min

  # the real run
  python -m analysis.lambda_error_trace --ckpt ... --alg slerp_sm \\
      --tag slerp --n-seqs 64 --steps 256 --out results_slerp.json

Run it once per arm with the same --seed and --mask-frac, then:
  python -m analysis.lambda_error_trace --compare results_lerp.json results_slerp.json
"""

import argparse
import json
import os
import sys

import torch


# ---------------------------------------------------------------- comparison


def compare(paths):
    """Print the side-by-side table from saved per-arm result files."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))

    keys = [
        ("flag_rate", "flagged / masked positions", "{:.4f}"),
        ("lambda_threshold", "lambda threshold used", "{:.5f}"),
        ("lambda_mean_masked", "mean lambda (masked)", "{:.5f}"),
        ("acc_flagged", "final acc AT flagged", "{:.4f}"),
        ("acc_neighbours", "final acc at +/-5 of flagged", "{:.4f}"),
        ("acc_far", "final acc far from any flag", "{:.4f}"),
        ("neighbour_penalty", "far - neighbours (amplification)", "{:+.4f}"),
        ("acc_overall", "final acc, all masked", "{:.4f}"),
    ]
    w = max(len(lbl) for _, lbl, _ in keys) + 2
    hdr = " " * w + "".join(r["tag"].rjust(14) for r in runs)
    print(hdr)
    print("-" * len(hdr))
    for k, lbl, fmt in keys:
        row = lbl.ljust(w)
        for r in runs:
            v = r.get(k)
            row += (fmt.format(v) if isinstance(v, (int, float)) else "-").rjust(14)
        print(row)
    print()
    print("neighbour_penalty > 0 means positions near a high-lambda error end up")
    print("LESS accurate than masked positions far from one -- the amplification")
    print("the hypothesis predicts. Compare the penalty ACROSS arms, not the")
    print("flag rate: a larger lambda raises the flag rate on its own.")


# ---------------------------------------------------------------- main run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", help="result JSONs to tabulate; skips inference")
    ap.add_argument("--ckpt")
    ap.add_argument("--alg", help="transparency_alg of the checkpoint")
    ap.add_argument("--tag", help="name for this arm in the output")
    ap.add_argument("--fixed-lambda", type=float, default=None)
    ap.add_argument("--data-cache-dir", default=os.path.expanduser("~/mdlm/data/owt_cache"))
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256, help="denoising steps (NFE)")
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--lambda-quantile", type=float, default=0.75,
                    help="'high lambda' = above this quantile of masked-position lambdas")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    for req in ("ckpt", "alg", "tag"):
        if getattr(args, req) is None:
            sys.exit(f"--{req.replace('_','-')} is required (or use --compare)")

    # Imports deferred so --compare works without torch/hydra//the repo env.
    import hydra
    import lightning as L
    import numpy as np
    from omegaconf import open_dict

    import dataloader
    import algo as algo_mod
    import main as main_mod
    from trainer_base import sample_categorical

    L.seed_everything(args.seed)

    # ---- config: mirror the eval path in main.py -------------------------
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "algo=mdlm_sm",
                "model=small",
                "data=openwebtext-split",
                f"data.cache_dir={args.data_cache_dir}",
                f"algo.tran_head.transparency_alg={args.alg}",
                "algo.tran_head.mixinputs_k=3",
                "algo.tran_head.slerp_n_iter=3",
                "model.length=1024",
                f"loader.batch_size={args.batch}",
                f"loader.eval_batch_size={args.batch}",
                f"loader.global_batch_size={args.batch}",
                f"loader.eval_global_batch_size={args.batch}",
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "trainer.accumulate_grad_batches=1",
                f"eval.checkpoint_path={args.ckpt}",
                f"seed={args.seed}",
                "+wandb.offline=true",
            ],
        )
    if args.fixed_lambda is not None:
        with open_dict(cfg):
            cfg.algo.tran_head.fixed_lambda = args.fixed_lambda

    # dataloader.get_dataloaders asserts
    #   global_batch_size == batch_size * num_nodes * torch.cuda.device_count() * accum
    # and reads the GPU count from the device itself, not from trainer.devices.
    # Pin the run to one visible GPU and make that identity hold trivially.
    n_vis = torch.cuda.device_count()
    if n_vis != 1:
        sys.exit(
            f"[fatal] {n_vis} GPUs visible; this analysis expects exactly 1.\n"
            f"        Re-run with e.g. CUDA_VISIBLE_DEVICES=0"
        )

    tokenizer = dataloader.get_tokenizer(cfg)
    _, valid_ds = dataloader.get_dataloaders(
        cfg, tokenizer, skip_train=True, valid_seed=args.seed
    )

    # Non-strict: checkpoints written before initial_nll / current_nll_ema /
    # R_ema were registered lack those buffers. They initialise to -1.0 and
    # MDLM_SM.forward treats anything < 0 as "not set", so the NLL-annealed
    # centre is simply skipped -- which is what those runs did anyway. R_ema
    # is only read when reliability_conditioned is on, and it is off.
    model = algo_mod.MDLM_SM.load_from_checkpoint(
        args.ckpt, tokenizer=tokenizer, config=cfg, strict=False
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing = [k for k in ("initial_nll", "current_nll_ema", "R_ema")
               if k not in ck["state_dict"]]
    if missing:
        print(f"[load] buffers absent from checkpoint, left at init: {missing}")
    del ck
    model = model.to("cuda").eval()
    if cfg.eval.disable_ema:
        model.ema = None

    mask_index = model.mask_index
    t_min = float(cfg.optim.sm_t_min)
    t_max = float(cfg.optim.sm_t_max)
    print(f"[cfg] alg={args.alg} mask_index={mask_index} band=[{t_min},{t_max}] "
          f"steps={args.steps} mask_frac={args.mask_frac}")

    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    lam_all, wrong_all = [], []      # per masked position, pooled across batches
    correct_all, seq_id_all, pos_all = [], [], []
    examples = []
    seen = 0

    for batch in valid_ds:
        if seen >= args.n_seqs:
            break
        x0 = batch["input_ids"][: args.n_seqs - seen].to("cuda")
        if x0.numel() == 0:
            break
        B, L = x0.shape
        seen += B

        # ---- mask ~mask_frac of the positions ---------------------------
        keep = torch.rand(x0.shape, generator=rng).to("cuda") >= args.mask_frac
        xt = torch.where(keep, x0, torch.full_like(x0, mask_index))
        masked0 = xt == mask_index          # the set we score on

        # lambda / top-1 recorded the FIRST time each position is seen while
        # still masked, which is when the feedback for it is actually formed.
        lam_rec = torch.full(x0.shape, float("nan"), device="cuda")
        top1_rec = torch.full(x0.shape, -1, dtype=torch.long, device="cuda")

        # ---- denoise ----------------------------------------------------
        eps_t = 1e-5
        timesteps = torch.linspace(1, eps_t, args.steps + 1, device="cuda")
        dt = (1 - eps_t) / args.steps
        x = xt.clone()
        log_p_cache = None

        with torch.no_grad():
            for i in range(args.steps):
                # Shapes follow MDLM_SM._ddpm_caching_update exactly: t is
                # squeezed to 1-D and the move chances are (B,1,1) so they
                # broadcast against p_x0's (B,L,V).
                t = timesteps[i] * torch.ones(B, device="cuda")
                sigma = model._sigma_from_alphat(model.noise(t[:, None])[1])
                tv = float(timesteps[i])
                in_band = t_min <= tv <= t_max
                feedback = log_p_cache if in_band else None

                log_p = model.forward(x, sigma, feedback)
                p_x0 = log_p.exp()

                cur_mask = x == mask_index
                if cur_mask.any():
                    # lambda exactly as TransparencyHead.forward computes it
                    if feedback is not None:
                        ne = torch.zeros(x.shape, device="cuda", dtype=log_p.dtype)
                        ne_m, _ = model.tran_head.get_neg_entropy_and_probabilities(
                            feedback[cur_mask]
                        )
                        ne[cur_mask] = ne_m
                        lam = model.tran_head.calculate_lambda_tensor(
                            ne, cur_mask, None, None, 1.0
                        )
                    else:
                        lam = torch.zeros(x.shape, device="cuda")

                    top1 = p_x0.argmax(-1)
                    fresh = cur_mask & torch.isnan(lam_rec)
                    lam_rec = torch.where(fresh, lam.float(), lam_rec)
                    top1_rec = torch.where(fresh, top1, top1_rec)

                log_p_cache = log_p

                # DDPM update, verbatim from MDLM_SM._ddpm_caching_update
                move_chance_t = t[:, None, None]
                move_chance_s = (t - dt)[:, None, None]
                q_xs = p_x0 * (move_chance_t - move_chance_s)
                q_xs[:, :, mask_index] = move_chance_s[:, :, 0]
                _x = sample_categorical(q_xs)
                copy_flag = (x != mask_index).to(x.dtype)
                x = copy_flag * x + (1 - copy_flag) * _x

        # ---- score ------------------------------------------------------
        final_correct = (x == x0) & masked0
        rec_ok = masked0 & ~torch.isnan(lam_rec)
        wrong_at_rec = rec_ok & (top1_rec != x0)

        for b in range(B):
            idx = rec_ok[b].nonzero(as_tuple=True)[0]
            lam_all.append(lam_rec[b, idx].cpu())
            wrong_all.append(wrong_at_rec[b, idx].cpu())
            correct_all.append(final_correct[b, idx].cpu())
            pos_all.append(idx.cpu())
            seq_id_all.append(torch.full_like(idx.cpu(), len(seq_id_all)))

        if len(examples) < args.examples:
            examples.append(
                _trace_example(tokenizer, x0, xt, x, lam_rec, top1_rec, masked0, args)
            )
        print(f"[run] {seen}/{args.n_seqs} sequences")

    lam = torch.cat(lam_all)
    wrong = torch.cat(wrong_all)
    correct = torch.cat(correct_all)
    pos = torch.cat(pos_all)
    seq = torch.cat(seq_id_all)

    thr = torch.quantile(lam, args.lambda_quantile).item()
    flagged = (lam > thr) & wrong

    # neighbours of flagged positions, within the same sequence
    near = torch.zeros_like(flagged)
    for s in seq.unique():
        m = seq == s
        fp = pos[m][flagged[m]]
        if fp.numel() == 0:
            continue
        d = (pos[m].unsqueeze(1) - fp.unsqueeze(0)).abs()
        near[m] = (d <= args.radius).any(1)
    near = near & ~flagged

    def acc(sel):
        return correct[sel].float().mean().item() if sel.any() else float("nan")

    far = ~near & ~flagged
    res = {
        "tag": args.tag,
        "alg": args.alg,
        "ckpt": args.ckpt,
        "steps": args.steps,
        "mask_frac": args.mask_frac,
        "n_masked_positions": int(lam.numel()),
        "lambda_quantile": args.lambda_quantile,
        "lambda_threshold": thr,
        "lambda_mean_masked": lam.mean().item(),
        "flag_rate": flagged.float().mean().item(),
        "wrong_rate": wrong.float().mean().item(),
        "acc_flagged": acc(flagged),
        "acc_neighbours": acc(near),
        "acc_far": acc(far),
        "acc_overall": correct.float().mean().item(),
        "examples": examples,
    }
    res["neighbour_penalty"] = res["acc_far"] - res["acc_neighbours"]

    out = args.out or f"lambda_trace_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
    print(f"\nwrote {out}")


def _trace_example(tokenizer, x0, xt, x, lam_rec, top1_rec, masked0, args):
    """One human-readable trace: the highest-lambda wrong position in sequence 0."""
    b = 0
    ok = masked0[b] & ~torch.isnan(lam_rec[b]) & (top1_rec[b] != x0[b])
    if not ok.any():
        return {"note": "no high-lambda error in this sequence"}
    i = int(torch.where(ok, lam_rec[b], torch.full_like(lam_rec[b], -1)).argmax())
    lo, hi = max(0, i - 12), min(x0.shape[1], i + 13)
    dec = tokenizer.decode
    return {
        "position": i,
        "lambda": float(lam_rec[b, i]),
        "true_token": dec([int(x0[b, i])]),
        "top1_prediction": dec([int(top1_rec[b, i])]),
        "final_token": dec([int(x[b, i])]),
        "context_true": dec([int(v) for v in x0[b, lo:hi]]),
        "context_final": dec([int(v) for v in x[b, lo:hi]]),
        "neighbour_final_acc": float(
            ((x[b, lo:hi] == x0[b, lo:hi]) & masked0[b, lo:hi]).float().mean()
        ),
    }


if __name__ == "__main__":
    main()
