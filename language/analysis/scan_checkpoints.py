#!/usr/bin/env python3
"""Report the arm, seed, step and lambda regime of every checkpoint found.

Lightning stores the resolved Hydra config under `hyper_parameters`, so a
checkpoint knows which transparency_alg, seed and fixed_lambda produced it --
none of which is reliably recoverable from directory names. Directories called
`fixed_chckpts` turned out to hold fixed-lambda runs; others named `_learned`
hold learned-lambda ones. Read the file instead of the path.

USAGE
  python -m analysis.scan_checkpoints ~/sm_outputs ~/fixed_chckpts
  python -m analysis.scan_checkpoints --learned-only ~
"""

import argparse
import os
import sys

import torch


def get(cfg, *path, default=None):
    cur = cfg
    for k in path:
        if cur is None:
            return default
        try:
            cur = cur[k]
        except Exception:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def scan(path):
    try:
        # mmap keeps the 2.6GB of weights off the heap; we only want metadata.
        try:
            c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"path": path, "error": type(e).__name__}

    hp = c.get("hyper_parameters") or {}
    cfg = hp.get("config", hp) if isinstance(hp, dict) else hp
    sd = c.get("state_dict", {})
    rs = sd.get("tran_head.raw_scale")

    return {
        "path": path,
        "step": c.get("global_step"),
        "alg": get(cfg, "algo", "tran_head", "transparency_alg", default="?"),
        "fixed_lambda": get(cfg, "algo", "tran_head", "fixed_lambda", default="?"),
        "seed": get(cfg, "seed", default="?"),
        "scale": round(torch.sigmoid(rs).item(), 5) if rs is not None else None,
        "has_head": any(k.startswith("tran_head") for k in sd),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--learned-only", action="store_true",
                    help="show only learned-lambda soft-masked checkpoints")
    ap.add_argument("--min-step", type=int, default=0)
    args = ap.parse_args()

    files = []
    for root in args.roots:
        for dp, _, fns in os.walk(os.path.expanduser(root)):
            for fn in fns:
                if fn.endswith(".ckpt"):
                    files.append(os.path.join(dp, fn))
    files = sorted(set(files))
    print(f"scanning {len(files)} checkpoints ...", file=sys.stderr)

    rows = []
    for f in files:
        r = scan(f)
        if "error" in r:
            print(f"  ! {r['error']}: {f}", file=sys.stderr)
            continue
        if args.learned_only and not (r["has_head"] and r["fixed_lambda"] in (None, "None")):
            continue
        if (r["step"] or 0) < args.min_step:
            continue
        rows.append(r)

    rows.sort(key=lambda r: (str(r["alg"]), str(r["seed"]), r["step"] or 0))
    print(f"{'alg':<20}{'seed':>5}{'step':>8}{'fixed_lambda':>14}{'scale':>9}  path")
    for r in rows:
        fl = r["fixed_lambda"]
        fl = "learned" if fl in (None, "None") else str(fl)
        print(f"{str(r['alg']):<20}{str(r['seed']):>5}{str(r['step']):>8}"
              f"{fl:>14}{str(r['scale']):>9}  {r['path']}")
    print("\nlambda regime comes from the checkpoint's own config, not its path.")
    print("For the amplification test you want two LEARNED-lambda rows with the")
    print("same seed and similar step, one mixinputs_with_topk and one slerp_sm.")


if __name__ == "__main__":
    main()
