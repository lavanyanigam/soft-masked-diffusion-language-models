#!/usr/bin/env python3
"""
Build a blind pairwise human-evaluation set from existing MDLM generations.

Run this ON vitallab1, where the ~23MB generation JSONs live. It emits two small
files:

  pairs.json  -> PUBLISHED to the rating site. Contains no arm labels at all.
  key.json    -> STAYS LOCAL. Maps each pair back to which side was which arm.

Keeping the arm identity out of pairs.json is what makes the study blind: a rater
who views the page source learns nothing.

Usage:
  python3 prep_pairs.py                      # uses the default paths below
  python3 prep_pairs.py SLERP.json TOPK.json # explicit paths
"""

import json
import os
import random
import sys

BASE = os.path.expanduser("~/sm_outputs/mauve_nfe")
FNAME = "mdlm_sm_mdlm_T-256_topp-0.9_eta-0.02_ton-0.55_toff-0.05_alphaon-0.9.json"

DEFAULT_SLERP = os.path.join(BASE, "slerp_T256", FNAME)
DEFAULT_TOPK = os.path.join(BASE, "topk_T256", FNAME)

N_PAIRS = 50
MAX_TOKENS = 100      # truncate so length is not a cue
SEED = 1              # fixed: index choice and side assignment are reproducible

# Table 6, seed 1, T=256 (1/4 budget) -- used only to report provenance.
EXPECTED = {
    "slerp": {"gen_ppl": 54.2024, "mauve": 0.040841},
    "topk": {"gen_ppl": 62.4244, "mauve": 0.025190},
}


def load(path, tag):
    with open(path) as f:
        d = json.load(f)
    if "text_samples" not in d:
        sys.exit(f"[fatal] {path} has no 'text_samples' key (found: {list(d)})")
    n = len(d["text_samples"])
    exp = EXPECTED[tag]
    gp, mv = d.get("gen_ppl"), d.get("MAUVE")
    gp_ok = gp is not None and abs(gp - exp["gen_ppl"]) < 0.01
    mv_ok = mv is not None and abs(mv - exp["mauve"]) < 1e-5
    print(f"[{tag}] n={n}")
    print(f"       gen_ppl={gp:.4f}  (Table 6: {exp['gen_ppl']})  {'MATCH' if gp_ok else 'DIFFERS'}")
    print(f"       MAUVE  ={mv:.6f}  (Table 6: {exp['mauve']})  {'MATCH' if mv_ok else 'DIFFERS'}")
    if not (gp_ok and mv_ok):
        print(f"       ^ these are NOT the Table 6 run. Usable, but say so in the write-up.")
    return d["text_samples"]


SPECIAL = ("<|endoftext|>",)


def clean(text):
    """Drop control tokens and collapse whitespace.

    The generations are unconditional, so every sample begins with
    <|endoftext|> and may contain more at document boundaries. It is a control
    token, not English, and raters are judging English fluency -- so it is
    removed. Both arms get identical treatment, so this cannot bias the
    comparison.
    """
    for tok in SPECIAL:
        text = text.replace(tok, " ")
    return " ".join(text.split())


def truncate(text, max_tokens=MAX_TOKENS):
    return " ".join(clean(text).split()[:max_tokens])


def main():
    slerp_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SLERP
    topk_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TOPK

    slerp = load(slerp_path, "slerp")
    topk = load(topk_path, "topk")

    pool = min(len(slerp), len(topk))
    rng = random.Random(SEED)
    # Same indices for both arms: pairs are index-matched, not cherry-picked.
    idxs = sorted(rng.sample(range(pool), N_PAIRS))

    pairs, key = [], []
    for i, idx in enumerate(idxs):
        pid = f"p{i:02d}"
        a, b = truncate(slerp[idx]), truncate(topk[idx])
        if rng.random() < 0.5:
            left, right, la, ra = a, b, "slerp", "topk"
        else:
            left, right, la, ra = b, a, "topk", "slerp"
        pairs.append({"pair_id": pid, "left": left, "right": right})
        key.append({"pair_id": pid, "source_index": idx, "left_arm": la, "right_arm": ra})

    # ---- sanity checks: refuse to write a broken or unblinded set ----
    problems = []
    if len(pairs) != N_PAIRS:
        problems.append(f"expected {N_PAIRS} pairs, built {len(pairs)}")
    for p in pairs:
        if not p["left"].strip() or not p["right"].strip():
            problems.append(f"{p['pair_id']}: empty side")
        if p["left"] == p["right"]:
            problems.append(f"{p['pair_id']}: identical sides")
        for side in ("left", "right"):
            if len(p[side].split()) > MAX_TOKENS:
                problems.append(f"{p['pair_id']}: {side} over {MAX_TOKENS} tokens")
    blob = json.dumps(pairs).lower()
    for banned in ("slerp", "topk", "lerp", "frechet", "euclid", "endoftext"):
        if banned in blob:
            problems.append(f"BLINDING LEAK: '{banned}' appears in pairs.json")
    if problems:
        print("\n[fatal] refusing to write:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    out = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out, "pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)
    with open(os.path.join(out, "key.json"), "w") as f:
        json.dump(key, f, indent=1)

    sides = sum(1 for k in key if k["left_arm"] == "slerp")
    print(f"\nWrote pairs.json ({N_PAIRS} pairs) and key.json to {out}")
    print(f"  slerp on the left in {sides}/{N_PAIRS} pairs (balance check)")
    print(f"  pairs.json is blind -- send it to the site; keep key.json local.")


if __name__ == "__main__":
    main()
