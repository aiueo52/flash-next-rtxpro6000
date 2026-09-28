"""A/B the fused vs 3-kernel MoE routing prologue on identical inputs.  GPU-ONLY.

The prologue is integer bookkeeping (a counting sort of the T*top_k
(token, slot) pairs by expert id), so the two paths must produce *bit-identical*
MoE outputs.  The kill switch is read once per process
(``FLASHINFER_MOE_FUSED_PROLOGUE``), so this runs one side per process and
compares the saved tensors.

    python -m moe_smallm.ab_prologue save  <out.pt> [--widths 4,16] [...]
    python -m moe_smallm.ab_prologue cmp   <a.pt> <b.pt>
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from . import model_shapes as MS
from . import routing as R


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("save", "cmp"))
    p.add_argument("paths", nargs="+")
    p.add_argument("--ckpt", default=MS.DEFAULT_CKPT)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--widths", default="1,4,16")
    p.add_argument("--rotation", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--routing", default="independent")
    p.add_argument("--carry", type=float, default=0.0)
    p.add_argument("--n-hot", type=int, default=5)
    p.add_argument("--replay", default=None)
    p.add_argument("--autotune-cache", default="auto")
    p.add_argument("--force-tactics", default=None,
                   help="'g1,g2' pinned CUTLASS tactic ids (see bench_moe)")
    p.add_argument("--tol", type=float, default=0.0,
                   help="cmp: max allowed relative error; 0 demands bit-identical")
    p.add_argument("--hidden", type=int, default=None)
    args = p.parse_args(argv)

    if args.action == "cmp":
        a, b = (torch.load(x, map_location="cpu") for x in args.paths[:2])
        assert a["keys"] == b["keys"], "different key sets"
        bad = 0
        worst_abs = worst_rel = 0.0
        for k in a["keys"]:
            ta, tb = a["out"][k], b["out"][k]
            same = torch.equal(ta, tb)
            if same:
                print(f"{k}: bit-identical  ({tuple(ta.shape)})")
                continue
            d = (ta.float() - tb.float()).abs()
            # relative to the tensor's own scale, not to the element: an MoE output
            # element can be ~0 while its neighbours are O(1).
            scale = ta.float().abs().max().clamp_min(1e-6)
            rel = (d / scale).max().item()
            worst_abs = max(worst_abs, d.max().item())
            worst_rel = max(worst_rel, rel)
            if rel > args.tol:
                bad += 1
            print(f"{k}: DIFFER  max_abs={d.max():.3e} max_rel_of_scale={rel:.3e} "
                  f"n_diff={(d > 0).sum().item()}/{d.numel()}")
        if args.tol > 0:
            print(f"worst max_abs={worst_abs:.3e} worst max_rel_of_scale={worst_rel:.3e} "
                  f"tol={args.tol:.3e}")
            print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad} tensors over tolerance)")
        else:
            print("RESULT:",
                  "PASS (bit-identical)" if bad == 0 else f"FAIL ({bad} tensors differ)")
        return 0 if bad == 0 else 1

    from .bench_moe import _load_weights, _routings
    from .harness import require_idle_gpu
    from .runners import build_inputs, ensure_sglang_imports, get, load_autotune_cache

    require_idle_gpu()
    ensure_sglang_imports()
    if args.autotune_cache != "none":
        load_autotune_cache(args.autotune_cache)
    if args.force_tactics:
        from .runners import force_tactics
        parts = [None if x.strip() in ("", "none") else int(x)
                 for x in args.force_tactics.split(",")]
        while len(parts) < 2:
            parts.append(None)
        force_tactics(parts[0], parts[1])
    shape = (MS.read_checkpoint_shape(args.ckpt)
             if os.path.exists(os.path.join(args.ckpt, "config.json")) else MS.MoEShape())
    lw = _load_weights(args, shape, "cuda")
    out, keys = {}, []
    for t in [int(x) for x in args.widths.split(",")]:
        rs = _routings(args, shape, t, args.rotation)
        ws = R.uniform_weights(rs)
        ins = build_inputs(lw, rs, ws, device="cuda", seed=args.seed)
        calls = get("flashinfer_cutlass")(lw, ins)
        for i, c in enumerate(calls):
            c()
        torch.cuda.synchronize()
        for i, ci in enumerate(ins):
            k = f"T{t}/call{i}"
            keys.append(k)
            out[k] = ci.output.detach().cpu().clone()
    env = {k: os.environ.get(k, "<unset>") for k in
           ("FLASHINFER_MOE_FUSED_PROLOGUE", "FLASHINFER_MOE_FUSED_STRIDES",
            "FLASHINFER_MOE_PACK_GROUPS", "FLASHINFER_MOE_SPLITK_GEMM1",
            "FLASHINFER_MOE_SPLITK_RAMP_PCT")}
    torch.save({"keys": keys, "out": out, "env": env}, args.paths[0])
    print(f"saved {len(keys)} tensors to {args.paths[0]} ({env})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
