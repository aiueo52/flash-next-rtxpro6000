"""fc-moe RQ1 (2026-10-01): dump the MoE outputs of this env's FlashInfer build for a bit-exact A/B.

  python -m fc_moe.dump_out --T 1,4,16 --out ../runs/fc_moe/dump-prod.pt
  python -m fc_moe.dump_out --compare ../runs/fc_moe/dump-prod.pt ../runs/fc_moe/dump-rq1.pt
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.path.expanduser("~/tools/flash-next-bench/bench"))
from fc_moe.fc_moe_bench import ROT, census_routings  # noqa: E402
from moe_smallm import model_shapes as MS  # noqa: E402
from moe_smallm.runners import build_inputs, ensure_sglang_imports, get, load_autotune_cache  # noqa: E402
from moe_smallm.weights import load_layer, prepare_runtime_scales  # noqa: E402


def dump(args):
    ensure_sglang_imports()
    import fc_moe.nojit  # noqa: F401  -- never compile inside the GPU scope
    import flashinfer
    shape = MS.read_checkpoint_shape(MS.DEFAULT_CKPT)
    lw = prepare_runtime_scales(load_layer(args.layer, MS.DEFAULT_CKPT, device="cuda", shape=shape))
    load_autotune_cache(args.autotune_cache)
    make = get("flashinfer_cutlass")
    res = {"flashinfer": flashinfer.__file__}
    for T in (int(x) for x in args.T.split(",")):
        for rep in range(2):  # rep 1 shows whether one build repeats itself bit for bit
            ids, ws = census_routings(T, ROT)
            ins = build_inputs(lw, ids, ws, device="cuda")
            for c in make(lw, ins):
                c()
            torch.cuda.synchronize()
            res[(T, rep)] = {"out": [ci.output.cpu() for ci in ins],
                             "ids": [ci.topk_ids.cpu() for ci in ins]}
    torch.save(res, args.out)
    print(f"[dump] {res['flashinfer']} -> {args.out}", flush=True)


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    print(f"[cmp] A {a['flashinfer']}\n[cmp] B {b['flashinfer']}")
    for key in sorted(k for k in a if isinstance(k, tuple)):
        T, rep = key
        pairs = [("A rep0 vs rep1", a[(T, 0)], a[(T, 1)])] if rep == 1 else []
        pairs += [(f"A vs B rep{rep}", a[key], b[key])]
        for name, x, y in pairs:
            same_out = sum(torch.equal(p, q) for p, q in zip(x["out"], y["out"]))
            same_ids = sum(torch.equal(p, q) for p, q in zip(x["ids"], y["ids"]))
            diff = max((p.float() - q.float()).abs().max().item() for p, q in zip(x["out"], y["out"]))
            print(f"T={T:2d} {name}: out equal {same_out}/{len(x['out'])}, ids equal "
                  f"{same_ids}/{len(x['ids'])}, max |diff| {diff:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="1,4,16")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--autotune-cache", default=os.path.expanduser(
        "~/.cache/sglang/flashinfer/autotune/0.6.17/sm120/594f7c285c817365/rank_tp0_pp0_dp0.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        dump(args)


if __name__ == "__main__":
    main()
