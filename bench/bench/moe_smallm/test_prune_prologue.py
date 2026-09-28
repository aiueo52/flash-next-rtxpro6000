"""P2 equivalence test: FlashInfer's in-prologue singleton prune vs SGLang's P1 Triton kernel.
GPU-ONLY.

Replays recorded routing (``runs/census-*.npz``: real top-k ids *and* weights of
T=16 / T=4 verify calls) through the production MoE path (``flashinfer_cutlass``
candidate, real 512-expert NVFP4 layer) and records, per call,

* the top-k ids / weights **after** the call -- both paths write the dropped
  routes back in place as id -1 / weight 0, so this is the (row, expert) set
  that was pruned;
* the MoE output.

In the same process the written-back ids are checked against the P1 Triton
kernel run on a copy of the inputs *and* against a numpy reference.  Across
processes (the env is read once per process on both sides) ``cmp`` demands
bit-identical ids/weights and, in deterministic mode, bit-identical outputs.

    # P1 path (Python kernel), deterministic mode
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08 \\
        python -m moe_smallm.test_prune_prologue save p1.pt --census runs/census-w16-code-edit.npz --T 16
    # P2 path (in-prologue), same inputs
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 SGLANG_MOE_PRUNE_SINGLETON_TAU=0.08 SGLANG_MOE_PRUNE_IN_PROLOGUE=1 \\
        python -m moe_smallm.test_prune_prologue save p2.pt --census runs/census-w16-code-edit.npz --T 16
    python -m moe_smallm.test_prune_prologue cmp p1.pt p2.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from . import model_shapes as MS


def ref_prune(ids, w, tau, min_rank):
    """numpy reference (same as test_prune_gpu.ref): multiplicity, within-row rank, mask."""
    n, k = ids.shape
    flat = ids.ravel()
    cnt = np.array([(flat == e).sum() for e in flat]).reshape(n, k)
    order = np.argsort(-w, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(k)[None, :], axis=1)
    prune = (cnt == 1) & (w < tau) & (rank >= min_rank)
    ids2, w2 = ids.copy(), w.copy()
    ids2[prune] = -1
    w2[prune] = 0.0
    return ids2, w2, prune


def load_census(path, T, n, stride):
    d = np.load(path)
    sel = np.nonzero(d["T"] == T)[0]
    if len(sel) == 0:
        raise SystemExit(f"{path}: no calls with T={T}")
    sel = sel[::stride][:n]
    k = int(d["k"][sel[0]])
    ids = d["ids"][sel, : T * k].astype(np.int32).reshape(len(sel), T, k)
    w = d["w"][sel, : T * k].astype(np.float32).reshape(len(sel), T, k)
    return ids, w, k


def distinct(ids):
    return len(set(int(x) for x in ids.ravel() if x >= 0))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("save", "cmp"))
    p.add_argument("paths", nargs="+")
    p.add_argument("--census", default=None)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--n", type=int, default=32, help="calls to replay")
    p.add_argument("--stride", type=int, default=97, help="take every stride-th recorded call")
    p.add_argument("--ckpt", default=MS.DEFAULT_CKPT)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--autotune-cache", default="none")
    p.add_argument("--tol", type=float, default=0.0)
    args = p.parse_args(argv)

    if args.action == "cmp":
        a, b = (torch.load(x, map_location="cpu") for x in args.paths[:2])
        assert a["keys"] == b["keys"], "different key sets"
        bad_ids = bad_out = 0
        worst = 0.0
        for k in a["keys"]:
            same_ids = torch.equal(a["ids"][k], b["ids"][k]) and torch.equal(a["w"][k], b["w"][k])
            bad_ids += 0 if same_ids else 1
            ta, tb = a["out"][k], b["out"][k]
            if torch.equal(ta, tb):
                continue
            d = (ta.float() - tb.float()).abs()
            scale = ta.float().abs().max().clamp_min(1e-6)
            rel = (d / scale).max().item()
            worst = max(worst, rel)
            if rel > args.tol:
                bad_out += 1
                print(f"{k}: OUTPUT DIFFERS max_abs={d.max():.3e} rel_of_scale={rel:.3e} "
                      f"n_diff={(d > 0).sum().item()}/{d.numel()} ids_same={same_ids}")
        n = len(a["keys"])
        pa = sum(int((a["ids"][k] < 0).sum()) for k in a["keys"])
        pb = sum(int((b["ids"][k] < 0).sum()) for k in b["keys"])
        print(f"calls={n}  pruned routes A={pa} B={pb}  env A={a['env']}  env B={b['env']}")
        print(f"ids/weights identical on {n - bad_ids}/{n} calls; outputs "
              f"{'bit-identical' if bad_out == 0 and worst == 0 else f'worst rel={worst:.3e}'} "
              f"on {n - bad_out}/{n} calls (tol={args.tol})")
        ok = bad_ids == 0 and bad_out == 0
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if not args.census:
        raise SystemExit("save needs --census")
    from .bench_moe import _load_weights
    from .harness import require_idle_gpu
    from .runners import CallInputs, ensure_sglang_imports, get, load_autotune_cache

    require_idle_gpu()
    ensure_sglang_imports()
    if args.autotune_cache != "none":
        load_autotune_cache(args.autotune_cache)
    from sglang.srt.layers.moe import prune_singleton as ps

    tau = ps.TAU
    min_rank = ps.MIN_RANK
    print(f"[p2test] prune config: {ps.describe()}  python-kernel-enabled={ps.ENABLED}")
    shape = (MS.read_checkpoint_shape(args.ckpt)
             if os.path.exists(os.path.join(args.ckpt, "config.json")) else MS.MoEShape())
    args.synthetic = False
    args.num_experts = None
    lw = _load_weights(args, shape, "cuda")

    ids_np, w_np, k = load_census(args.census, args.T, args.n, args.stride)
    g = torch.Generator().manual_seed(args.seed)
    H = lw.shape.hidden
    ins = []
    for i in range(len(ids_np)):
        x = (torch.randn((args.T, H), generator=g) * 0.5).to(torch.bfloat16).cuda()
        ins.append(CallInputs(
            hidden_states=x,
            topk_ids=torch.from_numpy(ids_np[i]).cuda(),        # int32: .to(torch.int) is a no-op
            topk_weights=torch.from_numpy(w_np[i]).cuda(),
            output=torch.empty((args.T, H), dtype=torch.bfloat16, device="cuda"),
        ))
    calls = get("flashinfer_cutlass")(lw, ins)
    # one warm-up call on a scratch copy so the JIT build / first-call setup is not in the loop
    scratch = CallInputs(ins[0].hidden_states.clone(), ins[0].topk_ids.clone(),
                         ins[0].topk_weights.clone(), ins[0].output.clone())
    get("flashinfer_cutlass")(lw, [scratch])[0]()
    torch.cuda.synchronize()

    out, keys, ids_out, w_out = {}, [], {}, {}
    n_bad_ref = n_bad_triton = 0
    d_before = d_after = pruned = 0
    for i, c in enumerate(calls):
        c()
        torch.cuda.synchronize()
        ci = ins[i]
        key = f"T{args.T}/call{i}"
        keys.append(key)
        out[key] = ci.output.detach().cpu().clone()
        ids_out[key] = ci.topk_ids.detach().cpu().clone()
        w_out[key] = ci.topk_weights.detach().cpu().clone()
        if tau > 0:
            ei, ew, pr = ref_prune(ids_np[i], w_np[i], tau, min_rank)
            oi, ow = ids_out[key].numpy(), w_out[key].numpy()
            if not ((oi == ei).all() and (ow == ew).all()):
                n_bad_ref += 1
                if n_bad_ref <= 3:
                    bad = np.argwhere(oi != ei)[:5]
                    print(f"  {key}: written-back ids differ from numpy ref at {bad.tolist()}")
            ti = torch.from_numpy(ids_np[i]).cuda()
            tw = torch.from_numpy(w_np[i]).cuda()
            ps.prune_singleton_routes_(ti, tw, tau, min_rank, 1, 256)
            torch.cuda.synchronize()
            if not (torch.equal(ti.cpu(), ids_out[key]) and torch.equal(tw.cpu(), w_out[key])):
                n_bad_triton += 1
            pruned += int(pr.sum())
            d_before += distinct(ids_np[i])
            d_after += distinct(ei)
    n = len(calls)
    if tau > 0:
        print(f"[p2test] {n} calls T={args.T}: pruned {pruned / n:.2f} routes/call, "
              f"D {d_before / n:.2f} -> {d_after / n:.2f}; written-back ids/weights == numpy ref on "
              f"{n - n_bad_ref}/{n}, == P1 Triton kernel on {n - n_bad_triton}/{n}")
    env = {k: os.environ.get(k, "<unset>") for k in
           ("SGLANG_MOE_PRUNE_SINGLETON_TAU", "SGLANG_MOE_PRUNE_IN_PROLOGUE",
            "FLASHINFER_MOE_PRUNE_SINGLETON_TAU", "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
            "FLASHINFER_MOE_PACK_GROUPS", "FLASHINFER_MOE_FOLD_EXPAND")}
    torch.save({"keys": keys, "out": out, "ids": ids_out, "w": w_out, "env": env}, args.paths[0])
    print(f"saved {n} calls to {args.paths[0]} ({env})")
    ok = n_bad_ref == 0 and n_bad_triton == 0
    print("IN-PROCESS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
