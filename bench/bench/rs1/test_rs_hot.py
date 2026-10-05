"""Statistical check: chain RS verify + truncated hot-vocab draft proposal is lossless.

Runs the real Triton kernel (TRITON_INTERPRET=1 -> CPU) and the real helper
functions extracted from the worktree sources (no sglang import needed).
"""
import ast, importlib.util, math, os, sys, time, types
import torch

WT = os.environ.get("WT", "/home/user/tools/sglang-rs") + "/python/sglang"
dev = os.environ.get("DEV", "cpu")

spec = importlib.util.spec_from_file_location(
    "reject_sampling", f"{WT}/kernels/ops/speculative/reject_sampling.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)

def extract(path, names, cls=None):
    src = open(path).read(); tree = ast.parse(src)
    out = {}
    nodes = tree.body
    if cls:
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out[n.name] = ast.get_source_segment(src, n)
    return out

import typing
ns = {"torch": torch, "Optional": typing.Optional}
funcs = extract(f"{WT}/srt/speculative/spec_utils.py",
                {"sample_draft_proposal_truncated", "fast_sample", "sample_draft_proposal"})
class _Envs:  # fast_sample reads envs.SGLANG_OPT_USE_GUMBEL_SAMPLE
    class SGLANG_OPT_USE_GUMBEL_SAMPLE:
        @staticmethod
        def get(): return True
ns["envs"] = _Envs
for f in funcs.values(): exec(f, ns)
meth_src = extract(f"{WT}/srt/speculative/eagle_worker_v2.py", {"_rs_draft_proposal"},
                   cls="EagleDraftWorker")["_rs_draft_proposal"]
import textwrap
meth_src = textwrap.dedent(meth_src).replace("Optional[torch.Tensor]", "object")

def make_worker(hot, vocab, k_cap, min_p_on=False):
    g = dict(ns); g["RS_DRAFT_TOPK"] = k_cap; g["SPEC_MIN_P"] = min_p_on
    exec(meth_src, g)
    w = types.SimpleNamespace(hot_token_id=hot, _rs_vocab_size=vocab)
    w._rs_draft_proposal = types.MethodType(g["_rs_draft_proposal"], w)
    return w

def target_probs(z, T, top_k, top_p, min_p=0.0):
    p = torch.softmax(z / T, -1)
    if top_k > 0:
        kth = p.topk(top_k, -1).values[..., -1:]
        p = torch.where(p >= kth, p, 0.0); p = p / p.sum(-1, keepdim=True)
    s, idx = p.sort(-1, descending=True)
    excl = s.cumsum(-1) - s
    s = torch.where(excl < top_p, s, 0.0)
    p = torch.zeros_like(p).scatter_(-1, idx, s); p = p / p.sum(-1, keepdim=True)
    if min_p > 0:
        p = torch.where(p >= p.amax(-1, keepdim=True) * min_p, p, 0.0); p = p / p.sum(-1, keepdim=True)
    return p

def chi2_pvalue(counts, probs, n):
    # merge cells with expected < 5 into one bucket
    exp = probs * n
    big = exp >= 5
    c = torch.cat([counts[big], counts[~big].sum().view(1)])
    e = torch.cat([exp[big], exp[~big].sum().view(1)])
    if e[-1] < 5: c, e = c[:-1], e[:-1]
    stat = ((c - e) ** 2 / e).sum().item(); dof = len(c) - 1
    if dof < 1: return stat, dof, float("nan")
    # Wilson-Hilferty normal approx of the chi2 tail
    zz = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return stat, dof, 0.5 * math.erfc(zz / math.sqrt(2))

def run(case, N, V=600, H=256, S=3, T=0.8, top_k=40, top_p=0.95, k_cap=64,
        draft_noise=0.7, seed=0, hot_head=30, min_p=0.0):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed + 1)
    zt = (torch.randn(S + 1, V, generator=g) * 2.0).to(dev)
    # hot set: most (not all) of every row's head, plus random filler, shuffled
    head = zt.cpu().topk(hot_head, -1).indices.flatten().unique()
    head = head[torch.randperm(len(head), generator=g)][: int(len(head) * 0.85)]
    rest = torch.tensor([t for t in torch.randperm(V, generator=g).tolist()
                         if t not in set(head.tolist())])[: H - len(head)]
    hot = torch.cat([head, rest])[torch.randperm(H, generator=g)].to(dev)
    p = target_probs(zt, T, top_k, top_p, min_p)                # (S+1, V)
    zd = (zt[:S][:, hot.cpu()].cpu() + torch.randn(S, H, generator=g) * draft_noise).to(dev)
    w = make_worker(hot, V, k_cap, min_p_on=min_p > 0)
    si = types.SimpleNamespace(
        temperatures=torch.full((N, 1), T, device=dev),
        top_ks=torch.full((N,), top_k if top_k > 0 else 1 << 30, dtype=torch.int32, device=dev),
        top_ps=torch.full((N,), top_p, device=dev),
        min_ps=torch.full((N,), min_p, device=dev))
    q_rows, cand = [], [torch.zeros(N, 1, dtype=torch.int64, device=dev)]
    for j in range(S):
        q, qx, x = w._rs_draft_proposal(zd[j].expand(N, H).contiguous(), si)
        assert q.shape == (N, V)
        assert torch.allclose(q.gather(1, hot[x]), qx), "q(X) mismatch"
        q_rows.append(q); cand.append(hot[x])
    draft_probs = torch.stack(q_rows, 1)
    tp = p.unsqueeze(0).expand(N, S + 1, V)
    cand = torch.cat(cand, 1).to(torch.int64)
    ns_ = S + 1
    predicts = torch.zeros(N * ns_, dtype=torch.int32, device=dev)
    acc_idx = torch.full((N, ns_), -1, dtype=torch.int32, device=dev)
    acc_num = torch.zeros(N, dtype=torch.int32, device=dev)
    retr = torch.arange(N * ns_, dtype=torch.int64, device=dev).view(N, ns_)
    coins = torch.rand(N, ns_, device=dev); coinf = torch.rand(N, device=dev)
    t0 = time.time()
    rs.chain_speculative_sampling_triton(predicts, acc_idx, acc_num, cand, retr, None, None,
        coins, coinf, tp, draft_probs, 1.0, 1.0, True)
    dt = time.time() - t0
    out = []
    for i in range(ns_):
        rows = acc_num >= i
        toks = predicts[acc_idx[rows, i].long()].long().cpu()
        cnt = torch.bincount(toks, minlength=V).double()
        n = int(rows.sum())
        pr = p[i].double().cpu()
        tv = 0.5 * (cnt / max(n, 1) - pr).abs().sum().item()
        stat, dof, pv = chi2_pvalue(cnt, pr, n) if n > 0 else (0, 0, 1)
        out.append((i, n, tv, dof, pv))
    alpha_emp = (acc_num >= 1).double().mean().item()
    alpha_th = torch.minimum(p[0], q_rows[0][0]).sum().item()
    print(f"[{case}] N={N} kernel {dt:.1f}s  accept@0 emp {alpha_emp:.4f} theory {alpha_th:.4f}"
          f"  mean acc {acc_num.double().mean().item():.3f}")
    for i, n, tv, dof, pv in out:
        print(f"   pos {i}: n={n:7d}  TV={tv:.4f}  chi2 dof={dof:3d} p={pv:.3f}")
    return out

if __name__ == "__main__":
    N = int(os.environ.get("N", "20000"))
    cases = os.environ.get("CASES", "")
    if cases == "minp":
        run("min_p 0.05 T0.8 k40 p0.95", N, min_p=0.05, seed=21)
        run("min_p 0.05, no min_p on q (k_cap=0)", N, min_p=0.05, k_cap=0, seed=21)
        raise SystemExit
    if cases == "minp_off":
        # same logits as the minp cases with min_p off: what the verify does today when min_p is requested
        run("min_p ignored, seed 21", N, seed=21)
        raise SystemExit
    if cases == "seeds":
        # fresh seeds for the case whose pos-0 p-value was low (0.011)
        for s in (11, 12, 13):
            run(f"no top-k, top_p 0.9, T1.0 seed{s}", N, T=1.0, top_k=0, top_p=0.9, seed=s)
        raise SystemExit
    run("trunc k64 T0.8 k40 p0.95", N)
    run("untrunc (k_cap=0)", N, k_cap=0)
    run("no top-k, top_p 0.9, T1.0", N, T=1.0, top_k=0, top_p=0.9, seed=3)
    run("good draft, k_cap 16 < top_k 40", N, draft_noise=0.3, k_cap=16, seed=5)
    run("greedy (top_k=1)", N // 4, T=1.0, top_k=1, top_p=1.0, seed=7)
