"""Sparse target-only verify (sglang-sv, SGLANG_OPT_SPEC_SPARSE_VERIFY) against the dense verify.

  DEV=cpu : TRITON_INTERPRET=1 runs the Triton kernels on the CPU; the references are Python ports of
            the dense chain (FlashInfer keep rules) and of TreeSpeculativeSamplingTargetOnly.
  DEV=cuda: the references are the real dense kernels (softmax, sgl_kernel top_k_renorm_prob /
            top_p_renorm_prob, tree_speculative_sampling_target_only); adds a GPU timing table.

  A. probabilities: kept sets and values on bf16-tied, peaked, flat and grammar-masked rows
  B. tree walk + final draw: equal coins -> equal accept_index / accepted tokens / bonus token
     (chain trees as in production, and a tree with siblings)
  C. distribution: first emitted token over many coins vs the target row (chi2)
  D. (cuda) GPU time per verify, dense vs sparse

Run (cpu):  TRITON_INTERPRET=1 DEV=cpu  ~/tools/sglang-rtxpro6000/.venv/bin/python bench/sv/test_sparse_verify.py
Run (cuda): under the GPU lock and a 40G scope, DEV=cuda.
WT=~/tools/sglang-sv2 TOPK=fi: FlashInfer top-k (needs the p3 FlashInfer + rq2u2h cache env, see bench/sv2/).
"""
import functools, importlib.util, inspect, math, os, statistics, sys, time
import numpy as np
import torch

WT = os.environ.get("WT", os.path.expanduser("~/tools/sglang-sv"))
dev = os.environ.get("DEV", "cpu")
V = int(os.environ.get("VOCAB", "248320"))
spec = importlib.util.spec_from_file_location(
    "sparse_verify", f"{WT}/python/sglang/kernels/ops/speculative/sparse_verify.py")
sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
if "use_flashinfer_topk" in inspect.signature(sv.sparse_target_probs).parameters:
    # sglang-sv2: TOPK=fi takes the top-KP with FlashInfer's top_k (DEV=cuda only), else torch.topk.
    sv.sparse_target_probs = functools.partial(sv.sparse_target_probs, use_flashinfer_topk=os.environ.get("TOPK") == "fi")
g = torch.Generator().manual_seed(1234)
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- inputs
def make_logits(n, kind):
    x = torch.randn(n, V, generator=g) * 1.5
    for r in range(n):
        head = torch.randperm(V, generator=g)[:80]
        s = [0.02, 0.08, 0.3, 1.0, 3.0][r % 5]  # flat .. peaked head
        x[r, head] = 11.0 - s * torch.arange(80, dtype=torch.float32) ** 0.8 + 0.05 * torch.randn(80, generator=g)
        if kind == "masked" and r % 3 == 0:  # grammar mask: 5 allowed tokens
            keep = head[:5]
            m = torch.full((V,), float("-inf")); m[keep] = 0.0
            x[r] = x[r] + m
    if kind == "bf16":
        x = x.bfloat16().float()  # bf16 logits: many exact ties inside the top 40
    return x


def make_params(bs, mix, top_k=40):
    temps = torch.tensor([[0.8, 0.6, 1.0, 1.3][i % 4] for i in range(bs)], dtype=torch.float32)
    top_ks = torch.tensor([[40, 20, 1, 40][i % 4] if mix else top_k for i in range(bs)], dtype=torch.int32)
    top_ps = torch.tensor([[0.95, 0.8, 1.0, 0.5][i % 4] if mix else 0.95 for i in range(bs)], dtype=torch.float32)
    min_ps = torch.tensor([[0.05, 0.0, 0.2, 0.05][i % 4] if mix else 0.05 for i in range(bs)], dtype=torch.float32)
    return temps, top_ks, top_ps, min_ps


# ---------------------------------------------------------------- dense references
def dense_probs(logits, temps, top_ks, top_ps, min_ps, S, apply_min_p):
    rep = lambda t: torch.repeat_interleave(t, S, dim=0)
    t_r, k_r, p_r, m_r = rep(temps), rep(top_ks), rep(top_ps), rep(min_ps)
    p = torch.softmax(logits / t_r[:, None], dim=-1)
    if dev == "cuda":
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
        p = top_k_renorm_prob(p, k_r)
        p = top_p_renorm_prob(p, p_r)
    else:  # FlashInfer keep rules: top-k p >= pivot, AIR top-p p >= crossing value
        srt = p.sort(-1, descending=True).values
        piv = srt.gather(1, (k_r.clamp(1, V) - 1).long()[:, None])
        p = torch.where(p >= piv, p, 0.0); p = p / p.sum(-1, keepdim=True)
        srt = p.sort(-1, descending=True).values
        c = srt.cumsum(-1)
        thr = torch.where(c >= p_r[:, None] * p.sum(-1, keepdim=True), srt, 0.0).amax(-1, keepdim=True)
        p = torch.where(p >= thr, p, 0.0); p = p / p.sum(-1, keepdim=True)
    if apply_min_p:  # eagle_sample's SPEC_MIN_P rule
        keep = p >= p.amax(dim=-1, keepdim=True) * m_r[:, None]
        p = p * keep; p = p / p.sum(dim=-1, keepdim=True)
    return p


def dense_tree_py(pred, acc_idx, acc_num, cands, ridx, rnext, rsib, coins, coinsf, tprobs):
    """Python port of TreeSpeculativeSamplingTargetOnly (threshold_single = threshold_acc = 1)."""
    bs, S = cands.shape
    num_spec = acc_idx.shape[1]
    f32 = np.float32
    for b in range(bs):
        rejected = {}  # row -> set of rejected token ids (the draft_probs buffer)
        prob_acc = f32(0); cur_row = 0; coin = f32(coins[b, 0].item())
        last = int(ridx[b, 0]); acc_idx[b, 0] = last; n = 0; cur = 0
        for j in range(1, num_spec):
            cur = int(rnext[b, cur])
            while cur != -1:
                di, tok = int(ridx[b, cur]), int(cands[b, cur])
                qt = f32(tprobs[b, cur_row, tok].item())
                prob_acc = f32(prob_acc + qt)
                if coin <= prob_acc or qt >= f32(1.0):
                    prob_acc = f32(0); cur_row = cur; coin = f32(coins[b, cur].item())
                    pred[last] = tok; n += 1; acc_idx[b, n] = di; last = di
                    break
                rejected.setdefault(cur_row, set()).add(tok)
                cur = int(rsib[b, cur])
            if cur == -1:
                break
        acc_num[b] = n
        w = tprobs[b, cur_row].clone()
        if n != num_spec - 1:
            for t in rejected.get(cur_row, ()):
                w[t] = 0.0
        u = f32(coinsf[b].item()) * w.sum()
        cum = torch.cumsum(w.double(), 0)
        hit = ((cum > u) & (w > 0)).nonzero()
        if len(hit):
            s = int(hit[0])
        else:
            valid = (w > 0).nonzero()
            s = int(valid[-1]) if len(valid) else V - 1
        pred[last] = s


def dense_tree(pred, acc_idx, acc_num, cands, ridx, rnext, rsib, coins, coinsf, tprobs):
    if dev == "cuda":
        from sgl_kernel import tree_speculative_sampling_target_only
        tree_speculative_sampling_target_only(
            predicts=pred, accept_index=acc_idx, accept_token_num=acc_num, candidates=cands,
            retrive_index=ridx, retrive_next_token=rnext, retrive_next_sibling=rsib,
            uniform_samples=coins, uniform_samples_for_final_sampling=coinsf,
            target_probs=tprobs, draft_probs=torch.zeros_like(tprobs),
            threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    else:
        dense_tree_py(pred, acc_idx, acc_num, cands, ridx, rnext, rsib, coins, coinsf, tprobs)


# ---------------------------------------------------------------- trees
def chain_tree(bs, S):
    ridx = (torch.arange(bs)[:, None] * S + torch.arange(S)[None, :]).long()
    rnext = torch.arange(1, S + 1).repeat(bs, 1).long(); rnext[:, -1] = -1
    rsib = torch.full((bs, S), -1, dtype=torch.long)
    parent = [-1] + list(range(S - 1))
    return ridx, rnext, rsib, parent, S  # num_spec = S


def sibling_tree(bs):
    # 0 -> (1, 2); 1 -> (3, 4); 2 -> (5, 6); 3 -> 7     depth 3 -> num_spec 4
    S = 8
    parent = [-1, 0, 0, 1, 1, 2, 2, 3]
    first_child = {0: 1, 1: 3, 2: 5, 3: 7}
    nxt_sib = {1: 2, 3: 4, 5: 6}
    ridx = (torch.arange(bs)[:, None] * S + torch.arange(S)[None, :]).long()
    rnext = torch.tensor([first_child.get(i, -1) for i in range(S)]).repeat(bs, 1).long()
    rsib = torch.tensor([nxt_sib.get(i, -1) for i in range(S)]).repeat(bs, 1).long()
    return ridx, rnext, rsib, parent, 4


def draw_candidates(q, idx, parent, bs, S):
    """Draft token of node i, from the target row of its parent: its argmax (like a good draft, so
    walks go deep), a sample from it, or a random token."""
    cands = torch.zeros(bs, S, dtype=torch.long)
    for b in range(bs):
        used = {}
        for i in range(1, S):
            pr = b * S + parent[i]
            r = torch.rand(1, generator=g).item()
            if r < 0.6:
                t = int(idx[pr, 0])
            elif r < 0.85:
                t = int(idx[pr, torch.multinomial(q[pr].cpu(), 1, generator=g)])
            else:
                t = int(torch.randint(0, V, (1,), generator=g))
            while t in used.get(parent[i], ()):  # siblings carry distinct tokens
                t = int(torch.randint(0, V, (1,), generator=g))
            used.setdefault(parent[i], set()).add(t)
            cands[b, i] = t
    return cands


def to(*ts):
    return [t.to(dev) for t in ts]


# ---------------------------------------------------------------- A. probabilities
print(f"device {dev}, vocab {V}")
print("A. probabilities")
for kind in ["peaked", "bf16", "masked"]:
    for mix, apply_min_p in [(False, False), (False, True), (True, True)]:
        bs, S = 5, 4
        logits = make_logits(bs * S, kind)
        temps, top_ks, top_ps, min_ps = make_params(bs, mix)
        kp = sv.sparse_verify_width(int(top_ks.max()))
        logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d = to(logits, temps, top_ks, top_ps, min_ps)
        dp = dense_probs(logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d, S, apply_min_p)
        q, idx = sv.sparse_target_probs(logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d, S, kp,
                                        apply_top_p=True, apply_min_p=apply_min_p)
        dg = dp.gather(1, idx)
        outside = (dp.sum(-1) - dg.sum(-1)).abs().max().item()
        set_diff = int(((dg > 0) != (q > 0)).any(-1).sum())
        maxdiff = (dg - q).abs().max().item()
        kept = (q > 0).sum(-1).float()
        check(f"{kind:6s} mix={int(mix)} min_p={int(apply_min_p)} kp={kp}",
              outside < 1e-6 and set_diff == 0 and maxdiff < 2e-6,
              f"mass outside {outside:.1e}, rows with a different kept set {set_diff}, max |dp-q| {maxdiff:.1e}, "
              f"kept {kept.min():.0f}-{kept.max():.0f}")

# ---------------------------------------------------------------- B. tree walk + final draw
print("B. tree walk and final draw, equal coins")
# top_k 1 (bf16 ties at the top still give 2+ tokens): drafts mostly accepted, covers full acceptance
for name, tree_fn, top_k in [("chain S=4", lambda bs: chain_tree(bs, 4), None),
                             ("chain S=16", lambda bs: chain_tree(bs, 16), None),
                             ("siblings", sibling_tree, None),
                             ("chain S=4 k1", lambda bs: chain_tree(bs, 4), 1),
                             ("chain S=16 k1", lambda bs: chain_tree(bs, 16), 1)]:
    for probs_src in ["dense chain", "scattered sparse"]:
        bs = 48 if dev == "cuda" else 12
        ridx, rnext, rsib, parent, num_spec = tree_fn(bs)
        S = ridx.shape[1]
        logits = make_logits(bs * S, "bf16")
        temps, top_ks, top_ps, min_ps = make_params(bs, top_k is None, top_k or 40)
        kp = sv.sparse_verify_width(int(top_ks.max()))
        logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d = to(logits, temps, top_ks, top_ps, min_ps)
        q, idx = sv.sparse_target_probs(logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d, S, kp,
                                        apply_top_p=True, apply_min_p=True)
        cands = draw_candidates(q.cpu(), idx.cpu(), parent, bs, S)
        if probs_src == "dense chain":
            dp = dense_probs(logits_d, temps_d, top_ks_d, top_ps_d, min_ps_d, S, True)
        else:
            dp = torch.zeros(bs * S, V, device=dev).scatter_(1, idx, q)
        coins = torch.rand(bs, S, generator=g); coinsf = torch.rand(bs, generator=g)
        cands_d, ridx_d, rnext_d, rsib_d, coins_d, coinsf_d = to(cands, ridx, rnext, rsib, coins, coinsf)
        outs = []
        for impl in ["dense", "sparse"]:
            pred = torch.zeros(bs * S, dtype=torch.int32, device=dev)
            acc_idx = torch.full((bs, num_spec), -1, dtype=torch.int32, device=dev)
            acc_num = torch.empty(bs, dtype=torch.int32, device=dev)
            if impl == "dense":
                dense_tree(pred, acc_idx, acc_num, cands_d, ridx_d, rnext_d, rsib_d, coins_d, coinsf_d,
                           dp.view(bs, S, V))
            else:
                sv.tree_speculative_sampling_target_only_sparse(
                    pred, acc_idx, acc_num, cands_d, ridx_d, rnext_d, rsib_d, coins_d, coinsf_d,
                    q.view(bs, S, kp), idx.view(bs, S, kp), V)
            outs.append((pred.cpu(), acc_idx.cpu(), acc_num.cpu()))
        (pd, ad, nd), (ps, as_, ns) = outs
        same_walk = int((ad == as_).all(-1).sum())
        same_tok = 0
        for b in range(bs):  # emitted tokens: predicts at the accepted path
            path = [int(i) for i in ad[b] if i >= 0]
            same_tok += int(all(pd[i] == ps[i] for i in path))
        acc = nd.float().mean().item(); full = int((nd == num_spec - 1).sum())
        # rounding may flip a coin sitting on a bucket edge; allow 1 in 48 per case
        check(f"{name:13s} probs={probs_src:16s}", bs - same_walk <= bs // 48 and bs - same_tok <= bs // 48,
              f"same accept path {same_walk}/{bs}, same emitted tokens {same_tok}/{bs}, "
              f"mean accepted drafts {acc:.2f}, all accepted {full}")

# ---------------------------------------------------------------- C. distribution
print("C. distribution of the first emitted token")
M = 40000 if dev == "cuda" else 3000
for S, rank, row in [(2, 1, 0), (2, 0, 0), (3, 3, 0), (2, 0, 3), (3, 1, 2)]:
    # one request's S rows, copied M times (full-vocab logits for M * S rows would not fit);
    # row 0 has a flat head, rows 2 and 3 are peaked (high acceptance)
    logits = make_logits(row + 1, "bf16")[row:].repeat(S, 1)
    kp = sv.sparse_verify_width(40)
    q, idx = sv.sparse_target_probs(*to(logits, *make_params(1, False)), S, kp, apply_top_p=True, apply_min_p=True)
    q, idx = q.repeat(M, 1), idx.repeat(M, 1)
    q0, i0 = q[0].cpu(), idx[0].cpu()
    ridx, rnext, rsib, parent, num_spec = chain_tree(M, S)
    cands = torch.zeros(M, S, dtype=torch.long); cands[:, 1:] = int(i0[rank])  # draft the rank-th token
    coins = torch.rand(M, S, generator=g); coinsf = torch.rand(M, generator=g)
    pred = torch.zeros(M * S, dtype=torch.int32, device=dev)
    acc_idx = torch.full((M, num_spec), -1, dtype=torch.int32, device=dev)
    acc_num = torch.empty(M, dtype=torch.int32, device=dev)
    sv.tree_speculative_sampling_target_only_sparse(pred, acc_idx, acc_num, *to(cands, ridx, rnext, rsib, coins, coinsf),
                                                    q.view(M, S, kp), idx.view(M, S, kp), V)
    first = pred.view(M, S)[:, 0].cpu().long()
    support = i0[q0 > 0]
    obs = torch.stack([(first == t).sum() for t in support]).double()
    exp = q0[q0 > 0].double() * M
    outside = M - int(obs.sum())
    big = exp >= 5  # pool small cells
    o = torch.cat([obs[big], obs[~big].sum()[None]]); e = torch.cat([exp[big], exp[~big].sum()[None]])
    if e[-1] < 5:
        o, e = o[:-1], e[:-1]
    chi2 = float(((o - e) ** 2 / e).sum()); dof = len(o) - 1
    # Wilson-Hilferty normal approximation of the chi2 upper tail
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    check(f"S={S} row {row} draft=rank {rank}", outside == 0 and pval > 1e-3,
          f"support {len(support)}, outside {outside}, chi2 {chi2:.1f} dof {dof}, p {pval:.3f}, "
          f"accept rate {acc_num.float().mean():.3f}")

# ---------------------------------------------------------------- D. GPU time
if dev == "cuda":
    print("D. GPU time per verify (us), kernels queued behind a sleep so launch cost is hidden")
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob, tree_speculative_sampling_target_only

    def dense_verify(lg, t, k, p, m, S, st):
        bs = lg.shape[0] // S
        pr = torch.softmax(lg / torch.repeat_interleave(t, S)[:, None], dim=-1)
        pr = top_k_renorm_prob(pr, torch.repeat_interleave(k, S))
        pr = top_p_renorm_prob(pr, torch.repeat_interleave(p, S))
        pr = pr.reshape(bs, S, -1)
        tree_speculative_sampling_target_only(
            predicts=st[0], accept_index=st[1], accept_token_num=st[2], candidates=st[3], retrive_index=st[4],
            retrive_next_token=st[5], retrive_next_sibling=st[6], uniform_samples=st[7],
            uniform_samples_for_final_sampling=st[8], target_probs=pr, draft_probs=torch.zeros_like(pr),
            threshold_single=1.0, threshold_acc=1.0, deterministic=True)

    def sparse_verify(lg, t, k, p, m, S, st):
        bs = lg.shape[0] // S
        kp = sv.sparse_verify_width(40)
        q, idx = sv.sparse_target_probs(lg, t, k, p, m, S, kp, apply_top_p=True, apply_min_p=False)
        sv.tree_speculative_sampling_target_only_sparse(*st, q.view(bs, S, kp), idx.view(bs, S, kp), V)

    for bs, S in [(1, 4), (1, 8), (1, 16), (4, 4)]:
        lg = make_logits(bs * S, "bf16").to(dev)
        t, k, p, m = to(*make_params(bs, False))
        ridx, rnext, rsib, parent, num_spec = chain_tree(bs, S)
        st = [torch.zeros(bs * S, dtype=torch.int32, device=dev), torch.full((bs, num_spec), -1, dtype=torch.int32, device=dev),
              torch.empty(bs, dtype=torch.int32, device=dev)] + to(torch.zeros(bs, S, dtype=torch.long), ridx, rnext, rsib,
                                                                 torch.rand(bs, S, generator=g), torch.rand(bs, generator=g))
        res, wall = {}, {}
        for name, fn in [("dense", dense_verify), ("sparse", sparse_verify)]:
            for _ in range(5):
                fn(lg, t, k, p, m, S, st)
            ts = []
            for _ in range(40):
                a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(3_000_000)
                a.record(); fn(lg, t, k, p, m, S, st); b_.record()
                torch.cuda.synchronize(); ts.append(a.elapsed_time(b_) * 1000)
            res[name] = statistics.median(ts)
            # back-to-back calls: max(CPU launch, GPU time) per call, as in the eager verify
            ws = []
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(50):
                    fn(lg, t, k, p, m, S, st)
                torch.cuda.synchronize(); ws.append((time.perf_counter() - t0) / 50 * 1e6)
            wall[name] = statistics.median(ws)
        print(f"  bs={bs} S={S:2d}: GPU dense {res['dense']:6.1f} sparse {res['sparse']:6.1f} "
              f"saved {res['dense'] - res['sparse']:6.1f} | back-to-back dense {wall['dense']:6.1f} "
              f"sparse {wall['sparse']:6.1f} saved {wall['dense'] - wall['sparse']:6.1f} us")
        # where the sparse GPU time goes: the top-k over V vs the two small kernels
        kp = sv.sparse_verify_width(40)
        parts = {"topk sorted": lambda: torch.topk(lg, kp, dim=-1, sorted=True),
                 "topk unsorted": lambda: torch.topk(lg, kp, dim=-1, sorted=False),
                 "topk+probs": lambda: sv.sparse_target_probs(lg, t, k, p, m, S, kp, apply_top_p=True, apply_min_p=False)}
        br = {}
        for name, fn in parts.items():
            for _ in range(5):
                fn()
            ts = []
            for _ in range(40):
                a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda._sleep(3_000_000)
                a.record(); fn(); b_.record()
                torch.cuda.synchronize(); ts.append(a.elapsed_time(b_) * 1000)
            br[name] = statistics.median(ts)
        print("           sparse parts (GPU us): " + ", ".join(f"{n} {v:.1f}" for n, v in br.items()))

print("RESULT:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
sys.exit(1 if fails else 0)
