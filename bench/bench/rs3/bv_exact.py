"""Exact enumeration oracle for block verification (RS3 spec, section 1).

For a small vocabulary V and a chain of G draft tokens, every draft path and every output is enumerated.
The output distribution of block verification and of token-level verification is compared with the target's,
and the expected accepted drafts of both are returned. `python bv_exact.py` runs the 360 cases of the spec.
Distributions are dicts from prefix tuples to numpy arrays: P (target) and Q (draft) for every prefix of
length 0..G.
"""
import itertools

import numpy as np


def bv_tau_probs(*, path, P, Q):
    """P(tau = t | path) for t = 0..G and pi_0..pi_G, with section 1's edge rules."""
    G = len(path)
    pi = [1.0]
    for i in range(1, G + 1):
        pt, qt = P[path[:i - 1]][path[i - 1]], Q[path[:i - 1]][path[i - 1]]
        pi.append(0.0 if pt <= 0 else (1.0 if qt == 0 else min(pi[-1] * pt / qt, 1.0)))
    h = [1.0]
    for i in range(1, G):
        n = np.maximum(pi[i] * P[path[:i]] - Q[path[:i]], 0).sum()
        h.append(n / (n + 1 - pi[i]) if n + 1 - pi[i] > 0 else 1.0)
    h.append(pi[G])
    probs = []
    for t in range(G + 1):
        pr = h[t]
        for j in range(t + 1, G + 1):
            pr *= 1 - h[j]
        probs.append(pr)
    return probs, pi


def token_tau_probs(*, path, P, Q):
    """P(tau = t | path) for token-level verification (accept while coin < p / q)."""
    G = len(path)
    probs, alive = [], 1.0
    for i in range(1, G + 1):
        pt, qt = P[path[:i - 1]][path[i - 1]], Q[path[:i - 1]][path[i - 1]]
        a = 0.0 if pt <= 0 else (1.0 if qt == 0 else min(1.0, pt / qt))
        probs.append(alive * (1 - a))
        alive *= a
    probs.append(alive)
    return probs, [1.0] * (G + 1)


def enumerate_outputs(*, P, Q, V, G, method):
    """Output distribution over length G + 1 sequences and the expected accepted drafts."""
    out = dict.fromkeys(itertools.product(range(V), repeat=G + 1), 0.0)
    expected = 0.0
    tau_probs = bv_tau_probs if method == "bv" else token_tau_probs
    for path in itertools.product(range(V), repeat=G):
        px = float(np.prod([Q[path[:i]][path[i]] for i in range(G)]))
        if px == 0:
            continue
        probs, pi = tau_probs(path=path, P=P, Q=Q)
        expected += px * sum(t * pr for t, pr in enumerate(probs))
        for t, pt in enumerate(probs):
            if pt == 0:
                continue
            if t < G:
                w = np.maximum(pi[t] * P[path[:t]] - Q[path[:t]], 0)
                w = w if w.sum() > 0 else P[path[:t]].copy()  # zero residual: draw the target
                w = w / w.sum()
            else:
                w = P[path]
            for y in range(V):
                prefix = path[:t] + (y,)
                for ext in itertools.product(range(V), repeat=G + 1 - len(prefix)):
                    seq = prefix + ext
                    pr = px * pt * w[y]
                    for k in range(len(prefix), G + 1):
                        pr *= P[seq[:k]][seq[k]]
                    out[seq] += pr
    return out, expected


def target_distribution(*, P, V, G):
    return {seq: float(np.prod([P[seq[:i]][seq[i]] for i in range(G + 1)]))
            for seq in itertools.product(range(V), repeat=G + 1)}


def make_dist(*, rng, V, kind):
    if kind == "onehot":
        d = np.zeros(V)
        d[rng.integers(V)] = 1.0
        return d
    w = rng.gamma(0.4, size=V)
    if kind == "trunc":
        w[rng.choice(V, size=rng.integers(1, V), replace=False)] = 0.0
        if w.sum() == 0:
            w[rng.integers(V)] = 1.0
    return w / w.sum()


def random_case(*, seed, V, G, mode):
    """mode: gen (mixed kinds), equal (q = p at about half the prefixes), greedy (one-hot targets)."""
    rng = np.random.default_rng(seed)
    P, Q = {}, {}
    for length in range(G + 2):
        for prefix in itertools.product(range(V), repeat=length):
            P[prefix] = make_dist(rng=rng, V=V, kind=rng.choice(["gen", "onehot", "trunc"], p=[.5, .25, .25]))
            if mode == "equal" and rng.random() < .5:
                Q[prefix] = P[prefix].copy()
            elif mode == "greedy":
                P[prefix] = make_dist(rng=rng, V=V, kind="onehot")
                Q[prefix] = make_dist(rng=rng, V=V, kind=rng.choice(["gen", "onehot"]))
            else:
                Q[prefix] = make_dist(rng=rng, V=V, kind=rng.choice(["gen", "onehot", "trunc"], p=[.6, .2, .2]))
    return P, Q


def main():
    worst, below, greedy_diff, cases = 0.0, 0, 0.0, 0
    for mode in ("gen", "equal", "greedy"):
        for seed in range(120):
            V, G = 3 + seed % 2, 1 + seed % 4
            P, Q = random_case(seed=seed, V=V, G=G, mode=mode)
            target = target_distribution(P=P, V=V, G=G)
            out_bv, e_bv = enumerate_outputs(P=P, Q=Q, V=V, G=G, method="bv")
            out_tok, e_tok = enumerate_outputs(P=P, Q=Q, V=V, G=G, method="token")
            worst = max(worst, max(abs(out_bv[s] - target[s]) for s in target),
                        max(abs(out_tok[s] - target[s]) for s in target))
            below += e_bv < e_tok - 1e-12
            if mode == "greedy":
                greedy_diff = max(greedy_diff, abs(e_bv - e_tok))
            cases += 1
    print(f"cases={cases} worst |output - target|={worst:.2e} BV below token-level={below} "
          f"greedy |E_bv - E_tok|={greedy_diff:.2e}")
    assert worst < 1e-12 and below == 0 and greedy_diff < 1e-12


if __name__ == "__main__":
    main()
