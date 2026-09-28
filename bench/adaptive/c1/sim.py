"""C1 offline policy simulation on RECORDED per-step traces.

Input: the two trace files produced by trace_run.sh (one fixed-W16 server, one
fixed-W4 server), each a JSONL stream of
    {"i":n, "steps":S, "bs":b, "p":[[p0..p_{S-1}], ...], "a":[accepted,...]}
plus a segments file giving the last line index of each workload.

The counterfactual problem: a trace taken at S=15 never says what S=3 would
have accepted on that same step.  Two estimators are used and reported side by
side, because they fail in opposite directions:

  empirical  -- draw the accepted count from the OTHER run's pool for the same
                position-0 confidence bucket.  Uses only measured numbers, but
                assumes the two runs visit comparable token distributions
                within a bucket.
  geometric  -- invert this step's own accepted count into a per-position rate
                r and evaluate E[accept | r, S'] for the other S'.  Uses this
                exact step, but assumes iid per-position acceptance.

A policy that wins under both is worth taking to the server.
"""
from __future__ import annotations
import json, os, random, sys, math, collections, statistics

if os.environ.get("SGLANG_PYTHON"):  # python/ of a flash-next-fast checkout; else the installed sglang
    sys.path.insert(0, os.environ["SGLANG_PYTHON"])
from sglang.srt.speculative.adaptive_confidence import (
    expected_accept, invert_accept, step_time_ms, ConfidenceStepSlot)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveStepSlot

D = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = ["code-edit", "prose-en", "prose-ja", "agent-loop"]
SWITCH_PENALTY = int(os.environ.get("SWPEN", "2"))   # cold batches after a swap
BUCKETS = tuple(
    float(x) for x in os.environ.get("C1_BUCKETS", "0.8,0.95,0.99,0.999").split(","))


def bucket(c):
    i = 0
    for e in BUCKETS:
        if c < e:
            return i
        i += 1
    return i


def load(profile):
    """Split a trace into per-workload segments.

    Prefers the wall-clock gap between fnbench invocations (the server is idle
    for seconds between them) over the recorded line counts: the line counts
    are taken with `wc -l` between runs and were wrong for the first w16 trace
    because the tracer was block-buffered.  Falls back to the counts when the
    trace carries no timestamps.
    """
    path = f"{D}/trace-{profile}.jsonl.rank0"
    seg = f"{D}/segments-{profile}.txt"
    rows = [json.loads(l) for l in open(path)]
    order = [l.split()[0] for l in open(seg)]
    # Both traces are now written line-buffered, so the counts taken with
    # `wc -l` between fnbench invocations are exact.  (They were not for the
    # first w16 trace; see specs/C1_LOG.md.  Sanity check when re-collecting:
    # each segment's mean accepted must match fnbench's own acc - 1.)
    counts = [int(l.split()[1]) for l in open(seg)]
    cuts = [0] + counts
    bounds = {order[k]: (cuts[k], cuts[k + 1]) for k in range(len(order))}
    out = {}
    for w, (a, b) in bounds.items():
        rs = rows[a:b]
        # one entry per (step, request); fnbench is single-stream so bs==1
        out[w] = [
            (r["p"][k][0] if r["p"] else None, r["a"][k], r["steps"])
            for r in rs
            for k in range(len(r["a"]))
            if r["p"] is None or k < len(r["p"])
        ]
    return out


def pools(samples):
    """confidence bucket -> list of accepted counts."""
    p = collections.defaultdict(list)
    for conf, acc, steps in samples:
        p[bucket(conf) if conf is not None else -1].append(acc)
    return p


def describe(name, samples):
    accs = [a for _, a, _ in samples]
    p = pools(samples)
    print(f"  {name:10s} n={len(accs):5d} mean_acc={statistics.mean(accs):5.2f} "
          f"p50={statistics.median(accs):4.1f} "
          f"frac0={sum(a == 0 for a in accs)/len(accs):.2f}")
    for b in sorted(p):
        v = p[b]
        lo = "-inf" if b == 0 else f"{BUCKETS[b-1]:.3f}"
        hi = f"{BUCKETS[b]:.3f}" if b < len(BUCKETS) else "1"
        print(f"      conf[{lo},{hi}) n={len(v):5d} ({len(v)/len(accs):5.1%}) "
              f"mean={statistics.mean(v):5.2f} frac0={sum(a==0 for a in v)/len(v):.2f}")


class Fixed:
    def __init__(self, s): self.current_steps = s; self.candidate_steps = [s]
    def observe_confidence(self, c): pass
    def update(self, a): return False


class Oracle:
    """Per-step best choice with perfect knowledge. Upper bound, not a policy."""
    def __init__(self, cands): self.candidate_steps = cands; self.current_steps = cands[-1]
    def observe_confidence(self, c): pass
    def update(self, a): return False


def simulate(slot, conf_seq, draw, batches=None, oracle=False):
    """Replay the confidence stream; draw acceptance for whatever S is live."""
    n = batches or len(conf_seq)
    tok = 0.0; ms = 0.0; sw = 0; cold = 0; prev = slot.current_steps
    inflight = [slot.current_steps] * 2      # decision->observation lag
    for t in range(n):
        conf = conf_seq[t % len(conf_seq)]
        if hasattr(slot, 'observe_confidence'):
            slot.observe_confidence([conf])
        if oracle:
            best, bt = None, -1.0
            for s in slot.candidate_steps:
                a = draw(s, conf, t)
                v = (1 + a) / step_time_ms(s)
                if v > bt: best, bt, ba = s, v, a
            slot.current_steps = best
            tok += 1 + ba; ms += step_time_ms(best)
            continue
        live = slot.current_steps
        a = draw(live, conf, t)
        if cold:
            a = min(a, 1); cold -= 1
        tok += 1 + a; ms += step_time_ms(live)
        produced_by = inflight.pop(0)
        obs = a if produced_by == live else draw(produced_by, conf, t)
        slot.update([obs])
        inflight.append(slot.current_steps)
        if slot.current_steps != prev:
            sw += 1; cold = SWITCH_PENALTY; prev = slot.current_steps
    return sw, tok / ms * 1000


def make_prefix_draw(seq15):
    """The faithful counterfactual, in trace order.

    A topk=1 draft is a greedy chain and the target accepts a prefix of it, and
    the first S tokens of a 15-chain are the tokens a S-chain would have
    drafted.  So a step recorded at S=15 says exactly what any shorter chain
    would have accepted: min(a, S).  Replaying in ORDER also keeps the temporal
    clustering of hard passages, which is the whole reason the shipped EMA
    controller dips -- an i.i.d. resample from a pool destroys it and makes
    every controller look stable.
    """
    def draw(S, conf, t):
        return min(seq15[t % len(seq15)], S)
    return draw


def make_draw(pool_by_steps, mode, rng):
    """draw(S, conf, t) -> accepted count."""
    real = {s: pools(v) for s, v in pool_by_steps.items()}
    flat = {s: [a for _, a, _ in v] for s, v in pool_by_steps.items()}
    seq = {s: v for s, v in pool_by_steps.items()}

    def draw(S, conf, t):
        if S in seq and mode != "geometric":
            b = bucket(conf)
            cand = real[S].get(b) or flat[S]
            return cand[rng.randrange(len(cand))]
        # geometric transfer from whichever run we do have at this step
        src = max(seq, key=lambda s: len(seq[s]))
        _, a_src, s_src = seq[src][t % len(seq[src])]
        r = invert_accept(a_src / 1.15, s_src)
        return int(round(1.15 * expected_accept(r, S)))
    return draw


def main():
    tr = {p: load(p) for p in ("w16", "w4")}
    steps_of = {"w16": 15, "w4": 3}
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1
                         else f"{D}/../w16_3_15.json"))["1"]
    conf_cfg = json.load(open(f"{D}/conf.json"))["1"] if os.path.exists(f"{D}/conf.json") else dict(cfg)

    for wl in WORKLOADS:
        if wl not in tr["w16"]:
            continue
        print(f"\n=== {wl}")
        for p in ("w16", "w4"):
            describe(p, tr[p][wl])
        pool_by_steps = {steps_of[p]: tr[p][wl] for p in tr}
        conf_seq = [c for c, _, _ in tr["w16"][wl] if c is not None] or [0.9]
        seq15 = [a for _, a, _ in tr["w16"][wl]]
        results = {}
        for mode in ("prefix", "empirical"):
            row = {}
            for name, mk in [
                ("fixed W4", lambda: Fixed(3)),
                ("fixed W16", lambda: Fixed(15)),
                ("EMA (shipped)", lambda: AdaptiveStepSlot(15, cfg)),
                ("confidence", lambda: ConfidenceStepSlot(15, conf_cfg)),
                ("conf (no buckets)", lambda: ConfidenceStepSlot(
                    15, {**conf_cfg, "confidence_weight": 0.0})),
            ]:
                if mode == "prefix":
                    accum = [simulate(mk(), conf_seq, make_prefix_draw(seq15))]
                else:
                    accum = [
                        simulate(mk(), conf_seq,
                                 make_draw(pool_by_steps, mode, random.Random(sd)))
                        for sd in range(5)
                    ]
                row[name] = (sum(x[0] for x in accum) / len(accum),
                             sum(x[1] for x in accum) / len(accum))
            d3 = (make_prefix_draw(seq15) if mode == "prefix"
                  else make_draw(pool_by_steps, mode, random.Random(0)))
            row["ORACLE"] = simulate(Oracle([3, 15]), conf_seq, d3, oracle=True)
            results[mode] = row
        names = list(results["empirical"])
        print(f"  {'policy':20s} " + "".join(f"{m:>22s}" for m in results))
        for nm in names:
            cells = "".join(
                f"   tps {results[m][nm][1]:6.0f} sw {results[m][nm][0]:4.0f}"
                for m in results)
            print(f"  {nm:20s} {cells}")


def sweep():
    """Grid over the confidence-policy knobs on the recorded traces.

    Scored by the worst-workload regret against the better fixed profile,
    because the whole point of one adaptive server is that a user must not be
    able to lose by not hand-picking a profile.
    """
    tr = {p: load(p) for p in ("w16", "w4")}
    steps_of = {"w16": 15, "w4": 3}
    base = json.load(open(f"{D}/conf.json"))["1"]
    grid = []
    for margin in (0.05, 0.10, 0.15):
        for wa in (0.01, 0.02):
            for ra in (0.05, 0.10):
                for grace in (20, 40, 80):
                    for interval in (10, 20):
                     for bo in (1.0, 2.0, 3.0):
                        grid.append({**base, "grace_backoff": bo, "switch_margin": margin,
                                     "weight_alpha": wa, "rate_alpha": ra,
                                     "switch_grace_batches": grace,
                                     "update_interval": interval,
                                     "warmup_batches": 15,
                                     "buckets": list(BUCKETS)})
    wls = [w for w in WORKLOADS if w in tr["w16"]]
    best_fixed, conf_seqs, draws = {}, {}, {}
    for wl in wls:
        conf_seqs[wl] = [c for c, _, _ in tr["w16"][wl] if c is not None] or [0.9]
        draws[wl] = make_prefix_draw([a for _, a, _ in tr["w16"][wl]])
        best_fixed[wl] = max(
            simulate(Fixed(s), conf_seqs[wl], draws[wl], batches=3000)[1]
            for s in (3, 15))
    scored = []
    for cfg in grid:
        rows = {}
        for wl in wls:
            rows[wl] = simulate(ConfidenceStepSlot(15, cfg), conf_seqs[wl],
                                draws[wl], batches=3000)
        worst = min(rows[w][1] / best_fixed[w] for w in wls)
        mean = sum(rows[w][1] / best_fixed[w] for w in wls) / len(wls)
        scored.append((worst, mean, cfg, rows))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    print(f"{'worst':>6} {'mean':>6}  margin  wa    ra    grace int   " +
          "  ".join(f"{w:>18s}" for w in wls))
    for worst, mean, cfg, rows in scored[:12]:
        print(f"{worst:6.3f} {mean:6.3f}  {cfg['switch_margin']:.2f}   "
              f"{cfg['weight_alpha']:.2f}  {cfg['rate_alpha']:.2f}  "
              f"{cfg['switch_grace_batches']:3d}  {cfg['update_interval']:3d}  " +
              "  ".join(f"{rows[w][1]:8.0f}/sw{rows[w][0]:4.0f}" for w in wls))
    print("\nbest fixed per workload:", {w: round(best_fixed[w]) for w in wls})
    return scored[0][2]


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sys.argv.remove("--sweep")
        best = sweep()
        json.dump({"1": best, "2": {**best, "candidate_steps": [15]}},
                  open(f"{D}/conf_tuned.json", "w"), indent=2)
        print(f"\nwrote {D}/conf_tuned.json")
    else:
        main()


