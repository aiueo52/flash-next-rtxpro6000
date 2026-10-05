"""ABBA arms with the acceptance noise taken out. Per mode, three least-squares fits with a B indicator and the
arm position (drift), each with one intercept per prompt (probe rows: one prompt per workload):

  ms/token   log(ms per output token), the user-facing cost (1 / tok/s); includes any acceptance change;
  tok/step   log(tokens per verify step), the acceptance;
  ms/step|a  log(ms/step) with per-workload slopes of log(tok/step): the per-step cost at equal acceptance.
             In wa, ms/step rises with tok/step (the adaptive width), so a run that accepted more is slower per
             step for a reason that is not the change under test; the slope removes that part.

usage: ancova_ab.py A1=<arm.jsonl> B1=<...> B2=<...> A2=<...> [A3=... B3=... ...]   (arm order = argument order)
  arm files: probe rows (prof/fc_sampling_probe.py --json-out) or fnbench rows (bench/stack/arm_stack.sh).

Prints per mode and fit: the B effect in % with a request-level 95% CI, and the covariate-adjusted arm means
(|A2-A1| and |B2-B1| show how much one server start moves the result). The request-level CI ignores that
server-start noise, so it is too narrow (SV1's greedy rows, which SV1 cannot change, gave -1.22% [-2.02, -0.40]);
with 5 or more arms the arm-level line (arm means on drift + B, one server start as the unit) is the honest one.
"""
import json
import sys

import numpy as np
from scipy import stats


def read_rows(label, pos, path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "server" in r:  # fnbench
            acc = r["server"]["acceptance"]["tokens_per_verify"]
            ms_tok = r["client"]["decode_seconds"] * 1000 / r["client"]["usage"]["completion_tokens"]
            rows.append((label, pos, r["sampling_mode"], r["workload"], r["prompt_id"], acc, ms_tok * acc, ms_tok))
        elif not r["profiled"]:
            rows.append((label, pos, r["mode"], r["workload"], r["workload"], r["tok_per_step"], r["ms_per_step"],
                         r["ms_per_step"] / r["tok_per_step"]))
    return rows


def fit(y, cols, pos, isb):
    X = np.column_stack(cols + [pos - pos.mean(), isb])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1])
    return beta, s2, se, dof


def pct(x):
    return 100 * (np.exp(x) - 1)


def main():
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    labels = [a[0] for a in arms]
    rows = [r for pos, (label, path) in enumerate(arms) for r in read_rows(label, pos, path)]
    for mode in sorted({r[2] for r in rows}):
        rs = [r for r in rows if r[2] == mode]
        units = sorted({r[4] for r in rs})
        wls = sorted({r[3] for r in rs})
        unit_cols = [np.array([r[4] == u for r in rs], float) for u in units]
        log_acc = np.log([r[5] for r in rs])
        slope_cols = [np.array([r[3] == w for r in rs], float) * log_acc for w in wls]
        pos = np.array([r[1] for r in rs], float)
        isb = np.array([r[0].startswith("B") for r in rs], float)
        print(f"== {mode}: {len(rs)} requests, {len(units)} prompts, {len(wls)} workloads, {len(labels)} arms")
        for name, y, cols in (("ms/token", np.log([r[7] for r in rs]), unit_cols),
                              ("tok/step", log_acc, unit_cols),
                              ("ms/step|a", np.log([r[6] for r in rs]), unit_cols + slope_cols)):
            beta, s2, se, dof = fit(y, cols, pos, isb)
            g, tq = beta[-1], stats.t.ppf(0.975, dof)
            # covariate-adjusted arm means: residual of the fit without arm terms, averaged per arm
            Xn = np.column_stack(cols)
            rn = y - Xn @ np.linalg.lstsq(Xn, y, rcond=None)[0]
            adj = {lab: pct(np.mean([rn[i] for i in range(len(rs)) if rs[i][0] == lab])) for lab in labels}
            extra = ("  slopes " + " ".join(f"{beta[len(units) + i]:.2f}" for i in range(len(wls)))
                     if name == "ms/step|a" else "")
            print(f"  {name:9s} B effect {pct(g):+6.2f}%  [{pct(g - tq * se):+6.2f} .. {pct(g + tq * se):+6.2f}]"
                  f"  resid sd {100 * np.sqrt(s2):.2f}%{extra}")
            print(f"  {'':9s} arm means (%): " + "  ".join(f"{lab} {adj[lab]:+.2f}" for lab in labels))
            # instance level: the arm means on drift + B; one server start is the unit (needs >= 5 arms)
            if len(labels) >= 5:
                m = np.log1p(np.array([adj[lab] for lab in labels]) / 100)
                ap = np.arange(len(labels), dtype=float)
                ab = np.array([lab.startswith("B") for lab in labels], float)
                ba, s2a, sea, dofa = fit(m, [np.ones(len(labels))], ap, ab)
                ta = stats.t.ppf(0.975, dofa)
                print(f"  {'':9s} arm level B effect {pct(ba[-1]):+6.2f}%  [{pct(ba[-1] - ta * sea):+6.2f} .."
                      f" {pct(ba[-1] + ta * sea):+6.2f}]  ({dofa} dof, arm sd {100 * np.sqrt(s2a):.2f}%)")


if __name__ == "__main__":
    main()
