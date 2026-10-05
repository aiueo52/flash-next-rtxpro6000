"""Secondary analysis for the RS package ABBA (specs/RS2D_DRAFT_SHARPEN_2026-10-01.md 4.4): bench/stats/ancova_ab.py's
fits with one more covariate, the log of the request's mean SM clock over its decode window (clocks.csv from
bench/rs2d/abba_after_queue.sh, one sample per 5 s). The RS2 ABBA's verify span drifted 35% over one evening on the
same code; this asks how much of the arm-to-arm noise the clock explains.

usage: ancova_clock.py <clocks.csv> A1=<arm.jsonl> B1=<...> ...   (fnbench rows; arm order = argument order)

Per fit it prints the B effect with a request-level CI, the clock elasticity (about -1 if the step is compute-bound,
about 0 if memory-bound; tok/step is a placebo and should show about 0) and the arm-level line as in ancova_ab.py.
It also prints the B - A clock difference: if RS moves the clock (power draw), the adjusted B effect leaves that
path out and is to be read with that caveat.
"""
import bisect
import json
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

sys.path.insert(0, "bench/stats")
sys.path.insert(0, "bench/rs2d")
from ancova_ab import fit, pct  # noqa: E402
from clock_phase import load_clocks, number  # noqa: E402


def request_clock(times, clocks, start, end):
    # mean of the samples inside the decode window; the nearest sample for windows shorter than the 5 s period
    lo, hi = bisect.bisect_left(times, start), bisect.bisect_right(times, end)
    if hi > lo:
        return float(np.mean(clocks[lo:hi]))
    mid = start + (end - start) / 2
    i = min(max(bisect.bisect_left(times, mid), 1), len(times) - 1)
    return clocks[i] if times[i] - mid < mid - times[i - 1] else clocks[i - 1]


def read_rows(label, pos, path, times, clocks):
    rows = []
    for line in open(path):
        r = json.loads(line)
        acc = r["server"]["acceptance"]["tokens_per_verify"]
        client = r["client"]
        ms_tok = client["decode_seconds"] * 1000 / client["usage"]["completion_tokens"]
        start = datetime.fromisoformat(r["timestamp"]).astimezone().replace(tzinfo=None)
        start += timedelta(seconds=client["ttft_seconds"])
        clock = request_clock(times, clocks, start, start + timedelta(seconds=client["decode_seconds"]))
        rows.append((label, pos, r["sampling_mode"], r["workload"], r["prompt_id"], acc, ms_tok * acc, ms_tok,
                     np.log(clock)))
    return rows


def main():
    samples = load_clocks(sys.argv[1])
    times = [s["timestamp"] for s in samples]
    clocks = [number(s["clocks.current.sm"]) for s in samples]
    arms = [a.split("=", 1) for a in sys.argv[2:]]
    labels = [a[0] for a in arms]
    rows = [r for pos, (label, path) in enumerate(arms) for r in read_rows(label, pos, path, times, clocks)]
    for mode in sorted({r[2] for r in rows}):
        rs = [r for r in rows if r[2] == mode]
        units = sorted({r[4] for r in rs})
        wls = sorted({r[3] for r in rs})
        unit_cols = [np.array([r[4] == u for r in rs], float) for u in units]
        log_acc = np.log([r[5] for r in rs])
        slope_cols = [np.array([r[3] == w for r in rs], float) * log_acc for w in wls]
        log_clock = np.array([r[8] for r in rs])
        clock_col = log_clock - log_clock.mean()
        pos = np.array([r[1] for r in rs], float)
        isb = np.array([r[0].startswith("B") for r in rs], float)
        print(f"== {mode}: {len(rs)} requests, {len(units)} prompts, {len(wls)} workloads, {len(labels)} arms")
        arm_clock = {lab: np.mean([np.exp(r[8]) for r in rs if r[0] == lab]) for lab in labels}
        print("  mean SM clock per arm (MHz): " + "  ".join(f"{lab} {arm_clock[lab]:.0f}" for lab in labels))
        m = np.log([arm_clock[lab] for lab in labels])
        ap = np.arange(len(labels), dtype=float)
        ab = np.array([lab.startswith("B") for lab in labels], float)
        if len(labels) >= 5:
            bc, _, sec, dofc = fit(m, [np.ones(len(labels))], ap, ab)
            tc = stats.t.ppf(0.975, dofc)
            print(f"  B - A clock {pct(bc[-1]):+6.2f}%  [{pct(bc[-1] - tc * sec):+6.2f} .. {pct(bc[-1] + tc * sec):+6.2f}]"
                  f"  (arm level, {dofc} dof)")
        for name, y, cols in (("ms/token", np.log([r[7] for r in rs]), unit_cols),
                              ("tok/step", log_acc, unit_cols),
                              ("ms/step|a", np.log([r[6] for r in rs]), unit_cols + slope_cols)):
            beta, s2, se, dof = fit(y, cols + [clock_col], pos, isb)
            g, tq = beta[-1], stats.t.ppf(0.975, dof)
            X = np.column_stack(cols + [clock_col, pos - pos.mean(), isb])
            se_clock = np.sqrt(s2 * np.linalg.inv(X.T @ X)[-3, -3])
            print(f"  {name:9s} B effect {pct(g):+6.2f}%  [{pct(g - tq * se):+6.2f} .. {pct(g + tq * se):+6.2f}]"
                  f"  resid sd {100 * np.sqrt(s2):.2f}%  clock elasticity {beta[-3]:+.2f} (se {se_clock:.2f})")
            Xn = np.column_stack(cols + [clock_col])
            rn = y - Xn @ np.linalg.lstsq(Xn, y, rcond=None)[0]
            adj = {lab: pct(np.mean([rn[i] for i in range(len(rs)) if rs[i][0] == lab])) for lab in labels}
            print(f"  {'':9s} arm means (%): " + "  ".join(f"{lab} {adj[lab]:+.2f}" for lab in labels))
            if len(labels) >= 5:
                ma = np.log1p(np.array([adj[lab] for lab in labels]) / 100)
                ba, s2a, sea, dofa = fit(ma, [np.ones(len(labels))], ap, ab)
                ta = stats.t.ppf(0.975, dofa)
                print(f"  {'':9s} arm level B effect {pct(ba[-1]):+6.2f}%  [{pct(ba[-1] - ta * sea):+6.2f} .."
                      f" {pct(ba[-1] + ta * sea):+6.2f}]  ({dofa} dof, arm sd {100 * np.sqrt(s2a):.2f}%)")


if __name__ == "__main__":
    main()
