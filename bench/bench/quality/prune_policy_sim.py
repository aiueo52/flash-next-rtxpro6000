#!/usr/bin/env python3
"""CPU-only route replay; Python 3.12 + NumPy, no server imports.

Usage (from the repository root, with NumPy available):
    python3 bench/quality/prune_policy_sim.py --selftest
    python3 bench/quality/prune_policy_sim.py runs/census-*.npz \
        --output bench/quality/runs/prune_policy_sim.md
No paths defaults to all runs/census-*.npz, relative to this repository.
NPZ: call/T/k vectors; ids/w padded matrices; first T*k entries are routes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAUS = (0, 0.08, 0.085, 0.09, 0.10)
EXPERTS = 512
STEP_US = {4: 8919, 16: 16083}


@dataclass(frozen=True)
class Policy:
    tau: float
    top1: bool = False
    cap: float | None = None
    count_two: bool = False

    @property
    def label(self):
        return (f"singleton {self.tau:g}"
                + (" + top1" if self.top1 else "")
                + (f" + cap{100*self.cap:g}%" if self.cap is not None else "")
                + (" + count-two <0.08" if self.count_two else ""))


POLICIES = [Policy(t, **kw) for t in TAUS
            for kw in ({}, {"top1": True}, {"cap": 0.1}, {"cap": 0.2})]
POLICIES += [Policy(0.08, count_two=True), Policy(0.10, top1=True, cap=0.2)]


def load_census(path):
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key] for key in ("call", "T", "k", "ids", "w")}
    n = len(d["call"])
    if not n or any(d[key].shape != (n,) for key in ("call", "T", "k")):
        raise ValueError("empty census or invalid call/T/k vectors")
    if (d["ids"].ndim != 2 or d["ids"].shape[0] != n
            or d["w"].shape != d["ids"].shape):
        raise ValueError("ids and w must be equally shaped padded matrices")
    if any(not np.issubdtype(d[key].dtype, np.integer) for key in ("call", "T", "k", "ids")):
        raise ValueError("call/T/k/ids must be integers")
    order = np.argsort(d["call"], kind="stable")
    d = {key: value[order] for key, value in d.items()}
    if d["call"][0] < 0 or np.any(np.diff(d["call"]) <= 0):
        raise ValueError("call indices must be nonnegative and unique")
    if (np.any(d["T"] <= 0) or np.any(d["k"] <= 0)
            or np.any(d["T"].astype(np.int64) * d["k"] > d["ids"].shape[1])):
        raise ValueError("invalid T*k or truncated route matrix")
    return d


def cadence(d, width):
    """Infer full width-call bursts between T=1 bursts; ignore ring edges/gaps."""
    c, t = d["call"], d["T"]
    cuts = np.r_[0, np.flatnonzero((np.diff(t) != 0) | (np.diff(c) != 1)) + 1, len(c)]
    full = []
    periods = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if (t[start] == width and start > 0 and end < len(c)
                and t[start-1] == t[end] == 1
                and c[start] == c[start-1] + 1 and c[end] == c[end-1] + 1):
            full.append(int(end - start))
            # Count from this width burst to the next one, including draft T=1.
            next_width = end
            while next_width < len(c) and t[next_width] == 1:
                next_width += 1
            if (next_width < len(c) and t[next_width] == width
                    and c[next_width] - c[start] == next_width - start):
                periods.append(int(next_width - start))
    if not full or len(set(full)) != 1 or not periods or len(set(periods)) != 1:
        raise ValueError("cannot infer a fixed MoE call cadence from complete steps")
    return full[0], periods[0], len(full)


class Replay:
    def __init__(self, ids, weights):
        self.ids = np.asarray(ids)
        self.w = np.asarray(weights, dtype=np.float64)
        if (self.ids.ndim != 3 or self.ids.shape != self.w.shape
                or not all(self.ids.shape) or not np.issubdtype(self.ids.dtype, np.integer)):
            raise ValueError("routes must have shape [calls, rows, top-k], with integer ids")
        if np.any(self.ids < 0) or np.any(self.ids >= EXPERTS):
            raise ValueError("invalid active expert id (padding must be sliced off)")
        if np.any(np.diff(np.sort(self.ids, axis=2), axis=2) == 0):
            raise ValueError("duplicate expert within a row: route count would not equal row count")
        if np.any(~np.isfinite(self.w)) or np.any(self.w < 0) or np.any(self.w.sum(2) <= 0):
            raise ValueError("routing weights must be finite, nonnegative, with positive row sums")
        n = len(ids)
        self.keys = np.arange(n)[:, None, None] * EXPERTS + self.ids
        counts = np.bincount(self.keys.ravel(), minlength=n * EXPERTS)
        self.mult = counts[self.keys]
        self.distinct = (counts.reshape(n, EXPERTS) > 0).sum(1)
        weak = np.bincount(self.keys[self.w < 0.08], minlength=n * EXPERTS)
        # Immutable original counts: both routes must qualify, removed jointly.
        self.two = (self.mult == 2) & (weak[self.keys] == 2)
        self.top = np.argmax(self.w, axis=2)[..., None]
        self.order = np.argsort(self.w, axis=2, kind="stable")
        self.row_sum = self.w.sum(2)

    def dropped(self, policy):
        drop = (self.mult == 1) & (self.w < policy.tau)
        if policy.count_two:
            drop |= self.two
        if policy.top1:
            np.put_along_axis(drop, self.top, False, axis=2)
        if policy.cap is not None:
            if policy.count_two:
                raise ValueError("row caps on count-two groups require a joint group scheduler")
            # Cheapest singleton first maximizes whole experts removed per row.
            # Stable ties use original slot order; every policy starts from raw routes.
            candidates = np.take_along_axis(drop, self.order, axis=2)
            weights = np.take_along_axis(self.w, self.order, axis=2)
            spent = np.cumsum(np.where(candidates, weights, 0), axis=2)
            chosen = candidates & (spent <= policy.cap * self.row_sum[..., None])
            np.put_along_axis(drop, self.order, chosen, axis=2)
        return drop

    def stats(self, policy):
        drop = self.dropped(policy)
        # Every selected expert loses ALL of its routes; count a pair only once.
        removed = (drop / self.mult).sum(axis=(1, 2))
        mass = (self.w * drop).sum(2) / self.row_sum
        return {
            "D": float((self.distinct - removed).mean()),
            "mass": (float(mass.mean()), *map(float, np.percentile(mass, [95, 99]))),
        }


def report_file(path):
    d = load_census(path)
    match = re.search(r"(?:^|-)w(4|16)(?:-|\.)", path.name)
    if not match:
        raise ValueError("filename must identify a w4 or w16 profile")
    width = int(match[1])
    calls_per_step, total_calls, complete = cadence(d, width)
    sel = d["T"] == width
    ks = np.unique(d["k"][sel])
    if len(ks) != 1:
        raise ValueError("width group must have one top-k value")
    k = int(ks[0]); n = int(sel.sum())
    ids = d["ids"][sel, :width*k].reshape(n, width, k)
    weights = d["w"][sel, :width*k].reshape(n, width, k)
    replay = Replay(ids, weights)
    stats = {policy: replay.stats(policy) for policy in POLICIES}
    base, prod = stats[Policy(0)]["D"], stats[Policy(0.08)]["D"]
    excluded = {int(t): int((d["T"] == t).sum()) for t in np.unique(d["T"]) if t != width}
    lines = [f"## {path.name}", "",
             f"W{width}, k={k}; {n:,} width-matched calls; excluded T:call counts {excluded}. "
             f"Inferred {calls_per_step} width-matched MoE calls/step, {total_calls} total calls/step "
             f"({complete} complete width bursts). All width-matched records, including partial "
             "ring-edge steps, enter D and row percentiles; gaps split cadence detection.",
             f"Original row sums: {replay.row_sum.min():.9f}..{replay.row_sum.max():.9f}.", "",
             "| policy | mean D | ΔD vs 0 | ΔD vs .08 | mass mean % | p95 % | p99 % | µs/step vs 0 / .08 | step % vs 0 / .08 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, s in stats.items():
        d0, d08 = base - s["D"], prod - s["D"]
        review = np.array([d0, d08]) * 69.9
        pair = lambda values: " / ".join(f"{v:+.2f}" for v in values)
        lines.append(f"| {policy.label} | {s['D']:.3f} | {d0:+.3f} | {d08:+.3f} | "
                     + " | ".join(f"{100*m:.3f}" for m in s["mass"])
                     + f" | {pair(review)} | {pair(review / STEP_US[width] * 100)} |")
    if path.name == "census-w4-code-edit.npz":
        mean, p95, _ = stats[Policy(0.08)]["mass"]
        agree = abs(100*mean - 14.2) < 0.1 and abs(100*p95 - 38.4) < 0.1
        lines += ["", f"Astra mass check: {'MATCH' if agree else 'MISMATCH'}; τ=.08 gives "
                  f"{100*mean:.3f}% mean / {100*p95:.3f}% p95 versus rounded 14.2% / 38.4%. "
                  "Uses the original W4 code-edit census, all T=4 rows and strict weight < τ; "
                  "fractions divide by the original row sum (within float32 rounding of 1). "
                  "Top-1 protection has no effect here at .08. Newer p2-c1 censuses are separate inputs, "
                  "not substitutes for this comparison."]
    return lines


def selftest():
    # D=11: expert 0 occurs in three rows; 1 is a weak pair; 2 has one strong route.
    ids = np.array([[[0, 1, 2, 3, 8], [0, 1, 4, 5, 9], [0, 2, 6, 7, 10]]])
    w = np.array([[[.80, .04, .05, .07, .04], [.81, .06, .04, .08, .01],
                   [.75, .09, .03, .08, .05]]])
    replay = Replay(ids, w)
    expected = [(Policy(0), 11, [0, 0, 0]),
                (Policy(.08), 5, [.11, .05, .08]),
                (Policy(.08, count_two=True), 4, [.15, .11, .08]),
                (Policy(.08, cap=.1), 6, [.04, .05, .08]),
                (Policy(.08, cap=.2), 5, [.11, .05, .08])]
    for policy, distinct, mass in expected:
        drop = replay.dropped(policy)
        np.testing.assert_allclose((drop * w).sum(2)[0], mass, atol=1e-15)
        actual = replay.stats(policy)
        assert actual["D"] == distinct, (policy, actual)
        assert len(np.unique(ids[~drop])) == distinct
        np.testing.assert_allclose(actual["mass"], [np.mean(mass), *np.percentile(mass, [95, 99])])
    assert not replay.dropped(Policy(.08))[0, 1, 3]  # equality at τ is retained
    # Unsorted, tied top-1 weights below τ; only the first maximum is protected.
    flat = Replay(np.arange(16).reshape(1, 1, 16),
                  np.array([.025, .065, .065, *([.065]*13)]).reshape(1, 1, 16))
    assert flat.stats(Policy(.08))["D"] == 0
    assert flat.stats(Policy(.08, top1=True))["D"] == 1
    assert not flat.dropped(Policy(.08, top1=True))[0, 0, 1]
    for policy in POLICIES:
        if policy.cap is not None:
            assert np.all((flat.dropped(policy) * flat.w).sum(2) <= policy.cap * flat.row_sum)
    # Padded NPZ roundtrip, unsorted ring slots, edge bursts, and full cadence.
    import tempfile
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        t = np.array([3, 1, 3, 3, 1, 3, 3, 1, 3])
        padded_ids = np.full((len(t), 20), -1, dtype=np.int16)
        padded_w = np.zeros((len(t), 20), dtype=np.float32)
        for i, rows in enumerate(t):
            padded_ids[i, :rows*5] = ids[0, :rows].ravel()
            padded_w[i, :rows*5] = w[0, :rows].ravel()
        path = Path(tmp) / "tiny.npz"
        np.savez(path, call=np.arange(len(t))[::-1], T=t[::-1], k=np.full(len(t), 5),
                 ids=padded_ids[::-1], w=padded_w[::-1])
        d = load_census(path)
        assert cadence(d, 3) == (2, 3, 2)
        sel = d["T"] == 3
        loaded = Replay(d["ids"][sel, :15].reshape(-1, 3, 5),
                        d["w"][sel, :15].reshape(-1, 3, 5))
        # float32 stores .08 just BELOW Python's .08; experts 5 and 7 now qualify.
        assert loaded.stats(Policy(.08, count_two=True))["D"] == 2
    np.testing.assert_array_equal(replay.ids, ids)
    np.testing.assert_array_equal(replay.w, w)
    print("SELFTEST PASS: singleton, strict threshold, joint count-two, top-1, mass caps, "
          "D/mass percentiles, immutable inputs, padded NPZ and cadence")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--contribution", action="store_true", help="P3 true-score frontiers and held-out norm-table proxy")
    ap.add_argument("--artifact-dir", type=Path, help="P3 CSV, calibration and proxy artifacts")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.contribution:
        if not args.paths or not args.output or not args.artifact_dir:
            ap.error("--contribution requires explicit eight paths, --output and --artifact-dir")
        from contrib_policy_sim import analyze
        analyze(args.paths, args.artifact_dir, args.output)
        return
    paths = args.paths or sorted((ROOT / "runs").glob("census-*.npz"))
    if not paths:
        ap.error("no census files found (route-census .npz snapshots are not published in this repository)")
    lines = ["# Pruning policy replay", "",
             "CPU-only replay of immutable recorded inputs; no rerouting, output-error estimate, "
             "acceptance measurement or measured speedup. D is the mean distinct experts per "
             "width-matched MoE call. ΔD = reference D minus policy D: positive means fewer experts.", "",
             "Singletons are experts used by exactly one row of the ORIGINAL call. "
             "Count-two adds whole experts used by exactly two rows, BOTH weights < .08. "
             "Thresholds are strict. Top-1 means the largest weight, stable first-slot tie break. "
             "Cap policies select singleton candidates from lowest weight upward within each row; "
             "caps are fractions of original row mass. Top-1 protection is independent unless labeled. "
             "No surviving route is renormalized. Mass percentiles pool all layer/token rows, including "
             "rows with zero dropped mass. Each census is a ring snapshot, not a cumulative histogram; "
             "snapshots are not differenced.", "",
             "**Timing units:** µs/step columns use ΔD × 69.9 µs, matching Astra's arithmetic, "
             "with percentages of 8,919 µs (W4) or 16,083 µs (W16). "
             "The cadence is inferred separately for each file. Width-matched bursts include the "
             "48 target layers plus one draft-extend MoE call; T=1 draft iterations are excluded. "
             "EXCLUSIVE_TIME_MAP_0907.md §4 defines 69.9 = 1.536 µs/call × 45.5 "
             "profiled calls/step, already a STEP coefficient; no further calls/step factor is applied. "
             "These are conditional linear projections, excluding policy overhead and acceptance changes. The census "
             "49-call cadence and the fit's 45.5 profiled-call normalization are distinct; "
             "no attempt is made to recalibrate the fit from route-only data.", ""]
    for path in paths:
        try:
            lines.extend(report_file(path))
            lines.append("")
        except (ValueError, KeyError, OSError) as exc:
            ap.error(f"{path}: {exc}")
    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(paths)} censuses × {len(POLICIES)} policies to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
