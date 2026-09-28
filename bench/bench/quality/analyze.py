#!/usr/bin/env python3
"""Q1 analysis: Wilson CIs, paired McNemar (exact) between arms, markdown table. Usage: analyze.py [--margin-pp 0.5] [base=prod] [arms...]

Only complete arms are analysed: runs/<arm>/DONE (written by run_arm.sh only after the whole arm succeeded) must
exist and, when runs/CURRENT_RUN (written by run_all.sh) or --run-id names a run, carry that run id; and neither
summary.json (incomplete flag, errors) nor any results .jsonl (finish="error") may record a failed request or item,
DONE marker or not (also under --allow-unmarked). Without explicit
arms, incomplete or stale arms are skipped and listed on stderr; an explicitly named incomplete arm (or base) is an error."""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs")
KNOWN_ARMS = ["prod", "noprune", "legacy", "prod2"]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("base", nargs="?", default="prod"); ap.add_argument("arms", nargs="*")
ap.add_argument("--margin-pp", type=float, default=0.5, help="non-inferiority margin in percentage points (default: 0.5)")
ap.add_argument("--run-id", help="accept only DONE markers from this run (default: runs/CURRENT_RUN if present; 'any' accepts any completed run)")
ap.add_argument("--allow-unmarked", action="store_true",
                help="accept arms without a DONE marker (only for callers that verify completeness themselves, e.g. hc1/report.py)")
ap.add_argument("--list-arms", action="store_true", help="print the complete default arms, one per line, and exit")
ap.add_argument("--runs", help="directory holding <arm>/summary.json (default: runs/ next to this script); the published "
                "results are in results/quality-q1 (use with --allow-unmarked: they predate DONE markers)")
args = ap.parse_args()
if args.runs: R = os.path.abspath(args.runs)
if not math.isfinite(args.margin_pp) or args.margin_pp < 0: ap.error("--margin-pp must be finite and nonnegative")
BASE = args.base
def current_run():
    try: return open(os.path.join(R, "CURRENT_RUN")).read().strip() or None
    except OSError: return None
RUN_ID = args.run_id or current_run()
def result_errors(a):
    """Reason if arm a's summary or per-item results record errors or an incomplete run, else None."""
    try: sm = json.load(open(os.path.join(R, a, "summary.json")))
    except (OSError, ValueError): return "unreadable summary.json"
    if sm.get("incomplete"): return "summary.json marks the run incomplete"
    if sm.get("errors"): return f"summary.json records {sm['errors']} errors"
    bad = [b for b, v in sm.get("bench", {}).items() if isinstance(v, dict) and v.get("errors")]
    if bad: return "errors recorded in " + ", ".join(bad)
    for name in os.listdir(os.path.join(R, a)):
        if not name.endswith(".jsonl"): continue
        try:
            with open(os.path.join(R, a, name)) as f:
                if any(json.loads(l).get("finish") == "error" for l in f if l.strip()): return f"error items in {name}"
        except (OSError, ValueError): return f"unreadable {name}"
    return None
def incomplete(a):
    """None if arm a is complete for RUN_ID, else the reason it is not."""
    if not os.path.exists(os.path.join(R, a, "summary.json")): return "no summary.json"
    why = result_errors(a)
    if why: return why
    if args.allow_unmarked: return None
    try: m = json.load(open(os.path.join(R, a, "DONE")))
    except (OSError, ValueError): return "no DONE marker (arm failed, was interrupted or is still running)"
    if RUN_ID not in (None, "any") and m.get("run_id") != RUN_ID:
        return f"stale: DONE marker is from run {m.get('run_id')}, not {RUN_ID}"
    return None
def default_arms():
    arms = [a for a in KNOWN_ARMS if not incomplete(a)]
    for a in KNOWN_ARMS:
        if a not in arms and os.path.exists(os.path.join(R, a)):
            print(f"analyze.py: skipping arm {a}: {incomplete(a)}", file=sys.stderr)
    return arms
if args.list_arms:
    print("\n".join(default_arms())); sys.exit(0)
bad = {a: incomplete(a) for a in dict.fromkeys([BASE] + args.arms) if incomplete(a)}
if bad:
    sys.exit("analyze.py: refusing incomplete arms (base and named arms must be complete):\n"
             + "\n".join(f"  {a}: {why}" for a, why in bad.items())
             + ("\n(the raw runs/ are not published in this repository; the published Q1 results are in"
                " results/quality-q1: use --runs results/quality-q1 --allow-unmarked)" if not args.runs else ""))
ARMS = args.arms or default_arms()
BENCH = ["gsm8k", "mmlu", "humaneval", "jcqa"]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def rows(arm, b):
    """Per-item records of arm/b by id, or None if the arm has neither the file nor a summary entry.
    Refuses a record without an id, a duplicate id, a file without a summary entry (or the reverse),
    and item/correct counts that do not match summary.json."""
    p = os.path.join(R, arm, b + ".jsonl"); s = summ[arm]["bench"].get(b)
    if not os.path.exists(p):
        if s: sys.exit(f"analyze.py: {arm}/{b}: summary.json has {s.get('n')} items but {b}.jsonl is missing")
        return None
    if not s: sys.exit(f"analyze.py: {arm}/{b}.jsonl exists but summary.json has no {b} entry")
    out = {}
    with open(p) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
            r = json.loads(line)
            if "id" not in r: sys.exit(f"analyze.py: {p}:{ln}: record without an id")
            if r["id"] in out: sys.exit(f"analyze.py: {p}:{ln}: duplicate id {r['id']!r}")
            out[r["id"]] = r
    k = sum(bool(r["correct"]) for r in out.values())
    if len(out) != s["n"] or k != s["correct"]:
        sys.exit(f"analyze.py: {arm}/{b}: {len(out)} items / {k} correct in {b}.jsonl, "
                 f"but summary.json says {s['n']} / {s['correct']}")
    # Every other per-benchmark field that is displayed or reused must agree with the items too.
    if "acc" in s and not (isinstance(s["acc"], (int, float)) and abs(s["acc"] - k / len(out)) <= 1e-9):
        sys.exit(f"analyze.py: {arm}/{b}: summary.json acc {s['acc']!r} does not match {k}/{len(out)} from {b}.jsonl")
    # A summary field is verified only if every item carries what it is derived from; otherwise refuse.
    for field, need in (("truncated", "finish"), ("mean_gen_tokens", "gen_tokens")):
        missing = [r["id"] for r in out.values() if need not in r]
        if field in s and missing:
            sys.exit(f"analyze.py: {arm}/{b}: summary.json has {field} but {len(missing)} item(s) in {b}.jsonl "
                     f"lack {need} (first id {missing[0]!r}), so it cannot be verified")
    tr = sum(r.get("finish") == "length" for r in out.values())
    if "truncated" in s and s["truncated"] != tr:
        sys.exit(f"analyze.py: {arm}/{b}: summary.json truncated {s['truncated']!r} but {b}.jsonl has {tr}")
    if "mean_gen_tokens" in s and out:
        mt = sum(r["gen_tokens"] for r in out.values()) / len(out)
        if not (isinstance(s["mean_gen_tokens"], (int, float)) and abs(s["mean_gen_tokens"] - mt) <= 0.05 + 1e-9):
            sys.exit(f"analyze.py: {arm}/{b}: summary.json mean_gen_tokens {s['mean_gen_tokens']!r} "
                     f"but {b}.jsonl gives {mt:.2f}")
    return out
def mean_tokens(arm, b):
    """Mean gen_tokens from the verified items, or '-' when the benchmark is absent or items lack gen_tokens."""
    r = rows(arm, b)
    if not r or not all("gen_tokens" in x for x in r.values()): return "-"
    return f"{sum(x['gen_tokens'] for x in r.values()) / len(r):.1f}"
def paired(a, b):
    """(base records, arm records) with identical id sets, None if the benchmark is in neither arm;
    refuses a benchmark present on one side only or two different item sets."""
    A, B = rows(BASE, b), rows(a, b)
    if A is None and B is None: return None
    if A is None or B is None: sys.exit(f"analyze.py: {b}: results for {BASE if A is None else a} are missing; refusing a one-sided comparison")
    if set(A) != set(B):
        sys.exit(f"analyze.py: {b}: {BASE} and {a} cover different items ({len(set(A) - set(B))} only in {BASE}, "
                 f"{len(set(B) - set(A))} only in {a}); refusing a paired comparison")
    if not A: sys.exit(f"analyze.py: {b}: no items in {BASE} or {a}")
    return A, B
def same_output(a, b):
    """Whether two records produced the same output: compares output_sha256 (written by run_bench.py) or,
    for KEEP_TEXT debug runs, output; None when neither record carries one (e.g. the published results)."""
    for key in ("output_sha256", "output"):
        if key in a and key in b: return a[key] == b[key]
    return None
def tango_bounds(b, c, n, z):
    """Tango (1998) score interval for a paired difference theta = (b - c) / n, where b and c are the two
    discordant counts. Valid for small n and few (even zero) discordances, unlike the Wald interval.
    Returns (lower, upper): the theta values where the score statistic equals +z and -z."""
    def score(t):
        w = b + c - t * (2 * n - b + c)
        q = (w + math.sqrt(max(w * w + 8 * n * c * t * (1 - t), 0.0))) / (4 * n)  # constrained MLE of p(c-cell)
        var = n * (2 * q + t - t * t)
        num = b - c - n * t
        if var <= 0: return 0.0 if num == 0 else math.copysign(math.inf, num)
        return num / math.sqrt(var)
    def solve(target, lo, hi):  # score is decreasing in t
        for _ in range(200):
            mid = (lo + hi) / 2
            if score(mid) > target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    est = (b - c) / n; eps = 1e-12
    lower = -1.0 if score(-1 + eps) <= z else solve(z, -1 + eps, est)
    upper = 1.0 if score(1 - eps) >= -z else solve(-z, est, 1 - eps)
    return lower, upper
def exact_p(x, y):
    # Two-sided Binomial(x+y, 0.5), using symmetry; no SciPy dependency.
    return min(1.0, 2 * sum(math.comb(x + y, j) for j in range(min(x, y) + 1)) / (1 << (x + y)))
def mcnemar(a, b):
    ids = sorted(set(a) & set(b)); x = sum(a[i]["correct"] and not b[i]["correct"] for i in ids); y = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
    p = exact_p(x, y)
    return x, y, p, len(ids)
out = []; summ = {a: json.load(open(os.path.join(R, a, "summary.json"))) for a in dict.fromkeys([BASE] + ARMS)}
out.append("| benchmark | " + " | ".join(ARMS) + " |"); out.append("|---|" + "---|" * len(ARMS))
for b in BENCH:
    cells = []
    for a in ARMS:
        r = rows(a, b)  # verified against summary.json; the displayed figures come from the items
        if r is None: cells.append("-"); continue
        k, n = sum(bool(x["correct"]) for x in r.values()), len(r)
        if n == 0: cells.append("no items"); continue
        lo, hi = wilson(k, n); cells.append(f"{100*k/n:.2f}% [{100*lo:.1f}, {100*hi:.1f}] ({k}/{n})")
    out.append(f"| {b} | " + " | ".join(cells) + " |")
out.append("| wall s, summary.json timing (gsm8k/mmlu/he/jcqa) | " + " | ".join("/".join(str(summ[a]["bench"][b]["wall_s"]) if b in summ[a]["bench"] else "-" for b in BENCH) for a in ARMS) + " |")
out.append("| mean gen tokens | " + " | ".join("/".join(mean_tokens(a, b) for b in BENCH) for a in ARMS) + " |")
out.append(""); out.append(f"Paired vs {BASE} (McNemar exact, two-sided; flips = base-right/other-wrong : base-wrong/other-right):"); out.append("")
out.append("| benchmark | arm | diff (pp) | diff 95% CI (pp) | flips +/- | p | n |"); out.append("|---|---|---|---|---|---|---|")
verdict = []; noninferiority = []; verdict_pairs = 0
for a in ARMS:
    if a == BASE: continue
    for b in BENCH:
        P = paired(a, b)
        if P is None: continue
        A, B = P
        x, y, p, n = mcnemar(A, B)
        diff = (y - x) / n * 100
        # 95% CI of the paired difference (other minus BASE): Tango score interval
        tl, th = tango_bounds(y, x, n, 1.959963984540054)
        out.append(f"| {b} | {a} | {diff:+.2f} | [{100*tl:+.2f}, {100*th:+.2f}] | {x}/{y} | {p:.3f} | {n} |")
        # diff is other minus BASE: positive significant differences mean BASE degraded.
        if a in ("noprune", "legacy"): verdict_pairs += 1
        if a in ("noprune", "legacy") and p < 0.05 and diff > 0: verdict.append(f"{b}: {BASE} vs {a} {diff:+.2f} pp (p={p:.3f})")
        lo, hi = (100 * v for v in tango_bounds(x, y, n, 1.6448536269514722))  # BASE minus other, one-sided 95% each
        status = "PASS" if lo > -args.margin_pp else "FAIL" if hi < -args.margin_pp else "INCONCLUSIVE"
        noninferiority.append(f"| {b} | {a} | {-diff:+.2f} | {lo:+.2f} | {hi:+.2f} | {args.margin_pp:g} | {status} | {n} |")
out.append("")
out.append("GSM8K restricted to questions that finished (finish_reason=stop) in BOTH arms (removes the 512-token-cap coin flips):"); out.append("")
out.append("| pair | n both finished | base acc | other acc | diff (pp) | flips +/- | p |"); out.append("|---|---|---|---|---|---|---|")
for a in ARMS:
    if a == BASE: continue
    P = paired(a, "gsm8k")
    if P is None: continue
    A, B = P
    ids = [i for i in A if A[i]["finish"] == "stop" and B[i]["finish"] == "stop"]
    ka = sum(A[i]["correct"] for i in ids); kb = sum(B[i]["correct"] for i in ids)
    x = sum(A[i]["correct"] and not B[i]["correct"] for i in ids); y = sum(B[i]["correct"] and not A[i]["correct"] for i in ids)
    p = exact_p(x, y)
    if not ids:
        out.append(f"| {BASE} vs {a} | 0 | not measured | not measured | - | - | - |"); continue
    out.append(f"| {BASE} vs {a} | {len(ids)} | {100*ka/len(ids):.2f}% | {100*kb/len(ids):.2f}% | {100*(kb-ka)/len(ids):+.2f} | {x}/{y} | {p:.3f} |")
out.append(""); out.append("Output identity vs " + BASE + " (greedy; differences come from the non-deterministic MoE finalize and, for prod, pruning):"); out.append("")
out.append("| arm | " + " | ".join(f"{b} identical" for b in BENCH) + " | gsm8k truncated (base/other) |"); out.append("|---|" + "---|" * (len(BENCH) + 1))
for a in ARMS:
    if a == BASE: continue
    cells = []
    for b in BENCH:
        P = paired(a, b)
        if P is None: cells.append("-"); continue
        A, B = P
        same = [same_output(A[i], B[i]) for i in A]
        cells.append("not measured" if None in same else f"{sum(same)}/{len(A)}")
    P = paired(a, "gsm8k")
    tr = f"{sum(r['finish']=='length' for r in P[0].values())}/{sum(r['finish']=='length' for r in P[1].values())}" if P else "-"
    out.append(f"| {a} | " + " | ".join(cells) + f" | {tr} |")
out.append("")
SANITY_KEYS = ("gen_tokens", "finish", "maxrun", "uniq_words", "rep_4gram")
def sanity_items(a):
    """summary.json sanity items checked against sanity.jsonl; None (not shown) when sanity.jsonl is absent."""
    items = summ[a]["bench"].get("sanity", {}).get("items", {})
    p = os.path.join(R, a, "sanity.jsonl")
    if not os.path.exists(p): return None if items else {}
    rs = [json.loads(l) for l in open(p) if l.strip()]
    got = {str(r["id"]): {k: r.get(k) for k in SANITY_KEYS} for r in rs}
    if len(got) != len(rs) or got != {str(k): {f: v.get(f) for f in SANITY_KEYS} for k, v in items.items()}:
        sys.exit(f"analyze.py: {a}: summary.json sanity items do not match sanity.jsonl")
    return items
SAN = {a: sanity_items(a) for a in ARMS}
unverified = [a for a in ARMS if SAN[a] is None]
if unverified:
    out.append("Sanity rows not shown for " + ", ".join(unverified) + ": no sanity.jsonl to check summary.json against.")
    out.append("")
out.append("| sanity | " + " | ".join(ARMS) + " |"); out.append("|---|" + "---|" * len(ARMS))
items = sorted({k for a in ARMS for k in (SAN[a] or {})})
for it in items:
    out.append(f"| {it} tokens/maxrun/uniq/rep4 | " + " | ".join(
        (lambda s: f"{s['gen_tokens']}/{s['maxrun']}/{s['uniq_words']}/{s['rep_4gram']}" if s else "-")((SAN[a] or {}).get(it)) for a in ARMS) + " |")
out.append(""); out.append("Needle 18.5k (pre-run / post-run, depths 0.1/0.5/0.9):"); out.append("")
for a in ARMS:
    p = os.path.join(R, a, "needle.log")
    if os.path.exists(p):
        ls = [l.strip() for l in open(p) if "PASS=" in l]
        out.append(f"- {a}: " + ", ".join(("PASS" if "PASS=True" in l else "FAIL") + f"({l.split('time=')[1].split('s')[0]}s)" for l in ls))
# No paired comparison against a reference arm means nothing was tested: say so instead of "no degradation".
out.append(""); out.append("VERDICT: " + ("not measured (no paired comparison of " + BASE + " against noprune or legacy)" if not verdict_pairs
                                          else "no significant degradation" if not verdict else "significant degradation in " + "; ".join(verdict)))
out.extend(["", f"Non-inferiority of {BASE} vs each other arm (paired BASE minus other; Tango score bounds, z=1.644854).",
            "Lower and upper bounds are each one-sided 95% bounds (not a two-sided 95% interval). PASS: lower > -margin; FAIL: upper < -margin; otherwise INCONCLUSIVE.",
            "Tango score bounds stay valid with few or zero discordant pairs (a Wald interval collapses there). The significance verdict above is not a non-inferiority verdict.", "",
            "| benchmark | arm | BASE - other (pp) | lower 95% (pp) | upper 95% (pp) | margin (pp) | status | n |",
            "|---|---|---|---|---|---|---|---|", *(noninferiority or ["| (none) | - | - | - | - | - | not measured | 0 |"])])
print("\n".join(out))
