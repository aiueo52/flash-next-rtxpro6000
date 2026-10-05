"""(run from ~/tools/flash-next-bench: python bench/sv/pool_acc.py)
Acceptance of the SV1 sparse arms against every dense, min_p-off lmstudio run of the day.
Per run and workload: mean over prompts of log(tokens_per_verify). Exact permutation test: which
2 of the runs are labelled sparse (all C(n,2) labellings), statistic = sparse mean - dense mean."""
import itertools, json, math, statistics
RUNS = {"rs1/A1": "dense", "rs1/A2": "dense", "rs1-minp/A1-minp": "dense", "rs1-minp/A2-minp": "dense",
        "sv/A1": "dense", "sv/A2": "dense", "sv/B1": "sparse", "sv/B2": "sparse"}
acc = {}
for run in RUNS:
    for line in open(f"runs/{run}-lmstudio.jsonl"):
        r = json.loads(line)
        acc.setdefault(run, {})[(r["workload"], r["prompt_id"])] = math.log(r["server"]["acceptance"]["tokens_per_verify"])
wls = sorted({k[0] for k in acc["sv/A1"]})
names = list(RUNS)
def per_run(w):
    return {n: statistics.mean(v for k, v in acc[n].items() if k[0] == w) for n in names}
rows = {w: per_run(w) for w in wls}
rows["pooled"] = {n: statistics.mean(rows[w][n] for w in wls) for n in names}
for w, m in rows.items():
    stat = lambda sp: statistics.mean(m[n] for n in sp) - statistics.mean(m[n] for n in names if n not in sp)
    actual = stat([n for n in names if RUNS[n] == "sparse"])
    perms = [stat(c) for c in itertools.combinations(names, 2)]
    p = sum(1 for s in perms if abs(s) >= abs(actual) - 1e-12) / len(perms)
    print(f"{w:11s} sparse-dense {100*(math.exp(actual)-1):+5.1f}%  two-sided p={p:.2f}  runs: " +
          " ".join(f"{n.split('/')[-1][:5]}:{math.exp(v):.2f}" for n, v in m.items()))
