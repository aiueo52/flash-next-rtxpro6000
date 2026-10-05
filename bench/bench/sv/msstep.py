"""(python bench/sv/msstep.py runs/sv lmstudio|greedy)
ABBA on the BN1 grid rows: ms/step = decode_ms / (completion_tokens / tokens_per_verify), per workload,
B/A log ratio averaged over the two blocks per prompt, bootstrap over prompts plus a t interval; pooled."""
import json, math, random, statistics, sys
d, s = sys.argv[1], sys.argv[2]
arms = ("A1", "B1", "B2", "A2")
data = {}
for a in arms:
    for line in open(f"{d}/{a}-{s}.jsonl"):
        r = json.loads(line)
        acc = r["server"]["acceptance"]["tokens_per_verify"]
        n = r["client"]["usage"]["completion_tokens"]
        ms_tok = r["client"]["decode_seconds"] * 1000 / n
        data.setdefault((r["workload"], r["prompt_id"]), {})[a] = {"ms_step": ms_tok * acc, "acc": acc, "ms_tok": ms_tok}
wls = sorted({k[0] for k in data})
def lr(m, pid):
    c = data[pid]
    b1 = math.log(c["B1"][m] / c["A1"][m]); b2 = math.log(c["B2"][m] / c["A2"][m])
    return (b1 + b2) / 2, b1, b2
random.seed(1)
for m in ("ms_step", "acc", "ms_tok"):
    print(f"== {s} {m}  (B/A, mean of the two blocks per prompt)")
    pooled = []
    for w in wls:
        pids = sorted(k for k in data if k[0] == w)
        x = [lr(m, p)[0] for p in pids]
        bl1 = statistics.mean(lr(m, p)[1] for p in pids); bl2 = statistics.mean(lr(m, p)[2] for p in pids)
        mu = statistics.mean(x); sd = statistics.stdev(x); t = 2.365 * sd / math.sqrt(len(x))
        bs = sorted(statistics.mean(random.choices(x, k=len(x))) for _ in range(4000))
        drift = statistics.mean(math.log(data[p]["A2"][m] / data[p]["A1"][m]) for p in pids)
        pooled.append(x)
        print(f"  {w:11s} {100*(math.exp(mu)-1):+6.2f}%  boot[{100*(math.exp(bs[100])-1):+6.2f} .. {100*(math.exp(bs[3899])-1):+6.2f}]"
              f"  t[{100*(math.exp(mu-t)-1):+6.2f} .. {100*(math.exp(mu+t)-1):+6.2f}]  blocks {100*(math.exp(bl1)-1):+5.1f} / {100*(math.exp(bl2)-1):+5.1f}"
              f"  A2/A1 {100*(math.exp(drift)-1):+5.1f}%")
    pm = [statistics.mean(statistics.mean(random.choices(x, k=len(x))) for x in pooled) for _ in range(4000)]
    pm.sort(); mu = statistics.mean(statistics.mean(x) for x in pooled)
    print(f"  {'pooled':11s} {100*(math.exp(mu)-1):+6.2f}%  boot[{100*(math.exp(pm[100])-1):+6.2f} .. {100*(math.exp(pm[3899])-1):+6.2f}]")
