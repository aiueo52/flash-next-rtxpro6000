"""support_sweep.py: IS E_bv' tok/step at (s, theta) = (0.7, 0.9) when the draft support is cut to K' ranks.
Run from the repo root on the D1 dump (runs/rs2d/dump, D1-lmstudio.jsonl); output in runs/rs2d/support_sweep.log.
"""
import importlib.util
import glob
import statistics
import torch

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("ao", "bench/rs2d/accept_offline.py")
ao = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ao)
SUPPORTS = (64, 32, 24, 16, 12, 8, 4)
S, THETA = 0.7, 0.9
windows = ao.request_windows(filename="runs/rs2d/D1-lmstudio.jsonl")


def variant(q, top_p, min_p, support):
    out = torch.zeros_like(q)
    for row in range(q.shape[0]):
        for j in range(q.shape[1]):
            w = q[row, j].clone()
            w[support:] = 0
            out[row, j] = ao.sharpen(q=w, scale=S, threshold=THETA, top_p=float(top_p[row]), min_p=float(min_p[row]))
    return out


sums = {}
for path in sorted(glob.glob("runs/rs2d/dump/rs-dump-*.pt")):
    for record in torch.load(path, weights_only=True):
        matches = []
        for row, length in enumerate(record["input_len"]):
            request, reason = ao.join_request(windows=windows, timestamp=record["time"], input_len=int(length))
            if reason is None:
                matches.append((row, request["workload"]))
        if not matches:
            continue
        steps = record["candidates"].shape[1] - 1
        p = record["target_probs"][:, :steps].double()
        pi = record["target_index"][:, :steps].long()
        raw = record["draft_support_probs"][:, :steps].double()
        q = torch.where(raw.isnan(), torch.zeros_like(raw), raw)
        qi = record["draft_support_tokens"][:, :steps].long()
        top_p = record["top_ps"].reshape(q.shape[0], -1)[:, 0]
        min_p = record["min_ps"].reshape(q.shape[0], -1)[:, 0]
        qs = torch.stack([q] + [variant(q, top_p, min_p, k) for k in SUPPORTS], dim=0)
        pd, qd = ao.merge_supports(p=p, pi=pi, q=qs, qi=qi)
        tokens = record["candidates"][:, 1:].unsqueeze(dim=-1)
        pt = (p * (pi == tokens)).sum(dim=-1)
        qt = (qs * (qi == tokens).unsqueeze(dim=0)).sum(dim=-1)
        e_tok, e_bv = ao.path_expectations(pt=pt, qt=qt, p=pd, q=qd)
        weights = torch.where(qt[:1] > 0, qt / qt[:1], 0.0).prod(dim=-1)
        kept = (qs[1:] > 0).sum(dim=-1)
        for row, workload in matches:
            acc = sums.setdefault(workload, dict(n=0, w=[0.0] * len(SUPPORTS), w2=[0.0] * len(SUPPORTS),
                                                 bv=[0.0] * len(SUPPORTS), rs2=0.0, kept=[0] * len(SUPPORTS)))
            acc["n"] += 1
            acc["rs2"] += float(e_bv[0, row])
            for i in range(len(SUPPORTS)):
                w = float(weights[i + 1, row])
                acc["w"][i] += w
                acc["w2"][i] += w * w
                acc["bv"][i] += w * float(e_bv[i + 1, row])
                acc["kept"][i] += int(kept[i, row].sum())
pooled = [[] for _ in SUPPORTS]
for workload in sorted(sums):
    acc = sums[workload]
    print(f"{workload}: verifies={acc['n']} RS2 E_bv tok/step={1 + acc['rs2'] / acc['n']:.6f}")
    for i, k in enumerate(SUPPORTS):
        bv = 1 + acc["bv"][i] / acc["w"][i]
        pooled[i].append(bv)
        ess = acc["w"][i] ** 2 / acc["w2"][i]
        print(f"  K'={k:2d}: IS E_bv' tok/step={bv:.6f}  vs K'=64 {100 * (bv / pooled[0][-1] - 1):+.3f}%  "
              f"ESS={ess:.0f}  kept ranks/position={acc['kept'][i] / acc['n']:.3f} (summed over positions)")
print("pooled (unweighted mean over workloads):")
for i, k in enumerate(SUPPORTS):
    value = statistics.mean(pooled[i])
    print(f"  K'={k:2d}: {value:.6f}  vs K'=64 {100 * (value / statistics.mean(pooled[0]) - 1):+.3f}%")
