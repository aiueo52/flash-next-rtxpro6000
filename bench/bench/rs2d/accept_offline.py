"""Estimate sparse-RS acceptance on dumped prefixes (CPU only)."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

import torch

SCALES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
GRID = tuple((s, theta) for s in SCALES for theta in (0.0, *THRESHOLDS))
PRODUCT_GRID = tuple((s, theta) for s in SCALES for theta in THRESHOLDS)


def sharpen(*, q, scale, threshold, top_p, min_p):
    assert scale > 0 and 0 <= threshold <= 1
    w = torch.where(condition=q.isnan(), input=torch.zeros_like(input=q, dtype=torch.float64), other=q.double())
    if scale != 1.0:
        positive = w > 0
        # Log weights preserve tiny kept probabilities for small scales.
        logs = torch.where(condition=positive, input=w.log(), other=-float("inf")) / scale
        w = torch.exp(input=logs - logs.max())
        w /= w.sum()
        exclusive = w.cumsum(dim=0) - w
        ranks = torch.arange(end=w.numel())
        w *= ((exclusive < top_p) | (ranks == 0)) & (w >= min_p * w[0])
        w /= w.sum()
    if threshold > 0 and w.max() >= threshold:
        w.zero_()
        w[0] = 1.0
    return w


def acceptance(*, p, pi, q, qi):
    # Sum duplicate ids on both supports before the overlap calculation.
    pd, qd = {}, {}
    for token, mass in zip(pi.tolist(), p.tolist()):
        pd[token] = pd.get(token, 0.0) + mass
    for token, mass in zip(qi.tolist(), q.tolist()):
        mass = 0.0 if math.isnan(mass) else mass
        qd[token] = qd.get(token, 0.0) + mass
    return sum(min(mass, pd.get(token, 0.0)) for token, mass in qd.items())


def expected_drafts(*, acceptances):
    product, total = 1.0, 0.0
    for value in acceptances:
        product *= value
        total += product
    return total


def sharpen_grid(*, q, grid, top_p, min_p):
    if not grid:
        return q.new_empty(size=(0, *q.shape))
    scales = tuple(dict.fromkeys(scale for scale, threshold in grid))
    assert all(scale > 0 and 0 <= threshold <= 1 for scale, threshold in grid)
    scale = torch.tensor(data=scales, dtype=torch.float64).reshape(-1, 1, 1, 1)
    logs = q.log().unsqueeze(dim=0) / scale
    w = torch.exp(input=logs - logs.amax(dim=-1, keepdim=True))
    w /= w.sum(dim=-1, keepdim=True)
    exclusive = w.cumsum(dim=-1) - w
    ranks = torch.arange(end=q.shape[-1])
    w *= ((exclusive < top_p) | (ranks == 0)) & (w >= min_p * w[..., :1])
    w /= w.sum(dim=-1, keepdim=True)
    w = torch.where(condition=scale == 1.0, input=q.unsqueeze(dim=0), other=w)
    w = w[[scales.index(scale) for scale, threshold in grid]]
    thresholds = torch.tensor(data=[threshold for scale, threshold in grid], dtype=torch.float64).reshape(-1, 1, 1, 1)
    onehot = (ranks == 0).double()
    return torch.where(condition=(thresholds > 0) & (w.amax(dim=-1, keepdim=True) >= thresholds),
                       input=onehot, other=w)


def merge_supports(*, p, pi, q, qi):
    # Position-local union ids: duplicates must be merged before min or residual clipping.
    ids, order = torch.sort(input=torch.cat(tensors=(pi, qi), dim=-1), dim=-1)
    starts = torch.cat(tensors=(torch.ones_like(input=ids[..., :1], dtype=torch.bool),
                               ids[..., 1:] != ids[..., :-1]), dim=-1)
    groups = torch.empty_like(input=order)
    groups.scatter_(dim=-1, index=order, src=starts.long().cumsum(dim=-1) - 1)
    pd = p.new_zeros(size=ids.shape)
    pd.scatter_add_(dim=-1, index=groups[..., :p.shape[-1]], src=p)
    qd = q.new_zeros(size=(*q.shape[:-1], ids.shape[-1]))
    qd.scatter_add_(dim=-1, index=groups[..., p.shape[-1]:].unsqueeze(dim=0).expand_as(other=q), src=q)
    return pd, qd


def path_expectations(*, pt, qt, p, q):
    alpha = torch.where(condition=pt <= 0, input=torch.zeros_like(input=pt),
                        other=torch.where(condition=qt == 0, input=torch.ones_like(input=qt), other=(pt / qt).clamp(max=1)))
    e_tok = alpha.cumprod(dim=-1).sum(dim=-1)
    previous = torch.ones_like(input=qt[..., 0])
    pis = []
    for position in range(qt.shape[-1]):
        previous = torch.where(condition=pt[..., position] <= 0, input=torch.zeros_like(input=previous),
                               other=torch.where(condition=qt[..., position] == 0, input=torch.ones_like(input=previous),
                                                 other=(previous * pt[..., position] / qt[..., position]).clamp(max=1)))
        pis.append(previous)
    pi = torch.stack(tensors=pis, dim=-1)
    n = (pi[..., :-1].unsqueeze(dim=-1) * p[:, 1:] - q[..., 1:, :]).clamp(min=0).sum(dim=-1)
    denominator = n + 1 - pi[..., :-1]
    h = torch.cat(tensors=(torch.where(condition=denominator <= 0, input=torch.ones_like(input=denominator), other=n / denominator),
                           pi[..., -1:]), dim=-1)
    suffix = (1 - h).flip(dims=(-1,)).cumprod(dim=-1).flip(dims=(-1,))
    later = torch.cat(tensors=(suffix[..., 1:], torch.ones_like(input=suffix[..., :1])), dim=-1)
    positions = torch.arange(start=1, end=qt.shape[-1] + 1)
    return e_tok, (positions * h * later).sum(dim=-1)


def record_metrics(*, record, grid=GRID, q_prime=None):
    steps = record["candidates"].shape[1] - 1
    p = record["target_probs"][:, :steps].double()
    pi = record["target_index"][:, :steps].long()
    raw_q = record["draft_support_probs"][:, :steps].double()
    q = torch.where(condition=raw_q.isnan(), input=torch.zeros_like(input=raw_q), other=raw_q)
    qi = record["draft_support_tokens"][:, :steps].long()
    top_p = record["top_ps"].reshape(q.shape[0], -1)[:, :1].reshape(1, -1, 1, 1)
    min_p = record["min_ps"].reshape(q.shape[0], -1)[:, :1].reshape(1, -1, 1, 1)
    alternatives = sharpen_grid(q=q, grid=grid, top_p=top_p, min_p=min_p)
    variants = [q.unsqueeze(dim=0), alternatives]
    if q_prime is not None:
        variants.append(q_prime[:, :steps].double().unsqueeze(dim=0))
    qs = torch.cat(tensors=variants, dim=0)
    pd, qd = merge_supports(p=p, pi=pi, q=qs, qi=qi)
    overlaps = torch.minimum(input=pd.unsqueeze(dim=0), other=qd).sum(dim=-1)
    expected = overlaps.cumprod(dim=-1).sum(dim=-1).T.tolist()
    tokens = record["candidates"][:, 1:].unsqueeze(dim=-1)
    pt = (p * (pi == tokens)).sum(dim=-1)
    qt = (qs * (qi == tokens).unsqueeze(dim=0)).sum(dim=-1)
    e_tok, e_bv = path_expectations(pt=pt, qt=qt, p=pd, q=qd)
    weights = torch.where(condition=qt[:1] > 0, input=qt / qt[:1], other=0.0).prod(dim=-1).T.tolist()
    e_tok, e_bv = e_tok.T.tolist(), e_bv.T.tolist()
    target = (p * (pi == qi[..., :1])).sum(dim=-1)
    target_e = target.cumprod(dim=-1).sum(dim=-1).tolist()
    a_rs, a_target = overlaps[0].tolist(), target.tolist()
    result = []
    for row, realised in enumerate(record["accept_len"].tolist()):
        metrics = dict(realised=float(realised), rs=expected[row][0], target_only=target_e[row],
                       grid=dict(zip(grid, expected[row][1:])), a_rs=a_rs[row], a_target_only=a_target[row],
                       E_tok=e_tok[row][0], E_bv=e_bv[row][0],
                       is_grid={pair: (weights[row][i], e_tok[row][i], e_bv[row][i])
                                for i, pair in enumerate(grid, start=1)})
        if q_prime is not None:
            metrics.update(w=weights[row][-1], E_tok_prime=e_tok[row][-1], E_bv_prime=e_bv[row][-1])
        result.append(metrics)
    return result


def row_metrics(*, record, row, grid=GRID, q_prime=None):
    fields = ("candidates", "target_probs", "target_index", "draft_support_probs", "draft_support_tokens",
              "top_ps", "min_ps", "accept_len")
    single = {name: record[name][row:row + 1] for name in fields}
    return record_metrics(record=single, grid=grid,
                          q_prime=None if q_prime is None else q_prime.unsqueeze(dim=0))[0]


def request_windows(*, filename):
    windows = []
    for line in Path(filename).read_text().splitlines():
        row = json.loads(line)
        client = row.get("client", {})
        ttft, decode = client.get("ttft_seconds"), client.get("decode_seconds")
        if ttft is None or decode is None:
            continue
        start = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
        windows.append(dict(start=start, end=start + ttft + decode + 2.0,
                            input_len=int(client["usage"]["prompt_tokens"]),
                            workload=row["workload"], prompt_id=row["prompt_id"]))
    return sorted(windows, key=lambda window: window["start"], reverse=True)


def join_request(*, windows, timestamp, input_len):
    # Padding can overlap the next serial request; the latest start owns that time.
    for window in windows:
        if window["start"] <= timestamp <= window["end"]:
            return (window, None) if input_len == window["input_len"] else (None, "length_mismatch")
    return None, "no_window"


def standard_error(*, values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan


def bv_summary(*, rows):
    tok_difference = [row["realised"] - row["E_tok"] for row in rows]
    bv_difference = [row["E_bv"] - row["E_tok"] for row in rows]
    return dict(realised=statistics.mean(row["realised"] for row in rows),
                E_tok=statistics.mean(row["E_tok"] for row in rows),
                E_bv=statistics.mean(row["E_bv"] for row in rows),
                sanity=statistics.mean(tok_difference), sanity_se=standard_error(values=tok_difference),
                gain=statistics.mean(bv_difference), gain_se=standard_error(values=bv_difference))


def print_bv(*, summary):
    denominator = 1 + summary["E_tok"]
    print(f"  path estimates at RS2 q: realised tok/step={1 + summary['realised']:.6f}; "
          f"E_tok tok/step={denominator:.6f}; E_bv tok/step={1 + summary['E_bv']:.6f}")
    print(f"  BV gain over E_tok={100 * summary['gain'] / denominator:+.3f}%; "
          f"paired SE={100 * summary['gain_se'] / denominator:.3f}% (relative to E_tok tok/step)")
    print(f"  sanity mean E_tok={summary['E_tok']:.6f}; mean accept_len={summary['realised']:.6f}; "
          f"difference={summary['sanity']:.6f} paired SE={summary['sanity_se']:.6f}")


def is_means(*, rows):
    samples = torch.tensor(data=[[row["is_grid"][pair] for pair in GRID] for row in rows], dtype=torch.float64)
    weights = samples[..., 0]
    total = weights.sum(dim=0)
    means = (weights.unsqueeze(dim=-1) * samples[..., 1:]).sum(dim=0) / total.unsqueeze(dim=-1)
    ess = torch.where(condition=total > 0, input=total.square() / weights.square().sum(dim=0), other=0.0)
    return {pair: (1 + tok, 1 + bv, effective)
            for pair, (tok, bv), effective in zip(GRID, means.tolist(), ess.tolist())}


def print_is(*, values, selections):
    print("  IS (self-normalised per workload):")
    for label, pair in selections:
        tok, bv, ess = values[pair]
        print(f"    {label} (s, theta)={pair}: IS E_tok' tok/step={tok:.6f}; "
              f"IS E_bv' tok/step={bv:.6f}; ESS={ess:.3f}")


def report(*, directory, fnbench, csv_path=None):
    started = time.perf_counter()
    windows = request_windows(filename=fnbench)
    exclusions = {"no_window": 0, "length_mismatch": 0}
    workloads = {}
    paths = sorted(Path(directory).glob("rs-dump-*.pt"))
    if not paths:
        raise ValueError(f"No rs-dump-*.pt chunks in {directory}")
    for path in paths:
        records = torch.load(f=path, map_location="cpu", weights_only=True)
        for record in records:
            matches = []
            for row, length in enumerate(record["input_len"]):
                request, reason = join_request(windows=windows, timestamp=record["time"], input_len=int(length))
                if reason is not None:
                    exclusions[reason] += 1
                    continue
                matches.append((row, request))
            if not matches:
                continue
            batch = record_metrics(record=record)
            for row, request in matches:
                metrics = batch[row]
                metrics["prompt_id"] = request["prompt_id"]
                workloads.setdefault(request["workload"], []).append(metrics)
    print(f"Excluded dump rows: no_window={exclusions['no_window']}; length_mismatch={exclusions['length_mismatch']}")
    if not workloads:
        raise ValueError("Dump contains no matched verify rows")
    print("Caveat: positions j >= 1 use prefixes drawn from RS2's q, not from the alternative q.")
    print("Approximation: s < 1 only sees kept q support; missing tail can cause slightly more top-p cuts.")
    print("IS reweights the complete draft path; zero total weight gives undefined estimates and ESS=0.")
    csv_rows, workload_means, target_means = [], [], []
    summaries = {name: bv_summary(rows=rows) for name, rows in workloads.items()}
    importance = {name: is_means(rows=rows) for name, rows in workloads.items()}
    pooled_is = {pair: (statistics.mean(values[pair][0] for values in importance.values()),
                        statistics.mean(values[pair][1] for values in importance.values()),
                        sum(values[pair][2] for values in importance.values())) for pair in GRID}
    grid_means = {name: {pair: statistics.mean(row["grid"][pair] for row in rows) for pair in GRID}
                  for name, rows in workloads.items()}
    pooled_grid = {pair: statistics.mean(values[pair] for values in grid_means.values()) for pair in GRID}
    best_product = max(PRODUCT_GRID, key=pooled_grid.__getitem__)
    best_bv = max((pair for pair in GRID if math.isfinite(pooled_is[pair][1])),
                  key=lambda pair: pooled_is[pair][1])
    selections = (("RS2", (1.0, 0.0)), ("best pooled E", best_product), ("best pooled IS E_bv'", best_bv))
    for workload, rows in sorted(workloads.items()):
        n = len(rows)
        realised = [row["realised"] for row in rows]
        expected = [row["rs"] for row in rows]
        differences = [actual - theory for actual, theory in zip(realised, expected)]
        means = grid_means[workload]
        workload_means.append({pair: 1 + mean for pair, mean in means.items()})
        target_means.append(1 + statistics.mean(row["target_only"] for row in rows))
        prompt_counts = {}
        for row in rows:
            prompt_counts[row["prompt_id"]] = prompt_counts.get(row["prompt_id"], 0) + 1
        best = max(PRODUCT_GRID, key=means.__getitem__)
        print(f"{workload}: verifies={n} realised tok/step={1 + statistics.mean(realised):.6f}; "
              f"RS2={1 + statistics.mean(expected):.6f}; "
              f"target-only={1 + statistics.mean(row['target_only'] for row in rows):.6f}")
        print("  verifies per prompt_id: " + ", ".join(f"{prompt_id}={count}" for prompt_id, count in sorted(prompt_counts.items())))
        print(f"  sanity mean E={statistics.mean(expected):.6f} SE={standard_error(values=expected):.6f}; "
              f"mean accept_len={statistics.mean(realised):.6f} SE={standard_error(values=realised):.6f}; "
              f"difference={statistics.mean(differences):.6f} paired SE={standard_error(values=differences):.6f}")
        print_bv(summary=summaries[workload])
        print_is(values=importance[workload], selections=selections)
        for label, pairs in (("s grid", [(s, 0.0) for s in SCALES]),
                             ("theta grid (s=1)", [(1.0, t) for t in THRESHOLDS])):
            print("  " + label + ": " + ", ".join(f"{pair}: {1 + means[pair]:.6f}" for pair in pairs))
        print(f"  best (s, theta)={best}: tok/step={1 + means[best]:.6f}")
        for key in ("a_rs", "a_target_only"):
            values = []
            for position in range(8):
                present = [row[key][position] for row in rows if len(row[key]) > position]
                values.append(f"j{position}={statistics.mean(present):.6f} (n={len(present)})" if present else f"j{position}=n/a")
            print(f"  {key}: " + ", ".join(values))
        csv_rows.extend({"workload": workload, "verifies": n, "temp_scale": s,
                         "onehot_above": t, "tok_per_step": 1 + means[(s, t)],
                         "tok_per_step_is_tok": importance[workload][(s, t)][0],
                         "tok_per_step_is_bv": importance[workload][(s, t)][1],
                         "ess": importance[workload][(s, t)][2]}
                        for s, t in GRID)
    pooled = {pair: statistics.mean(means[pair] for means in workload_means) for pair in GRID}
    rs2 = pooled[(1.0, 0.0)]
    best_s = max(((scale, 0.0) for scale in SCALES), key=pooled.__getitem__)
    best_product = max(PRODUCT_GRID, key=pooled.__getitem__)
    print(f"pooled: unweighted mean over {len(workloads)} workloads")
    pooled_summary = {key: statistics.mean(summary[key] for summary in summaries.values())
                      for key in ("realised", "E_tok", "E_bv", "sanity", "gain")}
    for key in ("sanity_se", "gain_se"):
        pooled_summary[key] = math.sqrt(sum(summary[key] ** 2 for summary in summaries.values())) / len(summaries)
    print_bv(summary=pooled_summary)
    print_is(values=pooled_is, selections=selections)
    for label, value in (("RS2 (s=1, theta=0)", rs2), ("target-only", statistics.mean(target_means)),
                         (f"best s-only (s, theta)={best_s}", pooled[best_s]),
                         (f"best product-grid (s, theta)={best_product}", pooled[best_product])):
        print(f"  {label}: tok/step={value:.6f}; gain over RS2={100 * (value / rs2 - 1):+.3f}%")
    csv_rows.extend({"workload": "pooled", "verifies": sum(len(rows) for rows in workloads.values()),
                     "temp_scale": scale, "onehot_above": theta, "tok_per_step": pooled[(scale, theta)],
                     "tok_per_step_is_tok": pooled_is[(scale, theta)][0],
                     "tok_per_step_is_bv": pooled_is[(scale, theta)][1],
                     "ess": pooled_is[(scale, theta)][2]}
                    for scale, theta in GRID)
    output = Path(csv_path) if csv_path else Path(directory) / "rs2d-accept-grid.csv"
    with output.open(mode="w", newline="") as handle:
        writer = csv.DictWriter(f=handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV: {output}")
    elapsed = time.perf_counter() - started
    count = sum(len(rows) for rows in workloads.values())
    print(f"Analysis: rows={count} elapsed={elapsed:.3f}s rows/s={count / elapsed:.3f}")
    return workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("fnbench_jsonl", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report(directory=args.dump_dir, fnbench=args.fnbench_jsonl, csv_path=args.csv)


if __name__ == "__main__":
    main()
