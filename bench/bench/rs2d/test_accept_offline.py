"""CPU self-test for request-window joins and offline acceptance."""

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import io
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch

import accept_offline as offline

spec = importlib.util.spec_from_file_location(name="bv_exact", location=Path(__file__).parents[1] / "rs3/bv_exact.py")
exact = importlib.util.module_from_spec(spec=spec)
spec.loader.exec_module(module=exact)


def path_record(*, path, P, Q, V):
    G = len(path)
    return dict(candidates=torch.tensor(data=[[0, *path]], dtype=torch.int32),
                target_probs=torch.tensor(data=np.array([P[path[:i]] for i in range(G + 1)])[None], dtype=torch.float64),
                target_index=torch.arange(end=V).expand(1, G + 1, V).int(),
                draft_support_probs=torch.tensor(data=np.array([Q[path[:i]] for i in range(G)])[None], dtype=torch.float64),
                draft_support_tokens=torch.arange(end=V).expand(1, G, V).int(),
                accept_len=torch.tensor(data=[0], dtype=torch.int32),
                top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))


def test_exact():
    cases, paths = 0, 0
    for mode, V, G in itertools.product(("gen", "equal", "greedy"), (3, 4), (1, 2, 3)):
        P, Q = exact.random_case(seed=100 * V + G, V=V, G=G, mode=mode)
        total_tok, total_bv = 0.0, 0.0
        for path in itertools.product(range(V), repeat=G):
            dumped = path_record(path=path, P=P, Q=Q, V=V)
            metrics = offline.row_metrics(record=dumped, row=0, grid=())
            bv, pi = exact.bv_tau_probs(path=path, P=P, Q=Q)
            tok, pi = exact.token_tau_probs(path=path, P=P, Q=Q)
            assert abs(metrics["E_bv"] - sum(t * pr for t, pr in enumerate(bv))) < 1e-6
            assert abs(metrics["E_tok"] - sum(t * pr for t, pr in enumerate(tok))) < 1e-6
            probability = float(np.prod([Q[path[:i]][token] for i, token in enumerate(path)]))
            total_bv += probability * metrics["E_bv"]
            total_tok += probability * metrics["E_tok"]
            paths += 1
        for method, total in (("bv", total_bv), ("token", total_tok)):
            out, expected = exact.enumerate_outputs(P=P, Q=Q, V=V, G=G, method=method)
            assert abs(total - expected) < 1e-6
        cases += 1
    print(f"PASS BV/token exact oracle: {cases} cases/{paths} paths; V=3/4 G=1/2/3 gen/equal/greedy; tolerance=1e-6")


def test_is_exact():
    cases, paths = 0, 0
    for V, G in itertools.product((3, 4), (1, 2, 3)):
        P, Q = exact.random_case(seed=700 + G, V=V, G=G, mode="gen")
        P2, Q2 = exact.random_case(seed=900 + G, V=V, G=G, mode="equal")
        # Arbitrary Q2 requires full-support sampling Q for the IS identity.
        Q = {prefix: (q + 0.1) / (q + 0.1).sum() for prefix, q in Q.items()}
        total_tok, total_bv, total_weight = 0.0, 0.0, 0.0
        for path in itertools.product(range(V), repeat=G):
            dumped = path_record(path=path, P=P, Q=Q, V=V)
            q_prime = torch.tensor(data=np.array([Q2[path[:i]] for i in range(G)]), dtype=torch.float64)
            metrics = offline.row_metrics(record=dumped, row=0, grid=(), q_prime=q_prime)
            probability = float(np.prod([Q[path[:i]][token] for i, token in enumerate(path)]))
            alternative = float(np.prod([Q2[path[:i]][token] for i, token in enumerate(path)]))
            assert abs(probability * metrics["w"] - alternative) < 1e-12
            total_bv += probability * metrics["w"] * metrics["E_bv_prime"]
            total_tok += probability * metrics["w"] * metrics["E_tok_prime"]
            total_weight += probability * metrics["w"]
            paths += 1
        assert abs(total_weight - 1) < 1e-12
        for method, total in (("bv", total_bv), ("token", total_tok)):
            out, expected = exact.enumerate_outputs(P=P, Q=Q2, V=V, G=G, method=method)
            assert abs(total - expected) < 1e-6
        cases += 1
    print(f"PASS IS exact oracle: {cases} cases/{paths} paths; direct Q2, full-support Q; tolerance=1e-6")


def test_sparse():
    path = (10, 20, 30)
    p = torch.tensor(data=[[[0.2, 0.4, 0.4, 0.0], [0.2, 0.4, 0.4, 0.0],
                           [0.6, 0.2, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    q = torch.tensor(data=[[[0.3, 0.2, 0.5, float("nan")], [0.3, 0.2, 0.5, float("nan")],
                           [0.3, 0.2, 0.5, float("nan")]]], dtype=torch.float64)
    dumped = dict(candidates=torch.tensor(data=[[0, *path]]), target_probs=p,
                  target_index=torch.tensor(data=[[[10, 10, 20, 99], [20, 20, 30, 99],
                                                   [10, 30, 30, 99], [10, 20, 30, 99]]]),
                  draft_support_probs=q, draft_support_tokens=torch.tensor(data=[[[10, 10, 40, 99],
                                                                                 [20, 20, 40, 99], [30, 30, 40, 99]]]),
                  accept_len=torch.tensor(data=[0]), top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))
    P, Q = {}, {}
    for i in range(3):
        P[path[:i]], Q[path[:i]] = np.zeros(100), np.zeros(100)
        for token, mass in zip(dumped["target_index"][0, i].tolist(), p[0, i].tolist()):
            P[path[:i]][token] += mass
        for token, mass in zip(dumped["draft_support_tokens"][0, i].tolist(), q[0, i].tolist()):
            Q[path[:i]][token] += 0 if math.isnan(mass) else mass
    metrics = offline.row_metrics(record=dumped, row=0)
    for method, name in ((exact.bv_tau_probs, "E_bv"), (exact.token_tau_probs, "E_tok")):
        probs, pi = method(path=path, P=P, Q=Q)
        assert abs(metrics[name] - sum(t * pr for t, pr in enumerate(probs))) < 1e-12
    for pair in offline.GRID:
        scale, threshold = pair
        Q2 = {}
        values = []
        for i in range(3):
            transformed = offline.sharpen(q=q[0, i], scale=scale, threshold=threshold, top_p=0.95, min_p=0.05)
            Q2[path[:i]] = np.zeros(100)
            for token, mass in zip(dumped["draft_support_tokens"][0, i].tolist(), transformed.tolist()):
                Q2[path[:i]][token] += mass
            values.append(offline.acceptance(p=p[0, i], pi=dumped["target_index"][0, i],
                                             q=transformed, qi=dumped["draft_support_tokens"][0, i]))
        weight, tok, bv = metrics["is_grid"][pair]
        assert abs(metrics["grid"][pair] - offline.expected_drafts(acceptances=values)) < 1e-12
        assert abs(weight - np.prod([Q2[path[:i]][path[i]] / Q[path[:i]][path[i]] for i in range(3)])) < 1e-12
        for method, actual in ((exact.bv_tau_probs, bv), (exact.token_tau_probs, tok)):
            probs, pi = method(path=path, P=P, Q=Q2)
            assert abs(actual - sum(t * pr for t, pr in enumerate(probs))) < 1e-12
    absent = {**dumped, "candidates": torch.tensor(data=[[0, 40, 20, 30]])}
    absent_metrics = offline.row_metrics(record=absent, row=0, grid=())
    assert absent_metrics["E_tok"] == absent_metrics["E_bv"] == 0
    print("PASS sparse duplicates, NaN q=0, missing p/q ids, 56 scalar/vector grid comparisons")


def record(*, timestamp, length, p):
    return dict(time=timestamp, input_len=[length], rid=["dump-request"],
                candidates=torch.tensor(data=[[0, 10, 20]], dtype=torch.int32),
                target_probs=torch.tensor(data=[[p, list(reversed(p)), p]]),
                target_index=torch.tensor(data=[[[10, 20]] * 3], dtype=torch.int32),
                draft_support_probs=torch.tensor(data=[[[0.6, 0.4], [0.7, 0.3]]]),
                draft_support_tokens=torch.tensor(data=[[[10, 20]] * 2], dtype=torch.int32),
                accept_len=torch.tensor(data=[1], dtype=torch.int32),
                top_ps=torch.tensor(data=[0.95]), min_ps=torch.tensor(data=[0.05]))


def request(*, timestamp, workload, prompt_id):
    return dict(timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                workload=workload, prompt_id=prompt_id,
                client=dict(ttft_seconds=1.0, decode_seconds=1.0, usage=dict(prompt_tokens=3)))


def test_batched_report():
    with tempfile.TemporaryDirectory(prefix="rs2d-bv-report-") as directory:
        root = Path(directory)
        records, requests = [], []
        for name, timestamp, paths in (("a", 2000, ((10, 10), (10, 20), (20, 10))),
                                       ("b", 2010, ((20, 20), (10, 20)))):
            singles = []
            for row, path in enumerate(paths):
                dumped = record(timestamp=timestamp + 1, length=3, p=[0.2 + 0.2 * row, 0.8 - 0.2 * row])
                dumped["candidates"] = torch.tensor(data=[[0, *path]], dtype=torch.int32)
                dumped["accept_len"][0] = row
                dumped["top_ps"][0] = 0.85 + 0.05 * row
                dumped["min_ps"][0] = 0.01 * row
                singles.append(dumped)
            fields = ("candidates", "target_probs", "target_index", "draft_support_probs", "draft_support_tokens",
                      "top_ps", "min_ps", "accept_len")
            batch = {**singles[0], **{field: torch.cat(tensors=[single[field] for single in singles]) for field in fields},
                     "input_len": [3] * len(singles)}
            batched = offline.record_metrics(record=batch)
            for row, single in enumerate(singles):
                assert batched[row] == offline.row_metrics(record=single, row=0)
            records.append(batch)
            requests.append(request(timestamp=timestamp, workload=name, prompt_id=name))
        torch.save(obj=records, f=root / "rs-dump-0-00000.pt")
        fnbench = root / "fnbench.jsonl"
        fnbench.write_text("".join(json.dumps(row) + "\n" for row in requests))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            workloads = offline.report(directory=root, fnbench=fnbench)
        csv_rows = list(csv.DictReader((root / "rs2d-accept-grid.csv").open()))
        for pair in offline.GRID:
            values = []
            for name, rows in workloads.items():
                samples = [row["is_grid"][pair] for row in rows]
                total = sum(weight for weight, tok, bv in samples)
                tok = 1 + sum(weight * tok for weight, tok, bv in samples) / total if total else math.nan
                bv = 1 + sum(weight * bv for weight, tok, bv in samples) / total if total else math.nan
                ess = total ** 2 / sum(weight ** 2 for weight, tok, bv in samples) if total else 0.0
                values.append((tok, bv, ess))
                actual = next(item for item in csv_rows if item["workload"] == name and
                              (float(item["temp_scale"]), float(item["onehot_above"])) == pair)
                for column, expected in zip(("tok_per_step_is_tok", "tok_per_step_is_bv", "ess"), values[-1]):
                    assert math.isnan(float(actual[column])) if math.isnan(expected) else math.isclose(float(actual[column]), expected, abs_tol=1e-12)
            pooled = next(item for item in csv_rows if item["workload"] == "pooled" and
                          (float(item["temp_scale"]), float(item["onehot_above"])) == pair)
            for index, column in enumerate(("tok_per_step_is_tok", "tok_per_step_is_bv", "ess")):
                expected = sum(value[index] for value in values) if index == 2 else statistics.mean(value[index] for value in values)
                assert math.isnan(float(pooled[column])) if math.isnan(expected) else math.isclose(float(pooled[column]), expected, abs_tol=1e-12)
        summaries = []
        for rows in workloads.values():
            differences = [row["E_bv"] - row["E_tok"] for row in rows]
            sanity = [row["realised"] - row["E_tok"] for row in rows]
            summary = offline.bv_summary(rows=rows)
            for samples, key in ((differences, "gain_se"), (sanity, "sanity_se")):
                mean = sum(samples) / len(samples)
                se = math.sqrt(sum((value - mean) ** 2 for value in samples) / (len(samples) * (len(samples) - 1)))
                assert math.isclose(summary[key], se, abs_tol=1e-12)
            summaries.append(summary)
        denominator = 1 + statistics.mean(summary["E_tok"] for summary in summaries)
        gain_se = math.sqrt(sum(summary["gain_se"] ** 2 for summary in summaries)) / len(summaries)
        pooled_text = output.getvalue().split("pooled: unweighted mean", maxsplit=1)[1]
        assert f"paired SE={100 * gain_se / denominator:.3f}%" in pooled_text
        assert f"paired SE={math.sqrt(sum(summary['sanity_se'] ** 2 for summary in summaries)) / len(summaries):.6f}" in pooled_text
    print("PASS batched metrics, unequal-workload SN IS/ESS CSV, zero-weight NaNs, paired and pooled SE")


def benchmark(*, rows, steps, batch_size):
    torch.manual_seed(seed=20261001)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rs2d-bv-speed-") as directory:
        root = Path(directory)
        fnbench = root / "fnbench.jsonl"
        requests = []
        for chunk, start in enumerate(range(0, rows, 50 * batch_size)):
            size = min(50 * batch_size, rows - start)
            q = torch.rand(size=(size, steps, 64)).square()
            q[:, :, 0] += q.sum(dim=-1)
            q /= q.sum(dim=-1, keepdim=True)
            q, qi = torch.sort(input=q, dim=-1, descending=True)
            p = torch.rand(size=(size, steps + 1, 64)).square()
            p[:, :, 0] += p.sum(dim=-1)
            p /= p.sum(dim=-1, keepdim=True)
            pi = torch.arange(end=64).expand(size, steps + 1, 64).clone()
            pi[..., -4:] += 64  # Exercise partially disjoint supports.
            draws = torch.multinomial(input=q.reshape(-1, 64), num_samples=1).reshape(size, steps, 1)
            candidates = torch.cat(tensors=(torch.zeros(size=(size, 1), dtype=torch.int32),
                                            qi.gather(dim=-1, index=draws).squeeze(dim=-1).int()), dim=-1)
            top_ps = torch.full(size=(size,), fill_value=0.95)
            min_ps = torch.full(size=(size,), fill_value=0.05)
            dumped = dict(candidates=candidates, target_probs=p, target_index=pi.int(),
                          draft_support_probs=q, draft_support_tokens=qi.int(),
                          accept_len=torch.zeros(size=(size,), dtype=torch.int32), top_ps=top_ps, min_ps=min_ps)
            # Independently drawn acceptance coins provide a realistic realised column.
            alpha = []
            for position in range(steps):
                token = candidates[:, position + 1:position + 2]
                pt = (p[:, position] * (pi[:, position] == token)).sum(dim=-1)
                qt = (q[:, position] * (qi[:, position] == token)).sum(dim=-1)
                alpha.append((pt / qt).clamp(max=1))
            passed = torch.rand(size=(size, steps)) < torch.stack(tensors=alpha, dim=-1)
            dumped["accept_len"] = passed.int().cumprod(dim=-1).sum(dim=-1).int()
            workload = f"synthetic-{chunk % 4}"
            timestamp = 1000 + 10 * chunk
            requests.append(request(timestamp=timestamp, workload=workload, prompt_id=f"chunk-{chunk}"))
            records = [{**{name: value[row:row + batch_size] for name, value in dumped.items()},
                        "time": timestamp + 1, "input_len": [3] * min(batch_size, size - row),
                        "rid": [f"synthetic-{start + row}"] * min(batch_size, size - row)}
                       for row in range(0, size, batch_size)]
            torch.save(obj=records, f=root / f"rs-dump-0-{chunk:05d}.pt")
        fnbench.write_text("".join(json.dumps(row) + "\n" for row in requests))
        print(f"Synthetic dump: rows={rows} G={steps} kp=K=64 batch_size={batch_size} grid={len(offline.GRID)} "
              f"generation={time.perf_counter() - started:.3f}s", flush=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            workloads = offline.report(directory=root, fnbench=fnbench)
        assert sum(len(values) for values in workloads.values()) == rows
        text = output.getvalue()
        # Report counts/CSV were computed for every row and all 56 grid pairs.
        csv_rows = list(csv.DictReader((root / "rs2d-accept-grid.csv").open()))
        assert len(csv_rows) == (len(workloads) + 1) * len(offline.GRID)
        for line in text.splitlines():
            if line.startswith(("pooled:", "Analysis:", "  path estimates", "  BV gain", "    best pooled")):
                print(line)
        print(f"PASS synthetic full report: rows={rows} G={steps} grid={len(offline.GRID)} CSV rows={len(csv_rows)}")


def main():
    torch.set_num_threads(4)
    test_exact()
    test_is_exact()
    test_sparse()
    test_batched_report()
    with tempfile.TemporaryDirectory(prefix="rs2d-join-") as directory:
        root = Path(directory)
        records = [record(timestamp=time, length=3, p=[0.9, 0.1]) for time in (1000, 1002, 1004)]
        records += [record(timestamp=1010, length=3, p=[0.2, 0.8]),
                    record(timestamp=1008, length=3, p=[0.9, 0.1]),
                    record(timestamp=1011, length=4, p=[0.9, 0.1])]
        torch.save(obj=records, f=root / "rs-dump-0-00000.pt")
        requests = [request(timestamp=1000, workload="code-edit", prompt_id="code-1"),
                    request(timestamp=1010, workload="prose-ja", prompt_id="ja-1")]
        missing = request(timestamp=1008, workload="ignored", prompt_id="missing")
        del missing["client"]["decode_seconds"]
        requests.append(missing)
        fnbench = root / "fnbench.jsonl"
        fnbench.write_text("".join(json.dumps(row) + "\n" for row in requests))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            workloads = offline.report(directory=root, fnbench=fnbench)
        text = output.getvalue()
        assert set(workloads) == {"code-edit", "prose-ja"}
        assert [len(workloads[name]) for name in ("code-edit", "prose-ja")] == [3, 1]
        assert all(row["prompt_id"] == "code-1" for row in workloads["code-edit"])
        assert workloads["prose-ja"][0]["prompt_id"] == "ja-1"
        assert "no_window=1; length_mismatch=1" in text
        assert "verifies per prompt_id: code-1=3" in text and "verifies per prompt_id: ja-1=1" in text
        windows = offline.request_windows(filename=fnbench)
        assert len(windows) == 2
        for dumped in records[:4]:
            metrics = offline.row_metrics(record=dumped, row=0, grid=((1.0, 0.0), (1.0, 1e-9)))
            for position in range(2):
                p, pi = dumped["target_probs"][0, position], dumped["target_index"][0, position]
                q, qi = dumped["draft_support_probs"][0, position], dumped["draft_support_tokens"][0, position]
                baseline = float(torch.minimum(input=p.double(), other=q.double()).sum())
                w = offline.sharpen(q=q, scale=1.0, threshold=0.0, top_p=0.95, min_p=0.05)
                assert math.isclose(offline.acceptance(p=p, pi=pi, q=w, qi=qi), baseline, abs_tol=1e-12)
                assert math.isclose(metrics["a_rs"][position], baseline, abs_tol=1e-12)
                onehot = offline.sharpen(q=q, scale=1.0, threshold=1e-9, top_p=0.95, min_p=0.05)
                target = float(p[pi == qi[0]].double().sum())
                assert math.isclose(offline.acceptance(p=p, pi=pi, q=onehot, qi=qi), target, abs_tol=1e-12)
            assert math.isclose(metrics["grid"][(1.0, 0.0)], metrics["rs"], abs_tol=1e-12)
            assert math.isclose(metrics["grid"][(1.0, 1e-9)], metrics["target_only"], abs_tol=1e-12)
        csv_rows = list(csv.DictReader((root / "rs2d-accept-grid.csv").open()))
        assert len(csv_rows) == 3 * len(offline.GRID)
        for pair in offline.GRID:
            pooled = next(row for row in csv_rows if row["workload"] == "pooled" and
                          (float(row["temp_scale"]), float(row["onehot_above"])) == pair)
            expected = 1 + sum(rows[0]["grid"][pair] for rows in workloads.values()) / 2
            assert math.isclose(float(pooled["tok_per_step"]), expected, abs_tol=1e-12)
            for name, index in (("tok_per_step_is_tok", 0), ("tok_per_step_is_bv", 1)):
                values = [offline.is_means(rows=rows)[pair][index] for rows in workloads.values()]
                mean = statistics.mean(values)
                actual = float(pooled[name])
                assert math.isnan(actual) if math.isnan(mean) else math.isclose(actual, mean, abs_tol=1e-12)
            ess = sum(offline.is_means(rows=rows)[pair][2] for rows in workloads.values())
            assert math.isclose(float(pooled["ess"]), ess, abs_tol=1e-12)
        pooled_rs = 1 + sum(rows[0]["rs"] for rows in workloads.values()) / 2
        pooled_target = 1 + sum(rows[0]["target_only"] for rows in workloads.values()) / 2
        assert f"RS2 (s=1, theta=0): tok/step={pooled_rs:.6f}; gain over RS2=+0.000%" in text
        assert f"target-only: tok/step={pooled_target:.6f}" in text
        assert "best s-only" in text and "best product-grid" in text
        assert "sanity mean E_tok=" in text and "BV gain over E_tok=" in text
        assert text.count("IS (self-normalised per workload)") == 3
        assert text.count("best pooled IS E_bv'") == 3
        baseline = offline.is_means(rows=workloads["code-edit"])[(1.0, 0.0)]
        assert math.isclose(baseline[0], 1 + workloads["code-edit"][0]["E_tok"], abs_tol=1e-12)
        assert math.isclose(baseline[1], 1 + workloads["code-edit"][0]["E_bv"], abs_tol=1e-12)
        assert baseline[2] == 3
        # Consecutive requests may overlap only through the two-second padding.
        newer = request(timestamp=1003, workload="next", prompt_id="next-1")
        fnbench.write_text("".join(json.dumps(row) + "\n" for row in [requests[0], newer]))
        match, reason = offline.join_request(windows=offline.request_windows(filename=fnbench),
                                            timestamp=1003.5, input_len=3)
        assert reason is None and match["prompt_id"] == "next-1"
        print(text, end="")
    print("PASS accept_offline CPU self-test: equal-length request join, inclusive windows, missing timings, "
          "no-window/wrong-length exclusions, verifies per prompt, per-position overlap/target-only, "
          "unweighted pooled grid and gains")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=int, metavar="ROWS")
    parser.add_argument("--steps", type=int, choices=(3, 7, 15), default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.benchmark is None:
        main()
    else:
        torch.set_num_threads(args.threads)
        benchmark(rows=args.benchmark, steps=args.steps, batch_size=args.batch_size)
