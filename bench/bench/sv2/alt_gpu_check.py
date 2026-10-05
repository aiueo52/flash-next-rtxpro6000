# The alternative kernel's GPU harness, archived from <worktree>/sv2_bench/gpu_check.py; needs alt_triton_topk.patch applied
# and the file placed back at <worktree>/sv2_bench/ (WORKTREE_ROOT = parents[1]).
import json
import statistics
import sys
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
sys.path = [str(WORKTREE_ROOT / "python"), *sys.path]

import torch

from sglang.kernels.ops.speculative.sparse_verify import sparse_topk

V = 248320
ROWS = (4, 8, 16)
KPS = (64, 128)
DTYPES = (torch.float32, torch.bfloat16)
MODES = ("random", "ties", "peaked", "masked", "edges")
WARMUPS = 3
SAMPLES = 7
REPEATS = 20


def _make_logits(*, mode, rows, kp, dtype, device):
    shape = (rows, V)
    if mode == "masked":
        logits = torch.full(
            size=shape,
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        finite_count = max(1, kp // 2)
        positions = torch.randint(
            low=0,
            high=V,
            size=(rows - 1, finite_count),
            device=device,
        )
        finite = torch.randn(
            size=(rows - 1, finite_count),
            dtype=torch.float32,
            device=device,
        )
        logits[1:].scatter_(dim=1, index=positions, src=finite)
    else:
        logits = torch.randn(size=shape, dtype=torch.float32, device=device) * 4.0
        if mode == "ties":
            logits = torch.randint(
                low=0,
                high=40,
                size=shape,
                device=device,
            ).to(dtype=torch.float32)
        elif mode == "peaked":
            logits[:, :8] = logits[:, :8] + 30.0
        elif mode == "edges":
            logits[:, : kp + 16] = 100.0
            logits[:, -3:] = 1e30
        elif mode != "random":
            raise ValueError(f"unknown mode: {mode}")
    return logits.to(dtype=dtype).contiguous()


def _check_result(*, logits, reference_values, values, ids, kp):
    problems = []
    expected_shape = (logits.shape[0], kp)
    if values.shape != expected_shape or ids.shape != expected_shape:
        problems.append(
            f"shape mismatch: values={tuple(values.shape)} "
            f"ids={tuple(ids.shape)}"
        )
    if values.dtype != logits.dtype:
        problems.append(f"values dtype mismatch: {values.dtype}")
    if ids.dtype != torch.int64:
        problems.append(f"indices dtype mismatch: {ids.dtype}")
    if problems:
        return problems
    if not torch.equal(input=values, other=reference_values):
        problems.append("values differ from torch.topk")
    in_range = torch.all((ids >= 0) & (ids < logits.shape[1]))
    if not bool(in_range.item()):
        problems.append("indices out of range")
        return problems
    sorted_ids = torch.sort(input=ids, dim=1).values
    if not bool(torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]).item()):
        problems.append("duplicate indices")
    if not torch.equal(input=logits.gather(dim=1, index=ids), other=values):
        problems.append("gather(indices) differs from values")
    selected = torch.zeros_like(input=logits, dtype=torch.bool)
    selected.scatter_(dim=1, index=ids, value=True)
    strictly_above = logits > reference_values[:, -1:]
    if bool(torch.any(strictly_above & ~selected).item()):
        problems.append("an index strictly above the pivot is missing")
    return problems


def _median_eager_us(*, function, device):
    for _ in range(WARMUPS):
        function()
    torch.cuda.synchronize(device=device)
    samples = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPEATS):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / REPEATS)
    return statistics.median(samples)


def _median_graph_us(*, function, device):
    for _ in range(WARMUPS):
        function()
    torch.cuda.synchronize(device=device)
    graph = torch.cuda.CUDAGraph()
    captured_outputs = []
    with torch.cuda.graph(cuda_graph=graph):
        for _ in range(REPEATS):
            captured_outputs.append(function())
    for _ in range(WARMUPS):
        graph.replay()
    torch.cuda.synchronize(device=device)
    samples = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / REPEATS)
    return statistics.median(samples)


def _emit(*, output, record):
    output.write(json.dumps(obj=record, sort_keys=True) + "\n")
    output.flush()


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <output.jsonl>")
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with open(file=output_path, mode="w", encoding="utf-8") as output:
        if not torch.cuda.is_available():
            _emit(
                output=output,
                record={"type": "error", "error": "CUDA is unavailable"},
            )
            return 2
        torch.manual_seed(seed=1234)
        torch.cuda.manual_seed_all(seed=1234)
        device = torch.device("cuda")
        for dtype in DTYPES:
            dtype_name = str(dtype).removeprefix("torch.")
            for rows in ROWS:
                for kp in KPS:
                    for mode in MODES:
                        case = {
                            "mode": mode,
                            "n": rows,
                            "v": V,
                            "kp": kp,
                            "dtype": dtype_name,
                        }
                        logits = _make_logits(
                            mode=mode,
                            rows=rows,
                            kp=kp,
                            dtype=dtype,
                            device=device,
                        )
                        reference_values, _ = torch.topk(
                            input=logits,
                            k=kp,
                            dim=-1,
                            largest=True,
                            sorted=True,
                        )

                        def torch_function():
                            return torch.topk(
                                input=logits,
                                k=kp,
                                dim=-1,
                                largest=True,
                                sorted=True,
                            )

                        def sparse_function():
                            return sparse_topk(logits=logits, kp=kp)

                        errors = []
                        try:
                            values, ids = sparse_function()
                            errors = _check_result(
                                logits=logits,
                                reference_values=reference_values,
                                values=values,
                                ids=ids,
                                kp=kp,
                            )
                        except Exception as exc:
                            errors.append(f"{type(exc).__name__}: {exc}")
                        _emit(
                            output=output,
                            record={
                                "type": "correctness",
                                **case,
                                "ok": not errors,
                                "errors": errors,
                            },
                        )
                        failed = failed or bool(errors)
                        for operator, function in (
                            ("torch.topk", torch_function),
                            ("sparse_topk", sparse_function),
                        ):
                            for timing_mode, benchmark in (
                                ("eager", _median_eager_us),
                                ("cuda_graph", _median_graph_us),
                            ):
                                try:
                                    median_us = benchmark(
                                        function=function,
                                        device=device,
                                    )
                                    record = {
                                        "type": "timing",
                                        **case,
                                        "operator": operator,
                                        "timing_mode": timing_mode,
                                        "median_us": median_us,
                                    }
                                except Exception as exc:
                                    failed = True
                                    record = {
                                        "type": "timing",
                                        **case,
                                        "operator": operator,
                                        "timing_mode": timing_mode,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                _emit(output=output, record=record)
                        del logits, reference_values, torch_function, sparse_function
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
