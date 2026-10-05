"""Shared draft rows must preserve every visible position through production packing."""

import json
import os
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
WT = Path(os.environ.get(key="WT", default="/home/user/tools/sglang-st1/python"))
if WT.name != "python":
    WT = WT / "python"
sys.path[:0] = [str(WT), str(BENCH)]
os.environ.setdefault(key="PYTHONDONTWRITEBYTECODE", value="1")
sys.dont_write_bytecode = True
os.environ.setdefault(key="SGLANG_CACHE_DIR", value="/tmp/st1-sglang")
os.environ.setdefault(key="FLASHINFER_WORKSPACE_BASE", value="/tmp/st1-sglang")
os.environ.setdefault(key="TORCHINDUCTOR_CACHE_DIR", value="/tmp/st1-sglang/inductor")
os.environ.setdefault(key="CUDA_CACHE_PATH", value="/tmp/st1-sglang/nv")
os.environ.setdefault(key="TRITON_CACHE_DIR", value="/tmp/st1-triton")

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    _mtp_shared_sparse_indices_lookup_kernel,
    _mtp_shared_sparse_indices_lookup_prefix_kernel,
)
from xa1.cases import BLOCK_TOPK, FINAL_TOPK, RATIO, fresh_row
from xa1.hole_demo import _packed_positions

ANCHORS = (1, 3, 4, 5, 1000, 2047, 2048, 2049, 2050, 2051, 2052, 8184, 8190, 8191)
WIDTHS = (4, 8, 16)
CHECKS = {}


@triton.jit
def production_lookup_kernel(
    indices, captured_len, req_pool_indices, current_positions, out,
    indices_row_stride, req_pool_indices_stride, current_positions_stride,
    out_row_stride, num_columns: tl.constexpr, tail_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(start=0, end=BLOCK)
    column_mask = columns < num_columns
    source_row = tl.load(req_pool_indices + row * req_pool_indices_stride).to(tl.int64)
    frozen = tl.load(
        pointer=indices + source_row * indices_row_stride + columns,
        mask=column_mask, other=0,
    )
    tail_start = num_columns - tail_width
    tail_offset = columns - tail_start
    base = tl.load(captured_len + source_row).to(tl.int64)
    position = tl.load(current_positions + row * current_positions_stride).to(tl.int64)
    tail_value = base + tail_offset
    tail_value = tl.where(condition=tail_value <= position, x=tail_value, y=-1)
    value = tl.where(condition=columns >= tail_start, x=tail_value, y=frozen)
    tl.store(pointer=out + row * out_row_stride + columns, value=value, mask=column_mask)


def record(*, name, ok, context):
    if name not in CHECKS:
        CHECKS[name] = [0, 0, []]
    counts = CHECKS[name]
    counts[0] += 1
    if not ok:
        counts[1] += 1
        if len(counts[2]) < 3:
            counts[2].append(context)


def same(*, a, b):
    return torch.equal(input=a, other=b)


def make_state(*, enabled, width, device="cpu"):
    with envs.SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX.override(enabled):
        return QSAMTPSharedSparseIndices(
            layer_ids=[0, 7], num_requests=3, token_topk=FINAL_TOPK,
            tail_width=width, device=device,
        )


def anchor_row(*, length, seed):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randperm(n=max(length // RATIO, 1), generator=gen)[:BLOCK_TOPK]
    return fresh_row(L=length, base_blocks=base, overlap=0.75, gen=gen)


def production_torch(*, state, req, positions, layer_id):
    slot = state.layer_slots[layer_id]
    rows = req.to(torch.long)
    out = state.indices[slot, rows]
    base = state.captured_len[slot, rows].to(torch.int64)
    tail = base.unsqueeze(1) + state._tail_offsets.unsqueeze(0)
    valid = tail <= positions.to(torch.int64).unsqueeze(1)
    out[:, out.shape[1] - state.tail_width:] = torch.where(
        condition=valid, input=tail, other=-1,
    ).to(out.dtype)
    return out


def launch(*, kernel, state, req, positions, layer_id, prefix, out=None):
    indices = state.indices[state.layer_slots[layer_id]]
    num_columns = indices.shape[1]
    if out is None:
        out = torch.full(
            size=(req.numel(), num_columns), fill_value=-99,
            dtype=torch.int32, device=indices.device,
        )
    block = triton.next_power_of_2(num_columns) if prefix else 256
    grid = (req.numel(),) if prefix else (req.numel(), (num_columns + block - 1) // block)
    kernel[grid](
        indices=indices, captured_len=state.captured_len[state.layer_slots[layer_id]],
        req_pool_indices=req, current_positions=positions, out=out,
        indices_row_stride=indices.stride(0), req_pool_indices_stride=req.stride(0),
        current_positions_stride=positions.stride(0), out_row_stride=out.stride(0),
        num_columns=num_columns, tail_width=state.tail_width, BLOCK=block, num_warps=8 if prefix else 4,
    )
    return out


def check_rows(*, off, on, req, positions, layer_id, context, pack=True):
    original = production_torch(state=off, req=req, positions=positions, layer_id=layer_id)
    torch_off = off.lookup(req_pool_indices=req, current_positions=positions, layer_id=layer_id)
    torch_on = on.lookup(req_pool_indices=req, current_positions=positions, layer_id=layer_id)
    kernel_off = launch(
        kernel=_mtp_shared_sparse_indices_lookup_kernel, state=off, req=req,
        positions=positions, layer_id=layer_id, prefix=False,
    )
    copy_off = launch(
        kernel=production_lookup_kernel, state=off, req=req,
        positions=positions, layer_id=layer_id, prefix=False,
    )
    kernel_on = launch(
        kernel=_mtp_shared_sparse_indices_lookup_prefix_kernel, state=on, req=req,
        positions=positions, layer_id=layer_id, prefix=True,
    )
    record(name="A torch equals production", ok=same(a=torch_off, b=original), context=context)
    record(name="A Triton equals production copy", ok=same(a=kernel_off, b=copy_off), context=context)
    record(name="A Triton equals torch", ok=same(a=kernel_off, b=torch_off), context=context)
    record(name="B Triton equals torch", ok=same(a=kernel_on, b=torch_on), context=context)
    for r, request in enumerate(req.tolist()):
        row = torch_on[r]
        position = int(positions[r])
        frozen = on.indices[on.layer_slots[layer_id], request, :FINAL_TOPK]
        frozen = frozen[frozen >= 0]
        base = int(on.captured_len[on.layer_slots[layer_id], request])
        drafted = torch.arange(start=base, end=position + 1, dtype=torch.int32)
        expected = torch.cat(tensors=[frozen, drafted])
        visible = row[(row >= 0) & (row <= position)]
        n = visible.numel()
        label = f"{context}, row={r}, req={request}"
        record(name="B visible entries form prefix", ok=same(a=row[:n], b=visible), context=label)
        record(name="B stable frozen order and tail", ok=same(a=row[:n], b=expected), context=label)
        record(name="B multiset preserved", ok=same(a=visible.sort().values, b=expected.sort().values), context=label)
        record(name="B every suffix entry is -1", ok=bool((row[n:] == -1).all()), context=label)
        dummy = request in (0, on.trash_row)
        if dummy:
            record(name="B never-captured/trash equals production", ok=same(a=row, b=original[r]), context=label)
        else:
            record(name="B no duplicates", ok=visible.unique().numel() == n, context=label)
        if pack:
            for enabled, packed_row in ((False, torch_off[r]), (True, row)):
                counts, packed = _packed_positions(row=packed_row, seq_len=position + 1, ctx=position + 2)
                visible_set = set(packed_row[(packed_row >= 0) & (packed_row <= position)].tolist())
                dropped = sorted(visible_set - set(packed.tolist()))
                stale = int((packed < 0).sum())
                if enabled:
                    record(name="C on: no dropped positions or stale slots", ok=not dropped and stale == 0 and counts == n, context=label)
                else:
                    holes = FINAL_TOPK - frozen.numel()
                    num_dropped = min(drafted.numel(), holes)
                    expected_dropped = list(range(position - num_dropped + 1, position + 1))
                    record(name="C off: production hole pattern", ok=dropped == expected_dropped and stale == num_dropped, context=label)


def run_captured_cpu(*, width):
    for length in ANCHORS:
        for bs in (1, 2):
            off = make_state(enabled=False, width=width)
            on = make_state(enabled=True, width=width)
            req = torch.tensor(data=[1, 0, 2, 0], dtype=torch.int32)[::2][:bs]
            anchors = torch.stack(tensors=[anchor_row(length=length, seed=r) for r in range(bs)])
            lengths = torch.full(size=(bs,), fill_value=length, dtype=torch.int32)
            layer_id = 7
            for state in (off, on):
                state.capture(topk_indices=anchors, req_pool_indices=req, captured_lens=lengths, layer_id=layer_id)
            snapshot = on.indices.clone()
            for j in range(1, width):
                positions = torch.full(size=(bs * 2,), fill_value=length + j - 1, dtype=torch.int64)[::2]
                check_rows(off=off, on=on, req=req, positions=positions, layer_id=layer_id,
                           context=f"L={length}, W={width}, bs={bs}, j={j}")
            record(name="B lookup leaves capture unchanged", ok=same(a=on.indices, b=snapshot), context=f"L={length}, W={width}, bs={bs}")
        print(f"CPU sweep W={width}: anchor {length} complete", flush=True)


def run_extra_cpu(*, width):
    off = make_state(enabled=False, width=width)
    on = make_state(enabled=True, width=width)
    for req_list in ([0], [3], [0, 3]):
        req = torch.tensor(data=req_list, dtype=torch.int32)
        positions = torch.zeros(size=(len(req_list),), dtype=torch.int64)
        check_rows(off=off, on=on, req=req, positions=positions, layer_id=0, context=f"dummy W={width}, req={req_list}")
    anchor = anchor_row(length=5, seed=0)
    # Interior holes exercise stable compaction beyond prefix-shaped anchors.
    shuffled = torch.full_like(input=anchor, fill_value=-1)
    shuffled[torch.tensor(data=[0, 4, 20, 100, 2050])] = anchor[:5]
    for state in (off, on):
        state.capture(topk_indices=shuffled[None], req_pool_indices=torch.tensor(data=[1]),
                      captured_lens=torch.tensor(data=[5]), layer_id=0)
    check_rows(off=off, on=on, req=torch.tensor(data=[1, 3]), positions=torch.tensor(data=[7, 0]),
               layer_id=0, context=f"interior holes/mixed trash W={width}", pack=False)
    check_rows(off=off, on=on, req=torch.tensor(data=[1, 0]), positions=torch.tensor(data=[7, 0]),
               layer_id=0, context=f"interior holes/mixed never-captured W={width}", pack=False)


def run_xa1_cpu():
    table = json.loads((BENCH / "xa1/results/xa1-hole-demo.json").read_text())
    for length_str, entries in table.items():
        length = int(length_str)
        off = make_state(enabled=False, width=8)
        off.capture(topk_indices=anchor_row(length=length, seed=0)[None], req_pool_indices=torch.tensor(data=[1]),
                    captured_lens=torch.tensor(data=[length]), layer_id=0)
        for entry in entries:
            row = off.lookup(req_pool_indices=torch.tensor(data=[1]), current_positions=torch.tensor(data=[entry["position"]]), layer_id=0)[0]
            count, packed = _packed_positions(row=row, seq_len=entry["position"] + 1, ctx=entry["position"] + 2)
            dropped = sorted(set(row[row >= 0].tolist()) - set(packed.tolist()))
            stale = int((packed < 0).sum())
            ok = (count == entry["valid_count"] and dropped == entry["dropped"]
                  and stale == entry["stale_slots"] and row.numel() == entry["width"]
                  and (entry["position"] in dropped) == entry["current_token_dropped"])
            record(name="C off: XA1 JSON anchors", ok=ok, context=f"L={length}, j={entry['step']}")


def run_cpu():
    for width in WIDTHS:
        run_captured_cpu(width=width)
        run_extra_cpu(width=width)
    run_xa1_cpu()


def graph_time_us(*, kernel, state, req, positions, prefix):
    out = torch.empty(size=(req.numel(), FINAL_TOPK + state.tail_width), dtype=torch.int32, device="cuda")
    for _ in range(10):
        launch(kernel=kernel, state=state, req=req, positions=positions, layer_id=7, prefix=prefix, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            launch(kernel=kernel, state=state, req=req, positions=positions, layer_id=7, prefix=prefix, out=out)
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 20)
    return sorted(times)[len(times) // 2]


def run_cuda():
    for width in WIDTHS:
        for bs in (1, 2):
            cpu = make_state(enabled=True, width=width)
            gpu = make_state(enabled=True, width=width, device="cuda")
            req = torch.arange(start=1, end=bs + 1, dtype=torch.int32)
            for length in ANCHORS:
                anchors = torch.stack(tensors=[anchor_row(length=length, seed=r) for r in range(bs)])
                lengths = torch.full(size=(bs,), fill_value=length, dtype=torch.int32)
                cpu.capture(topk_indices=anchors, req_pool_indices=req, captured_lens=lengths, layer_id=7)
                gpu.capture(topk_indices=anchors.cuda(), req_pool_indices=req.cuda(), captured_lens=lengths.cuda(), layer_id=7)
                for j in range(1, width):
                    positions = torch.full(size=(bs,), fill_value=length + j - 1, dtype=torch.int64)
                    expected = cpu.lookup(req_pool_indices=req, current_positions=positions, layer_id=7)
                    actual = gpu.lookup(req_pool_indices=req.cuda(), current_positions=positions.cuda(), layer_id=7).cpu()
                    record(name="D GPU equals CPU torch", ok=same(a=actual, b=expected), context=f"L={length}, W={width}, bs={bs}, j={j}")
            for req_list in ([0], [3], [0, 3]):
                check_cuda_rows(
                    cpu=cpu, gpu=gpu, req=torch.tensor(data=req_list),
                    positions=torch.zeros(size=(len(req_list),), dtype=torch.int64),
                    layer_id=0, context=f"dummy W={width}, req={req_list}",
                )
            anchor = anchor_row(length=5, seed=0)
            shuffled = torch.full_like(input=anchor, fill_value=-1)
            shuffled[torch.tensor(data=[0, 4, 20, 100, 2050])] = anchor[:5]
            for state, device in ((cpu, "cpu"), (gpu, "cuda")):
                state.capture(
                    topk_indices=shuffled[None].to(device),
                    req_pool_indices=torch.tensor(data=[1], device=device),
                    captured_lens=torch.tensor(data=[5], device=device), layer_id=0,
                )
            for dummy in (0, 3):
                check_cuda_rows(
                    cpu=cpu, gpu=gpu, req=torch.tensor(data=[1, dummy]),
                    positions=torch.tensor(data=[7, 0]), layer_id=0,
                    context=f"interior holes/mixed dummy={dummy}, W={width}",
                )
            req_gpu = req.cuda()
            positions_gpu = positions.cuda()
            old = graph_time_us(kernel=_mtp_shared_sparse_indices_lookup_kernel, state=gpu, req=req_gpu, positions=positions_gpu, prefix=False)
            new = graph_time_us(kernel=_mtp_shared_sparse_indices_lookup_prefix_kernel, state=gpu, req=req_gpu, positions=positions_gpu, prefix=True)
            # Budget +2 us per launch: about 6 lookups per step, so at most ~12 us of an ~11 ms step;
            # the A/B decides. The first budget, +1 us, failed at 4 warps (+2.0 us); 8 warps gave +1.4 us.
            record(name=f"D graph latency rows={bs} columns={FINAL_TOPK + width}", ok=new <= old + 2,
                   context=f"old={old:.3f} us, new={new:.3f} us, delta={new - old:.3f} us")
            print(f"D timing rows={bs} columns={FINAL_TOPK + width}: old={old:.3f} us new={new:.3f} us delta={new - old:.3f} us")


def check_cuda_rows(*, cpu, gpu, req, positions, layer_id, context):
    expected = cpu.lookup(
        req_pool_indices=req, current_positions=positions, layer_id=layer_id,
    )
    actual = gpu.lookup(
        req_pool_indices=req.cuda(), current_positions=positions.cuda(),
        layer_id=layer_id,
    ).cpu()
    record(name="D GPU dummy/interior holes equals CPU", ok=same(a=actual, b=expected), context=context)


def main():
    dev = os.environ.get(key="DEV", default="cpu")
    if dev not in ("cpu", "cuda"):
        sys.exit("DEV must be cpu or cuda")
    if dev == "cpu" and (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("TRITON_INTERPRET") != "1"):
        sys.exit("CPU checks require CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1")
    if dev == "cuda" and os.environ.get("TRITON_INTERPRET") == "1":
        sys.exit("D requires TRITON_INTERPRET=0 and an explicitly selected CUDA device")
    torch.set_num_threads(1)
    print(f"sglang source: {WT}", flush=True)
    try:
        if dev == "cpu":
            run_cpu()
            print("D skipped (cpu)")
        else:
            run_cuda()
    except Exception as exc:
        record(name="execution", ok=False, context=f"{type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    return report(dev=dev)


def report(*, dev):
    for name, (total, failures, examples) in CHECKS.items():
        detail = f"; {'; '.join(examples)}" if examples else ""
        print(f"{'FAIL' if failures else 'PASS'} {name}: {total - failures}/{total}{detail}")
    failed = sum(counts[1] for counts in CHECKS.values())
    total = sum(counts[0] for counts in CHECKS.values())
    print(f"Summary: {len(CHECKS)} checks, {total - failed}/{total} assertions passed, {failed} failures; DEV={dev}")
    return int(failed != 0)


if __name__ == "__main__":
    sys.exit(main())
