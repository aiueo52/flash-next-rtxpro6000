"""Deterministic probe of the BV thresholds on B5's accept paths (CPU interpreter).

For each path and position u: coins are 2.0 (never accept) except coin_u = h_u (float64 reference) * (1 -/+ 1e-4).
Expect tau = u just below h_u and tau = 0 just above, which pins the kernel's h_u to the reference.
Usage: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= WT=<RS3 worktree> G1_WT=<G1 worktree> python threshold_probe.py
Paths: B5 accept's 12 (slots 4/8/16 x 4 rows), rebuilt by replaying b5_timing.main()'s generator.
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(name="b5", location=Path(__file__).with_name("b5_timing.py"))
b5 = importlib.util.module_from_spec(spec)
sys.modules["b5"] = b5
spec.loader.exec_module(b5)
reference_h64 = b5.load_reference_h64()

generator = torch.Generator().manual_seed(20261001)
for bs, slots in itertools.product((1, 4, 16), (4, 8, 16)):
    b5.make_batch(bs=bs, slots=slots, generator=generator)
for slots in (4, 8, 16):
    b5.make_batch(bs=256, slots=slots, generator=generator)
checked, bad = 0, []
for slots in (4, 8, 16):
    bs, draws = 4, 4096
    batch = b5.make_batch(bs=bs, slots=slots, generator=generator)
    torch.rand(size=(bs * draws, slots), generator=generator)
    torch.rand(size=(bs * draws,), generator=generator)
    hs = reference_h64(batch=batch)
    rows, expect = [], []
    for row, u, side in itertools.product(range(bs), range(1, slots), (-1, 1)):
        h = float(hs[row, u - 1])
        if side < 0 and h <= 0:
            continue
        coins = torch.full(size=(slots,), fill_value=2.0)
        coins[u - 1] = h * (1 + side * 1e-4) + (1e-7 if side > 0 else 0.0)
        rows.append((row, coins))
        expect.append(u if side < 0 else 0)
    n = len(rows)
    big = {name: value[[row for row, _ in rows]] for name, value in batch.items()
           if name not in ("coins", "coins_final", "retrieve")}
    big["coins"] = torch.stack(tensors=[coins for _, coins in rows])
    big["coins_final"] = torch.full(size=(n,), fill_value=0.5)
    big["retrieve"] = torch.arange(end=n * slots).reshape(n, slots)
    outputs = dict(predicts=torch.full(size=(n * slots,), fill_value=-1, dtype=torch.int32),
                   accept_index=torch.full(size=(n, slots), fill_value=-1, dtype=torch.int32),
                   accept_token_num=torch.empty(size=(n,), dtype=torch.int32))
    b5.verify(module=b5.rs3, batch=big, outputs=outputs, block_verify=True)
    got = outputs["accept_token_num"].tolist()
    for index, (want, have) in enumerate(zip(expect, got)):
        checked += 1
        if want != have:
            bad.append((slots, rows[index][0], want, have))
    print(f"slots={slots}: probe rows={n} h={[[round(float(x), 4) for x in r] for r in hs]}", flush=True)
print(f"{'PASS' if not bad else 'FAIL'} threshold probe: {checked} rows, mismatches={bad[:20]}", flush=True)
