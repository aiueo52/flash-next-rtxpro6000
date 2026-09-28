#!/usr/bin/env python3
"""CPU checks of the unmodified production policy and WS1 OLS fit."""
import importlib.util
import json
import math
from pathlib import Path
import statistics

BENCH = Path(__file__).resolve().parents[1]
PROD = BENCH.parent / 'sglang-rtxpro6000'
spec = importlib.util.spec_from_file_location('wa2_policy', PROD / 'python/sglang/srt/speculative/adaptive_confidence.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original = json.loads((BENCH / 'adaptive/w16_conf.json').read_text())
cfg = json.loads(json.dumps(original))
cfg['1']['candidate_steps'] = [3, 7, 15]
assert cfg['2'] == original['2']
(BENCH / 'adaptive/w16_3_7_15.json').write_text(json.dumps(cfg, indent=2) + '\n')
c = cfg['1']
for start, want in [(3, 3), (7, 7), (15, 15), (5, 7)]:
    assert m.ConfidenceStepSlot(start, c).current_steps == want
print('PASS initial: explicit 3/7/15 preserved; unsupported 5 falls back to 7. Launcher wa explicitly starts at 15.')

# Exercise all candidates, with the middle as incumbent and both margins active.
for values, want in [({3: 1, 7: 2, 15: 1}, 7),
                     ({3: 1.13, 7: 1, 15: 1}, 3),
                     ({3: 1.11, 7: 1, 15: 1.45}, 7),
                     ({3: 1, 7: 1, 15: 1.46}, 15)]:
    slot = m.ConfidenceStepSlot(7, c)
    slot.value = values.__getitem__
    assert slot.best_steps()[0] == want
print('PASS best_steps considers all 3; middle down threshold 1.12x; up threshold 1.12*1.30=1.456x.')

# S=7 observations update exact shorter-prefix means, never the censored S=15.
slot = m.ConfidenceStepSlot(7, dict(c, warmup_batches=10000, rate_alpha=1.0, confidence_weight=0))
slot.update([0, 3, 6, 7, 7])
assert slot._pm[3] == 2.4 and slot._pm[7] == 4.6 and slot._pmn[15] == 0
r = 2 / 3
expected = 4.6 + .75 * .4 * sum(r**i for i in range(1, 9))
assert math.isclose(slot.expected_accept_for(15), expected)
slot._pm[15] = 999  # A stale larger-width observation must be ignored.
slot._pmn[15] = 100
assert math.isclose(slot.expected_accept_for(15), expected)
assert math.isclose(slot.value(15), (1 + expected) / m.step_time_ms(15))
print(f'PASS S7->S15: base=4.6, P(a>=7)=.4, P(a>=6)=.6, k=8, tail_bias=.75 => {expected:.9f} accepted drafts; stale S15 ignored.')
for bias in (0, .75, 1):
    s = m.ConfidenceStepSlot(7, dict(c, tail_bias=bias))
    for sat, satp in [(0, 0), (.1, .2), (1, 1)]:
        e = s._extrapolate(7 * sat, sat, satp, 8)
        assert math.isfinite(e) and 7 * sat <= e <= 15

slot = m.ConfidenceStepSlot(7, c)
slot._pm[7] = 3
slot._pmn[7] = 1
slot.value = {3: 1, 7: 1, 15: 2}.__getitem__
assert slot._recompute() and slot.current_steps == 15 and slot._grace == 40
slot._pmn[15] = 1
slot.value = {3: 1, 7: 2, 15: 1}.__getitem__
assert slot._recompute() and slot.current_steps == 7 and slot._grace == 80
slot.value = {3: 2, 7: 1, 15: 1}.__getitem__
assert slot._recompute() and slot.current_steps == 3 and slot._grace == 40
slot = m.ConfidenceStepSlot(7, c)
slot.value = {3: 1, 7: 1, 15: 2}.__getitem__
for _ in range(24):
    assert not slot.update([7])  # 15 warmup + decision interval 10
assert slot.update([7]) and slot.current_steps == 15
for _ in range(39):
    assert not slot.update([0])
print('PASS reversal doubles grace; continuation resets grace; warmup, interval and post-switch grace enforced.')

# Equal weight on five widths and both workload measurements; W4b is drift only.
ss = [3, 5, 7, 11, 15]
ys = {'code-edit': [9.47, 11.18, 12.01, 14.63, 16.05],
      'prose-en': [9.16, 10.82, 11.49, 14.16, 16.00]}
points = [(s, y) for values in ys.values() for s, y in zip(ss, values)]
xbar = statistics.mean(s for s, y in points)
ybar = statistics.mean(y for s, y in points)
b = sum((s-xbar)*(y-ybar) for s, y in points) / sum((s-xbar)**2 for s, y in points)
a = ybar - b*xbar
residuals = {w: [y-a-b*s for s, y in zip(ss, values)] for w, values in ys.items()}
rmse = math.sqrt(statistics.mean(r*r for rr in residuals.values() for r in rr))
result = dict(A=a, B=b, S=ss, observed=ys, residuals=residuals, rmse=rmse)
(BENCH / 'specs/wa2-fit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
print('CPU RESULT: PASS. No production code change needed; no worktree or branch created.')
