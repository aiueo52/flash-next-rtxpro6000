#!/usr/bin/env python3
"""WA3 logged-ratio screen and exact production-default parity, CPU only."""
import importlib.util
import json
from pathlib import Path
import random

B = Path(__file__).resolve().parents[1]

def module(name, tree):
    spec = importlib.util.spec_from_file_location(name, B.parent / tree / 'python/sglang/srt/speculative/adaptive_confidence.py')
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out

_trees = [B.parent / t / 'python/sglang/srt/speculative/adaptive_confidence.py' for t in ('sglang-rtxpro6000', 'sglang-wa2')]
if not all(p.is_file() for p in _trees):
    raise SystemExit('wa3_select.py: needs the production and WA2 SGLang trees next to the bench directory '
                     '(' + ', '.join(str(p) for p in _trees) + '); the SGLang trees are not published in this repository')
production = module('production_confidence', 'sglang-rtxpro6000')
worktree = module('wa3_confidence', 'sglang-wa2')
cfg = json.loads((B / 'adaptive/w16_3_7_15_c.json').read_text())
base = json.loads((B / 'adaptive/w16_conf.json').read_text())
assert cfg['2'] == base['2']
rng = random.Random(12345)
for key, slot_cfg in base.items():
    old = production.ConfidenceStepSlot(15, slot_cfg)
    new = worktree.ConfidenceStepSlot(15, slot_cfg)
    for i in range(3000):
        p = [.9999 if i % 400 < 200 else rng.random()]
        a = [old.current_steps if i % 400 < 200 else rng.randrange(3)]
        old.observe_confidence(p)
        new.observe_confidence(p)
        assert old.update(a) == new.update(a)
        assert old.best_steps() == new.best_steps()
        assert all(getattr(new, k) == v for k, v in vars(old).items())
    print(f'PASS production slot {key}: 3000 updates, decisions, scores and all legacy state fields exactly equal.')

examples = [
    ('b1 code 15->7 14:54:46', 15, {3: 0, 7: 431, 15: 392}, 15),
    ('b1 code 15->7 14:54:56', 15, {3: 0, 7: 489, 15: 437}, 15),
    ('b2 code 15->7 15:02:44', 15, {3: 0, 7: 368, 15: 326}, 15),
    ('b2 code 7->15 15:02:47', 7, {3: 0, 7: 584, 15: 666}, 7),
    ('b2 code 7->15 15:02:58', 7, {3: 0, 7: 631, 15: 746}, 15),
    ('b2 prose 7->3 batch1705', 7, {3: 224, 7: 215, 15: 165}, 3),
    ('b2 prose 7->3 batch2285', 7, {3: 245, 7: 229, 15: 180}, 3),
    ('b1 prose false 3->15 batch1745', 3, {3: 301, 7: 338, 15: 375}, 3),
    ('b2 agent 3->7 15:03:52', 3, {3: 329, 7: 382, 15: 0}, 7),
    ('b2 agent false 7->15 15:04:03', 7, {3: 0, 7: 533, 15: 607}, 7),
]
for name, start, values, want in examples:
    slot = worktree.ConfidenceStepSlot(start, cfg['1'])
    slot.value = values.__getitem__
    got = slot.best_steps()[0]
    assert got == want, (name, got, want)
    print(f'PASS {name}: S={start}, values={values} -> {got}')
print('Pairwise examples use zero for unreported irrelevant candidates; they validate named transition thresholds, not full trajectory predictions.')
print('Thresholds: upward 1.10*1.04=1.144; down to S7=1.15; down to S3=1.03. Grace cap=80; adjacent promotion=true.')
print('FINAL ITERATION: no further configuration/code iteration will be started after its gate.')
