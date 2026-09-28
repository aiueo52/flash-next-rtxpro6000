"""C1 policy regression guard -- CPU only, runs against the recorded traces.

Asserts the two properties the estimator is built on, and that the shipped
config decides correctly at both fixed points for all four workloads.
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import json, os, statistics, sys

D = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("SGLANG_PYTHON"):  # python/ of a flash-next-fast checkout; else the installed sglang
    sys.path.insert(0, os.environ["SGLANG_PYTHON"])
from sglang.srt.speculative.adaptive_confidence import (  # noqa: E402
    ConfidenceStepSlot, step_time_ms)

def segs(prof):
    rows = [json.loads(l) for l in open(f"{D}/trace-{prof}.jsonl.rank0")]
    out, prev = {}, 0
    for line in open(f"{D}/segments-{prof}.txt"):
        w, n = line.split(); out[w] = rows[prev:int(n)]; prev = int(n)
    return out

cfg = json.load(open(f"{D}/../w16_conf.json"))["1"]
w4, w16 = segs("w4"), segs("w16")
fail = 0

# 1. downward counterfactual is exact: mean(min(a_15,3)) ~ measured mean at W4
print("downward (min(a,3) from W16 vs measured W4):")
for w in w16:
    pred = statistics.mean(min(r["a"][0], 3) for r in w16[w])
    meas = statistics.mean(r["a"][0] for r in w4[w])
    bad = abs(pred / meas - 1) > 0.25
    fail += bad
    print(f"  {w:11s} {pred:5.2f} vs {meas:5.2f} ({pred/meas-1:+6.1%})"
          + ("  <-- FAIL" if bad else ""))

# 2. the policy settles on the better fixed profile from both fixed points
BETTER = {"code-edit": 15, "prose-en": 3, "prose-ja": 3, "agent-loop": 3}
print("\ndecision from each starting state (600 batches of the recorded trace):")
for w in w16:
    for start in (15, 3):
        slot = ConfidenceStepSlot(15, dict(cfg))
        slot.current_steps = start
        # Always replay the W16 trace: min(a_15, S) is the exact outcome any
        # shorter chain would have had, whereas the W4 trace is censored at 3
        # and so cannot say what 15 would have accepted.
        rows = w16[w]
        for i in range(600):
            r = rows[i % len(rows)]
            if r["p"] and r["p"][0][0] > 0:
                slot.observe_confidence([r["p"][0][0]])
            slot.update([min(r["a"][0], slot.current_steps)])
        bad = slot.current_steps != BETTER[w]
        fail += bad
        print(f"  {w:11s} start={start:2d} -> {slot.current_steps:2d} "
              f"(want {BETTER[w]:2d}) E={{3:{slot.expected_accept_for(3):.2f}, "
              f"15:{slot.expected_accept_for(15):.2f}}}"
              + ("  <-- FAIL" if bad else ""))

print("\nRESULT:", "PASS" if fail == 0 else f"FAIL ({fail})")
sys.exit(1 if fail else 0)
