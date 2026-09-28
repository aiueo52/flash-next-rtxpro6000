"""CPU policy cases and full P3 decision equivalence; no GPU initialization."""
import json
from pathlib import Path
import numpy as np
from contrib_policy_sim import load_capture, ContributionReplay, proxy_score
from p4_policy import reference

ROOT = Path(__file__).resolve().parents[2]

def main():
    manifest = json.loads((ROOT/'prune/p4-manifest.json').read_text())
    result = []
    for width in (4, 16):
        entry = manifest['widths'][str(width)]
        table = np.load(ROOT/'prune'/entry['file'])
        p3_table = np.load(ROOT/f'bench/quality/runs/p3-contrib-20260907-2300/proxy-w{width}.npz')['norm_table']
        for source in entry['sources']:
            c = load_capture(ROOT/source['file'])
            inv = np.float32(1)/c.h.astype(np.float32)
            for joint in (False, True):
                runtime_score = (c.w.astype(np.float32) * table[c.layer[:, None, None], c.ids]) * inv[..., None]
                replay = ContributionReplay(c, runtime_score)
                replay.group_score = replay.group_score.astype(np.float32)
                expected = replay.drop('joint' if joint else 'singleton', float(np.float32(entry['threshold'])))
                legacy = ContributionReplay(c, proxy_score(c, p3_table)).drop(
                    'joint' if joint else 'singleton', entry['p3_threshold'])
                got = reference(c.ids, c.w, table[c.layer], inv, entry['threshold'], joint)
                mismatch = int((got != expected).sum())
                result.append(dict(width=width, file=c.path.name, joint=joint,
                    calls=len(c.ids), routes=c.ids.size, mismatches=mismatch,
                    p3_float64_table_arithmetic_drift=int((got != legacy).sum())))
                assert mismatch == 0 and result[-1]['p3_float64_table_arithmetic_drift'] == 0, result[-1]
    ids = np.arange(40, dtype=np.int32).reshape(4, 10)
    w = np.full((4, 10), .01, dtype=np.float32); w[:, 0] = 1
    ids[1, 1] = ids[0, 1]  # pair with sum .02
    table = np.ones(512, dtype=np.float32); inv = np.ones(4, dtype=np.float32)
    drop = reference(ids, w, table, inv, .015, True)
    assert not drop[:, 0].any() and not drop[0, 1] and not drop[1, 1]
    assert reference(ids, w, table, inv, .03, True)[0, 1]
    bad = inv.copy(); bad[0] = np.nan
    assert not reference(ids, w, table, bad, .03).any()
    bad_ids = ids.copy(); bad_ids[3, 9] = -1
    assert not reference(bad_ids, w, table, inv, .03).any()
    print(json.dumps(dict(status='PASS', synthetic_cases=5, captures=result), indent=2))

if __name__ == '__main__':
    main()
