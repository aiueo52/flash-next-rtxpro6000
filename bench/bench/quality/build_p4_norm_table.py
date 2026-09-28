"""Rebuild P3 train-only per-width tables and freeze FP32 deployment artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contrib_policy_sim import load_capture, fit_proxy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'prune'
P3 = ROOT / 'bench/quality/runs/p3-contrib-20260907-2300'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if not (P3 / 'proxy-train-calibration.json').is_file():
        raise SystemExit(f'build_p4_norm_table.py: needs the P3 calibration run {P3} and its route captures; they are '
                         'not published in this repository (the frozen tables themselves are in prune/)')
    OUT.mkdir(exist_ok=True)
    cal = json.loads((P3 / 'proxy-train-calibration.json').read_text())
    manifest = dict(schema=1, dtype='float32', shape=[48, 512],
        split='first floor(complete_verify_steps/2) steps per workload train; remaining steps holdout',
        smoothing='8 layer-mean pseudo-observations per expert',
        policy='singleton plus count-two; group SUM strictly below cut; stable top-1 protected',
        widths={})
    for width in (4, 16):
        captures = [load_capture(ROOT / 'runs' / name) for name in cal[str(width)]['workloads']]
        table, counts = fit_proxy([c.subset(~c.split) for c in captures])
        original = np.load(P3 / f'proxy-w{width}.npz')['norm_table']
        np.testing.assert_array_equal(table, original)
        path = OUT / f'p4-norm-w{width}.npy'
        np.save(path, table.astype(np.float32), allow_pickle=False)
        manifest['widths'][str(width)] = dict(file=path.name, sha256=sha(path),
            p3_threshold=cal[str(width)]['threshold'],
            threshold=float(np.nextafter(np.float32(cal[str(width)]['threshold']), np.float32(np.inf))),
            threshold_rounding='one FP32 ULP toward +infinity after nearest rounding; strict-cut boundary preservation',
            min_observations=int(counts.min()),
            sources=[dict(file=str(c.path.relative_to(ROOT)), sha256=sha(c.path),
                train_steps=int((~c.split).sum())//48, holdout_steps=int(c.split.sum())//48)
                for c in captures])
    (OUT / 'p4-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
