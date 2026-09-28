#!/usr/bin/env python3
"""Paired prompt-cluster bootstrap and prospective paired-t power.

Requires numpy/scipy only for this analysis tool, not for fnbench.
python paired_ab.py --schedule ABBA --arms A1.jsonl B1.jsonl B2.jsonl A2.jsonl
All files must contain the same complete prompt/repeat grid; never intersect away missing rows.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

METRICS = ('tps', 'acceptance')


def read_arm(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        key = (row['workload'], row['prompt_id'], row['repeat'])
        if key in rows:
            raise ValueError(f'duplicate prompt/repeat: {path}: {key}')
        for field in ('prompt_sha256', 'max_tokens', 'sampling', 'model', 'prompt_set'):
            if field not in row:
                raise ValueError(f'missing matching field {field}: {path}')
        if not row['prompt_set']:
            raise ValueError('paired analysis requires explicit prompt sets')
        rows[key] = row
    if not rows:
        raise ValueError(f'empty arm {path}')
    return rows


def matched(paths):
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError('the same arm file was supplied more than once')
    arms = [read_arm(p) for p in paths]
    keys = set(arms[0])
    for arm in arms[1:]:
        if set(arm) != keys:
            raise ValueError('arm prompt/repeat grids differ; refusing incomplete pairs')
        if list(arm) != list(arms[0]):
            raise ValueError('request order differs between arms')
        for key in keys:
            for field in ('prompt_sha256', 'max_tokens', 'sampling', 'model', 'prompt_set', 'engine'):
                if arm[key][field] != arms[0][key][field]:
                    raise ValueError(f'mismatched {field} for {key}')
    for domain in {k[0] for k in keys}:
        grids = {}
        for d, pid, repeat in keys:
            if d == domain:
                grids.setdefault(pid, set()).add(repeat)
        hashes = {pid: next(row['prompt_sha256'] for key,row in arms[0].items() if key[:2] == (domain,pid)) for pid in grids}
        if len(set(hashes.values())) != len(hashes):
            raise ValueError('different prompt IDs reuse identical prompt content')
        expected = next(iter(grids.values()))
        if expected != set(range(1, max(expected)+1)) or any(v != expected for v in grids.values()):
            raise ValueError('unbalanced or noncontiguous repeats')
        if len(grids) < 2:
            raise ValueError('need at least two independent prompts per domain')
    return arms


def metric(row, name):
    if name == 'tps':
        x = row['client']['decode_tps']
    else:
        a = row['server'].get('acceptance', {})
        if a.get('status') != 'ok':
            raise ValueError('invalid/missing Prometheus acceptance; no SSE/log fallback')
        x = a['tokens_per_verify']
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError(f'invalid {name}: {x}')
    return float(x)


def matrix(arms, domain, name):
    pids = sorted({k[1] for k in arms[0] if k[0] == domain})
    # Repeats are technical replicates, collapsed within prompt in log space.
    return pids, np.array([[np.mean([math.log(metric(row, name)) for k, row in arm.items()
                                     if k[:2] == (domain, pid)]) for pid in pids] for arm in arms])


def bootstrap(x, alpha=.05, draws=20000, seed=20260908):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return dict(log_mean=float(x.mean()), change_pct=float(100*np.expm1(x.mean())),
                ci_pct=[float(100*np.expm1(v)) for v in np.quantile(means, [alpha/2, 1-alpha/2])],
                lower_one_sided_pct=float(100*np.expm1(np.quantile(means, alpha))),
                upper_one_sided_pct=float(100*np.expm1(np.quantile(means, 1-alpha))),
                n_prompts=len(x))


def power(effect_log, se, df, alpha=.05):
    if se == 0:
        return 1.0 if effect_log > 0 else alpha
    return float(stats.nct.sf(stats.t.ppf(1-alpha, df), df, effect_log/se))


def mde(se, df, alpha=.05, target=.80):
    """One-sided noncentral-t planning effect; variance is a plug-in estimate."""
    if se == 0:
        return 0.0
    critical = optimize.brentq(lambda nc: power(nc, 1.0, df, alpha)-target, 0, 100)
    return float(100*np.expm1(critical*se))


def analyze(paths, schedule='ABAB', alpha=.05, draws=20000, seed=20260908):
    if schedule not in ('ABAB', 'ABBA') or len(paths) < 4 or len(paths) % 4:
        raise ValueError('complete four-arm ABAB or ABBA cycles required')
    cycles = len(paths) // 4
    expanded_schedule = schedule * cycles
    arms = matched(paths)
    result = dict(schedule=schedule, cycles=cycles, alpha=alpha, bootstrap_draws=draws, seed=seed,
                  files=[dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths],
                  estimand='equal-weight mean per-prompt log B/A, averaged over all adjacent blocks',
                  uncertainty_scope='Bootstrap resamples prompts, preserving all blocks and collapsed repeats. Conditional on the observed restarts; common restart shifts are not independent prompts. Inspect drift and variance-study restart-aware MDE.',
                  domains={})
    ai = [i for i, c in enumerate(schedule) if c == 'A']
    for domain in sorted({k[0] for k in arms[0]}):
        result['domains'][domain] = {}
        for name in METRICS:
            pids, y = matrix(arms, domain, name)
            pairs = [(i if expanded_schedule[i]=='A' else i+1, i+1 if expanded_schedule[i]=='A' else i) for i in range(0,len(paths),2)]
            blocks = np.array([y[b]-y[a] for a,b in pairs])
            d = blocks.mean(axis=0)
            summary = bootstrap(d, alpha, draws, seed)
            summary['block_means'] = [dict(block=i+1, log_ratio=float(v.mean()), change_pct=float(100*np.expm1(v.mean()))) for i,v in enumerate(blocks)]
            summary['per_prompt'] = [dict(prompt_id=pid, log_ratios=blocks[:,i].tolist(), mean_log_ratio=float(d[i])) for i,pid in enumerate(pids)]
            summary['mde_pct_80_power'] = mde(float(d.std(ddof=1)/math.sqrt(len(d))), len(d)-1, alpha)
            summary['power_at_3pct_plugin'] = power(math.log1p(.03), float(d.std(ddof=1)/math.sqrt(len(d))), len(d)-1, alpha)
            drifts = []
            for cycle in range(cycles):
                drift = bootstrap(y[4*cycle+ai[1]]-y[4*cycle+ai[0]], alpha, draws, seed)
                drift['flag'] = abs(drift['change_pct']) > 3 or drift['ci_pct'][0] > 0 or drift['ci_pct'][1] < 0
                drifts.append(dict(cycle=cycle+1, **drift))
            summary['A2_over_A1_drift'] = drifts[0]
            summary['control_drift_by_cycle'] = drifts
            cycle_effects = blocks.reshape(cycles,2,len(pids)).mean(axis=1)
            summary['cycle_mean_log_ratios'] = cycle_effects.mean(axis=1).tolist()
            summary['crossed_cycle_prompt_interval'] = None
            if cycles > 1:
                e = cycle_effects-cycle_effects.mean(axis=0)-cycle_effects.mean(axis=1,keepdims=True)+cycle_effects.mean()
                residual = float(np.sum(e**2)/((cycles-1)*(len(pids)-1)))
                parent = max(0, float(np.var(cycle_effects.mean(axis=0),ddof=1))-residual/cycles)
                common = max(0, float(np.var(cycle_effects.mean(axis=1),ddof=1))-residual/len(pids))
                se = math.sqrt(parent/len(pids)+common/cycles+residual/(cycles*len(pids)))
                df = min(cycles-1,len(pids)-1)
                mu = float(cycle_effects.mean())
                summary['crossed_cycle_prompt_interval'] = dict(
                    method='crossed cycle + parent random-effects plug-in t interval; conservative min df',
                    cycles=cycles, df=df, se_log=se,
                    lower_one_sided_pct=float(100*np.expm1(mu-stats.t.ppf(1-alpha,df)*se)),
                    ci_pct=[float(100*np.expm1(mu+sign*stats.t.ppf(1-alpha/2,df)*se)) for sign in (-1,1)])
            result['domains'][domain][name] = summary
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', required=True, type=Path)
    p.add_argument('--schedule', choices=['ABAB','ABBA'], default='ABAB')
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--draws', type=int, default=20000)
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if not 0 < a.alpha < .5 or a.draws < 1000: p.error('alpha must be (0,.5); draws >=1000')
    try: result = analyze(a.arms, a.schedule, a.alpha, a.draws, a.seed)
    except (ValueError, KeyError) as exc: p.error(str(exc))
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if a.out: a.out.write_text(text)
    else: print(text, end='')

if __name__ == '__main__': main()
