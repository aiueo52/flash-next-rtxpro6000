#!/usr/bin/env python3
"""Crossed prompt/restart ANOVA on log metrics; three controls per profile.
The raw within-restart CV includes prompt difficulty. Residual CV removes it.
With one observation/cell, interaction and request noise cannot be separated.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from paired_ab import METRICS, matched, matrix, mde, power


def cv_log(variance):
    return float(100*np.sqrt(np.expm1(max(0, variance))))


def analyze(paths):
    if len(paths) != 3: raise ValueError('three independent server restarts required')
    arms = matched(paths)
    if any(k[2] != 1 for k in arms[0]): raise ValueError('variance study requires exactly one repeat')
    result = {}
    for domain in sorted({k[0] for k in arms[0]}):
        result[domain] = {}
        for name in METRICS:
            pids, y = matrix(arms, domain, name); r, n = y.shape
            residual = y-y.mean(axis=0)-y.mean(axis=1, keepdims=True)+y.mean()
            sigma2 = float(np.sum(residual**2)/((r-1)*(n-1)))
            tau2 = max(0, float(np.var(y.mean(axis=1), ddof=1))-sigma2/n)
            raw = np.exp(y)
            out = dict(n_prompts=n, restarts=r,
                restart_means=raw.mean(axis=1).tolist(),
                between_restart_raw_cv_pct=float(100*raw.mean(axis=1).std(ddof=1)/raw.mean()),
                within_restart_raw_cv_pct=[float(100*v.std(ddof=1)/v.mean()) for v in raw],
                within_restart_raw_cv_mean_pct=float(np.mean([100*v.std(ddof=1)/v.mean() for v in raw])),
                within_restart_residual_cv_pct=cv_log(sigma2),
                between_restart_component_cv_pct=cv_log(tau2),
                residual_log_variance=sigma2, restart_log_variance=tau2,
                A2_over_A1_pct=float(100*np.expm1((y[1]-y[0]).mean())),
                A3_over_A1_pct=float(100*np.expm1((y[2]-y[0]).mean())), mde={})
            for size in (8,16):
                # 2A+2B: sum squared arm-mean coefficients = 1.
                se = math.sqrt(tau2+sigma2/size)
                out['mde'][str(size)] = dict(
                    conditional_pct=mde(math.sqrt(sigma2/size), size-1),
                    restart_aware_pct=mde(se, 2),
                    power_at_3pct_restart_aware=power(math.log1p(.03),se,2),
                    df=2)
            # A planning option; future independent 4-arm cycles reduce common shifts.
            options=[]
            for size in (8,16):
                # With fixed pilot df, noncentrality scales as sqrt(cycles).
                # Solve directly rather than silently capping noisy cases at 100 cycles.
                single_delta = math.log1p(out['mde'][str(size)]['restart_aware_pct']/100)
                cycles = max(1, math.ceil((single_delta/math.log1p(.03))**2))
                se = math.sqrt((tau2+sigma2/size)/cycles)
                pw = power(math.log1p(.03),se,2)
                options.append(dict(prompts=size, repeats=1, four_arm_cycles=cycles,
                                    requests=4*cycles*size, power_plugin=pw, pilot_df=2))
            out['planning_options'] = options
            out['restart_aggregate_counters'] = []
            for arm in arms:
                rows = [row for key, row in arm.items() if key[0] == domain]
                if all('usage' in row['client'] for row in rows):
                    tokens = sum(row['client']['usage']['completion_tokens'] for row in rows)
                    seconds = sum(row['client']['decode_seconds'] for row in rows)
                    verifies = sum(row['server']['acceptance']['verify_calls'] for row in rows)
                    out['restart_aggregate_counters'].append(dict(
                        requests=len(rows), completion_tokens=tokens, decode_seconds=seconds,
                        aggregate_decode_tps=(tokens-len(rows))/seconds,
                        verify_calls=verifies, aggregate_tokens_per_verify=tokens/verifies))
            result[domain][name] = out
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=dict(status='measured', assumption='Crossed random restart + fixed prompt + residual log model. Plug-in power, three restarts only (2 restart df); 16-prompt values extrapolate exchangeable independent parents. Does not certify 80% real-world power.', profiles={})
    for profile in ('w4','wa'):
        published=[a.root/f'{profile}-A{i}.jsonl' for i in (1,2,3)]
        if all(p.is_file() for p in published):
            # results/bn1-study-v2: the stripped request records of the completed study, one file per arm
            paths=published; result['input']='published stripped records (results/bn1-study-v2 layout)'
        else:
            paths=[a.root/f'{profile}-A{i}'/'requests.jsonl' for i in (1,2,3)]
            incomplete=[str(p.parent) for p in paths if not (p.parent/'complete.json').exists()]
            if incomplete:
                raise SystemExit('variance.py: incomplete or missing arms (no complete.json): '+', '.join(incomplete)
                                 +'; pass a study run directory or the published results/bn1-study-v2')
        result['profiles'][profile]=analyze(paths)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
