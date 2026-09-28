#!/usr/bin/env python3
"""Render the machine-readable variance study into the incremental BN1 spec."""
import argparse
import json
import re
from pathlib import Path


def render(result):
    lines=['## Measured variance results', '',
        'All values below are percentages; acceptance is completed tokens per verify. Within-restart CV is the mean of the three per-restart raw CVs (each retained in JSON); it includes parent difficulty. Residual CV removes the fixed parent effect.', '',
        '| Profile | Domain | Mean within raw CV t/s / accept | Between restart mean CV t/s / accept | Residual CV t/s / accept | Restart component CV t/s / accept |',
        '|---|---|---:|---:|---:|---:|']
    for profile,domains in result['profiles'].items():
        for domain,metrics in domains.items():
            pair=lambda k: ' / '.join(f'{metrics[m][k]:.2f}' for m in ('tps','acceptance'))
            lines.append(f'| {profile} | {domain} | {pair("within_restart_raw_cv_mean_pct")} | {pair("between_restart_raw_cv_pct")} | {pair("within_restart_residual_cv_pct")} | {pair("between_restart_component_cv_pct")} |')
    lines+=['', '### MDE at 80% plug-in power, one-sided alpha=.05', '',
        'Each cell is throughput / acceptance. Conditional = prompt residual only. Restart-aware includes common restart variance with 2 pilot restart df. n=16 is extrapolated. These estimates do not certify achieved power.', '',
        '| Profile | Domain | n=8 conditional | n=16 conditional | n=8 restart-aware | n=16 restart-aware |',
        '|---|---|---:|---:|---:|---:|']
    for profile,domains in result['profiles'].items():
        for domain,metrics in domains.items():
            cells=[' / '.join(f'{metrics[m]["mde"][str(n)][k]:.2f}' for m in ('tps','acceptance'))
                   for k,n in [('conditional_pct',8),('conditional_pct',16),('restart_aware_pct',8),('restart_aware_pct',16)]]
            lines.append(f'| {profile} | {domain} | '+' | '.join(cells)+' |')
    lines+=['', '### Prospective 3% planning options', '',
        'Each cell is the required independent four-arm cycles for throughput / acceptance, reaching 80% model-based power with conservative pilot df=2. Requests per domain = 4 × prompts × cycles (one repeat); running all four domains multiplies this by four. These are planning options, not authorization for additional GPU runs or a guarantee of real-world power.', '',
        '| Profile | Domain | 8 prompts: cycles t/s / accept | 16 prompts: cycles t/s / accept |',
        '|---|---|---:|---:|']
    for profile,domains in result['profiles'].items():
        for domain,metrics in domains.items():
            opts={m:{x['prompts']:x for x in metrics[m]['planning_options']}
                  for m in ('tps','acceptance')}
            cells=[' / '.join(str(opts[m][n]['four_arm_cycles'])
                              for m in ('tps','acceptance')) for n in (8,16)]
            lines.append(f'| {profile} | {domain} | '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--spec',type=Path)
    a=p.parse_args()
    if not a.result.is_file():
        raise SystemExit(f'render_variance.py: no variance result at {a.result} (produce it with variance.py --out; variance results are not published in this repository, but variance.py runs on results/bn1-study-v2)')
    text=render(json.loads(a.result.read_text()))
    if a.spec:
        original=a.spec.read_text()
        if '\n## Measured variance results\n' in original:
            original=re.sub(r'\n## Measured variance results\n.*?(?=\n## |\Z)',
                            lambda match: '\n'+text+'\n',original,flags=re.S)
        else:
            original=original.rstrip()+'\n\n'+text
        a.spec.write_text(original)
    else: print(text,end='')

if __name__=='__main__': main()
