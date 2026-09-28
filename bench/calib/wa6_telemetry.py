#!/usr/bin/env python3
"""Summarize recorded telemetry only; does not contact the GPU."""
import datetime,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'specs/wa6'
manifest=json.loads((ROOT/'final-arms.json').read_text());result={}
for policy in ('current','prior'):
 for name in manifest[policy]:
  rs=[json.loads(l) for l in (Path(name)/'memory.jsonl').open()];rs=[r for r in rs if r['phase']=='steady'];active=[r for r in rs if float(r['gpu'][1])>=10]
  result[Path(name).name]=dict(policy=policy,steady_samples=len(rs),steady_min_free_mib=min(r['free_mib'] for r in rs),max_temperature_c=max(float(r['gpu'][2]) for r in rs),active_sm_clock_median_mhz=statistics.median(float(r['gpu'][4]) for r in active),active_memory_clock_median_mhz=statistics.median(float(r['gpu'][5]) for r in active),sw_power_cap_active_samples=sum(r['gpu'][6]=='Active' for r in rs),hw_thermal_active_samples=sum(r['gpu'][7]=='Active' for r in rs),other_gpu_app_present_samples=sum(bool(r.get('other_gpu_app')) for r in rs))
  reqs=[json.loads(l) for l in (Path(name)/'requests.jsonl').open()]
  windows=[(r['wall_start']+r['ttft_s'],r['wall_start']+r['ttft_s']+r['decode_s']) for r in reqs]
  decode=[r for r in rs if any(a<=datetime.datetime.fromisoformat(r['t']).timestamp()<=b for a,b in windows)]
  result[Path(name).name].update(sampled_decode_points=len(decode),sampled_decode_power_cap_active=sum(r['gpu'][6]=='Active' for r in decode),sampled_decode_thermal_active=sum(r['gpu'][7]=='Active' for r in decode))
(ROOT/'telemetry-summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['\n## Recorded timed-arm telemetry\n','| arm | min free MiB | max temperature C | active median SM MHz | memory MHz | power-cap / thermal active samples | other-GPU-app present samples | decode power-cap / samples |','|---|---:|---:|---:|---:|---:|---:|---:|']
for label,r in result.items():
 lines.append(f"| {label} | {r['steady_min_free_mib']} | {r['max_temperature_c']:.0f} | {r['active_sm_clock_median_mhz']:.0f} | {r['active_memory_clock_median_mhz']:.0f} | {r['sw_power_cap_active_samples']} / {r['hw_thermal_active_samples']} | {r['other_gpu_app_present_samples']} | {r['sampled_decode_power_cap_active']} / {r['sampled_decode_points']} |")
lines.append('\nTelemetry is sampled every approximately 2 seconds across the full ready-state arm, including validation requests. Decode samples are matched to client wall-clock first/last-token intervals; coarse sampling does not prove absence between samples. Active-clock medians use samples with GPU utilization >=10%. These are context diagnostics, not time-aligned per-kernel measurements or a claim of identical background desktop activity.\n')
text='\n'.join(lines);print(text)
p=ROOT.parent/'WA6_REQUEST_PRIOR.md'
existing=p.read_text()
if '\n## Recorded timed-arm telemetry\n' in existing:
 start=existing.index('\n## Recorded timed-arm telemetry\n');end=existing.find('\n## ',start+1)
 if end<0:end=len(existing)
 p.write_text(existing[:start]+text+existing[end:])
else:
 with p.open('a') as f:f.write(text)
