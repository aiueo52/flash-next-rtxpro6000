#!/usr/bin/env python3
"""Assemble final review evidence after the frozen sequence; CPU only."""
import json
from pathlib import Path
from wa6_analyze import read, metric
ROOT=Path(__file__).resolve().parents[1]/'specs/wa6'
if not (ROOT/'sequence-complete').exists():raise SystemExit(f'wa6_finalize.py: frozen sequence is incomplete (no {ROOT}/sequence-complete); WA6 run directories (specs/wa6) are not published in this repository (results/ keeps only stripped records)')
assert (ROOT/'sequence-complete').exists(), 'Frozen sequence is incomplete'
gate=json.loads((ROOT/'gate.json').read_text())
mechanism=json.loads((ROOT/'mechanism.json').read_text())
audit=json.loads((ROOT/'audit.json').read_text())
manifest=json.loads((ROOT/'final-arms.json').read_text())
rows={p:[r for name in names for r in read(Path(name)/'requests.jsonl') if r['split']=='eval'] for p,names in manifest.items()}
metrics={p:metric(rs) for p,rs in rows.items()}
source_files=['adaptive_confidence.py','adaptive_runtime_state.py','adaptive_request_prior.py','eagle_worker_v2.py','wa6_client.py','eval-stream.jsonl']
source_sets={}
for p in ('current','prior'):
 for name in manifest[p]:
  hashes=json.loads((Path(name)/'source-hashes.json').read_text())
  source_sets[Path(name).name]=hashes
# Compare only implementation files, not deliberately different config files.
source_parity={}
for f in source_files:
 values=[v for h in source_sets.values() for k,v in h.items() if k.endswith('/'+f)]
 source_parity[f]=len(values)==4 and len(set(values))==1
assert len(rows['current'])==len(rows['prior'])>0, 'policy arms must be paired and non-empty'
assert len({len(rows[p]) for p in ('fixed4','fixed8','fixed16')})==1 and len(rows['fixed4'])>0, 'fixed-width references must be equal-sized and non-empty'
cleanup=json.loads((ROOT/'process-cleanup-audit.json').read_text())
valid=(cleanup['all_owned_groups_stopped'] and all(audit['production_hashes_unchanged'].values()) and not audit['model_stat_changes'] and audit['under_gpu_budget'] and audit['under_disk_budget'] and audit['parent_disjoint'] and all(source_parity.values()))
progress=json.loads((ROOT/'sequence-progress.json').read_text())
prior_diag=json.loads((Path(progress['prior-diagnostic'])/'summary.json').read_text())['diagnostics']
expected={r['id']:(r['designation'] if r['designation'] is not None else r['predictor']) for r in read(ROOT/'eval-stream.jsonl')}
task_mismatches=[dict(id=k,expected=v,actual=prior_diag[k]['task']) for k,v in expected.items() if v!=prior_diag[k]['task']]
valid=valid and not task_mismatches
passed=gate['numerical_gate'] and mechanism['mechanism_supported'] and valid
result=dict(all_owned_server_groups_stopped=cleanup['all_owned_groups_stopped'],frozen_task_prediction_mismatches=task_mismatches,decision='PASS' if passed else 'FAIL — do not adopt',metrics=metrics,numerical_gate=gate['numerical_gate'],mechanism_supported=mechanism['mechanism_supported'],audit_valid=valid,timed_implementation_hashes_identical=source_parity,production_change='Unapplied change proposal required' if passed else 'None; retain current config c + STEP 7.943/0.5554 + X3 autotune fix',scope='BS1 mixed stream; two repeats/policy; fixed references one repeat; no general output-quality equivalence claim')
(ROOT/'final-summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['\n## Final decision and audit\n',f"Decision: **{result['decision']}**. Numerical gate {gate['numerical_gate']}; initial-width/dwell diagnostic support {mechanism['mechanism_supported']}; artifact/environment audit {valid}.",'','The following table uses evaluation parents only. Current/prior each contain two complete replays of the same parent stream; fixed references each contain one. Decode tokens exclude the initial streaming chunk; completion tokens include it. Throughput uses actual decode tokens and decode seconds. Early32 is latency from the first streaming chunk to 32 additional delivered tokens; naturally shorter outputs are excluded from this mean, and eligible counts are retained in final-summary.json.','', '| arm | requests | completion tokens | decode tokens | decode s | delivered t/s | early32 ms | TTFT ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
for p in ('current','prior','fixed4','fixed8','fixed16'):
 m=metrics[p];lines.append(f"| {p} | {m['requests']} | {m['completion_tokens']} | {m['decode_tokens']} | {m['decode_s']:.4f} | {m['tps']:.3f} | {1000*m['early32_mean_s']:.3f} | {1000*m['ttft_mean_s']:.3f} |")
lines+=['',f"Frozen task/caller outcomes match the {len(expected)} evaluation inputs: {not task_mismatches}. All four timed policy arms use identical implementation/input/client hashes: {source_parity}. Production manifest unchanged: {all(audit['production_hashes_unchanged'].values())}; {audit['model_manifest_files']} model-file stat/symlink records unchanged: {not audit['model_stat_changes']}. Branch {audit['branch']}, base HEAD {audit['head']}; no commit/push. Occupied GPU-lock time including the invalid first preparation arm: {audit['gpu_lock_hours']:.4f} hours. New saved artifacts including private cache: {audit['disk_bytes']/1e9:.3f} GB. All completed arm VRAM/needle evidence is in audit.json; timed current/prior reserve+needle gate: {gate['reserve_and_needle']}.",'', 'Interpretation limits: fixed-reference greedy trajectories and natural EOS differ across widths; fixed-best labels are optimistic diagnostics, never policy inputs. Timed throughput is uninstrumented; occupancy/switch/recovery events come from separate matched diagnostic replays. The small block count and measured confidence intervals bound generalization. The live overlap smoke is not a test of bit-exact cross-width outputs or a general NI-quality battery. The -2% domain tolerance was predeclared as the provisional default in the absence of a user override.','']
if not passed:
 lines+=['Production change: **none**. Keep candidates [3,7,15], adaptive/w16_3_7_15_c.json, STEP 7.943/0.5554 and the X3 autotune fix. The WA6 code remains opt-in and uncommitted in the named worktree. Do not add W12 or retune this predictor on the final holdout.','']
text='\n'.join(lines)+'\n'
(ROOT/'final-review.md').write_text(text)
p=ROOT.parent/'WA6_REQUEST_PRIOR.md';s=p.read_text();start=s.index('Status:');end=s.index('\n',start);s=s[:start]+f"Status: COMPLETE — {result['decision']}. No production change, commit or push. Final tables and gate evidence are below."+s[end:]
if '\n## Final decision and audit\n' in s:s=s[:s.index('\n## Final decision and audit\n')]
p.write_text(s+text)
print(text)
