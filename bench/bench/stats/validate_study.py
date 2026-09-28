#!/usr/bin/env python3
"""Verify the completed six-arm BN1 artifact and write a compact audit."""
import argparse
import json
import os
from pathlib import Path

from paired_ab import matched, metric
from run_variance import B, check_frozen, digest, occupied_seconds


def provenance_check(provenance, digest=digest):
    """Source-provenance verification. 'verified' only when a real audit verdict (status PASS) exists and
    every listed source has a hash that still matches; a published method-only audit (no verdict, no
    hashes) or an empty source list is 'unverified', never a pass. A non-PASS audit verdict is an error."""
    if not (provenance.get('method') and provenance.get('scope')):
        raise AssertionError('provenance-audit.json lacks method/scope')
    verdict = provenance.get('status')
    if verdict is not None and verdict != 'PASS':
        raise AssertionError(f'provenance audit verdict is {verdict}')
    sources = provenance.get('sources') or []
    if verdict is None:
        return dict(status='unverified', sources=len(sources),
                    reason='the audit has no verdict (the published copy documents only the method; its result '
                           'was computed on private prompt files). Set PROVENANCE_AUDIT to a local audit_sets.py output.')
    if not sources:
        return dict(status='unverified', sources=0, reason='the audit lists no sources')
    unhashed = [s['path'] for s in sources if 'sha256' not in s]
    if unhashed:
        return dict(status='unverified', sources=len(sources), reason=f'{len(unhashed)} sources carry no hash')
    for source in sources:
        if digest(Path(source['path'])) != source['sha256']:
            raise AssertionError(f"provenance source changed: {source['path']}")
    return dict(status='verified', sources=len(sources), reason='audit verdict PASS and every source hash matches')


def frozen_check(frozen):
    """check_frozen() has already compared every listed hash/inventory entry; empty lists verify nothing."""
    if not frozen.get('hashes') or not frozen.get('model_inventory'):
        return dict(status='unverified', reason='frozen.json lists no input hashes or no model inventory')
    return dict(status='verified', inputs=len(frozen['hashes']), model_files=len(frozen['model_inventory']))


def overall_status(checks, requests, duration_exceptions):
    """PASS only when every check verified something; skipped/empty checks give UNVERIFIED, never PASS."""
    if not requests:
        return 'UNVERIFIED'
    if any(c['status'] != 'verified' for c in checks.values()):
        return 'UNVERIFIED'
    return 'PASS_WITH_DURATION_EXCEPTION' if duration_exceptions else 'PASS'


def validate(root):
    frozen = check_frozen(root)
    result = dict(status='UNVERIFIED', frozen_inputs=len(frozen['hashes']),
                  model_inventory_files=len(frozen['model_inventory']),
                  lock_seconds=occupied_seconds(root), arms={}, duration_by_prompt={},
                  duration_exceptions=[], checks={})
    # The published provenance-audit.json omits the verdict (it was computed on private prompt files) and
    # only documents the method. A local re-run (audit_sets.py --out PATH, kept outside the repository because it
    # names private sources and their hashes) is used instead when PROVENANCE_AUDIT=PATH.
    provenance = json.loads(Path(os.environ.get('PROVENANCE_AUDIT') or B/'workloads/sets/provenance-audit.json').read_text())
    result['checks']['source_provenance'] = provenance_check(provenance)
    result['checks']['frozen_inputs'] = frozen_check(frozen)
    result['provenance_sources_listed'] = result['checks']['source_provenance']['sources']
    for profile in ('w4', 'wa'):
        paths = [root/f'{profile}-A{i}'/'requests.jsonl' for i in (1, 2, 3)]
        arms = matched(paths)
        infos, commands = [], []
        for path, arm in zip(paths, arms):
            folder = path.parent
            assert (folder/'complete.json').exists(), f'incomplete: {folder}'
            assert not (folder/'invalid.json').exists(), f'invalid: {folder}'
            assert len(arm) == 32, f'expected 32 requests: {folder}'
            durations = json.loads((folder/'durations.json').read_text())
            memory = [json.loads(s) for s in (folder/'memory.jsonl').read_text().splitlines()]
            info = json.loads((folder/'server-info.json').read_text())
            command = json.loads((folder/'command.json').read_text())
            command.pop('name')
            assert info['mem_fraction_static'] == .920
            assert command['env']['SERVE_DISPLAY_HZ'] == ''
            assert min(x['free_mib'] for x in memory) >= 4096
            assert info['speculative_adaptive'] == (profile == 'wa')
            for row in arm.values():
                assert row['repeat'] == 1
                metric(row, 'tps'); metric(row, 'acceptance')
                seconds = row['client']['ttft_seconds'] + row['client']['decode_seconds']
                if not 15 <= seconds <= 60:
                    result['duration_exceptions'].append(dict(
                        arm=folder.name, prompt_id=row['prompt_id'], seconds=seconds,
                        completion_tokens=row['client']['usage']['completion_tokens'],
                        max_tokens=row['max_tokens']))
                result['duration_by_prompt'].setdefault(row['prompt_id'], []).append(
                    dict(arm=folder.name, seconds=seconds))
            result['arms'][folder.name] = dict(
                requests=len(arm), random_seed=info['random_seed'],
                min_free_mib=min(x['free_mib'] for x in memory),
                min_steady_free_mib=min(x['free_mib'] for x in memory if x['steady']),
                duration_range_seconds=[min(x['seconds'] for x in durations),
                                        max(x['seconds'] for x in durations)])
            infos.append(info); commands.append(command)
        assert all(c == commands[0] for c in commands), f'launcher drift: {profile}'
        changed = sorted(k for k in infos[0] if any(infos[0][k] != x.get(k) for x in infos[1:]))
        # Production leaves seed unset; SGLang generates it on each startup.
        assert set(changed) <= {'random_seed', 'startup_time', 'internal_states'}, changed
        result.setdefault('server_info_runtime_differences', {})[profile] = changed
    result['requests'] = sum(x['requests'] for x in result['arms'].values())
    result['duration_target_passes'] = result['requests'] - len(result['duration_exceptions'])
    result['status'] = overall_status(result['checks'], result['requests'], result['duration_exceptions'])
    result['seed_scope'] = ('Identical explicit production launch configuration; auto-generated server '
                            'seeds vary by restart. Greedy requests, deterministic inference disabled. '
                            'This measures normal production restart noise, not fixed-seed noise.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not (args.root/'frozen.json').is_file():
        raise SystemExit(f'validate_study.py: {args.root} is not a BN1 study run directory (no frozen.json); '
                         'study run directories are not published in this repository')
    result = validate(args.root)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    unverified = [k for k, c in result['checks'].items() if c['status'] != 'verified']
    print(f"{result['status']}: {result['requests']} requests; {result['frozen_inputs']} frozen inputs; "
          f"{len(result['duration_exceptions'])} duration exceptions retained"
          + (f"; unverified: {', '.join(unverified)}" if unverified else ''))
    if not result['status'].startswith('PASS'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
