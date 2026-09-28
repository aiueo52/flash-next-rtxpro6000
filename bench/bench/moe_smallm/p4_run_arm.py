"""One P4 arm; caller holds the GPU flock throughout speed and D census servers."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent/'sglang-p4'
PY = ROOT.parent/'sglang-rtxpro6000/.venv/bin/python'

def run(cmd, log, **kwargs):
    with Path(log).open('w') as f:
        subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, check=True, **kwargs)

def gpu(field):
    return subprocess.check_output(['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'], text=True).strip()

def stop(p):
    if p is None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        pass
    # A parent can exit before children; only our original PGID is eligible.
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()

def serve(profile, directory, env):
    existing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if existing:
        raise RuntimeError('existing compute application despite GPU lock; no processes killed')
    directory.mkdir(parents=True, exist_ok=False)
    log = directory/'server.log'
    fp = log.open('w')
    p = subprocess.Popen([str(REPO/'serve-fast.sh'), profile, '--max-running-requests', '1'],
        stdout=fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    fp.close()
    try:
        for _ in range(225):
            time.sleep(4)
            if 'ready to roll' in log.read_text(errors='replace'):
                free = int(gpu('memory.free'))
                (directory/'ready.json').write_text(json.dumps(dict(free_MiB=free, pid=p.pid, at=time.time())))
                if free < 4096:
                    raise RuntimeError(f'steady free VRAM {free} < 4096 MiB')
                return p
            if p.poll() is not None:
                raise RuntimeError(f'server exited {p.returncode}: {log}')
        raise RuntimeError(f'server readiness timeout: {log}')
    except BaseException:
        stop(p)
        raise

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label'); ap.add_argument('profile', choices=['w4','w16','wa'])
    ap.add_argument('arm', choices=['control','contrib','control2','prod','noprune','prod2'])
    ap.add_argument('--quality', action='store_true')
    a = ap.parse_args()
    out = ROOT/'runs/p4'/f'{a.label}-{a.profile}-{a.arm}'
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MEM_FRACTION='0.920', W16_MEM_FRACTION='0.920', WA_MEM_FRACTION='0.920',
        SERVE_DISPLAY_HZ='', MAMBA_SLOTS='10', MAX_TOTAL_TOKENS='131072',
        SGLANG_MOE_PRUNE_SINGLETON_TAU='0' if a.arm=='noprune' else '0.08',
        FLASHINFER_MOE_PRUNE_SINGLETON_TAU='0' if a.arm=='noprune' else '0.08',
        SGLANG_MOE_PRUNE_IN_PROLOGUE='1', PYTHONDONTWRITEBYTECODE='1', BS='1', CONC='1')
    if a.arm == 'contrib': env['SGLANG_MOE_PRUNE_POLICY'] = 'contrib'
    else: env.pop('SGLANG_MOE_PRUNE_POLICY', None)
    env.pop('SGLANG_P4_CENSUS', None)
    (out/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('SGLANG_', 'FLASHINFER_', 'MEM_', 'W16_', 'WA_', 'SERVE_', 'MAX_TOTAL', 'MAMBA_', 'TARGET_', 'PYTHONPATH'))}, indent=2))
    print('P4_ARM_START', out, time.ctime(), flush=True)
    for diagnostic in ([False] if a.quality else [False, True]):
        directory = out/('census' if diagnostic else 'timing')
        if diagnostic: env['SGLANG_P4_CENSUS'] = str(directory/'recorder')
        server = watcher = None
        try:
            server = serve(a.profile, directory, env)
            watcher = subprocess.Popen([str(PY), str(ROOT/'prof/p3_memory_watch.py'), str(server.pid),
                str(directory/'memory.log'), str(directory/'memory-failed.txt')])
            if a.quality:
                quality_arm = f'{a.label}-{a.arm}'
                for phase in ('pre','post'):
                    if phase == 'post':
                        smoke = quality_arm+'-smoke'
                        run([PY, ROOT/'bench/quality/run_bench.py', smoke], directory/'smoke.log', env=dict(env, LIMIT='3'))
                        summary = json.loads((ROOT/'bench/quality/runs'/smoke/'summary.json').read_text())
                        assert all(v.get('errors', 0)==0 and v.get('mean_gen_tokens', 1)>0 for v in summary['bench'].values())
                        run([PY, ROOT/'bench/quality/run_bench.py', quality_arm], directory/'quality.log', env=env)
                    for depth in (.1, .5, .9):
                        run([PY, ROOT/'prof/needle_test.py', 18500, depth, f'{quality_arm}-{phase}-{depth}'], directory/f'needle-{phase}-{depth}.log', env=env)
            elif not diagnostic:
                # Warm the full workload mix before timed repeats.
                for workload in ('code-edit','prose-en','prose-ja','agent-loop'):
                    for repeat in range(3):
                        label = f'{a.label}-{a.profile}-{a.arm}-{workload}-r{repeat}'
                        logpath = directory/'server.log'
                        offset = logpath.stat().st_size
                        run([ROOT/'.venv-review/bin/python', '-m', 'fnbench', 'run',
                            '--endpoint','http://127.0.0.1:8001/v1','--engine','sglang',
                            '--workloads',workload,'--repeats','1','--sampling','greedy',
                            '--allow-proc','sglang','--label',label,'--out',str(directory/f'{workload}-r{repeat}.jsonl')],
                            directory/f'{workload}-r{repeat}.client.log', cwd=ROOT, env=env)
                        with logpath.open('rb') as f: f.seek(offset); lines=f.read().decode(errors='replace')
                        accepts = [float(v) for v in re.findall(r'accept len:\s*([\d.]+)', lines)]
                        (directory/f'{workload}-r{repeat}.accept.json').write_text(json.dumps(dict(samples=accepts,
                            mean=sum(accepts)/len(accepts) if accepts else None)))
                for workload in ('code-edit','prose-en'):
                    for repeat in range(3):
                        trace = directory/f'trace-{workload}-r{repeat}'
                        run([PY, ROOT/'prof/profile_decode2.py', trace, 30, ROOT/f'workloads/{workload}.txt'],
                            directory/f'trace-{workload}-r{repeat}.log', env=env)
                        assert list(trace.glob('*.gz')), trace
                for depth in (.1, .5, .9):
                    run([PY, ROOT/'prof/needle_test.py',18500,depth,f'{a.label}-{a.profile}-{a.arm}-{depth}'],directory/f'needle-{depth}.log',env=env)
            else:
                for workload in ('code-edit','prose-en','prose-ja','agent-loop'):
                    prefix = directory/'recorder'
                    run([PY, ROOT/'bench/moe_smallm/p4_control.py',prefix,'reset'],directory/f'{workload}.reset.log')
                    run([ROOT/'.venv-review/bin/python','-m','fnbench','run','--endpoint','http://127.0.0.1:8001/v1',
                        '--engine','sglang','--workloads',workload,'--repeats','1','--sampling','greedy',
                        '--allow-proc','sglang','--label',f'{a.label}-D-{a.profile}-{a.arm}-{workload}',
                        '--out',directory/f'{workload}.jsonl'], directory/f'{workload}.log', cwd=ROOT, env=env)
                    run([PY, ROOT/'bench/moe_smallm/p4_control.py',prefix,'dump','--path',directory/f'{workload}.npz'], directory/f'{workload}.D.json')
            if a.quality:
                import shutil
                qout = ROOT/'bench/quality/runs'/quality_arm
                shutil.copy2(directory/'server.log', qout/'server.log')
                (qout/'needle.log').write_text(''.join(p.read_text() for p in sorted(directory.glob('needle-*.log'))))
            assert not (directory/'memory-failed.txt').exists()
        finally:
            if watcher: watcher.terminate(); watcher.wait()
            stop(server)
            time.sleep(5)
    needle_logs = list((out/'timing').glob('needle-*.log'))
    needle_pass = sum('PASS=True' in p.read_text() for p in needle_logs)
    summary = dict(label=a.label, profile=a.profile, arm=a.arm, quality=a.quality,
                   needle_pass=needle_pass, needle_total=len(needle_logs), finished=time.ctime())
    (out/'arm-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (ROOT/'specs/P4_CONTRIB_PRUNE_SHIP.md').open('a') as f:
        f.write(f"\nArm `{out.name}` complete at {summary['finished']}: needle {needle_pass}/{len(needle_logs)}; BS=1, memory .920, private logs `{out.relative_to(ROOT)}`.\n")
    (out/'COMPLETE').write_text(time.ctime()+'\n')
    print('P4_ARM_COMPLETE',out,flush=True)

if __name__ == '__main__':
    main()
