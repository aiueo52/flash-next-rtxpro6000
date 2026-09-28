#!/usr/bin/env python3
"""One WA6 arm. Invoke under flock -w 28800 /home/user/.gpu.lock."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
OTHER_GPU_APP = os.environ.get('OTHER_GPU_APP', '').lower()  # substring of another GPU application to log (e.g. a game engine); empty = off

B = Path('/home/user/tools/flash-next-bench')
R = Path('/home/user/tools/sglang-rtxpro6000')
W = Path('/home/user/tools/sglang-wa6')
ROOT = B / 'specs/wa6'
NOTE = B / 'specs/WA6_REQUEST_PRIOR.md'
arm, label, split, mode = sys.argv[1:]
parent = Path(f'/proc/{os.getppid()}')
if (parent/'comm').read_text().strip() != 'flock' or b'/home/user/.gpu.lock' not in (parent/'cmdline').read_bytes().split(b'\0'):
    raise RuntimeError('Invoke this arm directly under the prescribed GPU flock')
assert arm in ('current', 'prior', 'fixed4', 'fixed8', 'fixed16')
assert split in ('prep','eval','reference') and mode in ('timed','diagnostic')
out = ROOT / label
out.mkdir()  # Existing labels are never reused.
py = R / '.venv/bin/python'
env = os.environ.copy()
for k in list(env):
    if k.startswith(('SGLANG_', 'PROF_', 'WA_', 'W16_')) or k in ('PYTHONPATH', 'ADAPTIVE_CONFIG', 'MAX_TOTAL_TOKENS'):
        env.pop(k)
env.update(PYTHONDONTWRITEBYTECODE='1', MEM_FRACTION='0.920', W16_MEM_FRACTION='0.920',
           WA_MEM_FRACTION='0.920', SERVE_DISPLAY_HZ='', MAX_TOTAL_TOKENS='131072',
           MAMBA_SLOTS='10', SGLANG_SHARED_GATEUP_FUSED='0',
           PYTHONPATH=str(W/'python'), SGLANG_ADAPTIVE_TARGET_AUTOTUNE='1',
           SGLANG_ADAPTIVE_STEP_A='7.943',SGLANG_ADAPTIVE_STEP_B='0.5554',
           SGLANG_ADAPTIVE_POLICY='confidence')
if mode == 'diagnostic' and arm in ('current','prior'):
    env['SGLANG_WA6_EVENTS']=str(out/'steps.jsonl')
if arm == 'prior':
    env['ADAPTIVE_CONFIG']=str(ROOT/'prior.json')
else:
    env['ADAPTIVE_CONFIG']=str(B/'adaptive/w16_3_7_15_c.json')
profile = {'fixed4':'w4','fixed8':'w8','fixed16':'w16'}.get(arm,'wa')
cmd = ['bwrap', '--die-with-parent', '--dev-bind', '/', '/', '--ro-bind', str(R), str(R),
       '--bind', str(ROOT / 'runtime-cache'), str(R / '.cache')]
cmd += ['--ro-bind', str(W), str(W)]
cmd += [str(R / 'serve-fast.sh'), profile]
(out / 'command.json').write_text(json.dumps(dict(arm=arm, cmd=cmd, env={k:v for k,v in env.items()
    if k.startswith(('SGLANG_', 'PYTHON', 'MEM_', 'W16_', 'WA_', 'SERVE_', 'ADAPTIVE_', 'MAX_TOTAL'))}), indent=2)+'\n')

def note(s):
    with NOTE.open('a') as f:
        f.write('\n'+s+'\n')
    print(s, flush=True)

def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()

occupied = sum(json.loads(p.read_text())["seconds"] for p in ROOT.glob('wa6-*/lock-time.json'))
if occupied >= 4 * 3600:
    raise RuntimeError('WA6 occupied GPU budget exhausted')
lock_started=time.monotonic()
remaining_budget = 4*3600 - occupied
server = client = None
stop = threading.Event()
ready = threading.Event()
errors = []

def monitor():
    low_since = None
    try:
        with (out / 'memory.jsonl').open('x', buffering=1) as f:
            while not stop.is_set():
                if time.monotonic()-lock_started >= remaining_budget:
                    raise RuntimeError('WA6 GPU lock budget reached')
                if sum(p.stat().st_size for p in ROOT.glob('wa6-*/*') if p.is_file()) > 120*10**9:
                    raise RuntimeError('WA6 artifact budget reached')
                values = query('--query-gpu=memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown', '--format=csv,noheader,nounits').split(', ')
                free = int(values[0])
                processes = subprocess.check_output(['ps', '-eo', 'pid,pgid,comm'], text=True)
                other_gpu_app = [line for line in processes.splitlines() if OTHER_GPU_APP and OTHER_GPU_APP in line.lower()]
                phase = 'steady' if ready.is_set() else 'startup'
                f.write(json.dumps(dict(t=datetime.datetime.now().astimezone().isoformat(), phase=phase, free_mib=free, gpu=values, other_gpu_app=other_gpu_app))+'\n')
                if ready.is_set() and free < 4096:
                    low_since = low_since or time.monotonic()
                    if time.monotonic() - low_since >= 10:
                        raise RuntimeError('Steady free VRAM <4096 MiB for >=10 seconds')
                else:
                    low_since = None
                stop.wait(2)
    except BaseException as e:
        errors.append(str(e))
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

def run_client(command, log_name, client_env):
    global client
    with (out / log_name).open('x') as f:
        client = subprocess.Popen(command, env=client_env, cwd=B, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        while client.poll() is None:
            if errors or server.poll() is not None:
                raise RuntimeError(f'Server/watchdog failed: {errors}, server={server.poll()}')
            try:
                client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f'[{label}] {log_name} running', flush=True)
        if client.returncode:
            raise RuntimeError(f'{log_name} failed with {client.returncode}; see {out/log_name}')
    client = None

def terminate_owned(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Children can outlive a wrapper, so keep the grace even if leader exits.
    time.sleep(8)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()

def interrupted(signum, frame):
    raise RuntimeError(f'Interrupted by signal {signum}')

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
thread = None
success = False
try:
    import hashlib
    for source, digest in json.loads((ROOT/'production-manifest.json').read_text()).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen production source drifted: {source}')
    note(f'### Starting {label}\n\nArm {arm}, {datetime.datetime.now().astimezone().isoformat()}; per-arm flock held.')
    for i in range(80):
        pids = query('--query-compute-apps=pid', '--format=csv,noheader')
        if not pids:
            break
        if i % 3 == 0:
            print(f'[{label}] waiting for unowned GPU process: {pids}', flush=True)
        time.sleep(15)
    if pids:
        raise RuntimeError('GPU remains occupied; no unrelated process touched')
    time.sleep(10)
    (out / 'gpu-before.txt').write_text(query('-q')+'\n')
    (out / 'gpu-processes-before.txt').write_text(query()+'\n')
    (out / 'processes-before.txt').write_text(subprocess.check_output(['ps', '-eo', 'pid,pgid,comm'], text=True))
    import hashlib
    files=[W/'python/sglang/srt/speculative'/f for f in ('adaptive_confidence.py','adaptive_runtime_state.py','adaptive_request_prior.py','eagle_worker_v2.py')]
    files += [Path(env['ADAPTIVE_CONFIG']),ROOT/f'{split}-stream.jsonl',B/'calib/wa6_client.py']
    (out/'source-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    origin = subprocess.check_output([str(py), '-B', '-c',
        'import sglang.srt.speculative.adaptive_confidence as c; import sglang.srt.speculative.adaptive_runtime_state as r; print(c.__file__); print(r.__file__)'], env=env, cwd=R, text=True)
    (out / 'code-origin.txt').write_text(origin)
    expected = W / 'python'
    assert all(str(expected) in line for line in origin.splitlines()), origin
    with (out / 'server.log').open('x') as f:
        server = subprocess.Popen(cmd, env=env, cwd=R, stdout=f, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    (out / 'server-pgid.txt').write_text(str(server.pid)+'\n')
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    for i in range(450):
        if errors or server.poll() is not None:
            raise RuntimeError(f'Server startup failed: {errors}; rc={server.poll()}')
        if 'ready to roll' in (out / 'server.log').read_text():
            break
        if i % 20 == 0:
            print(f'[{label}] startup {i*2}s', flush=True)
        time.sleep(2)
    else:
        raise RuntimeError('Server readiness timeout, 15 minutes')
    assert os.getpgid(server.pid) == server.pid
    free = int(query('--query-gpu=memory.free', '--format=csv,noheader,nounits'))
    (out / 'ready-free-mib.txt').write_text(str(free)+'\n')
    (out / 'ready').touch()
    ready.set()
    note(f'{label}: ready; {free} MiB free VRAM.')
    run_client([str(py), '-B', str(B/'calib/wa6_client.py'), str(ROOT/f'{split}-stream.jsonl'),
                str(out), label], 'client.log', env)
    if arm == 'prior' and split == 'prep':
        run_client([str(py), '-B', str(B/'calib/wa6_state_smoke.py'), str(out)], 'state-smoke.log', env)
    run_client([str(py), '-B', str(B / 'prof/needle_test.py'), '18500', '0.4', label], 'needle.log', env)
    assert 'PASS=True' in (out / 'needle.log').read_text()
    if errors:
        raise RuntimeError(str(errors))
    (out / 'gpu-after.txt').write_text(query('-q')+'\n')
    (out / 'gpu-processes-after.txt').write_text(query()+'\n')
    success = True
except BaseException as e:
    note(f'{label}: ERROR {e}')
    raise
finally:
    stop.set()
    if thread:
        thread.join(timeout=10)
    (out / 'stopping').touch()
    terminate_owned(client)
    terminate_owned(server)
    (out/'lock-time.json').write_text(json.dumps({'seconds':time.monotonic()-lock_started,'success':success})+'\n')
    note(f'{label}: owned process groups cleaned up; lock released on command exit.')
if success:
    (out / 'complete').touch()
