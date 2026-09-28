#!/usr/bin/env python3
"""One WA4 arm. Invoke under flock -w 28800 /home/user/.gpu.lock."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

B = Path('/home/user/tools/flash-next-bench')
R = Path('/home/user/tools/sglang-rtxpro6000')
W = Path('/home/user/tools/sglang-wa4')
ROOT = B / 'specs/wa4'
NOTE = B / 'specs/WA4_LOG.md'
arm, label = sys.argv[1:]
assert arm in ('A', 'B', 'C', 'D')
out = ROOT / label
out.mkdir()  # Existing labels are never reused.
py = R / '.venv/bin/python'
env = os.environ.copy()
for k in list(env):
    if k.startswith(('SGLANG_', 'PROF_', 'WA_', 'W16_')) or k in ('PYTHONPATH', 'ADAPTIVE_CONFIG', 'MAX_TOTAL_TOKENS'):
        env.pop(k)
env.update(PYTHONDONTWRITEBYTECODE='1', MEM_FRACTION='0.920', W16_MEM_FRACTION='0.920',
           WA_MEM_FRACTION='0.920', SERVE_DISPLAY_HZ='', MAX_TOTAL_TOKENS='131072',
           SGLANG_ADAPTIVE_TRACE=str(out / 'trace.jsonl'), SGLANG_ADAPTIVE_DEBUG='1')
if arm != 'A':
    env.update(PYTHONPATH=str(W / 'python'), SGLANG_ADAPTIVE_TARGET_AUTOTUNE='1')
if arm in ('C', 'D'):
    env.update(ADAPTIVE_CONFIG=str(B / f'adaptive/w16_3_7_15_{arm.lower()}.json'),
               SGLANG_ADAPTIVE_STEP_A='7.943', SGLANG_ADAPTIVE_STEP_B='0.5554')
else:
    env['ADAPTIVE_CONFIG'] = str(B / 'adaptive/w16_conf.json')
cmd = ['bwrap', '--die-with-parent', '--dev-bind', '/', '/', '--ro-bind', str(R), str(R),
       '--bind', str(ROOT / 'runtime-cache'), str(R / '.cache')]
if arm != 'A':
    cmd += ['--ro-bind', str(W), str(W)]
cmd += [str(R / 'serve-fast.sh'), 'wa']
(out / 'command.json').write_text(json.dumps(dict(arm=arm, cmd=cmd, env={k:v for k,v in env.items()
    if k.startswith(('SGLANG_', 'PYTHON', 'MEM_', 'W16_', 'WA_', 'SERVE_', 'ADAPTIVE_', 'MAX_TOTAL'))}), indent=2)+'\n')

def note(s):
    with NOTE.open('a') as f:
        f.write('\n'+s+'\n')
    print(s, flush=True)

def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()

server = client = None
stop = threading.Event()
ready = threading.Event()
errors = []

def monitor():
    low_since = None
    try:
        with (out / 'memory.jsonl').open('x', buffering=1) as f:
            while not stop.is_set():
                free = int(query('--query-gpu=memory.free', '--format=csv,noheader,nounits'))
                phase = 'steady' if ready.is_set() else 'startup'
                f.write(json.dumps(dict(t=datetime.datetime.now().astimezone().isoformat(), phase=phase, free_mib=free))+'\n')
                if ready.is_set() and free < 1024:
                    low_since = low_since or time.monotonic()
                    if time.monotonic() - low_since >= 10:
                        raise RuntimeError('Steady free VRAM <1024 MiB for >=10 seconds')
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
    origin = subprocess.check_output([str(py), '-B', '-c',
        'import sglang.srt.speculative.adaptive_confidence as c; import sglang.srt.speculative.adaptive_runtime_state as r; print(c.__file__); print(r.__file__)'], env=env, cwd=R, text=True)
    (out / 'code-origin.txt').write_text(origin)
    expected = R / 'python' if arm == 'A' else W / 'python'
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
    client_env = env | {'WA2_EVENTS': str(out / 'events.jsonl')}
    run_client([str(B / '.venv-review/bin/python'), '-B', str(B / 'calib/wa2_fnbench.py'), 'run',
        '--endpoint', 'http://127.0.0.1:8001/v1', '--engine', 'sglang', '--workloads',
        'code-edit,prose-en,prose-ja,agent-loop', '--repeats', '3', '--sampling', 'greedy',
        '--allow-proc', 'sglang', '--label', label, '--out', str(out / 'runs.jsonl')], 'fnbench.log', client_env)
    run_client([str(py), '-B', str(B / 'prof/needle_test.py'), '18500', '0.4', label], 'needle.log', env)
    assert 'PASS=True' in (out / 'needle.log').read_text()
    if errors:
        raise RuntimeError(str(errors))
    (out / 'gpu-after.txt').write_text(query('-q')+'\n')
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
    note(f'{label}: owned process groups cleaned up; lock released on command exit.')
if success:
    subprocess.run([str(py), '-B', str(B / 'calib/wa4_report.py'), 'arm', str(out)], check=True, env=env)
    (out / 'complete').touch()
