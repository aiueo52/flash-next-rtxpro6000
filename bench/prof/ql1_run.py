#!/usr/bin/env python3
"""Measurement only. Invoke once per server under the shared GPU flock."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import requests

B = Path(__file__).resolve().parents[1]
R = B.parent / 'sglang-rtxpro6000'
ROOT = B / 'prof/traces'
W = B / 'workloads/longctx'
REPORT = B / 'specs/QL1_LONG_CONTEXT_MAP.md'
profile, runid = sys.argv[1:3]
assert profile in ('w4', 'w16')
assert os.environ.get('QL1_GPU_LOCKED') == '1'
O = ROOT / f'ql1-{runid}-{profile}'
O.mkdir(exist_ok=False)
EP = 'http://127.0.0.1:8001'
stop = threading.Event(); ready = threading.Event(); errors = []
server = None
lock_start = time.monotonic()

def save(name, value):
    (O / name).write_text(json.dumps(value, indent=2) + '\n')

def event(kind, **kw):
    row = dict(time=datetime.datetime.now().astimezone().isoformat(), event=kind, **kw)
    with (O/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)

def note(s):
    with REPORT.open('a') as f: f.write('\n'+s+'\n')
    event('note', text=s)

def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()

def guard():
    if errors: raise RuntimeError('; '.join(errors))
    if server is not None and server.poll() is not None:
        raise RuntimeError(f'Server exited: {server.returncode}')
    # Two server arms each bounded at two occupied hours; total <=4 hours.
    if time.monotonic() - lock_start > 7100:
        raise RuntimeError('Per-server lock budget reached (7100 seconds)')

def watch():
    try:
        with (O/'memory.jsonl').open('w', buffering=1) as f:
            while not stop.is_set():
                free = int(query('--query-gpu=memory.free','--format=csv,noheader,nounits'))
                f.write(json.dumps(dict(time=time.time(), ready=ready.is_set(), free_mib=free))+'\n')
                if ready.is_set() and free < 4096:
                    errors.append(f'Steady-state VRAM below 4096 MiB: {free}')
                    if server is not None: os.killpg(server.pid, signal.SIGTERM)
                    return
                if time.monotonic()-lock_start > 7100:
                    errors.append('GPU lock budget deadline')
                    if server is not None: os.killpg(server.pid, signal.SIGTERM)
                    return
                stop.wait(1)
    except BaseException as e:
        errors.append(f'Monitor failed: {e}')
        if server is not None:
            try: os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError: pass

def api(path, payload=None, timeout=120):
    guard()
    r = requests.get(EP+path, timeout=timeout) if payload is None else requests.post(EP+path, json=payload, timeout=timeout)
    r.raise_for_status()
    try: return r.json()
    except ValueError: return r.text

def generate(row, budget, name, profile_dir=None, ignore_eos=True):
    ids = json.loads((W/row['input_ids']).read_text())
    body = dict(input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=budget, ignore_eos=ignore_eos), stream=True)
    timeline = []; result = {}; thrown = []
    t0 = time.perf_counter()
    def stream():
        try:
            with requests.post(EP+'/generate', json=body, stream=True, timeout=(20,1800)) as r:
                r.raise_for_status()
                for line in r.iter_lines(chunk_size=1):
                    if not line.startswith(b'data:'): continue
                    data = line[5:].strip()
                    if data == b'[DONE]': break
                    j = json.loads(data)
                    if 'error' in j: raise RuntimeError(str(j))
                    meta = j.get('meta_info', {})
                    n = meta.get('completion_tokens', 0)
                    if n and (not timeline or n > timeline[-1]['tokens']):
                        timeline.append(dict(seconds=time.perf_counter()-t0, tokens=n,
                                             verify_ct=meta.get('spec_verify_ct')))
                    result.update(j)
        except BaseException as e: thrown.append(e)
    th = threading.Thread(target=stream, daemon=True); th.start()
    if profile_dir is not None:
        while len(timeline) < 8 and th.is_alive():
            guard(); th.join(timeout=0.05)
        if not th.is_alive(): raise RuntimeError('Generation ended before profile trigger')
        profile_dir.mkdir()
        trigger = dict(seconds=time.perf_counter()-t0, emitted_tokens=timeline[-1]['tokens'])
        response = api('/start_profile', dict(output_dir=str(profile_dir),num_steps=20,
                       activities=['CPU','GPU'],with_stack=False,record_shapes=False), timeout=600)
        save(name+'-profile-response.json',response)
        save(name+'-profile-window.json',dict(trigger=trigger, completed=dict(
            seconds=time.perf_counter()-t0,emitted_tokens=timeline[-1]['tokens']),
            input_tokens=row['context_tokens'],note='SSE-observed emission bounds, not GPU sequence-length tensor capture'))
    while th.is_alive():
        guard(); th.join(timeout=5)
    if thrown: raise thrown[0]
    guard()
    meta = result.get('meta_info', {})
    if meta.get('prompt_tokens') != row['context_tokens']:
        raise RuntimeError(f'Prompt length mismatch: {meta}')
    if ignore_eos and meta.get('completion_tokens') != budget:
        raise RuntimeError(f'Completion length mismatch: {meta}')
    dt = timeline[-1]['seconds']-timeline[0]['seconds'] if len(timeline)>1 else None
    sample = dict(prompt_id=row['id'],context_tokens=row['context_tokens'],family=row['family'],
                  profile=profile,budget=budget,timeline=timeline,meta_info=meta,
                  total_seconds=time.perf_counter()-t0,decode_seconds=dt,
                  decode_tps=(timeline[-1]['tokens']-timeline[0]['tokens'])/dt if dt else None,
                  conventional_budget_over_decode_seconds=budget/dt if dt else None,
                  text=result.get('text',''),ignore_eos=ignore_eos,profiled=profile_dir is not None)
    if row['family'].startswith('needle'):
        sample['needle_pass']='AURORA-CEDAR-7319' in sample['text']
    save(name+'.json',sample)
    if profile_dir is not None:
        paths=list(profile_dir.glob('*.gz'))
        if len(paths)!=1: raise RuntimeError(f'Expected one CUPTI trace: {paths}')
        with gzip.open(paths[0],'rt') as f: events=json.load(f)['traceEvents']
        nd=sum(e.get('cat')=='user_annotation' and e.get('name')=='draft' for e in events)
        nk=sum(e.get('cat')=='kernel' for e in events)
        if nd!=20 or not nk: raise RuntimeError(f'Invalid trace coverage: draft={nd} kernel={nk}')
        save(name+'-coverage.json',dict(draft_annotations=nd,kernels=nk,trace=str(paths[0])))
    event('sample',id=name,tps=sample['decode_tps'],tokens=meta.get('completion_tokens'),needle_pass=sample.get('needle_pass'))
    return sample

def interrupted(sig,frame):
    raise RuntimeError(f'Interrupted by signal {sig}')
signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
thread=None
try:
    frozen=json.loads((W/'frozen.json').read_text())
    for path,digest in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, 'Frozen input changed: '+path
    note(f'### {profile} / {runid}: lock acquired\n\nUTC start: {datetime.datetime.now(datetime.timezone.utc).isoformat()}.')
    apps=query('--query-compute-apps=pid,process_name','--format=csv,noheader')
    if apps: raise RuntimeError('Unowned GPU compute processes remain after lock acquisition: '+apps)
    sock=socket.socket(); bound=sock.connect_ex(('127.0.0.1',8001))==0; sock.close()
    if bound: raise RuntimeError('Port 8001 belongs to another server')
    (O/'gpu-before.txt').write_text(query('-q'))
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SGLANG_','PROF_','WA_','W16_','W4_','FLASHINFER_')) or k in (
            'PYTHONPATH','ADAPTIVE_CONFIG','MAX_TOTAL_TOKENS','TARGET_MODEL','TOKEN_MAP','PRUNE_TAU','MAMBA_SLOTS','CONTEXT_LENGTH','PORT'):
            env.pop(k)
    env.update(PYTHONDONTWRITEBYTECODE='1',MEM_FRACTION='0.920',W16_MEM_FRACTION='0.920',
               SERVE_DISPLAY_HZ='',MAX_TOTAL_TOKENS='131072',PORT='8001',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(R/'python'))
    cache=W/'runtime-cache'
    cmd=['bwrap','--die-with-parent','--dev-bind','/','/','--ro-bind',str(R),str(R),
         '--bind',str(cache),str(R/'.cache'),str(R/'serve-fast.sh'),profile]
    save('command.json',dict(command=cmd,environment={k:v for k,v in env.items() if k in (
         'MEM_FRACTION','W16_MEM_FRACTION','SERVE_DISPLAY_HZ','MAX_TOTAL_TOKENS','PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTORCH_CUDA_ALLOC_CONF','PORT')},frozen=frozen))
    with (O/'server.log').open('w') as log:
        server=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,start_new_session=True)
    save('server-pgid.json',server.pid)
    thread=threading.Thread(target=watch,daemon=True); thread.start()
    until=time.monotonic()+1200
    while 'ready to roll' not in (O/'server.log').read_text():
        guard()
        if time.monotonic()>until: raise RuntimeError('Server readiness timeout')
        time.sleep(2)
    ready.set()
    info=api('/get_server_info');save('server-info-ready.json',info)
    free=int(query('--query-gpu=memory.free','--format=csv,noheader,nounits'))
    if free<4096: raise RuntimeError(f'Insufficient ready VRAM: {free}')
    note(f'{profile}: ready with {free} MiB free; mem_fraction_static=0.920, requested KV cap=131072. Runtime server info and allocation log recorded.')
    manifest=json.loads((W/'manifest.json').read_text())
    # Capacity is checked from actual server response before any ladder request.
    def numbers(x,key):
        if isinstance(x,dict):
            return ([x[key]] if key in x else []) + [z for v in x.values() for z in numbers(v,key)]
        if isinstance(x,list): return [z for v in x for z in numbers(v,key)]
        return []
    budgets=[v for v in numbers(info,'max_total_num_tokens') if isinstance(v,int)]
    if not budgets:
        import re
        budgets=[int(v) for v in re.findall(r'max_total_num_tokens[=:]\s*(\d+)',(O/'server.log').read_text())]
    if not budgets: raise RuntimeError('Cannot verify actual KV token capacity')
    capacity=min(budgets)
    save('kv-budget.json',dict(capacity_tokens=capacity,required_for_96k_trace=98304+600+256,
                              supports_96k=capacity>=98304+600+256,conservative_prompt_max=capacity-856))
    note(f'{profile}: observed KV capacity {capacity} tokens; 96k trace needs 99160 including 600 output tokens and 256-token reserve; supports_96k={capacity>=99160}.')
    for row in manifest['prompts']:
        if row['context_tokens']+856>capacity:
            event('skipped_capacity',id=row['id'],capacity=capacity);continue
        api('/flush_cache',{})
        if row['perf']:
            generate(row,128,row['id']+'-warmup')
            samples=[generate(row,128,row['id']+f'-wall{rep}') for rep in range(1,4)]
            trace_dir=ROOT/f'ql1-{runid}-{profile}-{row["id"]}'
            generate(row,600,row['id']+'-trace-request',profile_dir=trace_dir)
            pooled=sum(s['timeline'][-1]['tokens']-s['timeline'][0]['tokens'] for s in samples)/sum(s['decode_seconds'] for s in samples)
            note(f'{profile} {row["id"]}: 3 x 128-token wall measurements complete, pooled decode={pooled:.3f} tokens/s; 20-step CUPTI trace validated at `{trace_dir.relative_to(B)}`.')
        else:
            sample=generate(row,48,row['id']+'-qa',ignore_eos=False)
            note(f'{profile} {row["id"]}: needle PASS={sample["needle_pass"]}, completion={sample["meta_info"]["completion_tokens"]}.')
    guard(); save('server-info-final.json',api('/get_server_info'))
    (O/'complete').touch()
except BaseException as e:
    save('error.json',dict(type=type(e).__name__,error=str(e)))
    note(f'{profile}: ERROR: {e}')
    raise
finally:
    if server is not None:
        for sig,timeout in [(signal.SIGTERM,15),(signal.SIGKILL,15)]:
            try: os.killpg(server.pid,sig)
            except ProcessLookupError: break
            try: server.wait(timeout=timeout)
            except subprocess.TimeoutExpired: continue
            # Also stop surviving descendants in our owned process group.
            try: os.killpg(server.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            break
    stop.set()
    if thread: thread.join(timeout=10)
    hours=(time.monotonic()-lock_start)/3600
    save('lock-hours.json',hours)
    note(f'{profile}: owned server process group cleaned, occupied lock-hours={hours:.4f}; flock releases on process exit.')
