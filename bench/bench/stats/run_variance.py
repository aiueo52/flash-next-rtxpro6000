#!/usr/bin/env python3
"""BN1 CPU freeze/orchestrator + one locked server arm. Never modify production.
Run CPU freeze first, then sequence; each arm acquires/relinquishes flock itself.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

B = Path(__file__).resolve().parents[2]
R = B.parent/'sglang-rtxpro6000'
PY = B/'.venv-review/bin/python'
LOCK = '/home/user/.gpu.lock'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, budget_seconds=7200):
    root.mkdir(parents=True, exist_ok=False)
    paths = set((R/'python').rglob('*.py'))
    paths.update([R/'serve-fast.sh', R/'serve-local.sh', B/'adaptive/w16_3_7_15_c.json',
                  B/'tokenmaps/hot2_49152.pt'])
    paths.update((B/'fnbench').glob('*.py'))
    paths.update((B/'workloads/sets').rglob('*.txt'))
    paths.update((B/'workloads/sets').rglob('*.json'))
    model = Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5')
    paths.update(p for p in model.iterdir() if p.suffix in ('.json','.jinja'))
    # Reuse read-only production cache via a private copy; launcher's fixed path is bind-mounted.
    cache = root/'runtime-cache'; cache.mkdir()
    subprocess.run(['cp','-a','--reflink=auto',str(R/'.cache')+'/.',str(cache)],check=True)
    result = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  hashes={str(p):digest(p) for p in sorted(paths)},
                  model_inventory={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                        resolved=str(p.resolve())) for p in model.glob('*.safetensors')},
                  production_head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
                  production_diff=subprocess.check_output(['git','-C',str(R),'diff','--binary'],text=True),
                  schedule=['w4-A1','w4-A2','w4-A3','wa-A1','wa-A2','wa-A3'],
                  memory_fraction=.920, display_hz='', min_free_mib=4096,
                  warmup='one fixed non-holdout story, 1024 output budget; no per-domain warmup',
                  runtime_budget_hours=budget_seconds/3600, runtime_budget_seconds=budget_seconds)
    (root/'frozen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Frozen {len(paths)} inputs in {root}',flush=True)


def check_frozen(root):
    frozen=json.loads((root/'frozen.json').read_text())
    for p,h in frozen['hashes'].items():
        if digest(Path(p)) != h: raise RuntimeError('Frozen input changed: '+p)
    for p,s in frozen['model_inventory'].items():
        f=Path(p)
        if f.stat().st_size!=s['size'] or f.stat().st_mtime_ns!=s['mtime_ns'] or str(f.resolve())!=s['resolved']:
            raise RuntimeError('Model asset metadata changed: '+p)
    return frozen


def query(*args):
    return subprocess.check_output(['nvidia-smi',*args],text=True,stderr=subprocess.STDOUT).strip()


def cleanup(proc):
    if proc is None: return
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    proc.wait(timeout=10)


def occupied_seconds(root):
    total = 0.0
    for path in root.glob('*/events.jsonl'):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'cleanup_complete':
                total += event['lock_seconds']
    return total


def arm(root, name):
    if os.environ.get('BN1_LOCKED')!='1': raise RuntimeError('use sequence or flock wrapper')
    frozen=check_frozen(root)
    out=root/name;out.mkdir()
    profile=name.split('-')[0]
    server=client=None; stop=threading.Event(); ready=threading.Event(); errors=[]; monitor_thread=None
    started=time.time()
    occupied=occupied_seconds(root)
    remaining=frozen.get("runtime_budget_seconds",7200)-occupied
    if remaining <= 0: raise RuntimeError("2 lock-hour budget exhausted")
    def mark(event,**kw):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),event=event,**kw))+'\n')
    def interrupted(sig,frame): raise RuntimeError(f'interrupted {sig}')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','FLASHINFER_','WA_','W16_','PROF_')) and k not in ('PYTHONPATH','ADAPTIVE_CONFIG','TARGET_MODEL','TOKEN_MAP','PRUNE_TAU','LD_PRELOAD')}
    env.update(MEM_FRACTION='0.920', W4_MEM_FRACTION='0.920', W16_MEM_FRACTION='0.920', WA_MEM_FRACTION='0.920',
               SERVE_DISPLAY_HZ='',PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(R/'python'),
               MAX_TOTAL_TOKENS='131072', PORT='8001',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd=['bwrap','--die-with-parent','--dev-bind','/','/','--ro-bind',str(R),str(R),
         '--bind',str(root/'runtime-cache'),str(R/'.cache'),str(R/'serve-fast.sh'),profile]
    (out/'command.json').write_text(json.dumps(dict(cmd=cmd,env={k:v for k,v in env.items() if k in ('MEM_FRACTION','WA_MEM_FRACTION','SERVE_DISPLAY_HZ','MAX_TOTAL_TOKENS','PYTHONPATH','PYTORCH_CUDA_ALLOC_CONF')},name=name),indent=2))
    def monitor():
        try:
            with (out/'memory.jsonl').open('x',buffering=1) as f:
                while not stop.is_set():
                    if time.time()-started >= remaining: raise RuntimeError("2 lock-hour budget exhausted")
                    free=int(query('--query-gpu=memory.free','--format=csv,noheader,nounits'))
                    f.write(json.dumps(dict(time=time.time(),free_mib=free,steady=ready.is_set()))+'\n')
                    if free <4096: raise RuntimeError(f'free VRAM {free} <4096 MiB')
                    stop.wait(1)
        except BaseException as exc: errors.append(str(exc))
    try:
        mark('lock_acquired')
        drain_deadline=time.monotonic()+min(120,remaining)
        while True:
            apps=query('--query-compute-apps=pid,process_name','--format=csv,noheader')
            if not apps: break
            mark('waiting_for_prior_gpu_cleanup',applications=apps)
            if time.monotonic() >= drain_deadline:
                raise RuntimeError('Unowned GPU compute applications remain after drain wait: '+apps)
            stop.wait(5)
        (out/'gpu-before.txt').write_text(query('-q'))
        with (out/'server.log').open('x') as f:
            server=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        mark('server_start',pgid=server.pid)
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if errors or server.poll() is not None: raise RuntimeError(f'Startup failure {errors}, rc={server.poll()}')
            if 'ready to roll' in (out/'server.log').read_text(): break
            stop.wait(2)
        else: raise RuntimeError('readiness timeout')
        ready.set();mark('ready')
        with urllib.request.urlopen('http://127.0.0.1:8001/get_server_info',timeout=20) as f:
            (out/'server-info.json').write_bytes(f.read())
        sys.path.insert(0,str(B))
        from fnbench.http_client import OpenAIStreamClient
        from fnbench.models import Sampling
        client_api=OpenAIStreamClient('http://127.0.0.1:8001/v1')
        client_api.complete(prompt='Write a detailed practical guide to maintaining a small community bicycle workshop. Discuss tools, storage, inspection and staff handover in connected paragraphs.',max_tokens=1024,sampling=Sampling(0.0),model='flash-next')
        client_api.session.close();mark('warmup_complete')
        bench=[str(PY),'-B','-m','fnbench','run','--endpoint','http://127.0.0.1:8001/v1','--engine','sglang',
            '--workloads','code-edit,prose-en,prose-ja,agent-loop','--prompt-sets',str(B/'workloads/sets'),
            '--prompt-limit','8','--repeats','1','--sampling','greedy','--require-acceptance','--allow-proc','sglang',
            '--label',name,'--out',str(out/'requests.jsonl')]
        with (out/'client.log').open('x') as f:
            client=subprocess.Popen(bench,cwd=B,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+2400
        while client.poll() is None:
            if errors or server.poll() is not None: raise RuntimeError(f'watchdog {errors}, server={server.poll()}')
            if time.monotonic()>deadline: raise RuntimeError('benchmark timeout')
            try: client.wait(timeout=30)
            except subprocess.TimeoutExpired: print(f'{name}: measuring',flush=True)
        if client.returncode: raise RuntimeError(f'fnbench failed: {client.returncode}')
        rows=[json.loads(l) for l in (out/'requests.jsonl').read_text().splitlines()]
        if len(rows)!=32: raise RuntimeError('incomplete prompt set')
        duration=[dict(id=r['prompt_id'],seconds=r['client']['decode_seconds']+r['client']['ttft_seconds']) for r in rows]
        (out/'durations.json').write_text(json.dumps(duration,indent=2))
        bad=[r for r in duration if not 15<=r['seconds']<=60]
        if bad: mark('duration_out_of_target',prompts=bad)
        if errors: raise RuntimeError(str(errors))
        check_frozen(root)
        (out/'measurements-valid.json').write_text(json.dumps(dict(records=32,duration_out_of_target=bad)))
        mark('measured')
    except BaseException as exc:
        (out/'invalid.json').write_text(json.dumps(dict(error=str(exc))))
        mark('error',error=str(exc));raise
    finally:
        cleanup(client);cleanup(server);stop.set()
        if monitor_thread: monitor_thread.join(timeout=10)
        if server is not None:
            # Process exit can precede NVML/CUDA context teardown. Keep flock
            # until teardown finishes so the next owner sees an idle device.
            drain_deadline=time.monotonic()+120
            while True:
                apps=query('--query-compute-apps=pid,process_name','--format=csv,noheader')
                if not apps: break
                mark('post_cleanup_gpu_drain',applications=apps)
                if time.monotonic()>=drain_deadline:
                    error='GPU contexts remain after owned process-group cleanup: '+apps
                    (out/'invalid.json').write_text(json.dumps(dict(error=error)))
                    mark('cleanup_complete',lock_seconds=time.time()-started,error=error)
                    raise RuntimeError(error)
                time.sleep(1)
        mark('cleanup_complete',lock_seconds=time.time()-started)
    (out/'complete.json').write_text(json.dumps(dict(lock_seconds=time.time()-started,finished=time.time())))
    print(f'{name}: complete',flush=True)


def sequence(root):
    frozen=check_frozen(root)
    for name in frozen['schedule']:
        if (root/name/'complete.json').exists(): continue
        occupied=occupied_seconds(root)
        if occupied>=frozen.get('runtime_budget_seconds',7200): raise RuntimeError('2 lock-hour budget exhausted')
        subprocess.run(['flock','-w','28800',LOCK,'env','BN1_LOCKED=1',str(PY),'-B',str(Path(__file__).resolve()),
                        'arm',str(root),'--name',name],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','sequence','arm']);p.add_argument('root',type=Path)
    p.add_argument('--budget-seconds',type=int,default=7200)
    p.add_argument('--name',choices=[f'{p}-A{i}' for p in ('w4','wa') for i in (1,2,3)])
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='freeze': freeze(root,a.budget_seconds)
    elif a.mode=='sequence': sequence(root)
    elif a.name: arm(root,a.name)
    else: p.error('--name required for arm')

if __name__=='__main__': main()
