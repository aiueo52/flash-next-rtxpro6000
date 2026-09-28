#!/usr/bin/env python3
"""HC1 isolated server arms. Run sequence; each server holds the shared flock."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

B=Path('/home/user/tools/flash-next-bench'); R=B.parent/'sglang-rtxpro6000'; W=B.parent/'sglang-hc1'
HERE=B/'bench/hc1'; OUT=B/'specs/hc1'; PY=R/'.venv/bin/python'; CLIENT=B/'.venv-review/bin/python'
LOCK='/home/user/.gpu.lock'; DOMAINS=('code-edit','prose-en','prose-ja','agent-loop')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def query(*a):return subprocess.check_output(['nvidia-smi',*a],text=True).strip()
def cleanup(p):
    if p is None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait(timeout=15)
def freeze():
    cache=OUT/'runtime-cache';cache.mkdir(exist_ok=True)
    subprocess.run(['cp','-a','--reflink=auto',str(R/'.cache')+'/.',str(cache)],check=True)
    q=HERE/'quality';q.mkdir(exist_ok=True)
    for name in ('run_bench.py','analyze.py','he_exec.py'):shutil.copy2(B/'bench/quality'/name,q/name)
    if not (q/'data').exists():(q/'data').symlink_to(B/'bench/quality/data',target_is_directory=True)
    files=list((W/'python').rglob('*.py'))+list((B/'workloads/sets').rglob('*.txt'))+list((B/'workloads/sets').rglob('*.json'))
    files+=list((B/'bench/quality/data').glob('*.jsonl'))+[R/'serve-fast.sh',R/'serve-local.sh',B/'tokenmaps/hot2_49152.pt']
    files+=list(HERE.glob('*.py'))+list(q.glob('*.py'))
    model=Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5')
    files+=list(model.glob('*.json'))+list(model.glob('*.jinja'))
    (OUT/'frozen.json').write_text(json.dumps(dict(created=time.time(),hashes={str(p):sha(p) for p in files},
         head=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),
         model_inventory={str(p):[p.stat().st_size,p.stat().st_mtime_ns,str(p.resolve())] for p in model.glob('*.safetensors')},
         quality_profile='w4',bs=1,memory_fraction=.900,min_free_mib=4096,budget_seconds=86400),indent=1))
    print('Frozen source, quality data, BN1 inputs, launchers and private cache',flush=True)

def check():
    f=json.loads((OUT/'frozen.json').read_text())
    for p,h in f['hashes'].items():
        if sha(Path(p))!=h:raise RuntimeError('Frozen input changed: '+p)
    for p,v in f['model_inventory'].items():
        p=Path(p)
        if [p.stat().st_size,p.stat().st_mtime_ns,str(p.resolve())]!=v:raise RuntimeError('Model drift: '+str(p))

def arm(name):
    if os.environ.get('HC1_LOCKED')!='1':raise RuntimeError('Must hold shared flock')
    check();folder=OUT/name;folder.mkdir(exist_ok=False)
    start=time.time();server=None;client=None;stop=threading.Event();ready=threading.Event();errors=[]
    used=sum(json.loads(p.read_text()).get('lock_seconds',0) for p in OUT.glob('*/complete.json'))
    remaining=86400-used
    def mark(event,**kw):
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),event=event,**kw))+'\n')
    def monitor():
        try:
            with (folder/'memory.jsonl').open('w',buffering=1) as f:
                while not stop.is_set():
                    free=int(query('--query-gpu=memory.free','--format=csv,noheader,nounits'))
                    f.write(json.dumps(dict(time=time.time(),free_mib=free,steady=ready.is_set()))+'\n')
                    if ready.is_set() and free<4096:raise RuntimeError(f'Steady VRAM floor violated: {free} MiB')
                    if time.time()-start>remaining:raise RuntimeError('24 lock-hour budget exhausted')
                    stop.wait(5)
        except BaseException as e:errors.append(str(e))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','FLASHINFER_','WA_','W16_','PROF_')) and k not in ('PYTHONPATH','ADAPTIVE_CONFIG','TARGET_MODEL','TOKEN_MAP','PRUNE_TAU','LD_PRELOAD','LIMIT','MOCK')}
    env.update(MEM_FRACTION='0.900',W4_MEM_FRACTION='0.900',W16_MEM_FRACTION='0.900',WA_MEM_FRACTION='0.900',
        SERVE_DISPLAY_HZ='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(W/'python'),MAX_TOTAL_TOKENS='131072',
        PORT='8001',BS='1',CONC='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        SGLANG_HC1_ARM='off' if name=='capture' else name,SGLANG_HC1_MANIFEST=str(OUT/'masks.json'))
    cmd=['bwrap','--die-with-parent','--dev-bind','/','/','--ro-bind',str(R),str(R),
         '--bind',str(OUT/'runtime-cache'),str(R/'.cache'),str(R/'serve-fast.sh'),'w4','--max-running-requests','1']
    if name=='capture':
        (OUT/'capture-data').mkdir(exist_ok=True)
        env['SGLANG_HC1_RECORD_DIR']=str(OUT/'capture-data')
        cmd+=['--disable-cuda-graph']
    (folder/'command.json').write_text(json.dumps(dict(cmd=cmd,env={k:v for k,v in env.items() if k.startswith(('SGLANG_','MEM_','W4_','W16_','WA_','SERVE_','PYTHON','MAX_TOTAL','BS','CONC'))}),indent=1))
    def run(command,log,cwd=B,extra=None):
        nonlocal client
        with log.open('a') as f:
            client=subprocess.Popen(command,cwd=cwd,env=env| (extra or {}),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while client.poll() is None:
            if errors or server.poll() is not None:raise RuntimeError(f'Watchdog: {errors}; server {server.poll()}')
            try:client.wait(timeout=30)
            except subprocess.TimeoutExpired:print(f'{name}: {log.name} running ({int(time.time()-start)} lock seconds)',flush=True)
        if client.returncode:raise RuntimeError(f'Client failed {command}: {client.returncode}')
    def interrupt(sig,frame):raise RuntimeError(f'Interrupted {sig}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        mark('lock_acquired');deadline=time.time()+120
        while query('--query-compute-apps=pid,process_name','--format=csv,noheader'):
            if time.time()>deadline:raise RuntimeError('Unowned compute applications remain; no signals sent')
            stop.wait(5)
        with (folder/'server.log').open('w') as f:
            server=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        mark('server_start',pgid=server.pid)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        deadline=time.time()+1200
        while 'ready to roll' not in (folder/'server.log').read_text():
            if errors or server.poll() is not None:raise RuntimeError(f'Startup: {errors}, rc={server.poll()}')
            if time.time()>deadline:raise RuntimeError('Startup timeout')
            stop.wait(2)
        ready.set();mark('ready');print(name+': server ready',flush=True)
        with urllib.request.urlopen('http://127.0.0.1:8001/get_server_info',timeout=30) as f:(folder/'server-info.json').write_bytes(f.read())
        if name=='capture':
            run([str(CLIENT),'-B',str(HERE/'capture_client.py')],folder/'capture-client.log')
            run([str(PY),'-B',str(HERE/'build_masks.py'),str(OUT/'capture-data')],folder/'masks.log')
        else:
            run([str(CLIENT),'-B',str(HERE/'warmup.py')],folder/'warmup.log')
            run([str(CLIENT),'-B','-m','fnbench','run','--endpoint','http://127.0.0.1:8001/v1','--engine','sglang',
                '--workloads',','.join(DOMAINS),'--prompt-sets',str(B/'workloads/sets'),'--prompt-limit','8','--repeats','1',
                '--sampling','greedy','--require-acceptance','--allow-proc','sglang','--label','hc1-'+name,'--out',str(folder/'requests.jsonl')],folder/'acceptance.log')
            for depth in ('0.1','0.5','0.9'):
                run([str(CLIENT),'-B',str(B/'prof/needle_test.py'),'18500',depth,'hc1-'+name],folder/'needle.log')
            q=HERE/'quality'
            run([str(PY),'-B',str(q/'run_bench.py'),name+'-smoke','gsm8k,mmlu,humaneval,jcqa'],folder/'quality-smoke.log',extra={'LIMIT':'3'})
            smoke=json.loads((q/'runs'/(name+'-smoke')/'summary.json').read_text())
            if any(v['errors'] or v['mean_gen_tokens']<=0 for v in smoke['bench'].values()):raise RuntimeError('Quality smoke transport failure')
            run([str(PY),'-B',str(q/'run_bench.py'),name,'gsm8k,mmlu,humaneval,jcqa'],folder/'quality.log')
            summary=json.loads((q/'runs'/name/'summary.json').read_text())
            if any(v['errors'] for v in summary['bench'].values()):raise RuntimeError('Quality transport errors')
            for depth in ('0.1','0.5','0.9'):
                run([str(CLIENT),'-B',str(B/'prof/needle_test.py'),'18500',depth,'hc1-'+name+'-post'],folder/'needle.log')
            shutil.copy2(folder/'needle.log',q/'runs'/name/'needle.log')
        check()
        if errors:raise RuntimeError(str(errors))
    except BaseException as e:
        (folder/'invalid.json').write_text(json.dumps(dict(error=str(e))))
        mark('error',error=str(e));raise
    finally:
        cleanup(client);cleanup(server);stop.set()
        deadline=time.time()+120
        while query('--query-compute-apps=pid,process_name','--format=csv,noheader'):
            if time.time()>deadline:raise RuntimeError('Compute contexts did not drain after owned cleanup')
            time.sleep(2)
        mark('cleanup_complete',lock_seconds=time.time()-start)
    (folder/'complete.json').write_text(json.dumps(dict(lock_seconds=time.time()-start,finished=time.time())))
    print(name+': complete',flush=True)

def sequence():
    for name in ('capture','prod','hc256','hc160','hc128','gate-const'):
        if (OUT/name/'complete.json').exists():continue
        print(name+': waiting for shared GPU lock',flush=True)
        subprocess.run(['flock','-w','28800',LOCK,'env','HC1_LOCKED=1',str(CLIENT),'-B',str(Path(__file__).resolve()),'arm',name],check=True)
        if (HERE/'report.py').exists():subprocess.run([str(CLIENT),'-B',str(HERE/'report.py')],check=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='sequence':sequence()
    elif sys.argv[1]=='arm':arm(sys.argv[2])
