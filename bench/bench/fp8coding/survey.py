"""CPU-only exhaustive serving FP8 byte survey. No serving imports or GPU calls.

Huffman is an exact encoded-size model, not a timed codec. ANS is explicitly
a normalized-probability cross-entropy model. zlib/zstd are real round trips.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
import argparse
import collections
import concurrent.futures
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import struct
import time
import zlib
import numpy as np
import torch
from safetensors import safe_open
try:
    import zstandard as zstd
except ImportError:
    zstd = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT.parent
MODEL = Path('/home/user/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft5')
MAP = ROOT/'tokenmaps/hot2_49152.pt'
OUT = HERE/'results'
BASE = 16384
BLOCKS = (16384, 32768, 65536)
torch.set_num_threads(12)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def entropy(c):
    c = np.asarray(c, dtype=np.float64)
    p = c[c > 0]/c.sum()
    return float(-np.dot(p, np.log2(p)))

def lengths(c):
    """Deterministic binary Huffman lengths (one-symbol alphabet uses 1 bit)."""
    h = [(int(v), i, [i]) for i,v in enumerate(c) if v]
    heapq.heapify(h)
    out = np.zeros(len(c), dtype=np.int64)
    if len(h) == 1:
        out[h[0][1]] = 1
    serial = len(c)
    while len(h) > 1:
        a, _, aa = heapq.heappop(h)
        b, _, bb = heapq.heappop(h)
        out[aa+bb] += 1
        heapq.heappush(h, (a+b, serial, aa+bb))
        serial += 1
    return out

def ans_costs(c):
    """12-bit normalized static frequencies, exact cross entropy, not bitstream."""
    c = np.asarray(c, dtype=np.float64)
    present = c > 0
    f = np.zeros(c.size, dtype=np.int64)
    ideal = c/c.sum()*(4096-int(present.sum()))
    f[present] = 1 + np.floor(ideal[present]).astype(np.int64)
    missing = 4096-int(f.sum())
    order = np.argsort(-(ideal-np.floor(ideal)), kind='stable')
    order = [i for i in order if present[i]]
    f[order[:missing]] += 1
    assert f.sum() == 4096 and np.all(f[present] > 0)
    cost = np.zeros(c.size)
    cost[present] = -np.log2(f[present]/4096)
    return cost

def inventory():
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    groups = {}
    excluded = []
    for key in sorted(wm):
        if not key.endswith('.weight') or not (key.startswith('model.language_model.layers.') or key.startswith('mtp.') or key == 'lm_head.weight'):
            continue
        name = key[:-7]
        cat = None
        suffix = name.rsplit('.',1)[-1]
        if '.mlp.shared_expert.' in name and suffix in ('gate_proj','up_proj','down_proj'):
            cat = 'shared_expert'
            if suffix != 'down_proj': name = name.rsplit('.',1)[0]+'.gate_up_proj'
        elif '.self_attn.' in name and '.indexer.' not in name and suffix in ('q_proj','k_proj','v_proj','o_proj'):
            cat = 'attn'
            if suffix != 'o_proj': name = name.rsplit('.',1)[0]+'.qkv_proj'
        elif '.linear_attn.' in name and suffix in ('in_proj_qkv','in_proj_z','out_proj'):
            cat = 'linear_attn'
            if suffix != 'out_proj': name = name.rsplit('.',1)[0]+'.in_proj_qkvz'
        elif key == 'lm_head.weight': cat = 'lm_head'
        elif key in ('mtp.fc_embedding.weight','mtp.fc_hidden.weight'): cat = 'mtp_dense'
        if cat and key.startswith('mtp.'): cat = 'mtp_dense'
        if cat:
            rec = groups.setdefault(name, dict(name=name, category=cat, sources=[]))
            rec['sources'].append(key)
        else:
            excluded.append(key)
    order = {'q_proj':0,'k_proj':1,'v_proj':2,'gate_proj':0,'up_proj':1,'in_proj_qkv':0,'in_proj_z':1}
    for g in groups.values():
        g['sources'].sort(key=lambda k:order.get(k.split('.')[-2],0))
    groups['draft.lm_head.hot2_49152'] = dict(name='draft.lm_head.hot2_49152', category='lm_head', sources=['lm_head.weight'], token_map=str(MAP))
    # Category selector is pure Python; load by filename, never import the server.
    p = TOOLS/'sglang-rtxpro6000/python/sglang/srt/qwen4_exp_dense_fp8.py'
    spec = importlib.util.spec_from_file_location('category_selector',p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    enabled = frozenset(('shared_expert','attn','linear_attn','lm_head','mtp_dense'))
    for g in groups.values():
        prefix = 'lm_head' if 'token_map' in g else g['name']
        assert mod.select_qwen4_exp_dense_fp8_category(prefix, enabled) == g['category'], g
    return wm, list(groups.values()), excluded

def quantize(g, wm):
    """DH1 per-row conversion, plus reciprocal-multiply serving-path audit.

    Production CUDA uses fp32 max/448 then fp32 reciprocal multiply. CPU
    reproduces these operations; the DH1 division expression is compared
    exhaustively and both byte hashes are recorded. No CUDA parity claimed.
    """
    arrays, scales = [], []
    rawhash, dhhash = hashlib.sha256(), hashlib.sha256()
    changed = zero_rows = floor_rows = 0
    for key in g['sources']:
        with safe_open(MODEL/wm[key], framework='pt', device='cpu') as f:
            raw = f.get_tensor(key)
            assert raw.dtype == torch.bfloat16 and raw.ndim == 2
            if 'token_map' in g:
                ids = torch.as_tensor(torch.load(MAP,map_location='cpu',weights_only=True),dtype=torch.long)
                assert ids.numel() == ids.unique().numel() == 49152
                raw = raw[ids]
            dest = np.empty(tuple(raw.shape),np.uint8)
            ss = np.empty(raw.shape[0],np.float32)
            for start in range(0, raw.shape[0], 1024):
                rr = raw[start:start+1024].contiguous()
                rawhash.update(rr.view(torch.uint8).numpy().tobytes())
                w = rr.float()
                assert bool(torch.isfinite(w).all())
                a = w.abs().amax(1,keepdim=True)
                zero_rows += int((a == 0).sum())
                floor_rows += int((a < 1e-10).sum())
                s = a / 448.0
                inv = torch.where(s == 0, 0.0, 1.0/s)
                q = (w*inv).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
                ds = a.clamp_min(1e-10)/448.0
                dq = (w/ds).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
                changed += int(np.count_nonzero(q != dq))
                dhhash.update(dq.tobytes())
                dest[start:start+len(q)] = q
                ss[start:start+len(q)] = s.numpy().ravel()
            arrays.append(dest)
            scales.append(ss)
    q = np.concatenate(arrays) if len(arrays)>1 else arrays[0]
    s = np.concatenate(scales)
    return q, s, dict(raw_bf16_sha256=rawhash.hexdigest(), fp8_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        dh1_fp8_sha256=dhhash.hexdigest(), dh1_byte_differences=changed, zero_rows=zero_rows, floor_rows=floor_rows,
        scales_sha256=hashlib.sha256(s.tobytes()).hexdigest())

SYMBOLS = np.arange(256)
EXP = (SYMBOLS >> 3) & 15
MANT = SYMBOLS & 7
SM = ((SYMBOLS >> 7) << 3) | MANT

def field_hist(c, field, n):
    return np.bincount(field, weights=c, minlength=n).astype(np.int64)

def statistics(q, s):
    flat = q.ravel()
    bc = np.stack([np.bincount(flat[i:i+BASE],minlength=256) for i in range(0,flat.size,BASE)])
    c = bc.sum(0)
    es = {name:entropy(field_hist(c,field,n)) for name,field,n in [('exponent',EXP,16),('mantissa',MANT,8),('sign',SYMBOLS>>7,2),('sign_mantissa',SM,16)]}
    es['byte'] = entropy(c)
    es['byte_given_scale_bucket'] = {}
    for div in (1,4):
        buckets = np.floor(div*np.log2(np.maximum(s, np.finfo(np.float32).tiny))).astype(np.int32)
        groups = []
        for bucket in np.unique(buckets):
            counts = np.zeros(256,np.int64)
            rows = np.flatnonzero(buckets == bucket)
            for i in range(0,len(rows),128):
                counts += np.bincount(q[rows[i:i+128]].ravel(),minlength=256)
            groups.append(dict(bucket=int(bucket), rows=len(rows), bytes=int(counts.sum()), entropy=entropy(counts)))
        es['byte_given_scale_bucket'][str(div)] = dict(entropy=sum(x['bytes']*x['entropy'] for x in groups)/flat.size, groups=groups)
    return bc, c, es

def compress_batch(blocks):
    compressor = zstd.ZstdCompressor(level=19,threads=0,write_checksum=True) if zstd else None
    decoder = zstd.ZstdDecompressor() if zstd else None
    result = collections.Counter()
    for data in blocks:
        z = zlib.compress(data,9)
        assert zlib.decompress(z) == data
        result['zlib9'] += len(z)
        result['zlib9_raw_fallback'] += min(len(z),len(data))
        if compressor:
            z = compressor.compress(data)
            assert decoder.decompress(z) == data
            result['zstd19'] += len(z)
            result['zstd19_raw_fallback'] += min(len(z),len(data))
    return result

def coding(q, bc, c, workers):
    flat = q.ravel()
    ec, sc = field_hist(c,EXP,16), field_hist(c,SM,16)
    models = {
        'huffman_byte': (lengths(c),256),
        'huffman_split': (lengths(ec)[EXP]+lengths(sc)[SM],32),
        'ans12_byte_model': (ans_costs(c),512),
        'ans12_split_model': (ans_costs(ec)[EXP]+ans_costs(sc)[SM],64),
        'huffman_exp_raw_sm': (lengths(ec)[EXP]+4,16),
        'ans12_exp_raw_sm_model': (ans_costs(ec)[EXP]+4,32),
    }
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for size in BLOCKS:
            totals = collections.Counter()
            # Bound queue/memory; no full-stream copies or unbounded futures.
            for start in range(0,flat.size,size*workers*32):
                batches = []
                for j in range(start,min(start+size*workers*32,flat.size),size*32):
                    batches.append([flat[i:min(i+size,flat.size)].tobytes() for i in range(j,min(j+size*32,flat.size),size)])
                for r in pool.map(compress_batch,batches): totals.update(r)
            nblock = (flat.size+size-1)//size
            # Common random-access container: 32B tensor header + (n+1) u64 offsets
            # and one per-block u32 mode/uncompressed-size word. 4B row scales stay raw.
            framing = 32+8*(nblock+1)+4*nblock
            sums = np.add.reduceat(bc,np.arange(0,len(bc),size//BASE),axis=0)
            for name,(cost,table) in models.items():
                # Split planes each need their own byte alignment (<=1 extra byte/block).
                bits = sums @ cost
                split = 'split' in name
                if name == 'huffman_split':
                    payload = int(np.ceil((sums @ lengths(ec)[EXP])/8).sum()+np.ceil((sums @ lengths(sc)[SM])/8).sum())
                else:
                    payload = int(np.ceil(bits/8).sum()) + (nblock if split else 0)
                # ANS model includes 32 independent states+u32 offsets per plane per block.
                # 4-byte state + 4-byte offset, with an additional 1B rounding allowance.
                if 'ans12' in name:
                    payload += nblock*32*9*(2 if split else 1)
                totals[name] = payload+table
            out[str(size)] = dict(blocks=nblock, framing_bytes=framing,
                formats={k:dict(payload_bytes=int(v),container_bytes=int(v)+framing,
                    with_scales_bytes=int(v)+framing+q.shape[0]*4) for k,v in totals.items()})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    wm, groups, excluded = inventory()
    sources = [TOOLS/'sglang-rtxpro6000/serve-fast.sh', TOOLS/'mtp-train/scripts/dh1/screen.py',
        TOOLS/'sglang-rtxpro6000/python/sglang/srt/layers/quantization/fp8.py',
        TOOLS/'sglang-rtxpro6000/python/sglang/kernels/ops/quantization/fp8_kernel.py',
        TOOLS/'sglang-rtxpro6000/python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh',
        ROOT/'specs/EXCLUSIVE_TIME_MAP_0907.md',MODEL/'model.safetensors.index.json',MAP,Path(__file__)]
    manifest = dict(torch=torch.__version__,numpy=np.__version__,zlib=zlib.ZLIB_RUNTIME_VERSION,
        zstandard=zstd.__version__ if zstd else None,workers=args.workers,omp_threads=12,nice=os.getpriority(os.PRIO_PROCESS,0),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),
        source_hashes={str(p):digest(p) for p in sources}, tensors=groups,excluded_weights=excluded)
    old = OUT/'manifest.json'
    if old.exists():
        previous = json.loads(old.read_text())
        assert previous['source_hashes'] == manifest['source_hashes'], 'Sources changed; use a fresh results directory'
    write(old,manifest)
    for i,g in enumerate(groups[:args.limit]):
        path = OUT/(g['name']+'.json')
        if path.exists(): continue
        start = time.monotonic()
        q,s,proof = quantize(g,wm)
        bc,c,es = statistics(q,s)
        code = coding(q,bc,c,args.workers)
        result = dict(**g,shape=list(q.shape),bytes=q.size,scale_bytes=s.nbytes,entropy=es,
            counts=c.tolist(),coding=code,proof=proof,seconds=time.monotonic()-start)
        write(path,result)
        print(f'{i+1}/{len(groups)} {g["name"]} {q.size/1e6:.2f} MB H={es["byte"]:.5f} {result["seconds"]:.1f}s',flush=True)
    assert not torch.cuda.is_initialized()
    print('Done; CUDA remains uninitialized.',flush=True)

if __name__ == '__main__': main()
