"""Actual CPU canonical-Huffman encoder/decoder; exhaustive byte round trips.

Each 256-weight segment is independently byte-aligned, with one u32 offset.
The exponent-only variant stores the four sign/mantissa bits verbatim.
Numba is already installed; JIT cache and outputs stay in this directory.
This is validation of the format, not a GPU implementation or speed proxy.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NUMBA_CACHE_DIR'] = str(__import__('pathlib').Path(__file__).resolve().parent/'results/numba-cache')
from numba import njit
import numpy as np
import json
import time
import survey as s

def book(lens):
    codes = np.zeros(len(lens),np.int64)
    value, previous = 0,0
    nodes = [[-1,-1,-1]]
    for sym in sorted(np.flatnonzero(lens),key=lambda i:(lens[i],i)):
        n = int(lens[sym])
        value <<= n-previous
        codes[sym] = value
        node = 0
        for bit in range(n-1,-1,-1):
            b = (value>>bit)&1
            if nodes[node][b] == -1:
                nodes[node][b] = len(nodes)
                nodes.append([-1,-1,-1])
            node = nodes[node][b]
        assert nodes[node][2] == -1
        nodes[node][2] = int(sym)
        value += 1
        previous = n
    assert max(lens) < 32
    return codes,np.array(nodes,np.int64)

@njit(cache=True)
def roundtrip(data, lens, codes, tree, exponent_only):
    segments = (data.size+255)//256
    sizes = np.zeros(segments,np.int64)
    restored = np.empty(data.size,np.uint8)
    for seg in range(segments):
        begin, end = seg*256,min(data.size,(seg+1)*256)
        encoded = np.zeros(2048,np.uint8)
        bits = 0
        # Deliberately simple reference bit packing: no GPU throughput claim.
        for i in range(begin,end):
            x = int(data[i])
            sym = ((x>>3)&15) if exponent_only else x
            for j in range(lens[sym]-1,-1,-1):
                encoded[bits//8] |= ((codes[sym]>>j)&1) << (7-bits%8)
                bits += 1
        nbyte = (bits+7)//8
        low = np.zeros(128,np.uint8)
        if exponent_only:
            for i in range(begin,end):
                x = int(data[i])
                sm = ((x>>7)<<3)|(x&7)
                low[(i-begin)//2] |= sm << (4*((i-begin)%2))
        pos = 0
        for i in range(begin,end):
            node = 0
            while tree[node,2] < 0:
                b = (int(encoded[pos//8])>>(7-pos%8))&1
                pos += 1
                node = tree[node,b]
                assert node >= 0
            x = tree[node,2]
            if exponent_only:
                sm = (int(low[(i-begin)//2])>>(4*((i-begin)%2)))&15
                x = (x<<3)|(sm&7)|((sm>>3)<<7)
            restored[i] = x
        assert pos == bits
        sizes[seg] = nbyte + ((end-begin+1)//2 if exponent_only else 0)
    return restored,sizes

def tests():
    rng = np.random.default_rng(20260907)
    for data in [np.zeros(257,np.uint8),np.arange(256,dtype=np.uint8),rng.integers(0,256,65539,dtype=np.uint8)]:
        for exp in (False,True):
            symbols = ((data>>3)&15) if exp else data
            lens = s.lengths(np.bincount(symbols,minlength=16 if exp else 256))
            codes,tree = book(lens)
            out,sizes = roundtrip(data,lens,codes,tree,exp)
            assert np.array_equal(out,data)
            # Exact segment padding and sign/mantissa size accounting.
            expected = sum((int(lens[symbols[i:i+256]].sum())+7)//8 + ((min(256,len(data)-i)+1)//2 if exp else 0) for i in range(0,len(data),256))
            assert int(sizes.sum()) == expected
    # Independent nearest-even E4M3 reference, including signed zero/ties.
    codes = np.arange(127,dtype=np.uint8)
    e,m = (codes>>3)&15,codes&7
    values = np.where(e==0,m.astype(float)*2**-9,(1+m/8)*np.exp2(e.astype(float)-7)).astype(np.float32)
    x = np.concatenate([values,(values[:-1]+values[1:])/2])
    x = np.concatenate([x,-x])
    distance = np.abs(np.abs(x[:,None])-values[None,:])
    # Ties select the code with even low bit.
    reference = np.array([min(np.flatnonzero(d == d.min()),key=lambda v:(v%2,v)) for d in distance],np.uint8)
    reference |= np.signbit(x).astype(np.uint8)<<7
    actual = s.torch.from_numpy(x).to(s.torch.float8_e4m3fn).view(s.torch.uint8).numpy()
    assert np.array_equal(reference,actual)

def main():
    tests()
    wm,groups,_ = s.inventory()
    for i,g in enumerate(groups):
        p = s.OUT/(g['name']+'.huffman.json')
        if p.exists(): continue
        source = s.OUT/(g['name']+'.json')
        if not source.exists():
            print('Awaiting survey result: '+g['name'],flush=True)
            break
        r = json.loads(source.read_text())
        start = time.monotonic()
        q,scale,proof = s.quantize(g,wm)
        assert proof == r['proof']
        out = {}
        c = np.array(r['counts'],np.int64)
        for exp in (False,True):
            name = 'huffman_exp_raw_sm_restart256' if exp else 'huffman_byte_restart256'
            lens = s.lengths(s.field_hist(c,s.EXP,16) if exp else c)
            codes,tree = book(lens)
            restored,sizes = roundtrip(q.ravel(),lens,codes,tree,exp)
            assert np.array_equal(restored,q.ravel())
            for block in s.BLOCKS:
                blocks = (q.size+block-1)//block
                framing = 32+8*(blocks+1)+4*blocks
                # Each segment's start offset in its block, followed by an end offset.
                total = int(sizes.sum())+4*(len(sizes)+blocks)+len(lens)+framing
                out.setdefault(str(block),{})[name] = dict(container_bytes=total,with_scales_bytes=total+scale.nbytes)
        s.write(p,dict(name=g['name'],fp8_sha256=proof['fp8_sha256'],bytes=q.size,
            roundtrip_bytes=q.size*2,formats=out,seconds=time.monotonic()-start))
        print(f'{i+1}/{len(groups)} {g["name"]} verified {q.size*2/1e6:.2f} MB in {time.monotonic()-start:.1f}s',flush=True)
    assert not s.torch.cuda.is_initialized()

if __name__ == '__main__': main()
