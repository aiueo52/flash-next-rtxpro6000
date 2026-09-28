"""Generate the FP8C report exclusively from exhaustive CPU measurements."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import collections
import hashlib
from pathlib import Path
import numpy as np
import survey as s

SUPPLEMENTAL = {'mtp.fc_embedding','mtp.fc_hidden'}

def subgroup(r):
    name=r['name']
    if name == 'lm_head':return 'lm_head.target'
    if name.startswith('draft.'):return 'lm_head.draft'
    cat=r['category']
    tail=name.rsplit('.',1)[-1]
    return cat+'.'+dict(in_proj_qkvz='in',qkv_proj='in',out_proj='out',o_proj='out',gate_up_proj='up',down_proj='down')[tail]

def category(r):
    return subgroup(r) if r['category']=='lm_head' else r['category']

def main():
    if not (s.OUT/'manifest.json').is_file():
        raise SystemExit(f'report.py: needs the survey outputs in {s.OUT} (run survey.py, validate_huffman.py, '
                         'audit_conversion.py and time_weights.py first; they read the model weights). The survey '
                         'outputs are not published in this repository')
    m=json.loads((s.OUT/'manifest.json').read_text())
    assert m['tensors'] and m['source_hashes'], 'manifest lists no tensors or source hashes: nothing to report'
    rs=[]
    for g in m['tensors']:
        r=json.loads((s.OUT/(g['name']+'.json')).read_text())
        h=json.loads((s.OUT/(g['name']+'.huffman.json')).read_text())
        assert h['fp8_sha256']==r['proof']['fp8_sha256'] and h['roundtrip_bytes']==2*r['bytes']
        for b,formats in h['formats'].items():r['coding'][b]['formats'].update(formats)
        rs.append(r)
    supplemental=[r for r in rs if r['name'] in SUPPLEMENTAL]
    rs=[r for r in rs if r['name'] not in SUPPLEMENTAL]
    tw=json.loads((s.OUT/'time_weights.json').read_text())['weights']
    cat=collections.defaultdict(list)
    sub=collections.defaultdict(list)
    for r in rs:cat[category(r)].append(r);sub[subgroup(r)].append(r)
    cats=['shared_expert','attn','linear_attn','lm_head.target','lm_head.draft','mtp_dense']
    formats=list(rs[0]['coding']['65536']['formats'])
    formats=[f for f in formats if 'raw_fallback' not in f]
    def before(rows):return sum(r['bytes']+r['scale_bytes'] for r in rows)
    def after(rows,b,f):return sum(r['coding'][str(b)]['formats'][f]['with_scales_bytes'] for r in rows)
    def saving(rows,b,f):return 1-after(rows,b,f)/before(rows)
    def effective(profile,b,f,which='mean'):
        total=sum(tw[profile].values())
        us=0
        for group,t in tw[profile].items():
            if group.startswith('BF16') or group=='unattributed_residual':continue
            rows=sub[group]
            v=saving(rows,b,f) if which=='mean' else (min if which=='min' else max)(saving([r],b,f) for r in rows)
            us+=t*v
        return us/total,us
    def coded_fraction(profile):
        return sum(t for g,t in tw[profile].items() if not g.startswith('BF16') and g!='unattributed_residual')/sum(tw[profile].values())
    def hmean(rows,field):return sum(r['bytes']*field(r) for r in rows)/sum(r['bytes'] for r in rows)
    lines=[]
    def p(t=''):lines.append(t)
    def table(headers,rows):
        esc=lambda v:str(v).replace('|',r'\|')
        p('| '+' | '.join(map(esc,headers))+' |');p('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:p('| '+' | '.join(map(esc,row))+' |')
        p()
    fm=lambda n:f'{n:,}'
    pct=lambda x:f'{100*x:.3f}%'
    best=max(formats,key=lambda f:effective('w16',65536,f)[0])
    actual_best=max(['zlib9','zstd19','huffman_byte_restart256','huffman_exp_raw_sm_restart256'],key=lambda f:effective('w16',65536,f)[0])
    p('# FP8C — CPU lossless coding survey of serving dense FP8 weights')
    p('\nDate: 2026-09-07. Scope: CPU only; no GPU work, no server, no installation, no commit.\n')
    p('## Verdict')
    p('\n**Storage gate: PASS. In-GEMV performance gate: NOT ESTABLISHED; no production GO.** '
      'The dense streams contain useful redundancy. Several measured codecs and exact Huffman size models exceed '
      'the >11% W4 / >13% W16 effective dense-block storage thresholds. A specialized restartable Huffman decoder '
      'is arithmetically plausible at ~1.6 TB/s, so entropy alone does not kill this direction. However, this CPU '
      'survey does not establish a decoder that preserves the existing GEMV roof while delivering the required net saving. '
      'Published general ANS/Huffman results do not demonstrate that specific fused FP8 case.\n')
    p(f'Ignoring decode cost, the best storage format in the tested set is `{best}` at 64 KiB: '
      f'W4 {pct(effective("w4",65536,best)[0])}, W16 {pct(effective("w16",65536,best)[0])}. '
      f'Among actual exhaustive round-trip codecs, `{actual_best}` wins by storage. '
      'For the decoder assessment, byte Huffman with 256-weight restarts is the strongest entropy-only candidate; '
      'exponent Huffman plus raw sign/mantissa is simpler but saves less.\n')
    table(['64 KiB format','W4 effective dense saving','W16 effective dense saving','Evidence'],[
      [f,pct(effective('w4',65536,f)[0]),pct(effective('w16',65536,f)[0]),
       'all bytes round-tripped' if f in ('zlib9','zstd19','huffman_byte_restart256','huffman_exp_raw_sm_restart256') else ('normalized cross-entropy model' if 'ans12' in f else 'exact size model, no lane restarts')]
      for f in [best,'zlib9','zstd19','huffman_byte_restart256','huffman_exp_raw_sm_restart256']])
    p('## 1. Coverage, serving conversion, and exclusions\n')
    p(f'**All {len(rs)} actual serving FP8 tensors measured**, across all 48 target layers '
      '(36 linear-attention, 12 full-attention), the MTP dense attention/shared-expert projections, '
      'the full target head (248,320 × 2,560), and the hot2_49152 draft head (49,152 × 2,560). '
      f'FP8 payload: **{fm(sum(r["bytes"] for r in rs))} bytes**; unchanged FP32 row scales: '
      f'**{fm(sum(r["scale_bytes"] for r in rs))} bytes**. No layer or block sampling.\n')
    p('The measured tensor is the **serving matrix**, in contiguous `[N,K]` order. Checkpoint '
      '`q/k/v`, shared `gate/up`, and GDN `qkv/z` are concatenated in loader order into '
      '`qkv_proj`, `gate_up_proj`, and `in_proj_qkvz`. The transposed parameter in `fp8.py` is a '
      'view, and the GEMV transposes it back; no physical column-major byte permutation is assumed. '
      'The manifest lists every checkpoint source for every serving matrix. TP=1 matches `serve-local.sh`.\n')
    p('**Two scope corrections from the actual code:** (1) `mtp.fc_embedding` and `mtp.fc_hidden` '
      'are ordinary BF16 `nn.Linear` instances in `_init_linear_projections`; the category selector '
      'mentions these names but does not quantize those instances. We measured their counterfactual FP8 streams '
      'as two supplemental tensors, excluded from serving totals. (2) GDN `in_proj_ba` and router GEMVs '
      'are BF16 despite sharing the `_w8a16_gemv` kernel name; their time remains in the dense-block denominator '
      'with zero FP8 saving. Indexer, HC, expert FP4, embeddings, norms, convolution, gates, and vision weights '
      'are outside the requested five-category FP8 weight set. HC has a separate opt-in FP8 path and is not '
      'silently included.\n')
    table(['Category','Serving tensors','FP8 bytes before','Raw scale bytes','FP8 + scales before'],[
      [c,len(cat[c]),fm(sum(r['bytes'] for r in cat[c])),fm(sum(r['scale_bytes'] for r in cat[c])),fm(before(cat[c]))] for c in cats]+
      [['TOTAL',len(rs),fm(sum(r['bytes'] for r in rs)),fm(sum(r['scale_bytes'] for r in rs)),fm(before(rs))]])
    p('Conversion follows the DH1 CPU pattern, with the current production rounding path made explicit:\n')
    p('```python\nw = bf16_weight.float()\nscale = w.abs().amax(dim=1, keepdim=True) / 448.0\ninv_scale = torch.where(scale == 0, 0.0, 1.0 / scale)\nq = (w * inv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)\n```\n')
    p('`Fp8LinearMethod.process_weights_after_loading` passes `group_size = weight.shape[-1]`. '
      'On CUDA this reaches the dedicated whole-row JIT in `fp8_kernel.py`, which computes an FP32 reciprocal '
      'and multiplies before E4M3 conversion. SM120 uses `precise_math` (the wrapper enables fast math only on SM90). '
      'The dedicated whole-row path does not use the group kernel’s 1e-10 floor. **20 observed rows '
      '(14 in layer 1 GDN input, 6 in layer 2 GDN input) fall below that floor**, and no row is all zero. '
      'We follow the dedicated serving path, without adding the group-kernel floor, and preserve '
      'the resulting scales raw. [conversion_audit.json](../bench/fp8coding/results/conversion_audit.json) '
      'records the separate below-floor comparison and independent FP32 division checks.\n')
    p(f'The literal DH1 floor-plus-`w/scale` expression differs from the serving reconstruction on '
      f'**{fm(sum(r["proof"]["dh1_byte_differences"] for r in rs))} / {fm(sum(r["bytes"] for r in rs))} bytes** '
      f'({100*sum(r["proof"]["dh1_byte_differences"] for r in rs)/sum(r["bytes"] for r in rs):.6f}%). '
      'This comparison includes reciprocal rounding and the 20 below-floor rows. '
      'Both hashes are recorded per tensor. This is why blindly reusing the DH1 byte stream would be incorrect. '
      '**CPU reconstruction is exhaustive and source-aligned; direct CPU-vs-CUDA bit parity is Not verified**, '
      'as required by the no-GPU scope. The independent E4M3 nearest-even reference checks all positive finite '
      'codes, their adjacent midpoints, both signs, and signed zero. The lossless-codec claim applies exactly '
      'to the reconstructed bytes, not to an unperformed GPU comparison.\n')
    p('Source evidence: [DH1](../../mtp-train/scripts/dh1/screen.py), '
      '[category selector](../../sglang-rtxpro6000/python/sglang/srt/qwen4_exp_dense_fp8.py), '
      '[FP8 loader](../../sglang-rtxpro6000/python/sglang/srt/layers/quantization/fp8.py), '
      '[quantization dispatcher](../../sglang-rtxpro6000/python/sglang/kernels/ops/quantization/fp8_kernel.py), '
      '[whole-row CUDA body](../../sglang-rtxpro6000/python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh), '
      '[MTP constructor](../../sglang-rtxpro6000/python/sglang/srt/models/qwen4_exp_mtp.py), '
      '[GEMV layout and shape plans](../../sglang-rtxpro6000/python/sglang/srt/layers/quantization/w8a16_gemv.py).\n')
    p('## 2. Zero-order and scale-conditional entropy\n')
    p('E4M3FN fields: sign `b>>7`, exponent `(b>>3)&15`, mantissa `b&7`. '
      'Split coding uses exponent plus **sign-and-mantissa** `((b>>7)<<3)|(b&7)`; '
      'it never drops the sign or signed zero. Entropies are empirical bits per original FP8 byte. '
      'Category values below average per-tensor entropies by byte count, rather than pooling distinct '
      'tensor distributions. `H(B|Q)` uses `Q=floor(4*log2(scale))` (quarter-octave buckets); '
      'octave `floor(log2(scale))` results and bucket populations are also in each JSON. '
      'Scales are already available to the decoder, but conditional model tables and access costs are '
      'not free, so conditional entropy is an information bound, not a codec size.\n')
    table(['Category','H(byte)','H(exponent)','H(mantissa)','H(sign)','H(sign,mantissa)','H(byte | quarter-octave scale)'],[
      [c]+[f'{hmean(cat[c],lambda r,key=k:r["entropy"][key]):.5f}' for k in ['byte','exponent','mantissa','sign','sign_mantissa']]+
      [f'{hmean(cat[c],lambda r:r["entropy"]["byte_given_scale_bucket"]["4"]["entropy"]):.5f}'] for c in cats])
    p(f'Across actual serving tensors, scale conditioning removes only '
      f'{hmean(rs,lambda r:r["entropy"]["byte"]-r["entropy"]["byte_given_scale_bucket"]["4"]["entropy"]):.5f} '
      'additional bits/weight on average. Most redundancy is in the exponent. Full-byte coding can also '
      'exploit exponent/mantissa dependence, explaining its advantage over two independent fields.\n')
    p('## 3. Block formats and accounting\n')
    p('All 16/32/64 KiB boundaries are in the uncompressed contiguous byte stream, including partial final blocks. '
      'zlib-9 and zstandard-19 use independent blocks, no dictionary, and every compressed block is '
      'decompressed and compared byte-for-byte. zstd uses the already installed `zstandard` module, '
      '`threads=0`, checksum enabled; six CPU compression workers operate on bounded batches. '
      'Raw-fallback sizes are also recorded and never count expanded blocks as savings.\n')
    p('A common random-access container adds 32 bytes/tensor, `(blocks+1)` 64-bit offsets, and '
      '4 bytes/block for mode/uncompressed length. Unchanged FP32 row scales are included in **both** '
      'before and after totals. Framing and codebooks are not hidden. No external alignment padding '
      'or sector-fetch amplification is assumed; these are discussed as decoder risks.\n')
    p('* `huffman_byte`: tensor-static canonical Huffman, 256 one-byte code lengths; exact '
      '`ceil(sum(count*length)/8)` per block.\n'
      '* `huffman_split`: separate 16-symbol exponent and sign/mantissa codebooks, independently '
      'byte-aligned planes, 32 code-length bytes. `huffman_exp_raw_sm` leaves the low four sign/mantissa '
      'bits raw and needs only 16 code-length bytes.\n'
      '* `ans12_*_model`: tensor-static frequencies normalized to 4,096, all observed symbols '
      'assigned nonzero frequency; cross entropy, not Shannon entropy alone. Tables use 2 bytes/symbol. '
      'The model adds 32 independent states/plane/block, 4-byte state + 4-byte offset + 1-byte rounding '
      'allowance each. It is an estimate, **not an encoded ANS bitstream or a proven finite-state upper bound**.\n'
      '* `huffman_*_restart256`: actual reference encoder/decoder, each 256-weight segment independently '
      'byte-aligned, one u32 start per segment plus a final u32 offset per block. Exact padding is '
      'measured, both variants round-trip every tensor byte. These restart costs are needed for '
      'parallel consumption; the no-restart Huffman models are storage comparisons only. '
      'Huffman segment payloads are actually encoded and decoded; the surrounding container/index '
      'has an explicit exact size specification but is not emitted as a persistent archive.\n')
    p('### Whole-corpus rates at all block sizes\n')
    table(['Format','16 KiB after bytes','saving','32 KiB after bytes','saving','64 KiB after bytes','saving'],[
      [f]+[v for b in s.BLOCKS for v in (fm(after(rs,b,f)),pct(saving(rs,b,f)))] for f in formats])
    p('### Explicit category before/after bytes, 64 KiB\n')
    table(['Category','Before: FP8+scales','zlib-9','zstd-19','Huffman byte (size model)','Huffman byte + restart256','Exponent Huffman + raw sign/mantissa + restart256'],[
      [c,fm(before(cat[c]))]+[fm(after(cat[c],65536,f)) for f in ['zlib9','zstd19','huffman_byte','huffman_byte_restart256','huffman_exp_raw_sm_restart256']] for c in cats]+
      [['TOTAL',fm(before(rs))]+[fm(after(rs,65536,f)) for f in ['zlib9','zstd19','huffman_byte','huffman_byte_restart256','huffman_exp_raw_sm_restart256']]])
    p('All per-category × format × block-size byte totals are in [aggregate.json](../bench/fp8coding/results/aggregate.json); '
      'all per-tensor counts, entropies, per-bucket populations, per-block-size codec totals, and proofs '
      'are in the [results directory](../bench/fp8coding/results/). The appendix below is an index, '
      'not a sampling subset.\n')
    p('## 4. W4 / W16 exclusive-time weighting\n')
    p('The [09-07 exclusive-time map](EXCLUSIVE_TIME_MAP_0907.md) is the authority for the dense block '
      '(2,385 / 3,656 µs, rounded) and median wall (8,919 / 16,083 µs). We reran its existing CPU '
      '`prof/exclusive_time.py` sweep on the same four stored X2 traces to recover all small families '
      'omitted from the printed top tables; no new profiling. [time_weights.json](../bench/fp8coding/results/time_weights.json) '
      'contains both the published-map weights used below and every recovered trace family.\n')
    trace_weights=json.loads((s.OUT/'time_weights.json').read_text())['trace_weights']
    p(f'**The printed map does not fully reconcile with the trace sweep:** the matching `_w8a16_gemv*` '
      f'families sum to {sum(trace_weights["w4"].values()):.3f} µs W4 / '
      f'{sum(trace_weights["w16"].values()):.3f} µs W16, versus the printed 2,385 / 3,656. '
      'This exceeds rounding. We do not silently replace the requested denominator. The primary '
      'calculation uses exactly the named dense-family exclusive times printed in the map, with '
      '**165 µs W4 / 3 µs W16 of unlisted residual assigned zero saving**. This is conservative: '
      '93.08% / 99.92% of the published dense time has an explicit named attribution; the residual '
      'is not claimed as compressible. Recovered trace attribution is retained as a sensitivity '
      'calculation, not a purported exact decomposition of the published map.\n')
    p('**Label correction:** the map calls verify grid `[80,6,1]` “attn qkv”, but the source shape plan '
      'maps that grid to the 2,560×6,144 `o_proj`; 13,312×2,560 QKV uses `[416,1,1]`. '
      'Their combined attention time is unchanged, and the calculation uses the matching tensor distribution. '
      'Similarly `[32,10,1]` is the BF16 router, not a shared FP8 projection.\n')
    table(['Subgroup','W4 exclusive µs','W4 dense share','W16 exclusive µs','W16 dense share'],[
      [g,f'{tw["w4"].get(g,0):.6f}',pct(tw['w4'].get(g,0)/sum(tw['w4'].values())),f'{tw["w16"].get(g,0):.6f}',pct(tw['w16'].get(g,0)/sum(tw['w16'].values()))] for g in sorted(set(tw['w4'])|set(tw['w16']))]+
      [['TOTAL',f'{sum(tw["w4"].values()):.6f}','100%',f'{sum(tw["w16"].values()):.6f}','100%']])
    p('For subgroup g, `s_g = 1 - sum(after_bytes_g)/sum(before_bytes_g)`. '
      'Effective dense saving is `S_W = sum(t_W,g * s_g) / sum(t_W,g)`, assigning zero to BF16 families and the unlisted residual. '
      'Potential step-wall reduction before decode overhead is `sum(t_W,g*s_g)/median_wall_W`. '
      'This is a traffic-proportional estimate on **exclusive** time, not an observed speedup. '
      'Time traces do not identify individual layer durations within a repeated shape, so within each '
      'shape group the all-layer compression ratio is byte-weighted. Equal-shaped layer matrices '
      'therefore receive equal weight; min/max sensitivity below avoids overstating that assumption.\n')
    table(['Format','Block KiB','W4 dense saving','W16 dense saving','W4 ideal saved µs / step %','W16 ideal saved µs / step %'],[
      [f,b//1024,pct(effective('w4',b,f)[0]),pct(effective('w16',b,f)[0]),
       f'{effective("w4",b,f)[1]:.3f} / {100*effective("w4",b,f)[1]/8919:.3f}%',
       f'{effective("w16",b,f)[1]:.3f} / {100*effective("w16",b,f)[1]/16083:.3f}%'] for f in formats for b in s.BLOCKS])
    table(['Restartable byte Huffman, 64 KiB','Min layer ratio per shape','Mean estimate','Max layer ratio per shape'],[
      [w]+[pct(effective(w,65536,'huffman_byte_restart256',q)[0]) for q in ('min','mean','max')] for w in ('w4','w16')])
    table(['Trace-only sensitivity, restart byte Huffman 64 KiB','Trace dense µs','Weighted saving'],[
      [w,f'{sum(trace_weights[w].values()):.3f}',pct(sum(t*saving(sub[g],65536,'huffman_byte_restart256') for g,t in trace_weights[w].items() if not g.startswith('BF16'))/sum(trace_weights[w].values()))] for w in ('w4','w16')])
    p('## 5. Decode-side arithmetic and feasibility\n')
    p('Use 1.6×10^12 **uncompressed weight bytes/s**, 188 SMs, and the local '
      '[G1 clock observations](G1_LOG.md) / [megakernel measurements](MEGAKERNEL_SPEC.md) of about 2.3 GHz. '
      'This is an explicit clock assumption from prior local measurements, not a live GPU query. '
      'The required per-SM output is `1.6e12/188 = 8.511 GB/s = 3.700 bytes/SM/cycle`. '
      'At 1.5 GHz it rises to 5.674 bytes/SM/cycle.\n')
    p('At 2.3 GHz, raw streaming permits only 10.24 / 20.48 / 40.96 ns per '
      '16/32/64 KiB across the whole GPU, or 1.925 / 3.850 / 7.700 µs per SM at even load. '
      'A single dependent symbol decoder per SM would need 3.7 symbols/cycle and cannot suffice. '
      'Even one 32-lane decoding warp/SM has only `32/3.7 = 8.65` cycles per lane-symbol, '
      'before GEMV work. With four warps it becomes 34.59 cycles per lane-symbol, but issue '
      'bandwidth is still shared; more warps hide latency rather than create ALU throughput.\n')
    p('A practical canonical byte-Huffman decoder needs: refill/amortized loads, bit-window '
      'alignment, primary LUT lookup, code/length extraction, bit-pointer advance, and occasional '
      'secondary lookup for long codes. A reasonable **instruction estimate, not SASS measurement**, '
      'is 8–16 scalar integer/bit instructions plus ~1 table read per weight, excluding the existing '
      'FP8-to-BF16 conversion and GEMV math. Exponent-only coding adds raw nibble extraction and '
      'reassembly (~3–4 simple operations), but shrinks the alphabet and table. For the target head '
      'the exponent Huffman maximum is 13 bits and 0.3694% of exponents exceed 8 bits; an 8-bit '
      'primary table with a fallback is small, but the byte-Huffman maximum is 18 bits. A flat '
      '2^18 table is inappropriate for per-CTA shared memory.\n')
    p('rANS per symbol needs `slot=state&4095`, symbol/frequency/cumulative lookup, '
      '`state=freq*(state>>12)+(slot-cum)`, a threshold test, and conditional byte refill, '
      'plus indexing/state maintenance. Budget roughly 10–18 scalar instructions and one table '
      'lookup/decoded symbol, not an integer divide (division is encoder-side). Full split coding '
      'runs two dependent decoders/weight; exponent-only leaves the sign/mantissa nibble raw. '
      'These are algorithmic estimates; real issue cost depends on packing, compiler lowering, '
      'bank conflicts, register spills, and latency hiding.\n')
    p('The [NVIDIA RTX Blackwell architecture whitepaper, pp. 9–11 and Appendix A]('
      'https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) '
      'describes 128 unified FP32/INT32 CUDA lanes/SM and shared use of those lanes. Using the optimistic '
      'one simple lane instruction/cycle roof, `188*128*2.3e9 = 55.35e12` lane-instructions/s. '
      'At 1.6 TB/s, 8 / 12 / 16 / 20 added instructions/byte consume 12.8 / 19.2 / 25.6 / 32.0 '
      'Tinst/s, or **23.1 / 34.7 / 46.3 / 57.8%** of that ideal issue capacity. '
      'If the relevant instruction mix sustains only 64 lanes/cycle, these fractions double. '
      'Thus the arithmetic does **not** prove impossibility, but also leaves no basis to treat decode '
      'as free, especially in M=1 GEMV, which uses the same scalar execution resources.\n')
    p('**Random access/coalescing is the harder constraint.** The 256-weight reference segments '
      'are a concrete format with measured padding/offsets; each can be decoded independently. '
      'They are not yet an optimal mapping to the production 128/256-wide K tiles and split-K grids. '
      'For K=2,560 or 640, flat 16/32/64 KiB blocks are not always row/tile aligned. A CTA must '
      'either fetch only the segment bytes it owns, or the offline packer must reorder segments '
      'into its tile without changing reconstructed weights. Fetching a whole 64 KiB block for '
      'each small tile would amplify traffic and invalidate these savings. Different lane streams '
      'also produce scattered compressed reads; staging a compressed tile into shared memory '
      'restores coalescing but adds shared writes/reads, synchronization, and occupancy pressure. '
      '64 KiB of staging can collide with GEMV multistage shared-memory use. Table expansion, '
      'tile-local offset layout, bank conflicts, and sector-rounded fetch costs were not measured.\n')
    p('### How little decode overhead fits\n')
    table(['Candidate (64 KiB)','Profile','Storage saving S','Required net bar b','Remaining dense-time overhead S-b','Equivalent serial decode floor f*1.6/(S-b) TB/s'],[
      [f,w,pct(effective(w,65536,f)[0]),pct(bar),pct(effective(w,65536,f)[0]-bar),f'{coded_fraction(w)*1.6/(effective(w,65536,f)[0]-bar):.2f}' if effective(w,65536,f)[0]>bar else 'no margin']
      for f in ['huffman_byte_restart256','huffman_exp_raw_sm_restart256'] for w,bar in [('w4',.11),('w16',.13)]])
    p(f'Here `time_new/time_old ≈ (1-S) + f*D/B_decode`, with `D=1.6 TB/s` and '
      f'coded time fraction `f={coded_fraction("w4"):.6f}` W4 / `{coded_fraction("w16"):.6f}` W16, treats decoder '
      'time as additive. The table shows why a separate or poorly overlapped decoder cannot work. '
      'A perfectly overlapped design instead needs the slowest pipeline stage to deliver at least '
      f'`1.6/(1-b/f)` = **{1.6/(1-.11/coded_fraction("w4")):.3f} TB/s W4 / '
      f'{1.6/(1-.13/coded_fraction("w16")):.3f} TB/s W16** to preserve the requested net dense-time '
      'reduction; merely matching 1.6 TB/s would remove all speedup if decode becomes the bottleneck. '
      'This is a uniform-reference-bandwidth screening model: individual GEMVs have different actual '
      'bandwidths, and the equations neglect non-weight traffic and unequal CTA load. '
      'The usable overhead margin must also pay for any sector padding or reduced occupancy.\n')
    p('### Published evidence and its limits\n')
    p('* [DietGPU, author README](https://github.com/facebookresearch/dietgpu) reports '
      '250–410 GB/s for generalized byte rANS and 250–600 GB/s for its floating-point codec on A100. '
      'It describes 4 KiB warp segments and advises sufficiently large arrays. These are broad '
      'codec ranges, not a promised FP8 decode-only result on this RTX GPU. 1,600/410 = 3.90× '
      'and 1,600/600 = 2.67×; even the favorable endpoints do not establish this target.\n'
      '* [ZipServ, ASPLOS 2026, §3.2](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf) '
      'reports L40S decode throughput at 43.7% of peak bandwidth for DietGPU and 76.5% for '
      'Huffman-based DFloat11, and demonstrates why a fused format can beat a materialized '
      'decompression pipeline. Those results concern BF16 weights and different kernels; they '
      'are evidence for co-design, not a transferable FP8 throughput guarantee.\n'
      '* [CUHD / Weißenberger GPU Huffman decoder](https://github.com/weissenberger/gpuhd) '
      'is a primary published implementation of parallel unpartitioned Huffman decoding. '
      'Its README gives no directly comparable TB/s result; no uncertain remembered number is '
      'used as evidence here.\n')
    p('A decode-to-DRAM pass would read compressed bytes, write the full bytes, and then make '
      'GEMV read them again: roughly `(1-S)+1+1 ≈ 2.85` weight-byte traffic units for S≈15%, '
      'versus 1 originally. It is excluded from a viable design. The only surviving direction '
      'is a small-table, independently restartable, tile-aware decoder fused into consumption, '
      'with nearly complete overlap and byte-identical reconstructed weights. **Plausible candidate: yes. '
      'A format shown to clear both the storage and net performance bars: no.** No GPU experiment '
      'is performed or authorized by this report.\n')
    p('## 6. Completeness and reproduction\n')
    table(['Check','Status'],[
      ['48 target layers; both heads; MTP attention/shared expert','Complete: 198 actual FP8 serving matrices'],
      ['MTP fc_embedding/fc_hidden','2 supplemental counterfactual FP8 matrices; excluded from actual totals'],
      ['16 / 32 / 64 KiB zlib-9 and zstd-19','Every block encoded, decoded, exact byte comparison'],
      ['Byte and exponent/raw-sign-mantissa Huffman restart256','Every byte encoded and decoded; hashes match independent reconstruction'],
      ['Byte / exponent / mantissa / sign entropy; scale-conditional byte entropy','All tensors, all bytes; octave and quarter-octave buckets'],
      ['Static no-restart Huffman','Exact sizes from actual symbol counts, code lengths, and block padding'],
      ['ANS','12-bit normalized cross-entropy model only; no ANS bitstream, timing, or GPU kernel'],
      ['Serving CUDA parity / GPU decode / integrated GEMV numerics / throughput','Not verified: CPU-only scope'],
      ['Within-family per-layer time shares','Not available; all-layer byte weighting plus min/max sensitivity'],
      ['Printed dense-total vs recovered trace-family sum','Mismatch disclosed; published-map primary weights, unlisted residual receives zero credit'],
      ['Workspace','Only bench/fp8coding and this report changed; no commit or server']])
    p('Run with the requested production interpreter and low priority:\n')
    p('```bash\ncd /home/user/tools/flash-next-bench\nbash bench/fp8coding/run_all.sh\n```\n')
    p(f'Environment recorded in [manifest.json](../bench/fp8coding/results/manifest.json): '
      f'Torch {m["torch"]}, NumPy {m["numpy"]}, zlib {m["zlib"]}, zstandard {m["zstandard"]}, '
      f'OMP threads {m["omp_threads"]}, nice {m["nice"]}, empty CUDA_VISIBLE_DEVICES; '
      '`torch.cuda.is_initialized()` remained false. Tensor-at-a-time conversion uses 1,024-row '
      'FP32 chunks; no whole BF16 model load, quantized-model artifact, cache of full FP8 weights, '
      'or serving import. Only the pure category selector is loaded by filename. '
      'The report and JSON are resumable from existing results, with source-hash drift checks.\n')
    p('### Supplemental BF16 MTP linears, hypothetical FP8 conversion only\n')
    table(['Name','Hypothetical FP8 bytes','H(byte)','zstd-19 64 KiB bytes incl. scale/framing'],[
      [r['name'],fm(r['bytes']),f'{r["entropy"]["byte"]:.5f}',fm(after([r],65536,'zstd19'))] for r in supplemental])
    p('## Appendix — every serving tensor\n')
    p('Target names abbreviate `model.language_model.layers.` to `L`; MTP names retain their prefix. '
      '`Hcond` is quarter-octave scale-conditional byte entropy. Savings include scales and framing. '
      'Every row links to its complete per-tensor 16/32/64 KiB results; the matching `.huffman.json` '
      'contains actual restartable Huffman round-trip proof.\n')
    table(['Serving tensor','FP8 bytes','H(byte)','H(exp)','H(mant)','Hcond','zstd19 64K saving','restart Huffman byte 64K saving'],[
      [f'[{r["name"].replace("model.language_model.layers.","L")}](../bench/fp8coding/results/{r["name"]}.json)',fm(r['bytes']),
       f'{r["entropy"]["byte"]:.5f}',f'{r["entropy"]["exponent"]:.5f}',f'{r["entropy"]["mantissa"]:.5f}',
       f'{r["entropy"]["byte_given_scale_bucket"]["4"]["entropy"]:.5f}',pct(saving([r],65536,'zstd19')),pct(saving([r],65536,'huffman_byte_restart256'))] for r in rs])
    (s.ROOT/'specs/FP8_CODING_SURVEY.md').write_text('\n'.join(lines)+'\n')
    aggregate=dict(actual_tensors=len(rs),supplemental_tensors=len(supplemental),before_bytes=before(rs),
        category={c:{str(b):{f:dict(before=before(cat[c]),after=after(cat[c],b,f),saving=saving(cat[c],b,f)) for f in formats} for b in s.BLOCKS} for c in cats},
        weighted={w:{str(b):{f:dict(saving=effective(w,b,f)[0],saved_us=effective(w,b,f)[1]) for f in formats} for b in s.BLOCKS} for w in tw})
    s.write(s.OUT/'aggregate.json',aggregate)
    hashes={p:s.digest(p) for p in m['source_hashes']}
    assert hashes==m['source_hashes'],'Input/source drift'
    conversion=json.loads((s.OUT/'conversion_audit.json').read_text())
    assert len(conversion['below_floor_rows'])==sum(r['proof']['floor_rows'] for r in rs)
    assert conversion['source_hashes'], 'conversion audit lists no serving sources: nothing verified'
    assert all(s.digest(p)==h for p,h in conversion['source_hashes'].items()),'Additional serving source drift'
    s.write(s.OUT/'audit.json',dict(actual_tensors=len(rs),supplemental_tensors=len(supplemental),
        source_hashes_unchanged=True,zero_rows=sum(r['proof']['zero_rows'] for r in rs),floor_rows=sum(r['proof']['floor_rows'] for r in rs),
        zlib_zstd_uncompressed_bytes_each=sum(r['bytes'] for r in rs)*3,
        huffman_roundtrip_bytes=sum(r['bytes'] for r in rs)*2,
        cuda_initialized=s.torch.cuda.is_initialized(),
        artifact_hashes={str(p.relative_to(s.ROOT)):s.digest(p) for p in sorted(s.HERE.glob('*.py'))},
        report_sha256=s.digest(s.ROOT/'specs/FP8_CODING_SURVEY.md')))
    print(json.dumps(dict(best_model=best,best_actual=actual_best,weighted64={w:aggregate['weighted'][w]['65536'] for w in tw}),indent=2))

if __name__=='__main__':main()
