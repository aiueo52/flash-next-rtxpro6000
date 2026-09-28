"""Independent CPU arithmetic checks and detailed DH1 floor-path differences."""
import json
import numpy as np
import survey as s
from safetensors import safe_open

def main():
    need=[p for p in (s.MODEL/'model.safetensors.index.json',s.OUT/'manifest.json') if not p.is_file()]
    if need:
        raise SystemExit('audit_conversion.py: needs the model weights and the survey.py outputs ('
                         +', '.join(map(str,need))+'); these are not published in this repository')
    # All finite nonnegative BF16 values, including subnormals.
    a=s.torch.from_numpy(np.arange(0x7f80,dtype=np.uint16)).view(s.torch.bfloat16).float()
    pt=(a/448.0).numpy()
    ref=np.divide(a.numpy(),np.float32(448))
    assert np.array_equal(pt.view(np.uint32),ref.view(np.uint32))
    with np.errstate(over='ignore'):
        rinv=np.divide(np.float32(1),ref[ref>0])
    pinv=(1.0/s.torch.from_numpy(ref[ref>0])).numpy()
    assert np.array_equal(pinv.view(np.uint32),rinv.view(np.uint32))
    wm,groups,_=s.inventory()
    cases=[]
    for g in groups:
        r=json.loads((s.OUT/(g['name']+'.json')).read_text())
        assert r['counts'][127] == r['counts'][255] == 0, 'Unexpected FP8 NaN'
        if not r['proof']['floor_rows']:continue
        for key in g['sources']:
            with safe_open(s.MODEL/wm[key],framework='pt',device='cpu') as f:
                raw=f.get_tensor(key)
                for start in range(0,raw.shape[0],1024):
                    w=raw[start:start+1024].float()
                    a=w.abs().amax(1,keepdim=True)
                    ids=s.torch.nonzero((a<1e-10).ravel()).ravel()
                    if not len(ids):continue
                    w,a=w[ids],a[ids]
                    scale=a/448.0
                    q=(w*(1.0/scale)).clamp(-448,448).to(s.torch.float8_e4m3fn).view(s.torch.uint8)
                    d=(w/scale).clamp(-448,448).to(s.torch.float8_e4m3fn).view(s.torch.uint8)
                    dh=(w/(a.clamp_min(1e-10)/448.0)).clamp(-448,448).to(s.torch.float8_e4m3fn).view(s.torch.uint8)
                    for i,row in enumerate(ids):
                        cases.append(dict(source=key,row=start+int(row),absmax=float(a[i,0]),
                            scale=float(scale[i,0]),elements=w.shape[1],
                            serving_vs_unfloored_division_bytes=int((q[i]!=d[i]).sum()),
                            serving_vs_dh1_bytes=int((q[i]!=dh[i]).sum())))
    extra=[s.TOOLS/'sglang-rtxpro6000/python/sglang/srt/models/qwen4_exp_mtp.py',
        s.TOOLS/'sglang-rtxpro6000/python/sglang/srt/models/qwen4_exp.py',
        s.TOOLS/'sglang-rtxpro6000/python/sglang/srt/models/qwen3_5.py',
        s.TOOLS/'sglang-rtxpro6000/python/sglang/srt/qwen4_exp_dense_fp8.py',
        s.TOOLS/'sglang-rtxpro6000/python/sglang/kernels/ops/quantization/per_token_quant_fp8.py',
        s.TOOLS/'sglang-rtxpro6000/python/sglang/srt/layers/quantization/w8a16_gemv.py',
        s.MODEL/'config.json',s.ROOT/'prof/exclusive_time.py']
    s.write(s.OUT/'conversion_audit.json',dict(
        finite_nonnegative_bf16_values_checked=0x7f80,torch_numpy_scale_division_mismatches=0,
        torch_numpy_reciprocal_mismatches=0,fp8_nan_count=0,below_floor_rows=cases,
        source_hashes={str(p):s.digest(p) for p in extra},cuda_initialized=s.torch.cuda.is_initialized()))
    print('Independent division audit passed; below-floor rows:',len(cases))

if __name__=='__main__':main()
