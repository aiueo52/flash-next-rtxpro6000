"""Locked GPU replay: real NVFP4 expert weights, P3 routing, graph input changes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import model_shapes as MS
from .bench_moe import _load_weights
from .runners import ensure_sglang_imports

ROOT = Path(__file__).resolve().parents[2]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, choices=(0, 1), default=1)
    joint = bool(ap.parse_args().joint)
    import sys
    sys.path.insert(0, str(ROOT/'bench/quality'))
    from p4_policy import reference
    ensure_sglang_imports()
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
    args = SimpleNamespace(ckpt=MS.DEFAULT_CKPT, layer=4, synthetic=False, num_experts=None)
    shape = MS.read_checkpoint_shape(args.ckpt)
    lw = _load_weights(args, shape, 'cuda')
    qs = [lw.w13_input_scale_quant, lw.w13_blockscale_swizzled.view(torch.int32), lw.g1_alphas,
          lw.w2_input_scale_quant, lw.w2_blockscale_swizzled.view(torch.int32), lw.g2_alphas]
    manifest = json.loads((ROOT/'prune/p4-manifest.json').read_text())
    results = []
    for width in (4, 16):
        entry = manifest['widths'][str(width)]
        tables = np.load(ROOT/'prune'/entry['file'])
        ids = torch.empty((width, 10), dtype=torch.int32, device='cuda')
        w = torch.empty((width, 10), dtype=torch.float32, device='cuda')
        inv = torch.empty(width, dtype=torch.float32, device='cuda')
        table = torch.empty(512, dtype=torch.float32, device='cuda')
        out_ids, out_w = torch.empty_like(ids), torch.empty_like(w)
        x = torch.randn((width, lw.shape.hidden), device='cuda', dtype=torch.bfloat16)
        out = torch.empty_like(x)
        base = dict(output=out, input=x, token_selected_experts=ids, token_final_scales=w,
            fc1_expert_weights=lw.w13_weight.view(torch.long), fc2_expert_weights=lw.w2_weight.view(torch.long),
            output_dtype=torch.bfloat16, quant_scales=qs, activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=width, use_fused_finalize=False)
        extra = dict(contrib_table=table, contrib_inv_norm=inv, contrib_ids_out=out_ids,
            contrib_scales_out=out_w, contrib_threshold=entry['threshold'], contrib_joint=joint)
        samples = []
        for source in entry['sources']:
            with np.load(ROOT/source['file']) as z:
                for index in np.linspace(0, len(z['ids'])-1, 24, dtype=int):
                    samples.append((z['ids'][index].reshape(width, 10).copy(),
                        z['w'][index].reshape(width, 10).copy(),
                        np.float32(1)/z['h_norm'][index], tables[int(z['layer'][index])]))
        def set_input(sample):
            a, b, c, d = sample
            ids.copy_(torch.from_numpy(a)); w.copy_(torch.from_numpy(b))
            inv.copy_(torch.from_numpy(c)); table.copy_(torch.from_numpy(d))
        set_input(samples[0])
        for _ in range(3):
            cutlass_fused_moe(**base, **extra)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cutlass_fused_moe(**base, **extra)
        total_drop = 0
        for index, sample in enumerate(samples):
            set_input(sample)
            graph.replay()
            torch.cuda.synchronize()
            a, b, c, d = sample
            drop = reference(a, b, d, c, entry['threshold'], joint=joint)
            expected_ids = np.where(drop, -1, a)
            expected_w = np.where(drop, np.float32(0), b)
            np.testing.assert_array_equal(ids.cpu(), a)
            np.testing.assert_array_equal(w.cpu(), b)
            np.testing.assert_array_equal(out_ids.cpu(), expected_ids)
            np.testing.assert_array_equal(out_w.cpu(), expected_w)
            candidate_out = out.clone()
            # Same effective routing provided explicitly must produce identical deterministic output.
            ref_kwargs = dict(base, token_selected_experts=out_ids, token_final_scales=out_w)
            cutlass_fused_moe(**ref_kwargs)
            torch.cuda.synchronize()
            assert torch.equal(candidate_out, out), (width, index, 'finalize mismatch')
            total_drop += int(drop.sum())
        # Nonfinite denominator and invalid original route fail closed for whole call.
        for kind in ('nan', 'invalid_id'):
            set_input(samples[0])
            if kind == 'nan': inv[0] = float('nan')
            else: ids[0, 1] = -1
            graph.replay(); torch.cuda.synchronize()
            assert torch.equal(ids, out_ids) and torch.equal(w, out_w)
        results.append(dict(width=width, joint=joint, calls=len(samples), routes_dropped=total_drop,
            immutable=True, graph_replay=True, deterministic_output_equal=True, invalid_fallback=True))
        print(json.dumps(results[-1]), flush=True)
    print('P4_GPU_PASS ' + json.dumps(results), flush=True)

if __name__ == '__main__':
    main()
