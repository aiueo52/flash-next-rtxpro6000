"""C1: numerical check of the top-1 probability added to draft_topk1_postprocess.

Also asserts that with chain_probs=None the kernel output is bit-identical to
the pre-C1 behaviour (topk_p == 1.0, same argmax, same position advance).
"""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import os, sys, torch
if os.environ.get("SGLANG_PYTHON"):  # python/ of a flash-next-fast checkout; else the installed sglang
    sys.path.insert(0, os.environ["SGLANG_PYTHON"])
from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess

dev = "cuda"
torch.manual_seed(0)
fails = 0
for bs, vocab in [(1, 49152), (4, 49152), (8, 32768), (3, 100000), (1, 8192)]:
    for scale in (0.5, 3.0, 12.0):
        logits = (torch.randn(bs, vocab, device=dev, dtype=torch.float32) * scale)
        ref_p = torch.softmax(logits.double(), dim=-1).amax(dim=-1).float()
        ref_i = logits.argmax(dim=-1)

        pos_a = torch.zeros(bs, dtype=torch.int64, device=dev)
        p_a, i_a = draft_topk1_postprocess(logits, pos_a)

        pos_b = torch.zeros(bs, dtype=torch.int64, device=dev)
        chain = torch.zeros((bs, 4), dtype=torch.float32, device=dev)
        p_b, i_b = draft_topk1_postprocess(logits, pos_b, chain_probs=chain,
                                           draft_token_column=2)

        ok_ident = (torch.equal(p_a, p_b) and torch.equal(i_a, i_b)
                    and torch.equal(pos_a, pos_b) and bool((p_a == 1.0).all()))
        err = (chain[:, 2] - ref_p).abs().max().item()
        ok_idx = torch.equal(i_a.view(-1), ref_i)
        ok_zero = bool((chain[:, [0, 1, 3]] == 0).all())
        bad = (not ok_ident) or (not ok_idx) or (not ok_zero) or err > 2e-6
        fails += bad
        print(f"bs={bs:2d} vocab={vocab:6d} scale={scale:4.1f} "
              f"max|p-ref|={err:.3e} argmax_ok={ok_idx} identical_default={ok_ident} "
              f"other_cols_untouched={ok_zero} p_range=[{chain[:,2].min():.4f},{chain[:,2].max():.4f}]"
              + ("  <-- FAIL" if bad else ""))

# hot_token_id mapping path (production config uses a 49152-entry map)
vocab = 49152
logits = torch.randn(2, vocab, device=dev) * 4.0
hot = torch.randperm(200000, device=dev)[:vocab].to(torch.int64)
pos = torch.zeros(2, dtype=torch.int64, device=dev)
dt = torch.zeros((2, 5), dtype=torch.long, device=dev)
chain = torch.zeros((2, 5), dtype=torch.float32, device=dev)
p, i = draft_topk1_postprocess(logits, pos, dt, 3, hot_token_id=hot, chain_probs=chain)
ref_p = torch.softmax(logits.double(), -1).amax(-1).float()
ok = torch.equal(dt[:, 3], hot[logits.argmax(-1)]) and (chain[:, 3] - ref_p).abs().max() < 2e-6
fails += not ok
print(f"hot_token_id path: token_ok+prob_ok={ok}  max|p-ref|={(chain[:,3]-ref_p).abs().max():.3e}")
print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
