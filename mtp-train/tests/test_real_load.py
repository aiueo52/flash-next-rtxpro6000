"""Load the REAL 31 mtp.* tensors on CPU and run one tiny forward (no GPU).

Skipped automatically when the checkpoint is not on disk (set MTP_MODEL_DIR).  Peak RSS is about
8 GiB (5.2 GiB of BF16 MTP weights + 2.5 GiB of shared embed/lm_head).
"""

from __future__ import annotations

import os
import sys
import time

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.config import DEFAULT_MODEL_DIR, MTPConfig  # noqa: E402
from mtptrain.model import MTPHead  # noqa: E402
from mtptrain.weights import export_state_dict, load_into, mtp_tensor_names  # noqa: E402

MODEL_DIR = DEFAULT_MODEL_DIR


def test_real_checkpoint_loads_and_runs():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors.index.json")):
        import pytest

        pytest.skip(f"checkpoint not found at {MODEL_DIR} (set MTP_MODEL_DIR)")
    cfg = MTPConfig.from_pretrained(MODEL_DIR)
    assert (cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank) == (2560, 4, 320)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (24, 2, 256)
    assert (cfg.num_experts, cfg.num_experts_per_tok) == (512, 10)
    assert cfg.rotary_dim == 64 and cfg.rope_theta == 10_000_000.0
    assert cfg.vocab_size == 248320
    assert cfg.dense_attention_max_len == 2048  # tokens, not blocks -- see config.py

    t0 = time.time()
    model = MTPHead(cfg).to(torch.bfloat16)
    names = load_into(model, MODEL_DIR, device="cpu", dtype=torch.bfloat16)
    model.freeze_non_trainable()
    print(f"  loaded {len(names)} tensors in {time.time()-t0:.1f}s")
    assert len(names) == 31 == len(mtp_tensor_names(MODEL_DIR))

    total, trainable = model.num_parameters()
    print(f"  params total={total/1e9:.4f}B trainable={trainable/1e9:.4f}B")
    assert abs(total - 2_607_161_856) < 5_000_000, total
    # frozen = router (512x2560) + indexer (640x2560 + 2x128)
    assert total - trainable == 512 * 2560 + 640 * 2560 + 2 * 128

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name

    torch.manual_seed(0)
    t = 8
    ids = torch.randint(0, cfg.vocab_size, (1, t))
    hc = (torch.randn(1, t, cfg.hc_size) * 0.5).to(torch.bfloat16)
    pos = torch.arange(t).view(1, t)
    t0 = time.time()
    with torch.no_grad():
        mixed, own_hc = model.forward_mixed(ids, hc, pos)
        logits = model.project_logits(mixed)
    print(
        f"  forward {t} rows in {time.time()-t0:.1f}s  "
        f"logits {tuple(logits.shape)} std={logits.float().std():.3f} "
        f"own_hc std={own_hc.float().std():.3f}"
    )
    assert logits.shape == (1, t, cfg.vocab_size)
    assert torch.isfinite(logits).all() and torch.isfinite(own_hc).all()
    assert 0.5 < float(logits.float().std()) < 100.0
    top = logits[0, -1].argmax().item()
    assert 0 <= top < cfg.vocab_size

    exported = export_state_dict(model)
    assert set(exported) == set(mtp_tensor_names(MODEL_DIR))
    assert all(v.dtype == torch.bfloat16 for v in exported.values())


def main() -> int:
    try:
        test_real_checkpoint_loads_and_runs()
        print("PASS test_real_checkpoint_loads_and_runs")
        return 0
    except BaseException as exc:  # noqa: BLE001  (includes pytest's Skipped)
        import traceback

        traceback.print_exc()
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
