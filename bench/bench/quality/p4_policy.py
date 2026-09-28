"""Independent batched CPU reference with explicit FP32 device arithmetic."""
import numpy as np


def reference(ids, weights, table, inv_norm, threshold, joint=True):
    ids = np.asarray(ids)
    weights = np.asarray(weights, dtype=np.float32)
    inv_norm = np.asarray(inv_norm, dtype=np.float32)
    single_call = ids.ndim == 2
    if single_call:
        ids, weights, inv_norm = ids[None], weights[None], inv_norm[None]
    n, rows, k = ids.shape
    drop = np.zeros_like(ids, dtype=bool)
    if rows not in (4, 16) or k != 10 or not np.isfinite(threshold) or threshold <= 0:
        return drop[0] if single_call else drop
    good = ((ids >= 0) & (ids < 512) & np.isfinite(weights) & (weights >= 0)).all((1, 2))
    good &= (np.isfinite(inv_norm) & (inv_norm > 0)).all(1)
    safe_ids = np.clip(ids, 0, 511)
    table = np.asarray(table, dtype=np.float32)
    norm = table[safe_ids] if table.ndim == 1 else table[np.arange(n)[:, None, None], safe_ids]
    score = (weights * norm) * inv_norm[:, :, None]
    good &= (np.isfinite(score) & np.isfinite(norm) & (norm >= 0)).all((1, 2))
    keys = (np.arange(n)[:, None, None] * 512 + safe_ids).ravel()
    count = np.bincount(keys, minlength=n*512)
    top = np.zeros_like(ids, dtype=bool)
    np.put_along_axis(top, weights.argmax(2)[..., None], True, 2)
    protected = np.bincount(keys[top.ravel()], minlength=n*512) > 0
    # Each eligible group has at most two FP32 addends; float64 bin sum then
    # FP32 rounding equals the ordered single FP32 addition in the prologue.
    summed = np.bincount(keys, weights=score.ravel(), minlength=n*512).astype(np.float32)
    chosen = ((count == 1) | ((count == 2) & joint)) & ~protected & (summed < np.float32(threshold))
    drop = chosen[keys].reshape(ids.shape) & good[:, None, None]
    return drop[0] if single_call else drop
