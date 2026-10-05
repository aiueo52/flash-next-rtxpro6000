"""dg1: stand-ins for the module's Triton kernels (compile only, skip) and a context to swap them in."""
from __future__ import annotations

import contextlib


class Warmup:
    """kernel[grid](...) -> kernel.warmup(..., grid=grid): compile (or load from cache), never launch."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        return lambda *a, **kw: self._fn.warmup(*a, grid=grid, **kw)


class Skip:
    """kernel[grid](...) -> nothing; isolates the other kernel in a timing graph."""

    def __getitem__(self, grid):
        return lambda *a, **kw: None


@contextlib.contextmanager
def patched(dmg, *, k1=None, k2=None, pdl=None):
    """Temporarily replace dmg's kernels and/or its PDL flag (read at call time by draft_moe_gemv)."""
    saved = (dmg._draft_moe_up_gate_kernel, dmg._draft_moe_down_kernel, dmg.PDL)
    try:
        if k1 is not None:
            dmg._draft_moe_up_gate_kernel = k1
        if k2 is not None:
            dmg._draft_moe_down_kernel = k2
        if pdl is not None:
            dmg.PDL = pdl
        yield
    finally:
        dmg._draft_moe_up_gate_kernel, dmg._draft_moe_down_kernel, dmg.PDL = saved
