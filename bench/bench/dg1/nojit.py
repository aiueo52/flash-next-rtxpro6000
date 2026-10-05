"""dg1: never JIT-compile FlashInfer inside a GPU scope (copy of fc_moe/nojit.py).

JitSpec.build() reruns ninja even when the .so is current; load compiled modules as-is and
refuse to build anything that is missing. Triton JIT is unaffected.
"""
import flashinfer.jit.core as _core


def _build(self, *a, **k):
    if self.is_compiled:
        return
    raise RuntimeError(f"dg1 nojit: {self.name} is not compiled; refusing to build it here")


for _cls in (_core.JitSpec, *_core.JitSpec.__subclasses__()):
    if "build" in _cls.__dict__:
        _cls.build = _build
