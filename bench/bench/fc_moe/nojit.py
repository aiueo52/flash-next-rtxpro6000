"""fc-moe: never JIT-compile inside a GPU/bench scope (2026-10-01 freeze rule).

FlashInfer's JitSpec.build() writes build.ninja and runs ninja even when the .so is current; in
the private cache copy the regenerated build.ninja embeds a different -I path, so ninja would
rebuild all 97 TUs.  The .so there is byte-identical to production's (md5 checked), so load it
as-is and refuse to build anything that is missing.
"""
import flashinfer.jit.core as _core


def _build(self, *a, **k):
    if self.is_compiled:
        return
    raise RuntimeError(f"fc-moe nojit: {self.name} is not compiled; build it in a capped scope first")


for _cls in (_core.JitSpec, *_core.JitSpec.__subclasses__()):
    if "build" in _cls.__dict__:
        _cls.build = _build
