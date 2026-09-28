"""Copy this to a directory on PYTHONPATH as ``sitecustomize.py`` to arm the
routing logger inside the SGLang scheduler process without touching
``python/sglang``.

    mkdir -p /tmp/moe-route-hook
    cp bench/moe_smallm/sitecustomize_example.py /tmp/moe-route-hook/sitecustomize.py
    export PYTHONPATH=/tmp/moe-route-hook:/home/user/tools/flash-next-bench/bench
    export SGLANG_MOE_CENSUS_LOG=/tmp/routes
    export SGLANG_MOE_ROUTE_RING=4096        # optional: raw ids for replay / layer attribution
    ./serve-fast.sh w4

It wraps ``builtins.__import__`` and arms the moment SGLang's flashinfer-cutlass
runner module is *fully* imported.  Three things this must get right:

* **No imports inside the hook** and a reentrancy flag -- an ``import sys`` in
  the hook body recurses into itself (``RecursionError`` at interpreter start).
* Arm only when the module's ``__spec__._initializing`` is False, so we never
  fire in the middle of the quantization<->moe_runner circular import.
* Patch only; the CUDA buffers are allocated lazily on the first MoE call
  (``route_census``), because CUDA may not be up yet at import time.
"""
import builtins
import os
import sys

TARGET = "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass"


def _ready(name):
    m = sys.modules.get(name)
    if m is None:
        return False
    spec = getattr(m, "__spec__", None)
    return spec is None or not getattr(spec, "_initializing", False)


if os.environ.get("SGLANG_MOE_CENSUS_LOG"):
    _orig_import = builtins.__import__
    _st = {"armed": False, "busy": False}

    def _hooked(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        if not _st["armed"] and not _st["busy"] and _ready(TARGET):
            _st["busy"] = True
            try:
                builtins.__import__ = _orig_import
                from moe_smallm import route_census

                route_census.install()
                _st["armed"] = True
            except Exception as exc:
                print(f"[sitecustomize] route_census not armed: {exc!r}",
                      file=sys.stderr)
                builtins.__import__ = _hooked      # retry on a later import
            finally:
                _st["busy"] = False
        return mod

    builtins.__import__ = _hooked
