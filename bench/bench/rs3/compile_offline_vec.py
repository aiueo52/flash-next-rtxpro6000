import importlib.util
import os
from pathlib import Path
import time
import sys
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import JITFunction
from triton.runtime.interpreter import InterpretedFunction

# Build JIT ASTs explicitly under the prescribed interpreter environment.
for name, language_module in list(sys.modules.items()):
    if name.startswith('triton.language'):
        for attribute, value in list(vars(language_module).items()):
            if isinstance(value, InterpretedFunction):
                setattr(language_module, attribute, JITFunction(fn=value.fn))

start = time.perf_counter()
path = Path(os.environ['WT']) / 'python/sglang/kernels/ops/speculative/sparse_rs.py'
spec = importlib.util.spec_from_file_location(name='rs3_vec_compile', location=path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._bv_combine = JITFunction(fn=module._bv_combine.fn)
kernel = JITFunction(fn=module._chain_sampling_sparse_kernel.fn)
signature = {name: '*i32' for name in ('Predicts', 'AcceptIndex', 'AcceptTokenNum')}
signature.update({name: '*i64' for name in ('Candidates', 'RetriveIndex', 'PI', 'QI')})
signature.update({name: '*fp32' for name in ('Coins', 'CoinsFinal', 'P', 'Q')})
for slots in (4, 8, 16):
    constants = dict(NUM_SLOTS=slots, K=64, KQ=64, KP=64, VOCAB=248320,
                     BLOCK_VERIFY=True, SP=triton.next_power_of_2(slots))
    warps = 8 if slots > 8 else 4
    print(f'COMPILE CPU-only explicit target cuda sm_120, slots={slots} num_warps={warps}', flush=True)
    compiled = triton.compile(src=ASTSource(fn=kernel, signature=signature, constexprs=constants),
                              target=GPUTarget(backend='cuda', arch=120, warp_size=32),
                              options={'num_warps': warps})
    print(f'PASS slots={slots} shared_bytes={compiled.metadata.shared} cubin_bytes={len(compiled.kernel)}', flush=True)
print(f'Offline compile elapsed={time.perf_counter() - start:.1f}s', flush=True)
