"""Does a worktree's SGLang JIT cache (.cache/jit) hold every module it has built before?

The cache is content-addressed: a module loads when `cache.find_prebuilt` matches the hashes of
its recorded deps, whatever the file times say. So `ninja -n` in a build dir is not the check:
after `.cache` is copied to a new worktree it reports work although the server compiles nothing.
A MISS here means that module would compile at server start (unless the server never uses it).
Run through jitprobe.sh, which sets the serve-local.sh compile env.
"""
import pathlib
import sys

from sglang.kernels.jit.utils.compile import cache

root = pathlib.Path(sys.argv[1]) / ".cache" / "jit"
hit = miss = 0
for scope in sorted(root.glob("*/*/build-*")):
    if cache.find_prebuilt(scope=scope, module_name=scope.parent.name) is not None:
        hit += 1
        continue
    miss += 1
    leaves = [q for q in scope.iterdir() if q.name.startswith("deps-")]
    _, changed = cache._refresh(cache._read_deps(leaves[0]))
    print("MISS", scope.parent.name, scope.name, "changed:", list(changed)[:3] if changed else changed)
print(f"{root}: hit={hit} miss={miss}")
