#!/usr/bin/env python3
"""Write trained ``mtp.*`` tensors into a NEW model directory.

The result is a drop-in serving directory: every untouched shard is a symlink
to the original file, so the multi-GB checkpoint is not copied.  Only the shards
that actually hold ``mtp.*`` tensors are materialised.

Two strategies (both produce a directory whose ``model.safetensors.index.json``
is self-consistent and whose shards contain each tensor name exactly once --
that uniqueness matters because SGLang's loader iterates every tensor of every
file the index references, not only the names it looks up):

``rewrite`` (default)
    Rewrite the shards that contain ``mtp.*`` with the new values in place of
    the old ones.  Tensor names, shard names and the index are unchanged.
    Smallest diff, nothing else can go wrong.

``new-shard``
    Rewrite those same shards *without* the ``mtp.*`` tensors, put all 31 into
    one fresh ``model-mtpft-00001.safetensors``, and repoint the index.  It
    costs exactly the same I/O.

Safety: ``--src`` and ``--dst`` must not be the same directory or nested in
each other (checked on the resolved real paths, so symlinks do not get
around it).  Every trained tensor is checked against the source checkpoint --
the name set, and each tensor's dtype and shape -- before anything is written.
Shard names from the index must be plain file names (no ``/``, ``..`` or
absolute paths).  No file the output links to or reads may resolve into
``--dst`` -- not even through an intermediate symlink -- because ``--dst`` is
what gets replaced (a model built *from* ``--dst`` cannot be the ``--src`` of
a rebuild of ``--dst``).  Untouched files are linked to their real path, so
the output never depends on a chain of links; a side file that resolves into
``--dst`` is copied instead.

The output is built in a temporary directory next to ``--dst`` and fully
verified there (the ``verify`` checks below); only then is it renamed into
place.  With ``--force`` an existing ``--dst`` is moved aside first and removed
only once the new directory is in place.  On any failure ``--dst`` is left as
it was.

``--ckpt`` is read with ``torch.load(..., weights_only=True)`` (tensors and
plain containers only, no arbitrary pickled objects) or with safetensors.

Usage (CPU only, no CUDA):

    CUDA_VISIBLE_DEVICES="" python writeback/write_mtp.py \
        --src $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
        --dst $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-mtpft \
        --ckpt out/best-mtp.safetensors --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.cli import StrictParser  # noqa: E402
from mtptrain.config import DEFAULT_MODEL_DIR  # noqa: E402
from mtptrain.weights import MTP_PREFIX, read_weight_map  # noqa: E402

INDEX_FILE = "model.safetensors.index.json"
NEW_SHARD = "model-mtpft-00001.safetensors"

# safetensors header dtype -> torch dtype (the ones a checkpoint can hold)
_ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


def load_trained(path: str) -> Dict[str, torch.Tensor]:
    """Accept either a training checkpoint (.pt) or a plain ``mtp.*`` safetensors.

    A ``.pt`` file is loaded with ``weights_only=True``: only tensors and plain
    containers are unpickled, so a tampered file cannot run code.  Every
    checkpoint ``mtptrain.train`` writes loads that way.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"--ckpt {path}: no such file")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model"] if "model" in payload else payload
    if not state:
        raise SystemExit(f"{path}: holds no tensors")
    out = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            raise SystemExit(f"{path}: {k} is a {type(v).__name__}, not a tensor")
        key = k if k.startswith(MTP_PREFIX) else MTP_PREFIX + k
        out[key] = v.detach().to(torch.bfloat16).contiguous()
    return out


def shard_tensors(src: str, shard: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(os.path.join(src, shard))


def shard_metadata(src: str, shard: str) -> Dict[str, str]:
    from safetensors import safe_open

    with safe_open(os.path.join(src, shard), framework="pt") as f:
        return dict(f.metadata() or {})


def _inside(child: str, parent: str) -> bool:
    return child != parent and os.path.commonpath([child, parent]) == parent


def _within(path: str, root: str) -> bool:
    return path == root or _inside(path, root)


def check_name(name, what: str = "shard") -> str:
    """A file name from an index must be a plain name inside its directory."""
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name or "\0" in name or os.path.isabs(name)):
        raise SystemExit(f"{what} name {name!r} is not a plain file name")
    return name


def resolution(path: str, limit: int = 64) -> Tuple[str, List[str]]:
    """-> (real path, every path visited while resolving ``path``).

    Resolves one component at a time, like the kernel, so a path that only
    passes *through* a directory or link (and ends somewhere else) is seen.
    """
    visited: List[str] = []
    todo = [c for c in os.path.abspath(path).split(os.sep) if c]
    cur, hops = os.sep, 0
    while todo:
        part = todo.pop(0)
        if part == ".":
            continue
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        nxt = os.path.join(cur, part)
        visited.append(nxt)
        if os.path.islink(nxt):
            hops += 1
            if hops > limit:
                raise SystemExit(f"{path}: too many levels of symbolic links")
            target = os.readlink(nxt)
            if os.path.isabs(target):
                cur = os.sep
            todo = [c for c in target.split(os.sep) if c] + todo
        else:
            cur = nxt
    return cur, visited


def plan_sources(src: str, dst: str, shards: List[str]) -> Dict[str, Tuple[str, str]]:
    """Decide, before anything is written, how each file of ``src`` is reused.

    -> {name: ("link" | "copy" | "read", real path)}.  Refuses a referenced
    shard that is missing, not a regular file, or resolves into ``dst`` (or
    through it); a side file that resolves into ``dst`` is copied.
    """
    guards = {os.path.abspath(dst), os.path.realpath(dst)}

    def into_dst(visited: List[str]) -> bool:
        return any(_within(v, g) for v in visited for g in guards)

    plan: Dict[str, Tuple[str, str]] = {}
    for name in shards:
        real, visited = resolution(os.path.join(src, check_name(name)))
        if into_dst(visited):
            raise SystemExit(f"{src}/{name} resolves into --dst ({real}); replacing "
                             "--dst would destroy it -- write to a new directory")
        if not os.path.isfile(real):
            raise SystemExit(f"index references a missing or non-regular file: {src}/{name}")
        plan[name] = ("read", real)
    for name in sorted(os.listdir(src)):
        if name in plan or name == INDEX_FILE:
            continue
        path = os.path.join(src, name)
        if os.path.isdir(path):
            continue
        real, visited = resolution(path)
        if not os.path.isfile(real):
            raise SystemExit(f"{path} is not a regular file (dangling link?)")
        plan[name] = ("copy" if into_dst(visited) else "link", real)
    return plan


def resolve_dirs(src: str, dst: str) -> Tuple[str, str]:
    """-> (src, dst) as absolute paths; refuses overlapping directories.

    The comparison uses the real paths, so ``dst`` being a symlink to ``src``
    (or to a directory inside it) is caught as well.  ``--force`` on an
    overlapping pair would otherwise delete the source checkpoint.
    """
    for opt, val in (("--src", src), ("--dst", dst)):
        if not (val or "").strip():  # abspath("") would be the working directory
            raise SystemExit(f"{opt} is empty (MTP_MODEL_DIR set but empty?)")
    src = os.path.abspath(os.path.expanduser(src))
    dst = os.path.abspath(os.path.expanduser(dst))
    if not os.path.isfile(os.path.join(src, INDEX_FILE)):
        raise SystemExit(f"--src {src}: no {INDEX_FILE} (not a checkpoint directory)")
    if os.path.lexists(dst) and not os.path.isdir(dst):
        raise SystemExit(f"--dst {dst} exists and is not a directory")
    real_src, real_dst = os.path.realpath(src), os.path.realpath(dst)
    if real_src == real_dst:
        raise SystemExit(f"--src and --dst are the same directory ({real_src})")
    if _inside(real_dst, real_src):
        raise SystemExit(f"--dst ({real_dst}) is inside --src ({real_src})")
    if _inside(real_src, real_dst):
        raise SystemExit(f"--src ({real_src}) is inside --dst ({real_dst})")
    return src, dst


def validate(
    src: str, trained: Dict[str, torch.Tensor], strategy: str
) -> Tuple[dict, Dict[str, str], List[str], List[str]]:
    """Check every trained tensor against the source; touches no output.

    -> (index, weight_map, mtp tensor names, shards holding them).  Raises
    SystemExit listing every problem: shard names that are not plain file
    names, tensor names missing from or unknown to the index, a name the index
    places in a shard that does not hold it, and any dtype or shape that
    differs from the original tensor.
    """
    from safetensors import safe_open

    if strategy not in ("rewrite", "new-shard"):
        raise SystemExit(f"unknown strategy {strategy!r}")
    weight_map = dict(read_weight_map(src))
    with open(os.path.join(src, INDEX_FILE)) as handle:
        index = json.load(handle)
    for shard in set(weight_map.values()):
        check_name(shard)

    mtp_names = sorted(k for k in weight_map if k.startswith(MTP_PREFIX))
    if not mtp_names:
        raise SystemExit(f"{src}/{INDEX_FILE} lists no {MTP_PREFIX}* tensors")
    missing = set(mtp_names) - set(trained)
    extra = set(trained) - set(mtp_names)
    if missing or extra:
        raise SystemExit(
            f"trained tensors do not match the checkpoint: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    affected = sorted({weight_map[k] for k in mtp_names})
    if strategy == "new-shard" and NEW_SHARD in weight_map.values() and NEW_SHARD not in affected:
        raise SystemExit(f"{NEW_SHARD} already exists in {src} and holds other tensors")

    problems: List[str] = []
    for shard in affected:
        path = os.path.join(src, shard)
        if not os.path.exists(path):
            raise SystemExit(f"index references a missing file: {path}")
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for name in mtp_names:
                if weight_map[name] != shard:
                    continue
                if name not in keys:
                    problems.append(f"{name}: index says {shard}, which does not hold it")
                    continue
                meta = f.get_slice(name)
                want_shape = tuple(meta.get_shape())
                want_dtype = _ST_DTYPES.get(meta.get_dtype(), meta.get_dtype())
                got = trained[name]
                if tuple(got.shape) != want_shape:
                    problems.append(
                        f"{name}: trained shape {tuple(got.shape)} != {want_shape}"
                    )
                if got.dtype != want_dtype:
                    problems.append(f"{name}: trained dtype {got.dtype} != {want_dtype}")
    if problems:
        head = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise SystemExit(f"trained tensors do not fit the checkpoint:\n  {head}{more}")
    return index, weight_map, mtp_names, affected


def _write_tree(
    src: str,
    out: str,
    trained: Dict[str, torch.Tensor],
    strategy: str,
    index: dict,
    weight_map: Dict[str, str],
    mtp_names: List[str],
    affected: List[str],
    plan: Dict[str, Tuple[str, str]],
) -> dict:
    """Populate the (empty, temporary) directory ``out``."""
    from safetensors.torch import save_file

    real_out = os.path.realpath(out)

    def target(name: str) -> str:
        path = os.path.join(out, check_name(name, "output"))
        if os.path.realpath(os.path.dirname(path)) != real_out:
            raise SystemExit(f"{name} would be written outside the output directory")
        # Never write through a symlink: it would land in the source checkpoint.
        if os.path.lexists(path):
            raise SystemExit(f"refusing to overwrite {path}")
        return path

    # 1. link (to the real file) or copy every file we are not rewriting
    linked = copied = 0
    for name, (how, real) in sorted(plan.items()):
        if name in affected or how == "read":
            if name not in affected:  # an indexed shard without mtp tensors
                os.symlink(real, target(name))
                linked += 1
            continue
        if how == "copy":
            shutil.copy2(real, target(name))
            copied += 1
        else:
            os.symlink(real, target(name))
            linked += 1

    # 2. materialise the affected shards
    written_bytes = 0
    for shard in affected:
        tensors = shard_tensors(src, shard)
        meta = shard_metadata(src, shard)
        for name in list(tensors):
            if not name.startswith(MTP_PREFIX):
                continue
            if strategy == "rewrite":
                tensors[name] = trained[name]  # dtype/shape checked in validate()
            else:
                del tensors[name]
        if tensors:
            out_path = target(shard)
            save_file(tensors, out_path, metadata=meta or None)
            written_bytes += os.path.getsize(out_path)
        else:  # a shard that held nothing but mtp tensors
            weight_map = {k: v for k, v in weight_map.items() if v != shard}
        del tensors

    # 3. the new shard, if requested
    if strategy == "new-shard":
        payload = {k: trained[k] for k in mtp_names}
        out_path = target(NEW_SHARD)
        save_file(payload, out_path, metadata={"format": "pt"})
        written_bytes += os.path.getsize(out_path)
        for k in mtp_names:
            weight_map[k] = NEW_SHARD

    # 4. index
    total = 0
    for name in sorted({v for v in weight_map.values()}):
        total += os.path.getsize(os.path.realpath(os.path.join(out, name)))
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = total
    index["metadata"]["mtp_finetuned_from"] = src
    with open(target(INDEX_FILE), "w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)

    return {
        "strategy": strategy,
        "symlinked_files": linked,
        "copied_files": copied,
        "rewritten_shards": affected,
        "written_gib": round(written_bytes / 2**30, 3),
        "mtp_tensors": len(mtp_names),
    }


def _remove(path: str) -> None:
    if os.path.islink(path) or not os.path.isdir(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)  # does not follow the shard symlinks inside


def _sync_tree(root: str) -> None:
    """fsync every regular file written into ``root`` and ``root`` itself."""
    from mtptrain.fileio import fsync_dir, fsync_file

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.islink(path) and os.path.isfile(path):
            fsync_file(path)
    fsync_dir(root)


def _swap_into_place(tmp: str, dst: str) -> None:
    """Rename ``tmp`` to ``dst``; an existing ``dst`` is removed only afterwards.

    Two renames (Linux cannot atomically replace a non-empty directory): a
    crash exactly between them leaves the old tree as ``.<dst>.old-*`` and the
    new, verified one as ``.<dst>.tmp-*`` next to ``--dst``; nothing is lost.
    """
    from mtptrain.fileio import fsync_dir

    _sync_tree(tmp)
    old: Optional[str] = None
    if os.path.lexists(dst):
        old = tempfile.mkdtemp(prefix=f".{os.path.basename(dst)}.old-",
                               dir=os.path.dirname(dst))
        os.rmdir(old)  # only the unique name is wanted
        os.rename(dst, old)
    try:
        os.rename(tmp, dst)
    except BaseException:
        if old is not None:
            os.rename(old, dst)
        raise
    fsync_dir(os.path.dirname(dst))
    if old is not None:
        _remove(old)


def build(
    src: str,
    dst: str,
    trained: Dict[str, torch.Tensor],
    strategy: str = "rewrite",
    force: bool = False,
) -> dict:
    src, dst = resolve_dirs(src, dst)
    if os.path.lexists(dst) and not force:
        raise SystemExit(f"{dst} exists (pass --force to replace)")
    index, weight_map, mtp_names, affected = validate(src, trained, strategy)
    plan = plan_sources(src, dst, sorted(set(weight_map.values())))

    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    # Same filesystem as dst, so the final rename is atomic.
    tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(dst)}.tmp-", dir=parent)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o777 & ~umask)  # mkdtemp's 0700 would outlive the rename
        report = _write_tree(src, tmp, trained, strategy, index, weight_map,
                             mtp_names, affected, plan)
        # Full check of the new tree while the old --dst is still untouched.
        report["verify"] = verify(src, tmp, trained)
        _swap_into_place(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"dst": dst, **report}


def verify(src: str, dst: str, trained: Dict[str, torch.Tensor]) -> dict:
    """Every tensor in the index resolves, appears once, and matches.

    * the tensor set equals the source's, each shard name is a plain file
      name, each tensor sits in exactly the shard the index names, and no
      shard holds a tensor the index does not list;
    * every tensor's dtype and shape equal the source's;
    * every ``mtp.*`` tensor equals the trained one;
    * every other tensor of a rewritten shard is byte-identical to the
      source's;
    * every other file (linked shards, config, tokenizer, ...) has the same
      full content as the source's file of that name (``file_digest``,
      cached on size, mtime, inode and device), and no file is missing or
      extra.
    """
    from safetensors import safe_open

    src = os.path.abspath(os.path.expanduser(src))
    dst = os.path.abspath(os.path.expanduser(dst))
    src_map = read_weight_map(src)
    dst_map = read_weight_map(dst)
    if set(src_map) != set(dst_map):
        raise SystemExit(
            "tensor set changed: "
            f"{sorted(set(src_map) - set(dst_map))[:5]} / "
            f"{sorted(set(dst_map) - set(src_map))[:5]}"
        )
    for shard in set(dst_map.values()) | set(src_map.values()):
        check_name(shard)

    def header(root: str, shard: str) -> Dict[str, tuple]:
        with safe_open(os.path.join(root, shard), framework="pt") as f:
            return {k: (f.get_slice(k).get_dtype(), tuple(f.get_slice(k).get_shape()))
                    for k in f.keys()}

    src_meta: Dict[str, tuple] = {}
    for shard in sorted(set(src_map.values())):
        for k, v in header(src, shard).items():
            if src_map.get(k) == shard:
                src_meta[k] = v

    seen: Dict[str, str] = {}
    for shard in sorted(set(dst_map.values())):
        path = os.path.join(dst, shard)
        if not os.path.exists(path):
            raise SystemExit(f"index references a missing file: {shard}")
        for name, meta in header(dst, shard).items():
            if name in seen:
                raise SystemExit(
                    f"{name} appears in both {seen[name]} and {shard}"
                )
            seen[name] = shard
            if name in src_meta and meta != src_meta[name]:
                raise SystemExit(f"{name}: {meta} in the output != {src_meta[name]} in the source")
    unresolved = [k for k, v in dst_map.items() if seen.get(k) != v]
    if unresolved:
        raise SystemExit(f"index points at the wrong shard for {unresolved[:5]}")
    orphans = sorted(set(seen) - set(dst_map))
    if orphans:
        raise SystemExit(f"shards contain tensors the index does not list: {orphans[:5]}")

    checked = 0
    for name, shard in sorted(dst_map.items()):
        if not name.startswith(MTP_PREFIX):
            continue
        with safe_open(os.path.join(dst, shard), framework="pt") as f:
            got = f.get_tensor(name)
        want = trained[name]
        if got.dtype != want.dtype or got.shape != want.shape:
            raise SystemExit(f"{name}: {got.dtype}{tuple(got.shape)} != trained")
        if not torch.equal(got, want):
            raise SystemExit(f"{name}: values differ from the trained tensor")
        checked += 1

    # The shards that held mtp.* tensors are rewritten: every other tensor in
    # them must equal the source's.  Every other file (linked or copied) must
    # have the source file's full content -- compared by content digest, not
    # by where a link points.
    from mtptrain.fileio import file_digest

    rewritten_shards = {v for k, v in src_map.items() if k.startswith(MTP_PREFIX)}
    rewritten_shards |= {v for k, v in dst_map.items() if k.startswith(MTP_PREFIX)}
    rewritten = 0
    for shard in sorted(rewritten_shards & set(dst_map.values())):
        path = os.path.join(dst, shard)
        names = [k for k, v in dst_map.items() if v == shard and not k.startswith(MTP_PREFIX)]
        with safe_open(path, framework="pt") as f:
            for name in names:
                b = f.get_tensor(name)
                with safe_open(os.path.join(src, src_map[name]), framework="pt") as g:
                    a = g.get_tensor(name)
                if not (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
                    raise SystemExit(f"{name} changed unexpectedly")
                rewritten += 1

    def files(root: str) -> List[str]:
        return sorted(n for n in os.listdir(root)
                      if not os.path.isdir(os.path.join(root, n)))

    src_files, dst_files = files(src), files(dst)
    expected = {n for n in src_files if n not in rewritten_shards or n in dst_map.values()}
    expected |= set(dst_map.values())
    if set(dst_files) != expected:
        raise SystemExit(f"files differ from the source's: missing "
                         f"{sorted(expected - set(dst_files))[:5]}, unexpected "
                         f"{sorted(set(dst_files) - expected)[:5]}")
    same = 0
    for name in dst_files:
        if name == INDEX_FILE or name in rewritten_shards or name == NEW_SHARD:
            continue
        if file_digest(os.path.join(dst, name)) != file_digest(os.path.join(src, name)):
            raise SystemExit(f"{name}: content differs from {src}/{name}")
        same += 1
    return {"mtp_checked": checked, "rewritten_tensors_checked": rewritten,
            "unchanged_files_checked": same, "tensors": len(dst_map)}


def main() -> None:
    p = StrictParser(description=__doc__,
                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_MODEL_DIR,
                   help="original checkpoint (env MTP_MODEL_DIR); never modified")
    p.add_argument("--dst", required=True,
                   help="new model directory to create (must not exist, must not "
                        "overlap --src)")
    p.add_argument("--ckpt", required=True,
                   help="training checkpoint (.pt, loaded with weights_only=True) "
                        "or mtp safetensors")
    p.add_argument("--strategy", default="rewrite", choices=["rewrite", "new-shard"])
    p.add_argument("--force", action="store_true",
                   help="replace an existing --dst; the new directory is built "
                        "and validated first, the old one removed last")
    p.add_argument("--verify", action="store_true",
                   help="re-run the verification on the final --dst (it always "
                        "runs on the new tree before it replaces anything)")
    args = p.parse_args()

    trained = load_trained(args.ckpt)
    report = build(args.src, args.dst, trained, args.strategy, args.force)
    print(json.dumps(report, indent=2))
    if args.verify:
        print(json.dumps(verify(args.src, args.dst, trained), indent=2))
        print("VERIFY OK")


if __name__ == "__main__":
    main()
