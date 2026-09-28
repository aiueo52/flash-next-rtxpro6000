"""Write-back tests on a fake tiny checkpoint directory (CPU only).

Builds a miniature model dir whose layout mirrors the real one -- several
shards, one that mixes ``mtp.*`` with other tensors, one that holds *only*
``mtp.*`` tensors, plus config/tokenizer side files -- then runs both
write-back strategies and asserts:

  * unchanged shards and side files are symlinks into the source (no copies)
  * every ``mtp.*`` tensor in the output equals the trained value
  * every other tensor is byte-identical to the source
  * the index is self-consistent: each name appears in exactly one shard and
    the index points at that shard (this is what stops SGLang from loading the
    stale copy of a rewritten tensor)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import load_file, save_file  # noqa: E402

from writeback.write_mtp import build, load_trained, verify  # noqa: E402

MTP_SHAPES = {
    "mtp.fc_embedding.weight": (8, 8),
    "mtp.fc_hidden.weight": (8, 8),
    "mtp.pre_fc_norm_embedding.weight": (8,),
    "mtp.pre_fc_norm_hidden.weight": (32,),
    "mtp.layers.0.self_attn.q_proj.weight": (16, 8),
    "mtp.layers.0.mlp.experts.gate_up_proj": (4, 6, 8),
}
OTHER_SHAPES = {
    "model.language_model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.language_model.layers.1.mlp.gate.weight": (4, 8),
    "lm_head.weight": (12, 8),
    "model.language_model.embed_tokens.weight": (12, 8),
}


def _make_fake_checkpoint(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    g = torch.Generator().manual_seed(11)

    def rnd(shape):
        return (torch.randn(shape, generator=g) * 0.1).to(torch.bfloat16)

    shards = {
        # a big shard mixing mtp with everything else
        "model-bf16-00001.safetensors": {
            "model.language_model.layers.0.self_attn.q_proj.weight": rnd((8, 8)),
            "mtp.layers.0.self_attn.q_proj.weight": rnd((16, 8)),
            "mtp.layers.0.mlp.experts.gate_up_proj": rnd((4, 6, 8)),
            "mtp.pre_fc_norm_hidden.weight": rnd((32,)),
        },
        # a shard that holds ONLY mtp tensors (exercises the empty-shard path)
        "model-bf16-00002.safetensors": {
            "mtp.fc_embedding.weight": rnd((8, 8)),
            "mtp.fc_hidden.weight": rnd((8, 8)),
            "mtp.pre_fc_norm_embedding.weight": rnd((8,)),
        },
        # a shard with no mtp tensors at all -> must be symlinked
        "model-bf16-00003.safetensors": {
            "model.language_model.layers.1.mlp.gate.weight": rnd((4, 8)),
            "lm_head.weight": rnd((12, 8)),
            "model.language_model.embed_tokens.weight": rnd((12, 8)),
        },
    }
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(root, shard), metadata={"format": "pt"})
        for name in tensors:
            weight_map[name] = shard
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, handle)
    for side in ("config.json", "tokenizer.json", "chat_template.jinja"):
        with open(os.path.join(root, side), "w") as handle:
            handle.write("{}\n")


def _trained_state(scale: float = 7.0) -> dict:
    g = torch.Generator().manual_seed(99)
    return {
        name: (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)
        for name, shape in MTP_SHAPES.items()
    }


def _check_output(src: str, dst: str, trained: dict, strategy: str) -> None:
    index = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    assert set(weight_map) == set(MTP_SHAPES) | set(OTHER_SHAPES)

    # shard 3 and the side files are symlinks; rewritten shards are real files
    assert os.path.islink(os.path.join(dst, "model-bf16-00003.safetensors"))
    assert os.path.islink(os.path.join(dst, "config.json"))
    assert not os.path.islink(os.path.join(dst, "model-bf16-00001.safetensors"))

    seen = {}
    for shard in sorted(set(weight_map.values())):
        for name, tensor in load_file(os.path.join(dst, shard)).items():
            assert name not in seen, (name, shard, seen.get(name))
            seen[name] = shard
            if name.startswith("mtp."):
                assert torch.equal(tensor, trained[name]), name
    assert seen == weight_map

    if strategy == "new-shard":
        assert all(
            weight_map[n] == "model-mtpft-00001.safetensors" for n in MTP_SHAPES
        )
        # the mtp-only shard must have disappeared from the index entirely
        assert "model-bf16-00002.safetensors" not in set(weight_map.values())
    else:
        assert weight_map["mtp.fc_embedding.weight"] == "model-bf16-00002.safetensors"

    # untouched tensors are byte-identical
    src_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))[
        "weight_map"
    ]
    for name in OTHER_SHAPES:
        a = load_file(os.path.join(src, src_map[name]))[name]
        b = load_file(os.path.join(dst, weight_map[name]))[name]
        assert torch.equal(a, b), name


def test_writeback_rewrite_strategy():
    tmp = tempfile.mkdtemp(prefix="mtpft-rewrite-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        trained = _trained_state()
        report = build(src, dst, trained, strategy="rewrite")
        assert report["mtp_tensors"] == 6
        assert set(report["rewritten_shards"]) == {
            "model-bf16-00001.safetensors",
            "model-bf16-00002.safetensors",
        }
        assert report["symlinked_files"] == 4  # shard 3 + three side files
        _check_output(src, dst, trained, "rewrite")
        print("  verify:", json.dumps(verify(src, dst, trained)))
    finally:
        shutil.rmtree(tmp)


def test_writeback_new_shard_strategy():
    tmp = tempfile.mkdtemp(prefix="mtpft-newshard-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        trained = _trained_state(scale=3.0)
        report = build(src, dst, trained, strategy="new-shard")
        assert os.path.exists(os.path.join(dst, "model-mtpft-00001.safetensors"))
        _check_output(src, dst, trained, "new-shard")
        print("  verify:", json.dumps(verify(src, dst, trained)))
    finally:
        shutil.rmtree(tmp)


def test_writeback_rejects_mismatched_tensor_set():
    tmp = tempfile.mkdtemp(prefix="mtpft-bad-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        trained = _trained_state()
        trained.pop("mtp.fc_hidden.weight")
        try:
            build(src, dst, trained)
        except SystemExit as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("expected a SystemExit for a missing tensor")

        shutil.rmtree(dst, ignore_errors=True)
        trained = _trained_state()
        trained["mtp.fc_hidden.weight"] = torch.zeros(3, 3, dtype=torch.bfloat16)
        try:
            build(src, dst, trained)
        except SystemExit as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("expected a SystemExit for a bad shape")
    finally:
        shutil.rmtree(tmp)


def test_load_trained_accepts_training_checkpoint():
    tmp = tempfile.mkdtemp(prefix="mtpft-ckpt-")
    try:
        state = {k[len("mtp.") :]: v for k, v in _trained_state().items()}
        path = os.path.join(tmp, "latest.pt")
        torch.save({"step": 3, "model": state}, path)
        loaded = load_trained(path)
        assert set(loaded) == set(MTP_SHAPES)
        assert all(v.dtype == torch.bfloat16 for v in loaded.values())
    finally:
        shutil.rmtree(tmp)


def _snapshot(root: str) -> dict:
    """name -> (is symlink, link target or file bytes) for a whole directory."""
    out = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.islink(path):
            out[name] = ("link", os.readlink(path))
        else:
            with open(path, "rb") as handle:
                out[name] = ("file", handle.read())
    return out


def _expect_exit(fn, *words) -> str:
    try:
        fn()
    except SystemExit as exc:
        msg = str(exc)
        for word in words:
            assert word in msg, (word, msg)
        return msg
    raise AssertionError(f"expected a SystemExit mentioning {words}")


def _no_leftovers(parent: str) -> None:
    stray = [n for n in os.listdir(parent) if ".tmp-" in n or ".old-" in n]
    assert not stray, stray


def test_writeback_refuses_same_directory():
    tmp = tempfile.mkdtemp(prefix="mtpft-same-")
    try:
        src = os.path.join(tmp, "src")
        _make_fake_checkpoint(src)
        before = _snapshot(src)
        trained = _trained_state()
        _expect_exit(lambda: build(src, src, trained, force=True), "same directory")
        # a symlink to the source and a non-normalised spelling are the same dir
        alias = os.path.join(tmp, "alias")
        os.symlink(src, alias)
        _expect_exit(lambda: build(src, alias, trained, force=True), "same directory")
        _expect_exit(lambda: build(src, src + "/./", trained, force=True),
                     "same directory")
        assert _snapshot(src) == before
    finally:
        shutil.rmtree(tmp)


def test_writeback_refuses_nested_directories():
    tmp = tempfile.mkdtemp(prefix="mtpft-nested-")
    try:
        src = os.path.join(tmp, "model")
        _make_fake_checkpoint(src)
        before = _snapshot(src)
        trained = _trained_state()
        _expect_exit(lambda: build(src, os.path.join(src, "out"), trained, force=True),
                     "inside --src")
        # dst = a parent of src: --force would have removed the source with it
        _expect_exit(lambda: build(src, tmp, trained, force=True), "inside --dst")
        # the same through a symlinked parent
        link = os.path.join(tmp, "link")
        os.symlink(src, link)
        _expect_exit(lambda: build(link, os.path.join(src, "x", "y"), trained),
                     "inside --src")
        assert _snapshot(src) == before
        assert sorted(os.listdir(src)) == sorted(before)
    finally:
        shutil.rmtree(tmp)


def test_writeback_bad_shape_leaves_existing_dst_untouched():
    tmp = tempfile.mkdtemp(prefix="mtpft-badshape-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        good = _trained_state()
        build(src, dst, good)
        before = _snapshot(dst)
        for strategy in ("rewrite", "new-shard"):
            bad = _trained_state()
            bad["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(
                4, 6, 9, dtype=torch.bfloat16
            )
            _expect_exit(lambda: build(src, dst, bad, strategy=strategy, force=True),
                         "gate_up_proj", "shape")
            assert _snapshot(dst) == before
            _no_leftovers(tmp)
        verify(src, dst, good)
    finally:
        shutil.rmtree(tmp)


def test_writeback_rejects_wrong_dtype_and_unknown_name():
    tmp = tempfile.mkdtemp(prefix="mtpft-dtype-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        bad = _trained_state()
        bad["mtp.fc_hidden.weight"] = bad["mtp.fc_hidden.weight"].float()
        _expect_exit(lambda: build(src, dst, bad), "dtype")
        assert not os.path.lexists(dst)
        bad = _trained_state()
        bad["mtp.not_in_the_checkpoint.weight"] = torch.zeros(2, dtype=torch.bfloat16)
        _expect_exit(lambda: build(src, dst, bad, strategy="new-shard"), "extra")
        assert not os.path.lexists(dst)
        _no_leftovers(tmp)
    finally:
        shutil.rmtree(tmp)


def test_writeback_new_shard_checks_shapes():
    tmp = tempfile.mkdtemp(prefix="mtpft-newshard-shape-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        bad = _trained_state()
        bad["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(31, dtype=torch.bfloat16)
        _expect_exit(lambda: build(src, dst, bad, strategy="new-shard"),
                     "pre_fc_norm_hidden", "shape")
        assert not os.path.lexists(dst)
        _no_leftovers(tmp)
    finally:
        shutil.rmtree(tmp)


def test_writeback_force_replaces_with_a_valid_dst():
    tmp = tempfile.mkdtemp(prefix="mtpft-force-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        src_before = _snapshot(src)
        first = _trained_state(scale=7.0)
        build(src, dst, first)
        _expect_exit(lambda: build(src, dst, first), "exists")
        second = _trained_state(scale=2.0)
        report = build(src, dst, second, strategy="new-shard", force=True)
        assert report["dst"] == dst
        _check_output(src, dst, second, "new-shard")
        verify(src, dst, second)
        _no_leftovers(tmp)
        assert _snapshot(src) == src_before
        # the output directory gets the usual umask mode, not mkdtemp's 0700
        umask = os.umask(0)
        os.umask(umask)
        assert os.stat(dst).st_mode & 0o777 == 0o777 & ~umask
    finally:
        shutil.rmtree(tmp)


def test_writeback_failure_while_writing_keeps_old_dst():
    import safetensors.torch as st

    tmp = tempfile.mkdtemp(prefix="mtpft-fail-")
    real_save = st.save_file
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        good = _trained_state()
        build(src, dst, good)
        before = _snapshot(dst)

        def broken_save(*args, **kwargs):
            raise OSError("disk full (simulated)")

        st.save_file = broken_save
        try:
            build(src, dst, _trained_state(scale=2.0), force=True)
        except OSError as exc:
            assert "simulated" in str(exc)
        else:
            raise AssertionError("expected the simulated write failure")
        finally:
            st.save_file = real_save
        assert _snapshot(dst) == before
        _no_leftovers(tmp)
        verify(src, dst, good)
    finally:
        st.save_file = real_save
        shutil.rmtree(tmp)


def _links_into(root: str, forbidden: str) -> list:
    """Symlinks in ``root`` that resolve into ``forbidden`` or to nothing."""
    bad = []
    for name in os.listdir(root):
        if not os.path.islink(os.path.join(root, name)):
            continue
        real = os.path.realpath(os.path.join(root, name))
        if real == os.path.realpath(forbidden) or real.startswith(
            os.path.realpath(forbidden) + os.sep
        ) or not os.path.isfile(real):
            bad.append(name)
    return bad


def test_writeback_refuses_a_source_that_links_into_dst():
    """A built from B, then --src A --dst B --force: must not eat B."""
    tmp = tempfile.mkdtemp(prefix="mtpft-loop-")
    try:
        orig, b, a = (os.path.join(tmp, n) for n in ("orig", "b", "a"))
        _make_fake_checkpoint(orig)
        trained = _trained_state()
        build(orig, b, trained)
        # A as an older write-back would have made it: links into B's files
        os.makedirs(a)
        for name in os.listdir(b):
            os.symlink(os.path.join(b, name), os.path.join(a, name))
        before = _snapshot(b)
        _expect_exit(lambda: build(a, b, _trained_state(scale=2.0), force=True),
                     "resolves into --dst")
        assert _snapshot(b) == before
        verify(orig, b, trained)
        _no_leftovers(tmp)
        # the same when the chain only passes *through* B (a link to a link)
        os.remove(os.path.join(a, "config.json"))
        os.symlink(os.path.join(b, "config.json"), os.path.join(a, "config.json"))
        _expect_exit(lambda: build(a, b, trained, force=True), "into --dst")
        assert _snapshot(b) == before
    finally:
        shutil.rmtree(tmp)


def test_writeback_rebuild_from_own_output_has_no_link_chains():
    """Links go to real files, so B -> A -> B round trips stay intact."""
    tmp = tempfile.mkdtemp(prefix="mtpft-roundtrip-")
    try:
        orig, b, a = (os.path.join(tmp, n) for n in ("orig", "b", "a"))
        _make_fake_checkpoint(orig)
        build(orig, b, _trained_state(scale=7.0))
        build(b, a, _trained_state(scale=5.0))
        assert _links_into(a, b) == []          # A does not depend on B
        third = _trained_state(scale=3.0)
        build(a, b, third, force=True)          # rebuild B from A
        assert _links_into(b, b) == [] and _links_into(b, a) == []
        verify(orig, b, third)
        _no_leftovers(tmp)
    finally:
        shutil.rmtree(tmp)


def test_writeback_copies_a_side_file_that_lives_in_dst():
    tmp = tempfile.mkdtemp(prefix="mtpft-sidecopy-")
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        trained = _trained_state()
        build(src, dst, trained)
        with open(os.path.join(dst, "notes.json"), "w") as fh:
            fh.write('{"kept": true}\n')
        os.symlink(os.path.join(dst, "notes.json"), os.path.join(src, "notes.json"))
        build(src, dst, trained, force=True)
        path = os.path.join(dst, "notes.json")
        assert not os.path.islink(path)
        assert open(path).read() == '{"kept": true}\n'
        # a dangling side file is an error, not a silent skip
        os.symlink(os.path.join(tmp, "nowhere"), os.path.join(src, "dangling.json"))
        _expect_exit(lambda: build(src, os.path.join(tmp, "dst2"), trained),
                     "not a regular file")
        assert not os.path.lexists(os.path.join(tmp, "dst2"))
    finally:
        shutil.rmtree(tmp)


def test_writeback_rejects_unsafe_shard_names():
    for evil in ("../evil.safetensors", "/tmp/evil.safetensors",
                 "sub/evil.safetensors", "..", ""):
        tmp = tempfile.mkdtemp(prefix="mtpft-names-")
        try:
            src, dst = os.path.join(tmp, "m", "src"), os.path.join(tmp, "m", "dst")
            _make_fake_checkpoint(src)
            idx_path = os.path.join(src, "model.safetensors.index.json")
            idx = json.load(open(idx_path))
            # point one untouched tensor at the unsafe name
            idx["weight_map"]["lm_head.weight"] = evil
            json.dump(idx, open(idx_path, "w"))
            listing = sorted(os.listdir(tmp)), sorted(os.listdir(os.path.join(tmp, "m")))
            _expect_exit(lambda: build(src, dst, _trained_state()), "plain file name")
            assert (sorted(os.listdir(tmp)), sorted(os.listdir(os.path.join(tmp, "m")))) == listing
            assert not os.path.lexists(dst)
            # and an mtp shard name
            idx["weight_map"]["lm_head.weight"] = "model-bf16-00003.safetensors"
            for k in idx["weight_map"]:
                if k.startswith("mtp.fc_"):
                    idx["weight_map"][k] = evil
            json.dump(idx, open(idx_path, "w"))
            _expect_exit(lambda: build(src, dst, _trained_state(), strategy="new-shard"),
                         "plain file name")
            assert not os.path.lexists(dst)
        finally:
            shutil.rmtree(tmp)


def test_writeback_verifies_before_replacing():
    """A tree that fails verification never replaces the old --dst."""
    import safetensors.torch as st

    tmp = tempfile.mkdtemp(prefix="mtpft-preverify-")
    real_save = st.save_file
    try:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        _make_fake_checkpoint(src)
        good = _trained_state()
        build(src, dst, good)
        before = _snapshot(dst)

        def corrupting_save(tensors, path, metadata=None):
            tensors = dict(tensors)
            for k in tensors:
                if not k.startswith("mtp."):  # silently damage an untouched tensor
                    tensors[k] = tensors[k] + 1
                    break
            return real_save(tensors, path, metadata=metadata)

        st.save_file = corrupting_save
        try:
            _expect_exit(lambda: build(src, dst, _trained_state(scale=2.0), force=True),
                         "changed unexpectedly")
        finally:
            st.save_file = real_save
        assert _snapshot(dst) == before
        _no_leftovers(tmp)
        verify(src, dst, good)
    finally:
        st.save_file = real_save
        shutil.rmtree(tmp)


class _NotATensor:
    pass


def test_load_trained_refuses_arbitrary_pickles():
    """``weights_only=True``: a checkpoint carrying a pickled object is refused."""
    tmp = tempfile.mkdtemp(prefix="mtpft-pickle-")
    try:
        path = os.path.join(tmp, "evil.pt")
        torch.save({"model": {"fc_hidden.weight": torch.zeros(2)},
                    "payload": _NotATensor()}, path)
        try:
            load_trained(path)
        except Exception as exc:  # noqa: BLE001  (torch raises UnpicklingError)
            assert "weights_only" in str(exc) or "Unpickl" in type(exc).__name__, exc
        else:
            raise AssertionError("an arbitrary pickled object was loaded")
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
