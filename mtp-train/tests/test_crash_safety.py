"""Crash safety of the pipeline's files and resumable tools (CPU, no network).

* whole-file writes are atomic: a failing writer leaves the old file intact;
* append-only JSONL logs survive a partial last line: readers skip it, the
  next append cuts it off instead of gluing a record onto it;
* a complete last record that only lacks its newline is kept, not cut;
* ``extract/client.py`` resumes by (mode, bucket) and keeps dry-run rows and
  real rows apart.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtptrain.fileio import (  # noqa: E402
    open_jsonl_append,
    read_jsonl,
    repair_tail,
    write_json_atomic,
)


def _expect(exc_type, fn, word=""):
    try:
        fn()
    except exc_type as exc:
        assert word in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {exc_type.__name__}")


def test_atomic_write_keeps_the_old_file_when_the_writer_fails():
    from mtptrain.fileio import write_atomic

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.json")
        write_json_atomic(path, {"score": 1})

        def broken(tmp):
            with open(tmp, "w") as fh:
                fh.write('{"score": ')  # half a file
            raise OSError("disk full (simulated)")

        _expect(OSError, lambda: write_atomic(path, broken), "simulated")
        assert json.load(open(path)) == {"score": 1}
        assert os.listdir(d) == ["best.json"]  # no temp file left behind


def test_jsonl_partial_last_line_is_skipped_and_repaired():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "manifest.jsonl")
        with open(path, "w") as fh:
            fh.write('{"doc_hash": "a"}\n{"doc_hash": "b"}\n{"doc_ha')  # killed mid-write
        assert [r["doc_hash"] for r in read_jsonl(path)] == ["a", "b"]
        with open_jsonl_append(path) as app:
            app.write({"doc_hash": "c"})
        assert open(path).read() == '{"doc_hash": "a"}\n{"doc_hash": "b"}\n{"doc_hash": "c"}\n'
        assert repair_tail(path) == 0
        # a broken line in the middle is real corruption, not a crash tail
        with open(path, "a") as fh:
            fh.write('garbage\n{"doc_hash": "d"}\n')
        _expect(ValueError, lambda: read_jsonl(path), "line 4")


def test_jsonl_complete_last_line_without_newline_is_kept_on_append():
    """Round 11: a valid record missing only its newline was read but then cut."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "manifest.jsonl")
        with open(path, "w") as fh:
            fh.write('{"doc_hash": "a"}\n{"doc_hash": "b"}')  # complete, no newline
        assert [r["doc_hash"] for r in read_jsonl(path)] == ["a", "b"]
        with open_jsonl_append(path) as app:
            app.write({"doc_hash": "c"})
        assert [r["doc_hash"] for r in read_jsonl(path)] == ["a", "b", "c"]
        assert open(path).read() == '{"doc_hash": "a"}\n{"doc_hash": "b"}\n{"doc_hash": "c"}\n'
        # whitespace-only tail: nothing to keep
        with open(path, "a") as fh:
            fh.write("   ")
        assert repair_tail(path) == 3
        assert [r["doc_hash"] for r in read_jsonl(path)] == ["a", "b", "c"]


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _StubRenderer:
    """Stands in for the tokenizer: one id per character (no model needed)."""

    def __init__(self, model_dir, chat_kwargs=None):
        self.chat_kwargs = chat_kwargs or {}

    def render(self, messages, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text):
        return [ord(c) % 1000 for c in text]


def test_client_selfgen_writes_greedy_rows_and_resumes_each_document_once():
    """The guide's self-generation route: generate, then dump prompt + output."""
    import requests

    import extract.client as client

    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            with open(os.path.join(d, f"chat{i}.jsonl"), "w") as fh:
                fh.write(json.dumps({"messages": [
                    {"role": "user", "content": f"question {i}: " + "why? " * 20},
                    {"role": "assistant", "content": "because " * 20}]}) + "\n")
        with open(os.path.join(d, "flat.md"), "w") as fh:  # no prompt of its own
            fh.write("plain prose " * 40)
        man = os.path.join(d, "dump", "manifest.jsonl")
        calls = []

        def post(self, url, json=None, timeout=None):  # noqa: A002
            if crash["after"] is not None and len(calls) >= crash["after"]:
                raise KeyboardInterrupt("simulated crash")
            calls.append(json["rid"])
            out = [(0.0, 7 + i, None) for i in range(json["sampling_params"]["max_new_tokens"])]
            return _Resp({"meta_info": {"output_token_logprobs": out}})

        args = client.build_parser().parse_args(
            ["--mode", "selfgen", "--model-dir", d, "--manifest", man,
             "--files", os.path.join(d, "*.*"), "--gen-tokens", "80", "--max-len", "400",
             "--min-len", "16", "--concurrency", "1", "--bucket", "notes"])
        args.settings = {"gen_tokens": 80}
        real = (client.Renderer, requests.Session.post)
        client.Renderer, requests.Session.post = _StubRenderer, post
        crash = {"after": 5}  # dies after two full documents and one generation
        try:
            try:
                client.run_selfgen(args)
            except KeyboardInterrupt:
                pass
            assert len(read_jsonl(man)) == 2
            crash["after"] = None
            client.run_selfgen(args)
        finally:
            client.Renderer, requests.Session.post = real
        rows = read_jsonl(man)
        assert len(rows) == 5 and len({r["split_key"] for r in rows}) == 5
        for r in rows:
            assert r["mode"] == "selfgen" and r["temperature"] == 0.0
            assert r["generated_tokens"] == 80
            assert r["tokens"] == r["prompt_tokens"] + 80
        # every document was dumped (prefill of prompt + output) exactly once
        dumped = [c for c in calls if c.endswith("-sg")]
        assert len(dumped) == 5 and len(set(dumped)) == 5
        # only flat text: nothing to continue from, and that is an error
        flat = client.build_parser().parse_args(
            ["--mode", "selfgen", "--model-dir", d, "--manifest",
             os.path.join(d, "flat", "manifest.jsonl"), "--files",
             os.path.join(d, "*.md"), "--gen-tokens", "80", "--max-len", "400"])
        client.Renderer, requests.Session.post = _StubRenderer, post
        try:
            _expect(SystemExit, lambda: client.run_selfgen(flat), "without a prompt")
        finally:
            client.Renderer, requests.Session.post = real


def test_client_resume_state_separates_modes_buckets_and_dry_runs():
    from extract.client import resume_state

    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "manifest.jsonl")
        with open(man, "w") as fh:
            for r in ({"doc_hash": "a", "split_key": "s1", "mode": "corpus",
                       "bucket": "code", "tokens": 100},
                      {"doc_hash": "b", "split_key": "s1", "mode": "corpus",
                       "bucket": "code", "tokens": 50},
                      {"doc_hash": "c", "split_key": "s2", "mode": "corpus",
                       "bucket": "prose", "tokens": 70},
                      {"doc_hash": "d", "split_key": "s3", "mode": "selfgen",
                       "bucket": "code", "tokens": 90, "generated_tokens": 30}):
                fh.write(json.dumps(r) + "\n")
            fh.write('{"doc_hash": "e", "mo')  # killed mid-write
        args = types.SimpleNamespace(manifest=man, mode="extract", bucket="code",
                                     dry_run=False)
        hashes, keys, tokens = resume_state(args)
        assert hashes == {"a", "b"} and tokens == 150
        args.mode = "selfgen"
        hashes, keys, tokens = resume_state(args)
        assert keys == {"s3"} and tokens == 30
        # a dry run may not append to a real manifest, nor the other way round
        args.dry_run = True
        _expect(SystemExit, lambda: resume_state(args), "real rows")
        dry = os.path.join(d, "dry.jsonl")
        with open(dry, "w") as fh:
            fh.write(json.dumps({"doc_hash": "x", "mode": "corpus", "bucket": "code",
                                 "tokens": 5, "dry_run": True}) + "\n")
        args = types.SimpleNamespace(manifest=dry, mode="extract", bucket="code",
                                     dry_run=False)
        _expect(SystemExit, lambda: resume_state(args), "dry-run rows")


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
