#!/usr/bin/env python3
"""Stream documents through a running SGLang server so the hook dumps them.

Two modes:

``extract`` (default)
    Render every corpus conversation with the serving chat template, tokenize
    locally, cut into <= ``--max-len`` token chunks and POST each chunk to
    ``/generate`` as raw ``input_ids`` with ``max_new_tokens=1``.  Sending ids
    (not text) guarantees the dumped ``input_ids`` are byte-identical to what
    the client believes it sent, so the manifest can key documents by hash.

``selfgen``
    Draw prompts from the same corpora, let the server generate ``--gen-tokens``
    continuations greedily, then re-send ``prompt_ids + output_ids`` as one
    ``input_ids`` request so the hook dumps the model's *own* text.  On that
    data the corpus continuation is the greedy continuation by construction,
    which makes the chain-acceptance evaluation exact at every step.

The client never starts a server: start a hook-enabled one first
(``parity/serve.sh start-extract``, or ``launch/serve-fast.sh`` from the dump
branch with ``SGLANG_MTP_DUMP_DIR`` set and ``--disable-radix-cache``), and
point ``--manifest`` into the same directory as ``SGLANG_MTP_DUMP_DIR``.
Nothing here touches CUDA.

Resumable: re-running with the same ``--manifest`` skips every chunk
(``extract``) or document (``selfgen``) the manifest already records for the
same ``--mode`` and ``--bucket``, and those rows count toward
``--token-budget``, so an interrupted run continues instead of dumping the
same text twice.  A manifest row is written (and flushed) only after the
server accepted the request; a chunk sent but not yet recorded when the
process died is sent again (its duplicate dump sequence is harmless: the
index keeps one copy per doc_hash).  ``--dry-run`` rows are marked
``"dry_run": true``, and a real run refuses a manifest that holds them (and
vice versa), so a budget check can never make a real run skip its work.

    python extract/client.py --mode extract --model-dir MODEL \
        --codex-sessions DIR --claude-sessions DIR --files 'notes=DIR/*.md' \
        --manifest $SGLANG_MTP_DUMP_DIR/manifest.jsonl --token-budget 8000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import requests  # noqa: E402

# ------------------------------------------------------------------ corpora
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from corpus import formats as F  # noqa: E402
from mtptrain.cli import StrictParser, split_list  # noqa: E402
from mtptrain.config import DEFAULT_MODEL_DIR  # noqa: E402
from mtptrain.fileio import (  # noqa: E402
    check_run_settings,
    open_jsonl_append,
    read_jsonl,
)

CJK = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),  # CJK ideographs
    (0xFF66, 0xFF9F),  # halfwidth kana
)


# The user turn put in front of flat text (a file or record with no chat
# structure), so it renders like any other conversation.  It is not a prompt
# of the document's own: self-generation skips documents that have nothing else.
CONTINUE = "Continue the document."


def _lang_of(text: str) -> str:
    if not text:
        return "en"
    hits = sum(
        1
        for ch in text[:20000]
        if any(lo <= ord(ch) <= hi for lo, hi in CJK)
    )
    return "ja" if hits / max(1, min(len(text), 20000)) > 0.10 else "en"


def _truncate_tools(messages, max_chars: int):
    """Cap tool-response length.

    In coding-agent transcripts most tokens are usually tool output -- file
    dumps and command results the model never generates.  Teacher-forcing over
    all of it would spend most of the budget on text the draft head will never
    be asked to predict at serving time.
    """
    if max_chars <= 0:
        return messages, 0
    cut = 0
    out = []
    for m in messages:
        if m["role"] == "tool" and len(m["content"]) > max_chars:
            head = m["content"][: max_chars // 2]
            tail = m["content"][-max_chars // 2 :]
            m = dict(m, content=head + "\n...[truncated]...\n" + tail)
            cut += 1
        out.append(m)
    return out, cut


def _glob_root(pattern: str) -> str:
    """The directory part of a glob before its first wildcard (the pattern
    itself for a directory): file names are made relative to it, so two files
    with the same base name in different directories name different documents."""
    pattern = os.path.expanduser(pattern)
    if os.path.isdir(pattern):
        return pattern
    parts = []
    for part in pattern.split(os.sep):
        if any(c in part for c in "*?["):
            break
        parts.append(part)
    root = os.sep.join(parts)
    return root if os.path.isdir(root or ".") and root != pattern else os.path.dirname(root)


def input_hash(doc: dict) -> str:
    """Hash of everything a document's requests are rendered from.

    Stored in every manifest row: a resume compares it with the document the
    same split key names now, so an edited source is never silently taken as
    already done (or dumped a second time under the same key)."""
    blob = json.dumps([doc["source"], doc["messages"]], sort_keys=True,
                      ensure_ascii=False)
    return hashlib.blake2b(blob.encode("utf-8", "replace"), digest_size=12).hexdigest()


def _named(spec: str, default_name: str) -> Tuple[str, str]:
    """``NAME=PATH`` -> (NAME, PATH); a bare ``PATH`` gets ``default_name``."""
    if "=" in spec and not os.path.exists(spec):
        name, path = spec.split("=", 1)
        if not name.strip() or not path.strip():
            raise SystemExit(f"{spec!r}: empty NAME or PATH in NAME=PATH")
        return name, path
    return default_name, spec


def load_corpus(
    codex: Sequence[str] = (),
    claude: Sequence[str] = (),
    lmstudio: Sequence[str] = (),
    files: Sequence[str] = (),
    arrow: Optional[dict] = None,
    max_file_mb: float = 20.0,
    limit_files: Optional[int] = None,
    max_tool_chars: int = 0,
    txt_split_bytes: int = 20000,
):
    """-> list of {name, source, workload, lang, messages:[{role,content,...}]}.

    ``codex`` / ``claude`` / ``lmstudio``: directories of Codex CLI rollout
    JSONL, Claude Code session JSONL and LM Studio ``*.conversation.json``.
    ``files``: globs or directories of generic .jsonl/.json/.txt/.md files,
    optionally as ``NAME=GLOB`` (NAME becomes the manifest ``source``).
    ``arrow``: keyword arguments of ``corpus.formats.iter_arrow_docs`` plus
    ``name``.  Nothing is read from a default location.

    Every source given must exist and contribute at least one document: a
    typo'd directory or a glob that matches nothing is an error, not a
    silently smaller corpus.
    """
    raw = []  # (source flag, source, name, [Message | (Message, reasoning)])
    errors: List[str] = []

    def exists(flag: str, path: str) -> bool:
        if not os.path.exists(os.path.expanduser(path)):
            errors.append(f"{flag} {path}: no such file or directory")
            return False
        return True

    for path in codex:
        if not exists("--codex-sessions", path):
            continue
        convs, summary = F.load_codex_sessions([path], max_file_mb=max_file_mb,
                                               limit_files=limit_files)
        print(f"[corpus] codex: {len(convs)} conversations ({summary.files_seen} files)")
        raw += [(f"--codex-sessions {path}", "codex", getattr(c, "name", "?"), c.messages)
                for c in convs]
    for path in lmstudio:
        if not exists("--lmstudio-conversations", path):
            continue
        convs, _ = F.load_lmstudio_conversations(path)
        print(f"[corpus] lmstudio: {len(convs)} conversations")
        raw += [(f"--lmstudio-conversations {path}", "lmstudio", getattr(c, "name", "?"),
                 c.messages) for c in convs]
    for path in claude:
        if not exists("--claude-sessions", path):
            continue
        sessions, counter = F.load_claude_sessions(path, max_file_mb=max_file_mb)
        print(f"[corpus] claude-code: {len(sessions)} sessions of "
              f"{counter['files_seen']} files"
              + (f" ({counter['oversize_files']} over --max-file-mb skipped)"
                 if counter["oversize_files"] else ""))
        raw += [(f"--claude-sessions {path}", "claude", name, msgs)
                for name, msgs in sessions]
    for spec in files:
        name, pattern = _named(spec, "files")
        paths = F.expand_globs([pattern])
        if not paths:
            errors.append(f"--files {spec}: matches no file")
            continue
        n = 0
        root = _glob_root(pattern)
        for path in paths:
            rel = os.path.relpath(path, root) if root else os.path.basename(path)
            for i, msgs in enumerate(
                F.iter_file_docs(path, txt_split_bytes=txt_split_bytes)
            ):
                raw.append((f"--files {spec}", name, f"{rel}:{i}", msgs))
                n += 1
        print(f"[corpus] {name}: {n} documents from {len(paths)} files")
    if arrow:
        spec = dict(arrow)
        name = spec.pop("name", "arrow")
        if not F.expand_globs([spec["glob_pattern"]]):
            errors.append(f"--arrow {spec['glob_pattern']}: matches no file")
        else:
            n = 0
            skip = int(spec.get("skip") or 0)
            for i, msgs in enumerate(F.iter_arrow_docs(**spec)):
                # the row's absolute position, so --arrow-skip does not rename rows
                raw.append((f"--arrow {spec['glob_pattern']}", name, f"{name}:{skip + i}",
                            msgs))
                n += 1
            print(f"[corpus] {name}: {n} documents")
    given = ([f"--codex-sessions {p}" for p in codex]
             + [f"--lmstudio-conversations {p}" for p in lmstudio]
             + [f"--claude-sessions {p}" for p in claude]
             + [f"--files {s}" for s in files]
             + ([f"--arrow {arrow['glob_pattern']}"] if arrow else []))

    out = []
    per_flag: Dict[str, int] = {}
    tool_cuts = 0
    for flag, source, name, msgs in raw:
        norm, _stats = F.normalize(msgs)
        messages = [
            {"role": r, "content": c, "reasoning_content": g}
            for r, c, g in norm
            if (c and c.strip()) or (g and g.strip())
        ]
        if len(messages) == 1 and messages[0]["role"] == "assistant":
            # Flat prose files yield one assistant turn; give it a user turn so
            # the rendered document looks like anything else the server sees.
            messages = [
                {"role": "user", "content": CONTINUE,
                 "reasoning_content": ""}
            ] + messages
        if len(messages) < 2:
            continue
        messages, cut = _truncate_tools(messages, max_tool_chars)
        tool_cuts += cut
        per_flag[flag] = per_flag.get(flag, 0) + 1
        joined = "\n".join(m["content"] for m in messages)
        out.append(
            {
                "name": name,
                "source": source,
                "workload": F.classify_workload(msgs),
                "lang": _lang_of(joined),
                "messages": messages,
                "content_hash": hashlib.blake2b(
                    joined.encode("utf-8", "replace"), digest_size=12
                ).hexdigest(),
                "split_key": hashlib.blake2b(
                    f"{source}|{name}".encode(), digest_size=8
                ).hexdigest(),
            }
        )
    if tool_cuts:
        print(f"[corpus] truncated {tool_cuts} tool blocks to {max_tool_chars} chars")
    for flag in given:
        if flag not in per_flag and not any(e.startswith(flag) for e in errors):
            errors.append(f"{flag}: no usable document (empty, unsupported or "
                          "unparsable files, or all over --max-file-mb)")
    if errors:
        raise SystemExit("corpus:\n  " + "\n  ".join(errors))
    return out


def corpus_from_args(args):
    """The corpus the CLI flags describe (shared by --mode extract and selfgen)."""
    arrow = None
    if args.arrow:
        arrow = {
            "name": args.arrow_name,
            "glob_pattern": args.arrow,
            "text_field": args.arrow_text_field,
            "aux_field": args.arrow_aux_field or None,
            "instruction": args.arrow_instruction or None,
            "limit": args.arrow_limit or None,
            "skip": args.arrow_skip,
        }
    docs = load_corpus(
        codex=args.codex_sessions, claude=args.claude_sessions,
        lmstudio=args.lmstudio_conversations, files=args.files, arrow=arrow,
        max_file_mb=args.max_file_mb, limit_files=args.limit_files,
        max_tool_chars=args.max_tool_chars, txt_split_bytes=args.txt_split_bytes,
    )
    if args.jsonl:
        docs += load_jsonl_docs(args.jsonl, text_key=args.jsonl_text_key)
    if not docs:
        raise SystemExit(
            "empty corpus: pass at least one of --codex-sessions, --claude-sessions, "
            "--lmstudio-conversations, --files, --jsonl, --arrow"
        )
    for doc in docs:
        doc["input_hash"] = input_hash(doc)
    return docs


def add_corpus_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("corpus (every source is opt-in; repeat flags for more dirs)")
    g.add_argument("--codex-sessions", action="append", default=[], metavar="DIR",
                   help="directory of Codex CLI rollout *.jsonl files")
    g.add_argument("--claude-sessions", action="append", default=[], metavar="DIR",
                   help="directory tree of Claude Code session *.jsonl transcripts")
    g.add_argument("--lmstudio-conversations", action="append", default=[], metavar="DIR",
                   help="directory of LM Studio *.conversation.json files")
    g.add_argument("--files", action="append", default=[], metavar="[NAME=]GLOB",
                   help="generic .jsonl/.json/.txt/.md files (chat records, "
                        "instruction/output pairs or plain text)")
    g.add_argument("--jsonl", nargs="+", default=[],
                   help="plain-text JSONL files, one document per record")
    g.add_argument("--jsonl-text-key", default="text")
    g.add_argument("--arrow", default=None, metavar="GLOB",
                   help="Hugging Face datasets Arrow cache files, e.g. "
                        "'<HF cache>/datasets/abisee___cnn_dailymail/3.0.0/*/*/"
                        "cnn_dailymail-train-*.arrow'")
    g.add_argument("--arrow-name", default="arrow")
    g.add_argument("--arrow-text-field", default="text")
    g.add_argument("--arrow-aux-field", default="",
                   help="if set, the row becomes user=text, assistant=aux")
    g.add_argument("--arrow-instruction", default="",
                   help="prefix of the user turn when --arrow-aux-field is set")
    g.add_argument("--arrow-limit", type=int, default=0)
    g.add_argument("--arrow-skip", type=int, default=0)
    g.add_argument("--max-file-mb", type=float, default=20)
    g.add_argument("--limit-files", type=int, default=None,
                   help="newest N Codex rollouts per directory")
    g.add_argument("--txt-split-bytes", type=int, default=20000,
                   help="split flat .txt corpora into documents of this size, "
                        "so the by-document held-out split has something to hold out")
    g.add_argument("--max-tool-chars", type=int, default=2048,
                   help="cap each tool-response block (0 = no truncation)")


def dedup_docs(docs, enabled: bool = True):
    if not enabled:
        return docs
    seen = set()
    out = []
    for doc in docs:
        if doc["content_hash"] in seen:
            continue
        seen.add(doc["content_hash"])
        out.append(doc)
    if len(out) != len(docs):
        print(f"[corpus] dedup: {len(docs)} -> {len(out)} documents")
    return out


def filter_docs(docs, lang=None, workload=None):
    if lang and lang != "any":
        docs = [d for d in docs if d["lang"] == lang]
    if workload and workload != "any":
        wanted = set(split_list(workload, "--workload"))
        docs = [d for d in docs if d["workload"] in wanted]
    return docs


def load_jsonl_docs(paths: Sequence[str], text_key: str = "text"):
    out = []
    for path in paths:
        if not os.path.isfile(os.path.expanduser(path)):
            raise SystemExit(f"--jsonl {path}: no such file")
        before = len(out)
        with open(os.path.expanduser(path), "r", encoding="utf-8",
                  errors="replace") as handle:
            for i, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_key) if isinstance(obj, dict) else None
                if not isinstance(text, str) or len(text) < 200:
                    continue
                name, source = f"{os.path.basename(path)}:{i}", os.path.basename(path)
                out.append(
                    {
                        "name": name,
                        "source": source,
                        "workload": "prose",
                        "lang": _lang_of(text),
                        # the same keys load_corpus gives: dedup and the
                        # by-document split need them
                        "content_hash": hashlib.blake2b(
                            text.encode("utf-8", "replace"), digest_size=12
                        ).hexdigest(),
                        "split_key": hashlib.blake2b(
                            f"{source}|{name}".encode(), digest_size=8
                        ).hexdigest(),
                        "messages": [
                            {"role": "user", "content": CONTINUE},
                            {
                                "role": "assistant",
                                "content": text,
                                "reasoning_content": "",
                            },
                        ],
                    }
                )
        if len(out) == before:
            raise SystemExit(f"--jsonl {path}: no record with a {text_key!r} string of "
                             ">= 200 characters")
    return out


# ---------------------------------------------------------------- tokenizing
class Renderer:
    def __init__(self, model_dir: str, chat_kwargs: Optional[dict] = None):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.chat_kwargs = chat_kwargs or {}

    def render(self, messages: Sequence[dict], add_generation_prompt: bool = False) -> str:
        return self.tok.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.chat_kwargs,
        )

    def encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]


def chunk_ids(
    ids: Sequence[int],
    max_len: int,
    min_len: int,
    max_chunks: int = 0,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Cut a document into <= max_len windows.

    ``max_chunks`` samples that many windows at random instead of taking them
    all: agent sessions can run to hundreds of thousands of tokens, so an
    unrestricted budget would be spent on a few dozen conversations.  Sampling
    keeps the document count -- and therefore the diversity -- high.
    """
    pieces = []
    for start in range(0, len(ids), max_len):
        piece = list(ids[start : start + max_len])
        if len(piece) >= min_len:
            pieces.append((start // max_len, piece))
    if max_chunks and len(pieces) > max_chunks:
        rng = rng or random
        pieces = sorted(rng.sample(pieces, max_chunks), key=lambda x: x[0])
    return pieces


def ids_hash(ids: Sequence[int]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for value in ids:
        h.update(int(value).to_bytes(4, "little", signed=False))
    return h.hexdigest()


# -------------------------------------------------------------------- HTTP
class Server:
    def __init__(self, url: str, timeout: float = 900.0, dry_run: bool = False):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.dry_run = dry_run
        self.session = requests.Session()

    def generate(self, payload: dict) -> dict:
        r = self.session.post(f"{self.url}/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def prefill_only(self, ids: Sequence[int], rid: str) -> dict:
        if self.dry_run:
            return {"meta_info": {"dry_run": True, "prompt_tokens": len(ids)}}
        return self.generate(
            {
                "rid": rid,
                "input_ids": list(ids),
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            }
        )

    def complete(
        self, ids: Sequence[int], rid: str, gen_tokens: int, temperature: float
    ) -> List[int]:
        out = self.generate(
            {
                "rid": rid,
                "input_ids": list(ids),
                "sampling_params": {
                    "max_new_tokens": gen_tokens,
                    "temperature": temperature,
                },
                "return_logprob": True,
                "top_logprobs_num": 0,
            }
        )
        meta = out.get("meta_info", {})
        pairs = meta.get("output_token_logprobs") or []
        token_ids = [int(p[1]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
        return token_ids

    def flush_cache(self) -> None:
        try:
            self.session.post(f"{self.url}/flush_cache", timeout=60)
        except Exception:  # noqa: BLE001
            pass


# -------------------------------------------------------------------- runs
def check_args(args) -> None:
    """Refuse values that would make the run do nothing, or not what was asked."""
    errs = []
    # A local checkpoint only: a missing path would turn into a hub lookup.
    if not os.path.isdir(os.path.expanduser(args.model_dir)):
        errs.append(f"--model-dir {args.model_dir}: no such directory")
    if os.path.isdir(os.path.expanduser(args.manifest)):
        errs.append(f"--manifest {args.manifest} is a directory")
    if args.token_budget < 1:
        errs.append("--token-budget must be >= 1")
    if args.min_len < 1 or args.max_len < args.min_len:
        errs.append("need 1 <= --min-len <= --max-len")
    if args.concurrency < 1 or args.max_chunks_per_doc < 0 or args.limit_docs < 0:
        errs.append("--concurrency must be >= 1; --max-chunks-per-doc, --limit-docs >= 0")
    try:
        if not isinstance(json.loads(args.chat_kwargs), dict):
            raise ValueError("not an object")
    except ValueError as exc:
        errs.append(f"--chat-kwargs is not a JSON object ({exc})")
    if args.mode == "selfgen":
        if args.dry_run:
            # selfgen has to generate to have anything to record
            errs.append("--dry-run only applies to --mode extract (selfgen must "
                        "call the server to generate)")
        if args.gen_tokens < 1 or args.max_len - args.gen_tokens < args.min_len:
            errs.append(f"--max-len {args.max_len} - --gen-tokens {args.gen_tokens} "
                        f"leaves less than --min-len {args.min_len} for the prompt")
    elif args.limit_docs:
        errs.append("--limit-docs only applies to --mode selfgen")
    if errs:
        raise SystemExit("refusing to start:\n  " + "\n  ".join(errs))


def run_settings(args) -> dict:
    """What decides the token ids a row stands for; rows of the same --mode and
    --bucket must share it (a resume with another chunking or chat template
    would dump the same text a second time under new hashes)."""
    from mtptrain.weights import model_identity

    out = {"model": model_identity(args.model_dir),
           "chat_kwargs": json.loads(args.chat_kwargs),
           "max_len": args.max_len, "min_len": args.min_len, "seed": args.seed}
    if args.mode == "extract":
        out["max_chunks_per_doc"] = args.max_chunks_per_doc
    else:
        out.update(gen_tokens=args.gen_tokens, temperature=args.temperature)
    return out


def prepare_docs(args, seed: int) -> list:
    """Corpus -> input check -> dedup -> filters -> shuffled; refuses an empty result.

    Every loaded document is checked against the manifest (``check_inputs``)
    before dedup, the --lang/--workload filters, resume skips, --limit-docs,
    the token budget or any length cut can drop it: a document that was
    edited into a copy of another, or into something a filter removes, is
    still compared with the rows its split key already has."""
    docs = corpus_from_args(args)
    check_inputs(args, docs)
    n = len(docs)
    docs = dedup_docs(docs, not args.no_dedup)
    docs = filter_docs(docs, args.lang, args.workload)
    if not docs:
        raise SystemExit(f"no document left of {n} after dedup and --lang {args.lang} / "
                         f"--workload {args.workload}")
    random.Random(seed).shuffle(docs)
    print(f"[corpus] {len(docs)} documents after dedup/filter")
    return docs


def check_inputs(args, docs) -> None:
    """A split key must name the same input in this run and in the manifest.

    ``docs`` is the whole corpus as loaded, before dedup or any filter.
    Refuses (naming the key and the document):

    * two documents of this run under one split key with different content;
    * manifest rows without ``input_hash`` (written before it was recorded),
      which cannot be checked;
    * a row whose ``input_hash`` differs from the document its key names now
      (the source was edited: a resume would skip it as done, or dump a
      second version under the same key);
    * a row of a source this run loads (same ``source`` label) whose key names
      no loaded document: the document was removed, or edited into something
      the loader drops (empty, too short, over --max-file-mb, beyond
      --limit-files / --arrow-limit).  Its dumped content no longer reflects
      the corpus, and nothing here can tell it apart from a current one.
    """
    now: Dict[str, dict] = {}
    for d in docs:
        other = now.setdefault(d["split_key"], d)
        if other["input_hash"] != d["input_hash"]:
            raise SystemExit(
                f"split key {d['split_key']} names two different documents "
                f"({other['source']}:{other['name']} and {d['source']}:{d['name']}); "
                "give the sources distinct NAME= labels or file names")
    sources = {d["source"] for d in docs}
    rows = [r for r in read_jsonl(args.manifest, missing_ok=True)
            if r.get("mode") in ("corpus", "selfgen")]
    legacy = [r for r in rows if "input_hash" not in r]
    if legacy:
        raise SystemExit(
            f"{args.manifest}: {len(legacy)} rows have no input_hash (written by an "
            "older client.py), so a changed source could not be detected on resume; "
            "use a new dump directory and manifest")
    for r in rows:
        d = now.get(r.get("split_key"))
        if d is None:
            if r.get("source") in sources:
                raise SystemExit(
                    f"split key {r.get('split_key')} ({r.get('source')}:{r.get('name')}): "
                    f"{args.manifest} has rows for it, but this run loads no such "
                    "document (removed, or edited into something the loader drops: "
                    "empty, too short, over --max-file-mb, beyond --limit-files or "
                    "--arrow-limit).  Its dump would be trained on as if current.  "
                    "Use a new dump directory, or restore the source")
            continue
        if d["input_hash"] != r["input_hash"]:
            raise SystemExit(
                f"split key {r['split_key']} ({d['source']}:{d['name']}): the document "
                f"changed since {args.manifest} recorded it (input hash "
                f"{r['input_hash']} -> {d['input_hash']}); a resume would skip or "
                "duplicate it.  Use a new dump directory, or restore the source")


def finish(state: dict, what: str) -> None:
    """Non-zero exit when requests failed: the rows written are complete and a
    rerun resumes, but the run did not do what it was asked."""
    if state["errors"]:
        raise SystemExit(f"[{what}] {state['errors']} requests failed; rerun with the "
                         "same arguments to retry them")


def resume_state(args) -> Tuple[set, set, int]:
    """-> (doc_hashes, split_keys, tokens) already recorded for this mode/bucket."""
    rows = read_jsonl(args.manifest, missing_ok=True)
    dry = [r for r in rows if r.get("dry_run")]
    if dry and not args.dry_run:
        raise SystemExit(f"{args.manifest} holds {len(dry)} --dry-run rows; "
                         "use a fresh manifest for the real run")
    if args.dry_run and len(dry) != len(rows):
        raise SystemExit(f"{args.manifest} holds real rows; give --dry-run its own manifest")
    mode = "corpus" if args.mode == "extract" else "selfgen"
    mine = [r for r in rows if r.get("mode") == mode and r.get("bucket") == args.bucket]
    if getattr(args, "settings", None) is not None:
        check_run_settings(mine, "settings", args.settings, args.manifest)
    hashes = {r["doc_hash"] for r in mine}
    keys = {r.get("split_key") for r in mine}
    tokens = sum(int(r.get("generated_tokens" if args.mode == "selfgen" else "tokens", 0))
                 for r in mine)
    if mine:
        print(f"[resume] {args.manifest}: {len(mine)} {args.mode} rows for bucket "
              f"{args.bucket!r} already recorded ({tokens} tokens of the budget)")
    return hashes, keys, tokens


def run_extract(args) -> None:
    renderer = Renderer(args.model_dir, json.loads(args.chat_kwargs))
    docs = prepare_docs(args, args.seed)
    server = Server(args.url, dry_run=args.dry_run)
    lock = threading.Lock()
    done, _keys, used = resume_state(args)
    state = {"tokens": used, "chunks": 0, "docs": 0, "errors": 0, "skipped": 0}
    manifest = open_jsonl_append(args.manifest)
    t0 = time.time()

    def handle(doc) -> None:
        if state["tokens"] >= args.token_budget:
            return
        text = renderer.render(doc["messages"])
        ids = renderer.encode(text)
        pieces = chunk_ids(
            ids, args.max_len, args.min_len, args.max_chunks_per_doc,
            # split_key is hex: a seed that is the same in every process
            # (``hash()`` of a str is not), so a resumed run samples the same
            # windows and skips them instead of dumping new ones.
            random.Random(int(doc["split_key"], 16)),
        )
        for index, piece in pieces:
            digest = ids_hash(piece)
            with lock:
                if digest in done:  # recorded by an earlier run
                    state["skipped"] += 1
                    continue
                if state["tokens"] >= args.token_budget:
                    return
                state["tokens"] += len(piece)
                done.add(digest)
            rid = f"mtpx-{digest}-{index}"
            try:
                server.prefill_only(piece, rid)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    state["errors"] += 1
                print(f"[warn] {doc['name']} chunk {index}: {exc}")
                continue
            record = {
                "doc_hash": digest,
                "split_key": doc["split_key"],
                "rid": rid,
                "name": doc["name"],
                "source": doc["source"],
                "workload": doc["workload"],
                "lang": doc["lang"],
                "chunk": index,
                "tokens": len(piece),
                "mode": "corpus",
                "bucket": args.bucket,
                "input_hash": doc["input_hash"],
                "settings": getattr(args, "settings", None),
            }
            if args.dry_run:
                record["dry_run"] = True
            with lock:
                manifest.write(record)
                state["chunks"] += 1
                if state["chunks"] % 25 == 0:
                    rate = state["tokens"] / max(1e-6, time.time() - t0)
                    print(
                        f"[extract] docs={state['docs']} chunks={state['chunks']} "
                        f"tokens={state['tokens']} ({rate:.0f} tok/s) "
                        f"errors={state['errors']}",
                        flush=True,
                    )
        with lock:
            state["docs"] += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(handle, docs))
    manifest.close()
    print(f"[extract] done: {json.dumps(state)}  {time.time()-t0:.0f}s")
    finish(state, "extract")


def run_selfgen(args) -> None:
    renderer = Renderer(args.model_dir, json.loads(args.chat_kwargs))
    docs = prepare_docs(args, args.seed + 1)
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    server = Server(args.url, dry_run=args.dry_run)
    lock = threading.Lock()
    done_hashes, done_keys, used = resume_state(args)
    state = {"tokens": used, "docs": 0, "errors": 0, "skipped": 0, "no_prompt": 0,
             "short": 0, "duplicate": 0}
    manifest = open_jsonl_append(args.manifest)
    t0 = time.time()

    def handle(doc) -> None:
        if state["tokens"] >= args.token_budget:
            return
        with lock:
            if doc["split_key"] in done_keys:  # one selfgen row per document
                state["skipped"] += 1
                return
        # Prompt = every message up to (and excluding) the last assistant turn.
        messages = doc["messages"]
        cut = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                cut = i
                break
        prompt_messages = messages[:cut]
        if not any((m.get("content") or "").strip() not in ("", CONTINUE)
                   for m in prompt_messages):
            # Flat text has no prompt of its own; every such document would
            # get the same placeholder prompt, hence the same greedy output.
            with lock:
                state["no_prompt"] += 1
            return
        prompt_text = renderer.render(prompt_messages, add_generation_prompt=True)
        prompt_ids = renderer.encode(prompt_text)
        budget = args.max_len - args.gen_tokens
        if budget < args.min_len:
            return
        prompt_ids = prompt_ids[-budget:]
        try:
            output_ids = server.complete(
                prompt_ids, f"mtpg-{ids_hash(prompt_ids)}", args.gen_tokens, args.temperature
            )
        except Exception as exc:  # noqa: BLE001
            with lock:
                state["errors"] += 1
            print(f"[warn] selfgen {doc['name']}: {exc}")
            return
        if len(output_ids) < args.min_len:
            with lock:
                state["short"] += 1
            return
        full = list(prompt_ids) + list(output_ids)
        digest = ids_hash(full)
        with lock:
            if digest in done_hashes:  # identical text from another document
                state["duplicate"] += 1
                return
            done_hashes.add(digest)
        try:
            server.prefill_only(full, f"mtpx-{digest}-sg")
        except Exception as exc:  # noqa: BLE001
            with lock:
                state["errors"] += 1
            print(f"[warn] selfgen replay {doc['name']}: {exc}")
            return
        record = {
            "doc_hash": digest,
            "split_key": doc["split_key"],
            "rid": f"mtpx-{digest}-sg",
            "name": doc["name"],
            "source": doc["source"],
            "workload": doc["workload"],
            "lang": doc["lang"],
            "chunk": 0,
            "tokens": len(full),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(output_ids),
            "temperature": args.temperature,
            "mode": "selfgen",
            "bucket": args.bucket,
            "input_hash": doc["input_hash"],
            "settings": getattr(args, "settings", None),
        }
        if args.dry_run:
            record["dry_run"] = True
        with lock:
            manifest.write(record)
            state["tokens"] += len(output_ids)
            state["docs"] += 1
            if state["docs"] % 10 == 0:
                print(
                    f"[selfgen] docs={state['docs']} gen_tokens={state['tokens']} "
                    f"errors={state['errors']} ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(handle, docs))
    manifest.close()
    print(f"[selfgen] done: {json.dumps(state)}  {time.time()-t0:.0f}s")
    finish(state, "selfgen")
    if not state["docs"] and not state["skipped"] and state["tokens"] < args.token_budget:
        raise SystemExit(
            f"[selfgen] no document was self-generated ({state['no_prompt']} without a "
            f"prompt of their own, {state['short']} shorter than --min-len, "
            f"{state['duplicate']} duplicates); self-generation needs chat-shaped "
            "sources (sessions, chat JSONL, --arrow-aux-field)")


def build_parser() -> argparse.ArgumentParser:
    p = StrictParser(description=__doc__,
                     formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="extract", choices=["extract", "selfgen"])
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="serving checkpoint: tokenizer + chat template (env MTP_MODEL_DIR)")
    p.add_argument("--manifest", required=True,
                   help="manifest.jsonl to append to; put it in SGLANG_MTP_DUMP_DIR")
    p.add_argument("--max-len", type=int, default=2048,
                   help="QSA selects 2048 context tokens per row, so dense causal "
                        "training is exact only up to this length")
    p.add_argument("--min-len", type=int, default=64)
    p.add_argument("--token-budget", type=int, default=10_000_000)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--gen-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    # Must match launch/serve-local.sh --default-chat-template-kwargs exactly,
    # or the tokens we send differ from what the server would build from text.
    p.add_argument(
        "--chat-kwargs",
        default='{"enable_thinking": true, "preserve_thinking": true, '
                '"reasoning_effort": "medium"}',
    )
    p.add_argument("--max-chunks-per-doc", type=int, default=0,
                   help="sample at most N windows per document (0 = all)")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--lang", default="any", choices=["any", "ja", "en"])
    p.add_argument("--workload", default="any")
    p.add_argument("--bucket", default="mixed", help="label written to the manifest")
    p.add_argument("--limit-docs", type=int, default=0)
    add_corpus_args(p)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--dry-run", action="store_true",
                   help="tokenize/chunk/manifest but never contact the server")
    return p


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)
    args.settings = run_settings(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or ".", exist_ok=True)
    if args.mode == "extract":
        run_extract(args)
    else:
        run_selfgen(args)


if __name__ == "__main__":
    main()
