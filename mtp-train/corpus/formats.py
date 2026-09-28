"""Readers for the corpus formats the extraction client accepts.

Every reader returns *documents*: lists of messages, each either an
``ngramsim.models.Message`` or a ``(Message, reasoning_text)`` tuple.
``normalize`` then maps them onto the four roles the Qwen chat template accepts
(``system``, ``user``, ``assistant``, ``tool``).

Formats:

* **Codex CLI rollout JSONL** (a directory of ``*.jsonl`` session files) and
  **LM Studio conversation JSON** (a directory of ``*.conversation.json``):
  parsed by the n-gram simulator's parsers in ``../sim/ngramsim`` (imported,
  not duplicated; set ``NGRAMSIM_PATH`` if ``sim/`` is elsewhere).
* **Claude Code session JSONL** (a directory tree of ``*.jsonl`` transcripts):
  ``parse_claude_session`` below.
* **Generic files**: ``.jsonl`` / ``.ndjson`` / ``.json`` records in the common
  chat shapes (``messages`` / ``conversations`` with ``role``/``from`` and
  ``content``/``value``; ``instruction``/``input``/``output``;
  ``prompt``/``response``; or a bare ``text`` field), and ``.txt`` / ``.md``
  plain text: ``iter_file_docs``.
* **Hugging Face ``datasets`` Arrow cache files** (``*.arrow`` stream files of a
  dataset you already downloaded): ``iter_arrow_docs``.

Nothing here reads a default location: every path is an argument.  Only use
data you are entitled to use for training.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_ngramsim():
    """Import ``ngramsim`` from the environment, ``$NGRAMSIM_PATH`` or ``../sim``."""
    try:
        import ngramsim  # noqa: F401
    except ImportError:
        if "NGRAMSIM_PATH" in os.environ:  # set: used alone, never widened
            if not os.environ["NGRAMSIM_PATH"].strip():
                raise SystemExit("NGRAMSIM_PATH is set but empty")
            candidates = [os.environ["NGRAMSIM_PATH"]]
        else:
            candidates = [os.path.join(_HERE, "..", "..", "sim")]
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "ngramsim")):
                sys.path.insert(0, os.path.abspath(c))
                break
        import ngramsim  # noqa: F401
    import ngramsim.models
    import ngramsim.parser

    return ngramsim


def Message(role, text, model_tag=None, is_generated=None):  # noqa: N802
    """``ngramsim.models.Message`` (imported lazily)."""
    return _import_ngramsim().models.Message(role, text, model_tag, is_generated)


def classify_workload(messages) -> str:
    """code / mixed / prose, by code-fence density (``ngramsim.parser``)."""
    plain = [m[0] if isinstance(m, tuple) else m for m in messages]
    return _import_ngramsim().parser.classify_workload(plain)


def mk(role: str, text: str, reasoning: str = ""):
    """One message with an optional reasoning trace (``<think>`` content)."""
    return (Message(role, text, None, role == "assistant"), reasoning)


def normalize(messages):
    """Map corpus roles onto the 4 roles the template accepts, then merge
    consecutive same-role messages (the Codex parser splits one tool call into
    several assistant records; merging avoids inventing extra turn headers).

    ``developer`` becomes ``user``, a ``system`` message after the first turn
    becomes ``user``, anything else unknown is dropped.
    Returns ``([(role, text, reasoning)], stats)``.
    """
    out = []
    stats: Counter = Counter()
    for m in messages:
        if isinstance(m, tuple):
            m, reasoning = m
        else:
            reasoning = ""
        role, text = m.role, m.text
        if not text and not reasoning:
            continue
        if role == "developer":
            role = "user"
            stats["developer_as_user"] += 1
        elif role == "system" and out:
            role = "user"
            stats["late_system_as_user"] += 1
        elif role not in ("system", "user", "assistant", "tool"):
            stats["dropped_role:" + str(role)] += 1
            continue
        if out and out[-1][0] == role:
            out[-1][1].append(text)
            if reasoning:
                out[-1][2].append(reasoning)
            stats["merged"] += 1
        else:
            out.append([role, [text], [reasoning] if reasoning else []])
    return [(r, "\n".join(p), "\n".join(q)) for r, p, q in out], stats


# ------------------------------------------------------------ Codex / LM Studio
def load_codex_sessions(paths: Sequence[str], max_file_mb: float = 20.0,
                        limit_files: Optional[int] = None):
    """Codex CLI rollout JSONL -> (list of ngramsim Conversation, ParseSummary)."""
    _import_ngramsim()
    from ngramsim.codex_parser import load_codex_sessions as _load

    return _load([os.path.expanduser(p) for p in paths],
                 limit_files=limit_files, max_file_mb=max_file_mb)


def load_lmstudio_conversations(path: str):
    """LM Studio ``*.conversation.json`` dir -> (conversations, ParseSummary)."""
    _import_ngramsim()
    from ngramsim.parser import load_conversations

    return load_conversations(os.path.expanduser(path))


# --------------------------------------------------------------- Claude Code
def _cc_blocks(content):
    """-> (text, reasoning) for one Claude Code message ``content`` field.

    Tool calls are rendered in the Qwen tool-call syntax so they tokenise like
    the model's own tool calls; tool results keep their text.
    """
    if isinstance(content, str):
        return content, ""
    text, think = [], []
    if not isinstance(content, list):
        return "", ""
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text" and isinstance(b.get("text"), str):
            text.append(b["text"])
        elif t == "thinking" and isinstance(b.get("thinking"), str):
            think.append(b["thinking"])
        elif t == "tool_use":
            name = b.get("name") or ""
            try:
                args = json.dumps(b.get("input", {}), ensure_ascii=False)
            except (TypeError, ValueError):
                args = ""
            text.append(f"<tool_call>\n<function={name}>\n{args}\n</function>\n</tool_call>")
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, str):
                text.append(c)
            elif isinstance(c, list):
                for inner in c:
                    if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                        text.append(inner["text"])
    return "\n".join(text), "\n".join(think)


def parse_claude_session(path, counter: Optional[Counter] = None):
    """One Claude Code session transcript (JSONL) -> list of messages, or None.

    Records of type ``user`` / ``assistant`` / ``system`` are kept; a ``user``
    record that carries ``tool_result`` blocks becomes a ``tool`` turn.
    Sessions without any assistant turn return None.
    """
    counter = counter if counter is not None else Counter()
    msgs = []
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        counter["corrupt_files"] += 1
        return None
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counter["malformed_jsonl_records"] += 1
                continue
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            if rtype not in ("user", "assistant", "system"):
                counter["skipped_record:" + str(rtype)] += 1
                continue
            m = rec.get("message")
            if not isinstance(m, dict):
                if rtype == "system" and isinstance(rec.get("content"), str):
                    msgs.append(mk("user", rec["content"]))
                continue
            role = m.get("role") or rtype
            text, think = _cc_blocks(m.get("content"))
            if not text and not think:
                continue
            if role == "user":
                is_tool = isinstance(m.get("content"), list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in m["content"]
                )
                msgs.append(mk("tool" if is_tool else "user", text))
            elif role == "assistant":
                msgs.append(mk("assistant", text, think))
    if not msgs or not any(x[0].role == "assistant" for x in msgs):
        return None
    return msgs


def load_claude_sessions(root: str, max_file_mb: float = 20.0):
    """Every ``*.jsonl`` under ``root`` -> [(file name, messages)], plus counts."""
    counter: Counter = Counter()
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        raise SystemExit(f"{root}: no such file or directory")
    paths = ([root] if os.path.isfile(root)
             else sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)))
    cap = int(max_file_mb * 1024 * 1024)
    out = []
    for path in paths:
        if os.path.getsize(path) > cap:
            counter["oversize_files"] += 1
            continue
        msgs = parse_claude_session(path, counter)
        if msgs:
            # the path under ``root`` names the document (split key): session
            # files of the same name in two sub-directories stay distinct
            name = (os.path.relpath(path, root) if os.path.isdir(root)
                    else os.path.basename(path))
            out.append((name, msgs))
        else:
            counter["empty_conversations"] += 1
    counter["files_seen"] = len(paths)
    return out, counter


# ------------------------------------------------------------- generic files
TEXT_KEYS = ("text", "content", "output", "response", "completion", "body", "chosen")
ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant",
                "model": "assistant", "system": "system", "tool": "tool",
                "function": "tool"}


def msgs_from_record(rec):
    """One JSON record -> list of Message, or None when no text is found."""
    if isinstance(rec, str):
        return [Message("assistant", rec, None, True)]
    if not isinstance(rec, dict):
        return None
    for key in ("messages", "conversations", "conversation", "turns", "chat"):
        seq = rec.get(key)
        if isinstance(seq, list) and seq:
            out = []
            for m in seq:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from") or "user"
                role = ROLE_ALIASES.get(str(role).lower(), str(role).lower())
                txt = m.get("content") if isinstance(m.get("content"), str) else m.get("value")
                if not isinstance(txt, str):
                    continue
                if role not in ("system", "user", "assistant", "tool"):
                    role = "user"
                out.append(Message(role, txt, None, role == "assistant"))
            return out or None
    if "instruction" in rec or "prompt" in rec or "question" in rec:
        q = rec.get("instruction") or rec.get("prompt") or rec.get("question") or ""
        extra = rec.get("input") or ""
        a = (rec.get("output") or rec.get("response") or rec.get("answer")
             or rec.get("completion") or rec.get("chosen") or "")
        if isinstance(q, str) and isinstance(a, str) and (q or a):
            ms = []
            if isinstance(rec.get("system"), str) and rec["system"]:
                ms.append(Message("system", rec["system"], None, False))
            ms.append(Message("user", (q + ("\n\n" + extra if extra else "")), None, False))
            ms.append(Message("assistant", a, None, True))
            return ms
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return [Message("assistant", v, None, True)]
    return None


def iter_file_docs(path, max_bytes: int = 512 * 1024 * 1024,
                   txt_split_bytes: int = 400_000) -> Iterator[list]:
    """Yield message-list documents from a .jsonl / .ndjson / .json / .txt / .md file.

    Plain text longer than ``txt_split_bytes`` is split at paragraph
    boundaries, so one big file becomes many documents and the by-document
    held-out split has something to hold out.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if p.stat().st_size > max_bytes:
        print(f"[corpus] skipping {path}: larger than {max_bytes} bytes", file=sys.stderr)
        return
    if suffix in (".jsonl", ".ndjson"):
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = msgs_from_record(rec)
                if m:
                    yield m
    elif suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[corpus] skipping {path}: not readable JSON ({exc})", file=sys.stderr)
            return
        if isinstance(data, dict):
            data = data.get("data") or data.get("examples") or data.get("records") or [data]
        if isinstance(data, list):
            for rec in data:
                m = msgs_from_record(rec)
                if m:
                    yield m
    elif suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) <= txt_split_bytes:
            if text.strip():
                yield [Message("assistant", text, None, True)]
        else:
            buf, size = [], 0
            for para in text.split("\n\n"):
                buf.append(para)
                size += len(para) + 2
                if size >= txt_split_bytes:
                    yield [Message("assistant", "\n\n".join(buf), None, True)]
                    buf, size = [], 0
            if buf and "".join(buf).strip():
                yield [Message("assistant", "\n\n".join(buf), None, True)]


def expand_globs(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        pattern = os.path.expanduser(pattern)
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "**", "*")
        paths += sorted(glob.glob(pattern, recursive=True))
    return [p for p in dict.fromkeys(paths) if os.path.isfile(p)]


# ------------------------------------------------------------- Arrow datasets
def iter_arrow_docs(glob_pattern: str, text_field: str,
                    aux_field: Optional[str] = None,
                    instruction: Optional[str] = None,
                    limit: Optional[int] = None,
                    skip: int = 0) -> Iterator[list]:
    """Rows of Hugging Face ``datasets`` Arrow stream files -> documents.

    With ``aux_field`` a row becomes ``user = instruction + row[text_field]``,
    ``assistant = row[aux_field]`` (e.g. article -> summary); without it the
    whole ``row[text_field]`` is one assistant turn.  ``skip`` drops the first
    rows, e.g. to draw prompts from rows a previous extraction did not use.
    """
    import pyarrow.ipc as ipc

    rows = 0
    seen = 0
    for path in sorted(glob.glob(os.path.expanduser(glob_pattern))):
        with ipc.open_stream(path) as reader:
            for rb in reader:
                cols = {
                    c: rb.column(c).to_pylist()
                    for c in [text_field] + ([aux_field] if aux_field else [])
                }
                for j in range(rb.num_rows):
                    seen += 1
                    if seen <= skip:
                        continue
                    body = cols[text_field][j] or ""
                    if aux_field:
                        yield [
                            mk("user", (instruction or "") + body),
                            mk("assistant", cols[aux_field][j] or ""),
                        ]
                    else:
                        yield [mk("assistant", body)]
                    rows += 1
                    if limit and rows >= limit:
                        return


__all__ = [
    "Message",
    "classify_workload",
    "mk",
    "normalize",
    "load_codex_sessions",
    "load_lmstudio_conversations",
    "parse_claude_session",
    "load_claude_sessions",
    "msgs_from_record",
    "iter_file_docs",
    "expand_globs",
    "iter_arrow_docs",
]
