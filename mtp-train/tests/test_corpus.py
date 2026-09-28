"""Corpus readers and the extraction client's corpus assembly (CPU only).

All fixtures are synthetic strings written into a temp directory; nothing is
read from a default location and no tokenizer or server is needed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import formats as F  # noqa: E402


def _jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _claude_records():
    return [
        {"type": "summary", "summary": "ignored"},
        {"type": "user", "message": {"role": "user", "content": "please list files"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan: call the tool"},
            {"type": "text", "text": "Listing now."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
        ]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": [{"type": "text", "text": "a.txt\nb.txt"}]},
        ]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Two files."},
        ]}},
        "not a dict",
    ]


def _codex_records():
    return [
        {"type": "session_meta", "timestamp": "2026-01-01T00:00:00Z",
         "payload": {"id": "s1", "timestamp": "2026-01-01T00:00:00Z",
                     "base_instructions": {"text": "base rules"}}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "fix the bug"}]}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}]}},
    ]


def test_claude_session_roles_and_reasoning():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.jsonl")
        _jsonl(path, _claude_records())
        msgs = F.parse_claude_session(path)
        roles = [m[0].role for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert msgs[1][1] == "plan: call the tool"
        assert "<tool_call>" in msgs[1][0].text and '"command": "ls"' in msgs[1][0].text
        assert msgs[2][0].text == "a.txt\nb.txt"
        norm, _ = F.normalize(msgs)
        assert [r for r, _, _ in norm] == ["user", "assistant", "tool", "assistant"]
        sessions, counter = F.load_claude_sessions(d)
        assert len(sessions) == 1 and counter["files_seen"] == 1


def test_claude_session_without_assistant_is_dropped():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.jsonl")
        _jsonl(path, [{"type": "user", "message": {"role": "user", "content": "hi"}}])
        assert F.parse_claude_session(path) is None


def test_normalize_merges_and_remaps_roles():
    msgs = [F.mk("system", "sys"), F.mk("developer", "dev"), F.mk("user", "q"),
            F.mk("assistant", "a1"), F.mk("assistant", "a2", "why"),
            F.mk("system", "late"), F.mk("weird", "x")]
    norm, stats = F.normalize(msgs)
    assert norm == [("system", "sys", ""), ("user", "dev\nq", ""),
                    ("assistant", "a1\na2", "why"), ("user", "late", "")]
    assert stats["developer_as_user"] == 1 and stats["late_system_as_user"] == 1


def test_generic_file_shapes():
    with tempfile.TemporaryDirectory() as d:
        _jsonl(os.path.join(d, "chat.jsonl"), [
            {"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}]},
            {"conversations": [{"from": "human", "value": "q2"},
                               {"from": "gpt", "value": "a2"}]},
            {"instruction": "do it", "input": "data", "output": "ok", "system": "s"},
            {"text": "plain body"},
            {"unrelated": 1},
        ])
        docs = list(F.iter_file_docs(os.path.join(d, "chat.jsonl")))
        assert len(docs) == 4
        assert [m.role for m in docs[1]] == ["user", "assistant"]
        assert [m.role for m in docs[2]] == ["system", "user", "assistant"]
        assert docs[2][1].text == "do it\n\ndata"
        assert docs[3][0].role == "assistant"
        with open(os.path.join(d, "long.txt"), "w") as fh:
            fh.write("\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(40)))
        parts = list(F.iter_file_docs(os.path.join(d, "long.txt"), txt_split_bytes=300))
        assert len(parts) > 3
        assert F.expand_globs([d]) == sorted(
            os.path.join(d, n) for n in ("chat.jsonl", "long.txt"))


def test_codex_and_lmstudio_via_ngramsim():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "codex"))
        _jsonl(os.path.join(d, "codex", "rollout-1.jsonl"), _codex_records())
        convs, summary = F.load_codex_sessions([os.path.join(d, "codex")])
        assert summary.files_seen == 1 and len(convs) == 1
        assert any(m.role == "assistant" and m.text == "done" for m in convs[0].messages)

        os.makedirs(os.path.join(d, "lms"))
        with open(os.path.join(d, "lms", "x.conversation.json"), "w") as fh:
            json.dump({"name": "x", "createdAt": "2026-01-01T00:00:00Z", "messages": [
                {"currentlySelected": 0, "versions": [
                    {"type": "singleStep", "role": "user",
                     "content": [{"type": "text", "text": "hello"}]}]},
                {"currentlySelected": 0, "versions": [
                    {"type": "singleStep", "role": "assistant",
                     "content": [{"type": "text", "text": "hi there"}]}]},
            ]}, fh)
        convs, _ = F.load_lmstudio_conversations(os.path.join(d, "lms"))
        assert len(convs) == 1


def test_client_load_corpus_splits_by_document():
    from extract.client import chunk_ids, load_corpus

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "cc"))
        _jsonl(os.path.join(d, "cc", "s.jsonl"), _claude_records())
        _jsonl(os.path.join(d, "notes.jsonl"), [{"text": "note body one"},
                                                {"text": "note body two"}])
        docs = load_corpus(claude=[os.path.join(d, "cc")],
                           files=[f"notes={os.path.join(d, 'notes.jsonl')}"],
                           max_tool_chars=4)
        by_source = {}
        for doc in docs:
            by_source.setdefault(doc["source"], []).append(doc)
        assert set(by_source) == {"claude", "notes"}
        # a lone assistant turn gets a synthetic user turn
        assert by_source["notes"][0]["messages"][0]["role"] == "user"
        # tool output is truncated to max_tool_chars (head + tail)
        tool = [m for m in by_source["claude"][0]["messages"] if m["role"] == "tool"][0]
        assert "[truncated]" in tool["content"]
        keys = {doc["split_key"] for doc in docs}
        assert len(keys) == len(docs)

    pieces = chunk_ids(list(range(100)), max_len=30, min_len=5)
    assert [i for i, _ in pieces] == [0, 1, 2, 3]
    assert [len(p) for _, p in pieces] == [30, 30, 30, 10]
