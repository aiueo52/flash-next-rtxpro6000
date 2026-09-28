import json
import os

from ngramsim.codex_parser import load_codex_sessions, parse_codex_file


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _records():
    return [
        {
            "type": "session_meta",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "id": "session-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "model_provider": "openai",
                "base_instructions": {"text": "base rules"},
            },
        },
        {"type": "turn_context", "payload": {"model": "gpt-fixture"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer rules"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "assistant answer"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"cmd":"echo hello"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "output": "tool result",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": "patch body",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "output": [
                    {"type": "text", "text": "ok " + "A" * 600},
                    {"type": "image", "data": "ignored"},
                ],
            },
        },
        {"type": "response_item", "payload": {"type": "future_item"}},
        {
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "duplicate answer"},
        },
        {"type": "future_outer", "payload": {}},
    ]


def test_codex_parser_targets_assistant_and_tool_arguments_only(tmp_path):
    source = tmp_path / "rollout.jsonl"
    _write_jsonl(source, _records())
    from collections import Counter

    counts = Counter()
    warnings = []
    conversation = parse_codex_file(source, counts, warnings)

    assert conversation is not None
    assert conversation.corpus == "codex-agent"
    assert [message.text for message in conversation.messages if message.decode_target] == [
        "assistant answer",
        '{"cmd":"echo hello"}',
        "patch body",
    ]
    assert all(
        message.model_tag == "gpt-fixture"
        for message in conversation.messages
        if message.decode_target
    )
    assert "duplicate answer" not in [message.text for message in conversation.messages]
    assert counts["skipped_binary_blobs"] == 1
    assert counts["skipped_binary_content:image"] == 1
    assert counts["unknown_response_item_type:future_item"] == 1
    assert counts["unknown_codex_record_type:future_outer"] == 1


def test_codex_loader_uses_mtime_limit_and_accounts_for_oversize(tmp_path):
    for index in range(3):
        path = tmp_path / f"{index}.jsonl"
        _write_jsonl(path, _records())
        os.utime(path, (100 + index, 100 + index))
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_text("x" * 5000, encoding="utf-8")

    conversations, summary = load_codex_sessions(
        [tmp_path], limit_files=2, max_file_mb=0.004
    )

    assert [conversation.source.rsplit("/", 1)[-1] for conversation in conversations] == [
        "1.jsonl",
        "2.jsonl",
    ]
    assert summary.files_seen == 4
    assert summary.files_selected == 2
    assert summary.skipped_oversize == 1
    assert summary.skipped_by_limit == 1
