import json
from collections import Counter

from ngramsim.parser import load_conversations, parse_conversation_file


def _fixture():
    return {
        "name": "fixture",
        "createdAt": "2026-01-01T00:00:00Z",
        "systemPrompt": "system text",
        "messages": [
            {
                "currentlySelected": 0,
                "versions": [
                    {
                        "type": "singleStep",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "file", "name": "ignored"},
                            {"type": "text", "text": "world"},
                        ],
                    }
                ],
            },
            {
                "currentlySelected": 0,
                "versions": [
                    {
                        "type": "multiStep",
                        "role": "assistant",
                        "senderInfo": {"senderName": "fixture-model"},
                        "steps": [
                            {
                                "type": "contentBlock",
                                "style": {"type": "thinking", "ended": True},
                                "content": [{"type": "text", "text": "secret reasoning"}],
                            },
                            {"type": "toolStatus", "statusState": {}},
                            {
                                "type": "contentBlock",
                                "content": [
                                    {"type": "text", "text": "answer"},
                                    {"type": "toolCallResult", "content": "ignored"},
                                    {"type": "mystery", "text": "ignored"},
                                ],
                            },
                            {"type": "futureBlock", "value": 1},
                        ],
                    }
                ],
            },
        ],
    }


def test_parser_extracts_selected_body_text_and_warns_for_unknown_types(tmp_path):
    source = tmp_path / "sample.conversation.json"
    source.write_text(json.dumps(_fixture()), encoding="utf-8")
    counts = Counter()
    warnings = []

    conversation = parse_conversation_file(source, counts, warnings)

    assert conversation is not None
    assert [(message.role, message.text) for message in conversation.messages] == [
        ("system", "system text"),
        ("user", "hello world"),
        ("assistant", "answer"),
    ]
    assert conversation.messages[-1].model_tag == "fixture-model"
    assert counts["skipped_thinking_blocks"] == 1
    assert counts["unknown_content_type:mystery"] == 1
    assert counts["unknown_step_type:futureBlock"] == 1
    assert len(warnings) == 2


def test_directory_loader_accounts_for_corrupt_and_empty_files(tmp_path):
    (tmp_path / "good.conversation.json").write_text(json.dumps(_fixture()), encoding="utf-8")
    (tmp_path / "bad.conversation.json").write_text("{", encoding="utf-8")
    (tmp_path / "empty.conversation.json").write_text(json.dumps({"messages": []}), encoding="utf-8")

    conversations, summary = load_conversations(tmp_path)

    assert len(conversations) == 1
    assert summary.files_seen == 3
    assert summary.skipped_corrupt == 1
    assert summary.skipped_empty == 1
