from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Conversation, Message, ParseSummary

_KNOWN_SKIPPED_STEPS = {"toolStatus", "debugInfoBlock", "requestConfirmToolCall"}
_KNOWN_SKIPPED_CONTENT = {"file", "toolCallRequest", "toolCallResult"}


def _warn(counter: Counter[str], samples: list[str], key: str, detail: str) -> None:
    counter[key] += 1
    if len(samples) < 50:
        samples.append(detail)


def _selected_version(message: dict[str, Any]) -> dict[str, Any] | None:
    versions = message.get("versions")
    selected = message.get("currentlySelected", 0)
    try:
        if isinstance(versions, list):
            value = versions[selected]
        elif isinstance(versions, dict):
            value = versions[str(selected)] if str(selected) in versions else versions[selected]
        else:
            return None
    except (IndexError, KeyError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _single_text(version: dict[str, Any], counts: Counter[str], warnings: list[str]) -> str:
    pieces: list[str] = []
    content = version.get("content", [])
    if not isinstance(content, list):
        _warn(counts, warnings, "malformed_single_content", "singleStep content was not a list")
        return ""
    for block in content:
        if not isinstance(block, dict):
            _warn(counts, warnings, "malformed_content_block", "non-object singleStep content")
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            pieces.append(block["text"])
        elif kind not in _KNOWN_SKIPPED_CONTENT:
            _warn(counts, warnings, f"unknown_content_type:{kind}", f"unknown singleStep content type {kind!r}")
    return "".join(pieces)


def _multi_text(version: dict[str, Any], counts: Counter[str], warnings: list[str]) -> str:
    pieces: list[str] = []
    steps = version.get("steps", [])
    if not isinstance(steps, list):
        _warn(counts, warnings, "malformed_steps", "multiStep steps was not a list")
        return ""
    for step in steps:
        if not isinstance(step, dict):
            _warn(counts, warnings, "malformed_step", "non-object multiStep step")
            continue
        kind = step.get("type")
        if kind != "contentBlock":
            if kind not in _KNOWN_SKIPPED_STEPS:
                _warn(counts, warnings, f"unknown_step_type:{kind}", f"unknown multiStep step type {kind!r}")
            continue
        style = step.get("style")
        if isinstance(style, dict) and style.get("type") == "thinking":
            counts["skipped_thinking_blocks"] += 1
            continue
        content = step.get("content", [])
        if not isinstance(content, list):
            _warn(counts, warnings, "malformed_content_block", "contentBlock content was not a list")
            continue
        for block in content:
            if not isinstance(block, dict):
                _warn(counts, warnings, "malformed_content_item", "non-object contentBlock item")
                continue
            inner_kind = block.get("type")
            if inner_kind == "text" and isinstance(block.get("text"), str):
                pieces.append(block["text"])
            elif inner_kind not in _KNOWN_SKIPPED_CONTENT:
                _warn(counts, warnings, f"unknown_content_type:{inner_kind}", f"unknown contentBlock item {inner_kind!r}")
    return "".join(pieces)


def classify_workload(messages: list[Message]) -> str:
    """Classify using code-fence density and conversation length."""
    text = "\n".join(message.text for message in messages)
    fence_count = text.count("```")
    blocks = fence_count // 2
    density = fence_count / max(len(text) / 1000.0, 1.0)
    if blocks >= 1 and density >= 0.5:
        return "code"
    if fence_count > 0 or len(messages) >= 12:
        return "mixed"
    return "prose"


def parse_conversation_file(path: Path, counts: Counter[str], warnings: list[str]) -> Conversation | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _warn(counts, warnings, "corrupt_files", f"{path.name}: {exc}")
        return None
    if not isinstance(data, dict):
        _warn(counts, warnings, "corrupt_files", f"{path.name}: top level is not an object")
        return None

    parsed: list[Message] = []
    system_prompt = data.get("systemPrompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        parsed.append(Message(role="system", text=system_prompt))
    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list):
        _warn(counts, warnings, "malformed_messages", f"{path.name}: messages is not a list")
        return None
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            _warn(counts, warnings, "skipped_messages", f"{path.name} message {index}: not an object")
            continue
        version = _selected_version(raw)
        if version is None:
            _warn(counts, warnings, "skipped_messages", f"{path.name} message {index}: invalid selection")
            continue
        kind = version.get("type")
        role = version.get("role")
        if role not in {"system", "user", "assistant"}:
            _warn(counts, warnings, "skipped_messages", f"{path.name} message {index}: invalid role {role!r}")
            continue
        if kind == "singleStep":
            text = _single_text(version, counts, warnings)
        elif kind == "multiStep":
            text = _multi_text(version, counts, warnings)
        else:
            _warn(counts, warnings, f"unknown_version_type:{kind}", f"{path.name} message {index}: unknown version type {kind!r}")
            continue
        if not text:
            _warn(counts, warnings, "empty_messages", f"{path.name} message {index}: no body text")
            continue
        sender = version.get("senderInfo")
        model_tag = sender.get("senderName") if isinstance(sender, dict) and isinstance(sender.get("senderName"), str) else None
        parsed.append(Message(role=role, text=text, model_tag=model_tag))

    if not parsed:
        return Conversation(
            source=str(path),
            name=str(data.get("name") or path.stem),
            created_at=str(data.get("createdAt") or ""),
            messages=[],
        )
    conversation = Conversation(
        source=str(path),
        name=str(data.get("name") or path.stem),
        created_at=str(data.get("createdAt") or ""),
        messages=parsed,
    )
    conversation.workload = classify_workload(parsed)
    return conversation


def load_conversations(path: str | Path) -> tuple[list[Conversation], ParseSummary]:
    root = Path(path).expanduser()
    paths = [root] if root.is_file() else sorted(root.glob("*.conversation.json"))
    counts: Counter[str] = Counter()
    warnings: list[str] = []
    conversations: list[Conversation] = []
    for source in paths:
        conversation = parse_conversation_file(source, counts, warnings)
        if conversation is None:
            continue
        if not conversation.messages:
            counts["empty_conversations"] += 1
            continue
        conversations.append(conversation)
    conversations.sort(key=lambda item: (item.created_at, item.source))
    summary = ParseSummary(
        files_seen=len(paths),
        files_selected=len(paths),
        conversations_loaded=len(conversations),
        skipped_corrupt=counts.get("corrupt_files", 0),
        skipped_empty=counts.get("empty_conversations", 0),
        skipped_messages=counts.get("skipped_messages", 0),
        warning_counts=dict(sorted(counts.items())),
        warnings=warnings,
    )
    return conversations, summary
