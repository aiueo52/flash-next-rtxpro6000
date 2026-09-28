from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Conversation, Message, ParseSummary
from .parser import classify_workload

_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{512,}={0,2}(?![A-Za-z0-9+/])")
_DATA_URI = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/_=-]+", re.IGNORECASE)
_KNOWN_OUTER = {
    "session_meta",
    "response_item",
    "message",
    "event_msg",
    "turn_context",
    "world_state",
    "inter_agent_communication_metadata",
    "compacted",
}
_KNOWN_RESPONSE_SKIPS = {"reasoning"}


def _account(counter: Counter[str], warnings: list[str], key: str, detail: str | None = None) -> None:
    counter[key] += 1
    if detail and len(warnings) < 50:
        warnings.append(detail)


def _sanitize(text: str, counter: Counter[str]) -> str:
    def removed(match: re.Match[str]) -> str:
        counter["skipped_binary_blobs"] += 1
        return ""

    text = _DATA_URI.sub(removed, text)
    return _BASE64_RUN.sub(removed, text)


def _text_blocks(content: Any, counter: Counter[str], warnings: list[str], *, label: str) -> str:
    if isinstance(content, str):
        return _sanitize(content, counter)
    if not isinstance(content, list):
        if content is not None:
            _account(counter, warnings, "malformed_codex_content", f"{label}: content was {type(content).__name__}")
        return ""
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            _account(counter, warnings, "malformed_codex_content_block", f"{label}: non-object content block")
            continue
        kind = block.get("type")
        if kind in {"input_text", "output_text", "text"} and isinstance(block.get("text"), str):
            pieces.append(_sanitize(block["text"], counter))
        elif kind in {"image", "input_image", "audio", "input_audio", "encrypted_content"}:
            _account(counter, warnings, f"skipped_binary_content:{kind}")
        else:
            _account(counter, warnings, f"unknown_codex_content_type:{kind}", f"{label}: unknown content type {kind!r}")
    return "".join(pieces)


def _base_instructions(payload: dict[str, Any], counter: Counter[str]) -> str:
    value = payload.get("base_instructions")
    if isinstance(value, str):
        return _sanitize(value, counter)
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return _sanitize(value["text"], counter)
    return ""


def _response_messages(
    payload: dict[str, Any],
    current_model: str | None,
    counter: Counter[str],
    warnings: list[str],
    label: str,
) -> list[Message]:
    kind = payload.get("type")
    if kind == "message":
        role = payload.get("role")
        if role not in {"assistant", "user", "developer", "system", "tool"}:
            _account(counter, warnings, f"unknown_codex_role:{role}", f"{label}: unknown message role {role!r}")
            return []
        text = _text_blocks(payload.get("content"), counter, warnings, label=label)
        return [Message(str(role), text, current_model, role == "assistant")] if text else []
    if kind in {"function_call", "tool_call"}:
        arguments = payload.get("arguments")
        if not isinstance(arguments, str):
            _account(counter, warnings, "malformed_tool_arguments", f"{label}: {kind} arguments were not text")
            return []
        result: list[Message] = []
        name = payload.get("name")
        if isinstance(name, str) and name:
            result.append(Message("assistant", _sanitize(name, counter), current_model, False))
        clean = _sanitize(arguments, counter)
        if clean:
            result.append(Message("assistant", clean, current_model, True))
        return result
    if kind in {"custom_tool_call", "custom_call"}:
        value = payload.get("input")
        if not isinstance(value, str):
            _account(counter, warnings, "malformed_tool_arguments", f"{label}: {kind} input was not text")
            return []
        result = []
        name = payload.get("name")
        if isinstance(name, str) and name:
            result.append(Message("assistant", _sanitize(name, counter), current_model, False))
        clean = _sanitize(value, counter)
        if clean:
            result.append(Message("assistant", clean, current_model, True))
        return result
    if kind in {"function_call_output", "tool_call_output", "custom_tool_call_output"}:
        text = _text_blocks(payload.get("output"), counter, warnings, label=label)
        return [Message("tool", text, current_model, False)] if text else []
    if kind == "agent_message":
        text = _text_blocks(payload.get("content"), counter, warnings, label=label)
        return [Message("assistant", text, current_model, True)] if text else []
    if kind not in _KNOWN_RESPONSE_SKIPS:
        _account(counter, warnings, f"unknown_response_item_type:{kind}", f"{label}: unknown response item {kind!r}")
    return []


def parse_codex_file(path: Path, counter: Counter[str], warnings: list[str]) -> Conversation | None:
    messages: list[Message] = []
    current_model: str | None = None
    created_at = ""
    name = path.stem
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        _account(counter, warnings, "corrupt_files", f"{path.name}: {exc}")
        return None
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _account(counter, warnings, "malformed_jsonl_records", f"{path.name}:{line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                _account(counter, warnings, "malformed_jsonl_records", f"{path.name}:{line_number}: non-object record")
                continue
            outer = record.get("type")
            payload = record.get("payload")
            if outer not in _KNOWN_OUTER:
                _account(counter, warnings, f"unknown_codex_record_type:{outer}", f"{path.name}:{line_number}: unknown record {outer!r}")
                continue
            if not isinstance(payload, dict):
                if outer in {"session_meta", "response_item", "message", "turn_context"}:
                    _account(counter, warnings, "malformed_codex_payload", f"{path.name}:{line_number}: payload was not an object")
                continue
            if outer == "session_meta":
                created_at = str(payload.get("timestamp") or record.get("timestamp") or created_at)
                name = str(payload.get("id") or payload.get("session_id") or name)
                provider = payload.get("model_provider")
                if current_model is None and isinstance(provider, str):
                    current_model = provider
                base = _base_instructions(payload, counter)
                if base:
                    messages.append(Message("system", base, current_model, False))
            elif outer == "turn_context":
                model = payload.get("model")
                if isinstance(model, str) and model:
                    current_model = model
            elif outer in {"response_item", "message"}:
                if outer == "message" and "type" not in payload:
                    payload = {**payload, "type": "message"}
                messages.extend(
                    _response_messages(payload, current_model, counter, warnings, f"{path.name}:{line_number}")
                )
            # event_msg and state/compaction envelopes duplicate canonical
            # response items or contain bookkeeping, so they are known skips.
    if not messages or not any(message.decode_target for message in messages):
        return None
    conversation = Conversation(
        source=str(path),
        name=name,
        created_at=created_at,
        messages=messages,
        corpus="codex-agent",
    )
    conversation.workload = classify_workload(messages)
    return conversation


def _discover(paths: Iterable[str | Path]) -> list[Path]:
    discovered: dict[Path, None] = {}
    for value in paths:
        root = Path(value).expanduser()
        candidates = [root] if root.is_file() else root.rglob("*.jsonl")
        for candidate in candidates:
            if candidate.is_file():
                discovered[candidate.resolve()] = None
    return list(discovered)


def load_codex_sessions(
    paths: Iterable[str | Path], *, limit_files: int = 200, max_file_mb: float = 20.0
) -> tuple[list[Conversation], ParseSummary]:
    all_paths = _discover(paths)
    maximum_bytes = int(max_file_mb * 1024 * 1024)
    eligible: list[Path] = []
    oversized = 0
    for path in all_paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = maximum_bytes + 1
        if size > maximum_bytes:
            oversized += 1
        else:
            eligible.append(path)
    eligible.sort(key=lambda item: (item.stat().st_mtime, str(item)), reverse=True)
    selected = eligible[:limit_files]
    counter: Counter[str] = Counter()
    warnings: list[str] = []
    conversations: list[Conversation] = []
    for path in selected:
        conversation = parse_codex_file(path, counter, warnings)
        if conversation is None:
            counter["empty_conversations"] += 1
        else:
            conversations.append(conversation)
    conversations.sort(key=lambda item: (item.created_at, item.source))
    summary = ParseSummary(
        files_seen=len(all_paths),
        files_selected=len(selected),
        conversations_loaded=len(conversations),
        skipped_corrupt=counter.get("corrupt_files", 0),
        skipped_empty=counter.get("empty_conversations", 0),
        skipped_messages=counter.get("skipped_messages", 0),
        skipped_oversize=oversized,
        skipped_by_limit=max(0, len(eligible) - len(selected)),
        warning_counts=dict(sorted(counter.items())),
        warnings=warnings,
    )
    return conversations, summary
