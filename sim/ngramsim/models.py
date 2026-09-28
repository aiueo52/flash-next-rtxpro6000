from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Message:
    role: str
    text: str
    model_tag: str | None = None
    is_generated: bool | None = None

    @property
    def decode_target(self) -> bool:
        return self.is_generated if self.is_generated is not None else self.role == "assistant"


@dataclass(slots=True)
class Conversation:
    source: str
    name: str
    created_at: str
    messages: list[Message]
    workload: str = "prose"
    corpus: str = "lmstudio-chat"


@dataclass(slots=True)
class ParseSummary:
    files_seen: int = 0
    conversations_loaded: int = 0
    skipped_corrupt: int = 0
    skipped_empty: int = 0
    skipped_messages: int = 0
    files_selected: int = 0
    skipped_oversize: int = 0
    skipped_by_limit: int = 0
    warning_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
