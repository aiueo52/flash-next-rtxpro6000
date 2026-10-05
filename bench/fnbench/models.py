from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

type JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    def as_request_fields(self) -> JsonObject:
        fields: JsonObject = {"temperature": self.temperature}
        if self.top_p is not None:
            fields["top_p"] = self.top_p
        if self.top_k is not None:
            fields["top_k"] = self.top_k
        if self.min_p is not None:
            fields["min_p"] = self.min_p
        return fields


# LM Studio's default sampler (temperature 0.8, top_p 0.95, top_k 40, min_p 0.05).
LMSTUDIO_SAMPLING = Sampling(temperature=0.8, top_p=0.95, top_k=40, min_p=0.05)


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_path: Path
    max_tokens: int
    sampling: Sampling
    description: str
    repeat_prompt: int = 1
    postamble: str = ""
    prompt_id: str | None = None
    prompt_set: str | None = None
    input_tokens: int | None = None

    def read_prompt(self) -> str:
        text = self.prompt_path.read_text(encoding="utf-8").strip()
        if self.repeat_prompt > 1:
            sections = [
                f"[CONTEXT BLOCK {index + 1:04d}]\n{text}"
                for index in range(self.repeat_prompt)
            ]
            text = "\n\n".join(sections)
        if self.postamble:
            text = f"{text}\n\n{self.postamble.strip()}"
        return text
