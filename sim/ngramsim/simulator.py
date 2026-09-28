from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import Conversation, ParseSummary
from .stats import StatsBook
from .strategies import CorpusIndex, matching_prefix


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(slots=True)
class SimulationConfig:
    n_mins: list[int]
    n_max: int
    ngram_ns: list[int]
    draft_lengths: list[int]
    alphas: list[float]
    strategies: list[str]
    cross_conv: bool = False
    c1_ms: float = 7.46
    delta_ms: float = 0.5
    draft_ms: float = 1.2
    hybrid_mtp_steps: int = 3
    sensitivity_deltas: list[float] | None = None
    max_tokens_per_conv: int = 60000
    search_window_tokens: int = 60000


def _variants(config: SimulationConfig) -> list[tuple[str, str, int]]:
    variants: list[tuple[str, str, int]] = []
    if "suffix-match" in config.strategies:
        variants.extend(("suffix-match", f"suffix-match(n_min={n_min},n_max={config.n_max})", n_min) for n_min in config.n_mins)
    if "ngram-mod" in config.strategies:
        variants.extend(("ngram-mod", f"ngram-mod(n={n})", n) for n in config.ngram_ns)
    return variants


def simulate(
    conversations: list[Conversation],
    parse_summary: ParseSummary | dict[str, ParseSummary],
    encoder: Encoder,
    config: SimulationConfig,
    *,
    tokenizer_path: str | Path,
    conversations_path: Any,
) -> dict[str, object]:
    if not config.draft_lengths or min(config.draft_lengths) < 1:
        raise ValueError("draft lengths must be positive")
    if min(config.n_mins + config.ngram_ns) < 1 or config.n_max < max(config.n_mins):
        raise ValueError("invalid n-gram bounds")
    max_n = max([config.n_max, *config.ngram_ns])
    max_length = max(config.draft_lengths)
    variants = _variants(config)
    if not variants:
        raise ValueError("at least one strategy must be enabled")

    book = StatsBook(
        config.draft_lengths,
        config.alphas,
        config.hybrid_mtp_steps,
        "global-window" if config.cross_conv else "session-local",
    )
    token_counts = {"all": 0, "assistant": 0}
    conversation_rows: list[dict[str, object]] = []
    shared_indexes: dict[str, CorpusIndex] = {}
    seen_corpora: set[str] = set()
    truncations: dict[str, int] = {}

    for conversation in conversations:
        if config.cross_conv:
            index = shared_indexes.setdefault(
                conversation.corpus, CorpusIndex(max_n, config.search_window_tokens)
            )
        else:
            index = CorpusIndex(max_n, config.search_window_tokens)
        if config.cross_conv and conversation.corpus in seen_corpora:
            index.start_segment()
        seen_corpora.add(conversation.corpus)
        before_all = token_counts["all"]
        before_assistant = token_counts["assistant"]
        remaining = config.max_tokens_per_conv
        truncated = False
        for message_number, message in enumerate(conversation.messages):
            tokens = encoder.encode(message.text)
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                truncated = True
            elif len(tokens) == remaining and message_number < len(conversation.messages) - 1:
                truncated = True
            token_counts["all"] += len(tokens)
            if not message.decode_target:
                index.extend(tokens)
            else:
                token_counts["assistant"] += len(tokens)
                model_tag = message.model_tag or "<unknown>"
                for position, token in enumerate(tokens):
                    actual = tokens[position : position + max_length]
                    for family, label, parameter in variants:
                        if family == "suffix-match":
                            draft = index.suffix_match(parameter, config.n_max, max_length)
                        else:
                            draft = index.ngram_mod(parameter, max_length)
                        hit = bool(draft)
                        for length in config.draft_lengths:
                            match = matching_prefix(draft, actual, length) if hit else 0
                            book.add(
                                corpus=conversation.corpus,
                                family=family,
                                strategy=label,
                                draft_steps=length,
                                workload=conversation.workload,
                                model_tag=model_tag,
                                hit=hit,
                                match=match,
                            )
                    index.append(token)
            remaining -= len(tokens)
            if remaining <= 0:
                break
        if truncated:
            truncations[conversation.corpus] = truncations.get(conversation.corpus, 0) + 1
        conversation_rows.append(
            {
                "corpus": conversation.corpus,
                "source": conversation.source,
                "name": conversation.name,
                "created_at": conversation.created_at,
                "workload": conversation.workload,
                "tokens": token_counts["all"] - before_all,
                "assistant_tokens": token_counts["assistant"] - before_assistant,
                "messages": len(conversation.messages),
            }
        )

    summaries = (
        {"lmstudio-chat": parse_summary}
        if isinstance(parse_summary, ParseSummary)
        else parse_summary
    )
    parser_accounting = {name: asdict(value) for name, value in summaries.items()}
    combined_summary = ParseSummary()
    for summary in summaries.values():
        for field_name in (
            "files_seen",
            "files_selected",
            "conversations_loaded",
            "skipped_corrupt",
            "skipped_empty",
            "skipped_messages",
            "skipped_oversize",
            "skipped_by_limit",
        ):
            setattr(combined_summary, field_name, getattr(combined_summary, field_name) + getattr(summary, field_name))
        for key, value in summary.warning_counts.items():
            combined_summary.warning_counts[key] = combined_summary.warning_counts.get(key, 0) + value
        combined_summary.warnings.extend(summary.warnings)
    by_corpus: dict[str, dict[str, int]] = {}
    for row in conversation_rows:
        counts = by_corpus.setdefault(str(row["corpus"]), {"all": 0, "assistant": 0})
        counts["all"] += int(row["tokens"])
        counts["assistant"] += int(row["assistant_tokens"])

    return {
        "schema_version": 2,
        "config": asdict(config),
        "inputs": {
            "conversations": conversations_path,
            "tokenizer": str(Path(tokenizer_path).expanduser()),
        },
        "parse_summary": asdict(combined_summary),
        "parser_accounting": parser_accounting,
        "token_counts": {**token_counts, "by_corpus": by_corpus},
        "truncations": truncations,
        "conversations": conversation_rows,
        "statistics": book.results(),
    }
