from __future__ import annotations

from bisect import bisect_right
from itertools import pairwise


class CorpusIndex:
    """Online token corpus with bounded n-gram occurrence and transition indexes."""

    def __init__(self, max_n: int, window_tokens: int | None = None) -> None:
        self.max_n = max_n
        self.window_tokens = window_tokens
        self.tokens: list[int] = []
        self.segment_start = 0
        self.boundaries: list[int] = []
        # Only the two newest starts are needed: the newest can be the current
        # suffix itself (which has no continuation), while the previous one is
        # then the newest usable occurrence.  This avoids one list per unique
        # n-gram on large transcripts.
        self.occurrences: dict[int, dict[tuple[int, ...], tuple[int | None, int]]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.transitions: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.transition_positions: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }

    def append(self, token: int) -> None:
        end = len(self.tokens)
        segment_length = end - self.segment_start
        for n in range(1, min(self.max_n, segment_length) + 1):
            key = tuple(self.tokens[end - n : end])
            self.transitions[n][key] = token
            self.transition_positions[n][key] = end - n
        self.tokens.append(token)
        size = len(self.tokens)
        segment_size = size - self.segment_start
        for n in range(1, min(self.max_n, segment_size) + 1):
            start = size - n
            key = tuple(self.tokens[start:size])
            old = self.occurrences[n].get(key)
            previous = old[1] if old is not None else None
            self.occurrences[n][key] = (previous, start)
        if self.window_tokens is not None:
            slack = max(self.max_n, self.window_tokens // 2)
            if len(self.tokens) > self.window_tokens + slack:
                self._compact()

    def extend(self, tokens: list[int]) -> None:
        for token in tokens:
            self.append(token)

    def start_segment(self) -> None:
        """Close a conversation without creating cross-boundary transitions."""
        end = len(self.tokens)
        if end == self.segment_start:
            return
        segment_length = end - self.segment_start
        # An n-gram ending exactly at the boundary has no textual continuation.
        # Remove that newest occurrence while preserving the previous usable one.
        for n in range(1, min(self.max_n, segment_length) + 1):
            start = end - n
            key = tuple(self.tokens[start:end])
            pair = self.occurrences[n].get(key)
            if pair is None or pair[1] != start:
                continue
            if pair[0] is None:
                del self.occurrences[n][key]
            else:
                self.occurrences[n][key] = (None, pair[0])
        self.boundaries.append(end)
        self.segment_start = end

    def _compact(self) -> None:
        """Rebuild indexes over the newest bounded token window."""
        if self.window_tokens is None or len(self.tokens) <= self.window_tokens:
            return
        old_tokens = self.tokens
        old_boundaries = self.boundaries
        keep_from = len(old_tokens) - self.window_tokens
        cuts = [keep_from]
        cuts.extend(boundary for boundary in old_boundaries if keep_from < boundary < len(old_tokens))
        cuts.append(len(old_tokens))

        configured_window = self.window_tokens
        self.window_tokens = None
        self.tokens = []
        self.segment_start = 0
        self.boundaries = []
        self.occurrences = {n: {} for n in range(1, self.max_n + 1)}
        self.transitions = {n: {} for n in range(1, self.max_n + 1)}
        self.transition_positions = {n: {} for n in range(1, self.max_n + 1)}
        for index, (start, end) in enumerate(pairwise(cuts)):
            self.extend(old_tokens[start:end])
            if index < len(cuts) - 2:
                self.start_segment()
        self.window_tokens = configured_window

    def suffix_match(self, n_min: int, n_max: int, length: int) -> list[int]:
        upper = min(n_max, len(self.tokens) - self.segment_start, self.max_n)
        window_floor = (
            max(0, len(self.tokens) - self.window_tokens)
            if self.window_tokens is not None
            else 0
        )
        for n in range(upper, n_min - 1, -1):
            key = tuple(self.tokens[-n:])
            pair = self.occurrences[n].get(key)
            if pair is None:
                continue
            for start in (pair[1], pair[0]):
                if start is None:
                    continue
                if start < window_floor:
                    continue
                continuation = start + n
                if continuation < len(self.tokens):
                    boundary_index = bisect_right(self.boundaries, continuation)
                    source_end = self.boundaries[boundary_index] if boundary_index < len(self.boundaries) else len(self.tokens)
                    candidate = self.tokens[continuation : min(continuation + length, source_end)]
                    if candidate:
                        return candidate
        return []

    def ngram_mod(self, n: int, length: int) -> list[int]:
        if n > len(self.tokens) - self.segment_start or n > self.max_n:
            return []
        context = list(self.tokens[-n:])
        draft: list[int] = []
        window_floor = (
            max(0, len(self.tokens) - self.window_tokens)
            if self.window_tokens is not None
            else 0
        )
        for _ in range(length):
            key = tuple(context[-n:])
            token = self.transitions[n].get(key)
            if token is None:
                break
            if self.transition_positions[n].get(key, -1) < window_floor:
                break
            draft.append(token)
            context.append(token)
        return draft


def matching_prefix(draft: list[int], actual: list[int], limit: int) -> int:
    matched = 0
    for proposed, expected in zip(draft[:limit], actual[:limit]):
        if proposed != expected:
            break
        matched += 1
    return matched
