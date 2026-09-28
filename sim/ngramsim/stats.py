from __future__ import annotations

from dataclasses import dataclass, field

from .cost import mtp_expected_confirmed


def _quantile(histogram: list[int], q: float) -> int | None:
    total = sum(histogram)
    if total == 0:
        return None
    target = max(1, int((q * total) + 0.999999999))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return len(histogram) - 1


@dataclass(slots=True)
class Accumulator:
    draft_steps: int
    positions: int = 0
    hits: int = 0
    match_sum: int = 0
    histogram: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.histogram:
            self.histogram = [0] * (self.draft_steps + 1)

    def add(self, hit: bool, match: int) -> None:
        self.positions += 1
        if hit:
            self.hits += 1
            bounded = min(max(match, 0), self.draft_steps)
            self.match_sum += bounded
            self.histogram[bounded] += 1

    def result(self, alphas: list[float], hybrid_mtp_steps: int) -> dict[str, object]:
        if self.positions == 0:  # nothing measured: None, never 0
            hit_rate = expected = hit_contribution = None
        else:
            hit_rate = self.hits / self.positions
            expected = 1.0 + self.match_sum / self.positions
            hit_contribution = (self.hits + self.match_sum) / self.positions
        mean = self.match_sum / self.hits if self.hits else None
        no_hit = self.positions - self.hits
        hybrid = {
            f"{alpha:.2f}": (
                (self.hits + self.match_sum + no_hit * mtp_expected_confirmed(hybrid_mtp_steps, alpha))
                / self.positions
                if self.positions
                else None
            )
            for alpha in alphas
        }
        return {
            "positions": self.positions,
            "hits": self.hits,
            "hit_rate": hit_rate,
            "hit_match_mean": mean,
            "hit_match_median": _quantile(self.histogram, 0.5),
            "hit_match_p90": _quantile(self.histogram, 0.9),
            "hit_match_histogram": {str(i): count for i, count in enumerate(self.histogram)},
            "expected_confirmed_per_verification": expected,
            "hit_confirmed_sum_per_position": hit_contribution,
            "hybrid_expected_confirmed": hybrid,
        }


class StatsBook:
    def __init__(
        self,
        draft_lengths: list[int],
        alphas: list[float],
        hybrid_mtp_steps: int,
        lookup_mode: str,
    ) -> None:
        self.draft_lengths = draft_lengths
        self.alphas = alphas
        self.hybrid_mtp_steps = hybrid_mtp_steps
        self.lookup_mode = lookup_mode
        self._data: dict[tuple[str, str, str, int, str, str], Accumulator] = {}

    def add(
        self,
        *,
        corpus: str,
        family: str,
        strategy: str,
        draft_steps: int,
        workload: str,
        model_tag: str,
        hit: bool,
        match: int,
    ) -> None:
        for scope, group in (("overall", "all"), ("workload", workload), ("model", model_tag)):
            key = (corpus, family, strategy, draft_steps, scope, group)
            accumulator = self._data.get(key)
            if accumulator is None:
                accumulator = self._data[key] = Accumulator(draft_steps)
            accumulator.add(hit, match)

    def results(self) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for (corpus, family, strategy, length, scope, group), accumulator in sorted(self._data.items()):
            row: dict[str, object] = {
                "corpus": corpus,
                "lookup_mode": self.lookup_mode,
                "family": family,
                "strategy": strategy,
                "draft_steps": length,
                "verification_width": length + 1,
                "scope": scope,
                "group": group,
            }
            row.update(accumulator.result(self.alphas, self.hybrid_mtp_steps))
            output.append(row)
        return output
