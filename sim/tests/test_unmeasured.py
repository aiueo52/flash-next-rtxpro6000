"""An empty group is 'not measured' (None), never a 0 % hit rate (CPU)."""
from ngramsim.stats import Accumulator


def test_accumulator_with_no_positions_is_unmeasured():
    r = Accumulator(7).result([0.5, 0.8], 3)
    assert r["positions"] == 0
    assert r["hit_rate"] is None
    assert r["expected_confirmed_per_verification"] is None
    assert r["hit_confirmed_sum_per_position"] is None
    assert r["hit_match_mean"] is None
    assert all(v is None for v in r["hybrid_expected_confirmed"].values())


def test_measured_zero_hit_rate_stays_zero():
    acc = Accumulator(7)
    for _ in range(4):
        acc.add(False, 0)
    r = acc.result([0.5], 3)
    assert r["hit_rate"] == 0.0 and r["expected_confirmed_per_verification"] == 1.0
    assert r["hit_match_mean"] is None  # no hit: the hit-conditional mean is not measured


def test_unrequested_width_16_is_not_measured_not_zero():
    """--lengths 3 7: the width-16 sensitivity columns must say "not measured", never 0.0."""
    from ngramsim.models import Conversation, Message, ParseSummary
    from ngramsim.report import render_report
    from ngramsim.simulator import SimulationConfig, simulate
    from tests.test_integration import WordEncoder

    conversation = Conversation(
        source="fixture.json", name="repeat", created_at="2026-01-01", workload="prose",
        messages=[Message("user", "a b c d a b c"), Message("assistant", "d a b c d", "fixture-model")],
    )
    config = SimulationConfig(
        n_mins=[2], n_max=4, ngram_ns=[2], draft_lengths=[3, 7], alphas=[0.75],
        strategies=["suffix-match"], sensitivity_deltas=[0.5],
    )
    result = simulate([conversation], ParseSummary(files_seen=1, conversations_loaded=1), WordEncoder(), config,
                      tokenizer_path="fixture-tokenizer.json", conversations_path="fixtures")
    markdown = render_report(result)
    section = markdown.split("### Sensitivity (overall)", 1)[1].split("###", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| 0.75 |")]
    assert rows and all(line.endswith("| not measured | not measured |") for line in rows)
    assert " 0.0 |" not in section


def test_missing_counts_are_not_recorded_not_zero():
    from ngramsim.report import _rec

    assert _rec({}, "files_seen") == "not recorded"
    assert _rec({"files_seen": 0}, "files_seen") == "0"
    assert _rec({"all": 12345}, "all", ",") == "12,345"


def test_corpus_without_positions_is_listed_as_not_measured():
    """A corpus with recorded conversations but zero generated positions must stay in the report."""
    from ngramsim.models import Conversation, Message, ParseSummary
    from ngramsim.report import render_report
    from ngramsim.simulator import SimulationConfig, simulate
    from tests.test_integration import WordEncoder

    conversation = Conversation(
        source="fixture.json", name="repeat", created_at="2026-01-01", workload="prose",
        messages=[Message("user", "a b c d a b c"), Message("assistant", "d a b c d", "fixture-model")],
    )
    config = SimulationConfig(n_mins=[2], n_max=4, ngram_ns=[2], draft_lengths=[3], alphas=[0.75],
                              strategies=["suffix-match"], sensitivity_deltas=[0.5])
    result = simulate([conversation], ParseSummary(files_seen=1, conversations_loaded=1), WordEncoder(), config,
                      tokenizer_path="fixture-tokenizer.json", conversations_path="fixtures")
    result.setdefault("parser_accounting", {})["empty-corpus"] = {
        "conversations_loaded": 3, "files_seen": 4, "skipped_corrupt": 2}
    result["token_counts"].setdefault("by_corpus", {})["empty-corpus"] = {"all": 1234, "assistant": 0}
    assert all(row["corpus"] != "empty-corpus" for row in result["statistics"])
    markdown = render_report(result)
    assert "| empty-corpus | 3 | 1,234 | 0 | not recorded |" in markdown
    section = markdown.split("## Corpus: empty-corpus", 1)[1]
    assert "not measured for this corpus" in section
    assert "seen=4" in section and "corrupt=2" in section
