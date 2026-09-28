import json

from ngramsim.models import Conversation, Message, ParseSummary
from ngramsim.report import regenerate_report, render_report
from ngramsim.simulator import SimulationConfig, simulate


class WordEncoder:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text):
        result = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary) + 1
            result.append(self.vocabulary[word])
        return result


def test_small_conversation_to_json_and_markdown_report(tmp_path):
    conversation = Conversation(
        source="fixture.json",
        name="repeat",
        created_at="2026-01-01",
        workload="prose",
        messages=[
            Message("user", "a b c d a b c"),
            Message("assistant", "d a b c d", "fixture-model"),
        ],
    )
    config = SimulationConfig(
        n_mins=[2],
        n_max=4,
        ngram_ns=[2],
        draft_lengths=[3, 7, 15],
        alphas=[0.65, 0.75, 0.85],
        strategies=["suffix-match", "ngram-mod"],
        sensitivity_deltas=[0.5],
    )
    result = simulate(
        [conversation],
        ParseSummary(files_seen=1, conversations_loaded=1),
        WordEncoder(),
        config,
        tokenizer_path="fixture-tokenizer.json",
        conversations_path="fixtures",
    )

    assert result["token_counts"]["assistant"] == 5
    assert any(row["hits"] > 0 for row in result["statistics"])
    markdown = render_report(result)
    assert "Acceptance by workload" in markdown
    assert "Predicted throughput" in markdown
    assert "Sensitivity" in markdown

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "raw_stats.json").write_text(json.dumps(result), encoding="utf-8")
    report_path = regenerate_report(result_dir)
    assert report_path.is_file()
    assert "fixture-model" in report_path.read_text(encoding="utf-8")


def test_cross_conversation_mode_uses_only_prior_conversations():
    first = Conversation(
        source="first.json",
        name="first",
        created_at="1",
        workload="prose",
        messages=[Message("user", "a b c d a b"), Message("assistant", "c d", "model")],
    )
    second = Conversation(
        source="second.json",
        name="second",
        created_at="2",
        workload="prose",
        messages=[Message("user", "x a b"), Message("assistant", "c d", "model")],
    )
    config = SimulationConfig(
        n_mins=[2],
        n_max=3,
        ngram_ns=[2],
        draft_lengths=[3],
        alphas=[0.75],
        strategies=["suffix-match"],
        cross_conv=True,
    )

    result = simulate(
        [first, second],
        ParseSummary(files_seen=2, conversations_loaded=2),
        WordEncoder(),
        config,
        tokenizer_path="fixture",
        conversations_path="fixtures",
    )

    overall = next(row for row in result["statistics"] if row["scope"] == "overall")
    assert overall["hits"] > 0


def test_token_cap_counts_truncation_and_preserves_corpus_dimension():
    conversation = Conversation(
        source="codex.jsonl",
        name="codex",
        created_at="1",
        workload="code",
        corpus="codex-agent",
        messages=[
            Message("user", "one two three"),
            Message("assistant", "four five six", "gpt", True),
        ],
    )
    config = SimulationConfig(
        n_mins=[2],
        n_max=3,
        ngram_ns=[2],
        draft_lengths=[3],
        alphas=[0.75],
        strategies=["suffix-match"],
        max_tokens_per_conv=4,
        search_window_tokens=4,
    )

    result = simulate(
        [conversation],
        {"codex-agent": ParseSummary(files_seen=1, conversations_loaded=1)},
        WordEncoder(),
        config,
        tokenizer_path="fixture",
        conversations_path={"codex-agent": ["fixtures"]},
    )

    assert result["truncations"] == {"codex-agent": 1}
    assert result["token_counts"]["by_corpus"]["codex-agent"] == {
        "all": 4,
        "assistant": 1,
    }
    assert {row["corpus"] for row in result["statistics"]} == {"codex-agent"}


def test_report_can_include_local_and_global_lookup_rows():
    conversation = Conversation(
        source="codex.jsonl",
        name="codex",
        created_at="1",
        corpus="codex-agent",
        messages=[Message("user", "a b a b"), Message("assistant", "a b", "gpt", True)],
    )
    base = SimulationConfig(
        n_mins=[2],
        n_max=3,
        ngram_ns=[2],
        draft_lengths=[7, 15],
        alphas=[0.75],
        strategies=["suffix-match"],
    )
    summary = {"codex-agent": ParseSummary(files_seen=1, conversations_loaded=1)}
    local = simulate(
        [conversation], summary, WordEncoder(), base,
        tokenizer_path="fixture", conversations_path={"codex-agent": ["fixtures"]},
    )
    global_result = simulate(
        [conversation], summary, WordEncoder(),
        SimulationConfig(
            n_mins=[2], n_max=3, ngram_ns=[2], draft_lengths=[7, 15],
            alphas=[0.75], strategies=["suffix-match"], cross_conv=True,
        ),
        tokenizer_path="fixture", conversations_path={"codex-agent": ["fixtures"]},
    )
    local["statistics"].extend(global_result["statistics"])

    markdown = render_report(local)
    assert "Lookup mode: session-local" in markdown
    assert "Lookup mode: global-window" in markdown
