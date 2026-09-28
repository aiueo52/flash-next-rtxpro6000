from ngramsim.strategies import CorpusIndex, matching_prefix


def test_suffix_match_uses_longest_suffix_and_previous_occurrence():
    index = CorpusIndex(max_n=4)
    index.extend([1, 2, 3, 4, 1, 2, 3])

    assert index.suffix_match(n_min=2, n_max=4, length=3) == [4, 1, 2]
    assert matching_prefix([4, 1, 2], [4, 1, 9], 3) == 2


def test_suffix_match_repeated_sequence_can_match_full_draft():
    index = CorpusIndex(max_n=3)
    index.extend([7, 8, 9, 4, 5, 6, 7, 8, 9])

    assert index.suffix_match(n_min=2, n_max=3, length=4) == [4, 5, 6, 7]


def test_unique_sequence_has_no_usable_suffix_hit():
    index = CorpusIndex(max_n=4)
    index.extend([10, 11, 12, 13, 14])

    assert index.suffix_match(n_min=2, n_max=4, length=3) == []
    assert index.ngram_mod(n=2, length=3) == []


def test_ngram_mod_chains_latest_fixed_n_transitions():
    index = CorpusIndex(max_n=3)
    index.extend([1, 2, 3, 1, 2])

    assert index.ngram_mod(n=2, length=4) == [3, 1, 2, 3]


def test_segment_boundary_does_not_become_a_transition_or_source_continuation():
    index = CorpusIndex(max_n=3)
    index.extend([1, 2, 3, 1, 2])
    index.start_segment()

    assert (1, 2) not in index.transitions[2] or index.transitions[2][(1, 2)] == 3
    index.extend([9, 1, 2])
    assert index.suffix_match(n_min=2, n_max=3, length=3) == [3, 1, 2]


def test_search_window_compacts_and_excludes_old_transitions():
    index = CorpusIndex(max_n=2, window_tokens=5)
    index.extend([1, 2, 3, 4, 5, 6, 7, 1, 2])

    assert len(index.tokens) <= 7
    assert index.ngram_mod(n=2, length=2) == []
