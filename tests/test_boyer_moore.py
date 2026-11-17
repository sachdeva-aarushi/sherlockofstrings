"""Tests for Boyer-Moore String Matching Algorithm."""

import pytest
from algorithms.boyer_moore import BoyerMooreMatcher, boyer_moore_search


@pytest.fixture
def matcher() -> BoyerMooreMatcher:
    return BoyerMooreMatcher()


def test_bm_bad_character_table(matcher: BoyerMooreMatcher) -> None:
    pattern = "ANPANMAN"
    table = matcher.compute_bad_character_table(pattern)

    # Rightmost occurrences: A: 6, N: 7, P: 2, M: 5
    assert table["A"] == 6
    assert table["N"] == 7
    assert table["P"] == 2
    assert table["M"] == 5
    assert "Z" not in table


def test_bm_z_array(matcher: BoyerMooreMatcher) -> None:
    # Z-array for "aabac"
    # z[0] = 5, z[1] = 1 ("abac" vs "aabac"), z[2] = 0, z[3] = 1, z[4] = 0
    z, comps = matcher.compute_z_array("aabac")
    assert z == [5, 1, 0, 1, 0]
    assert comps > 0


def test_bm_good_suffix_table_progress(matcher: BoyerMooreMatcher) -> None:
    pattern = "ANPANMAN"
    gs_table, full_match_shift, prep_comps = matcher.compute_good_suffix_table(pattern)

    assert len(gs_table) == len(pattern)
    assert all(shift >= 1 for shift in gs_table)
    assert full_match_shift >= 1
    assert prep_comps > 0


def test_bm_basic_match(matcher: BoyerMooreMatcher) -> None:
    text = "HERE IS A SIMPLE EXAMPLE"
    pattern = "EXAMPLE"
    result = matcher.search(text, pattern)

    assert result.algorithm_name == "Boyer-Moore"
    assert result.matches == [17]
    assert result.preprocessing_comparisons > 0
    assert result.search_comparisons > 0
    assert result.total_comparisons == result.preprocessing_comparisons + result.search_comparisons


def test_bm_multiple_matches(matcher: BoyerMooreMatcher) -> None:
    text = "the cat in the hat sat on the mat"
    pattern = "the"
    result = matcher.search(text, pattern)

    assert result.matches == [0, 11, 26]


def test_bm_overlapping_matches(matcher: BoyerMooreMatcher) -> None:
    text = "AAAA"
    pattern = "AA"
    result = matcher.search(text, pattern)

    assert result.matches == [0, 1, 2]


def test_bm_sublinear_best_case(matcher: BoyerMooreMatcher) -> None:
    # Best-case: text does not contain any character of pattern
    # Pattern of length 5 slides by 5 after a single comparison at each step
    text = "A" * 100
    pattern = "B" * 5
    result = matcher.search(text, pattern)

    assert result.matches == []
    # Search comparisons should be strictly sublinear: ~100 / 5 = 20 comparisons
    assert result.search_comparisons <= 25
    assert result.search_comparisons < len(text)


def test_bm_empty_pattern(matcher: BoyerMooreMatcher) -> None:
    text = "example"
    result = matcher.search(text, "")

    assert result.matches == [0, 1, 2, 3, 4, 5, 6, 7]
    assert result.total_comparisons == 0


def test_bm_empty_text_and_empty_pattern(matcher: BoyerMooreMatcher) -> None:
    result = matcher.search("", "")
    assert result.matches == [0]


def test_bm_pattern_longer_than_text(matcher: BoyerMooreMatcher) -> None:
    text = "dog"
    pattern = "longerdog"
    result = matcher.search(text, pattern)

    assert result.matches == []
    assert result.preprocessing_comparisons > 0
    assert result.search_comparisons == 0


def test_bm_single_character(matcher: BoyerMooreMatcher) -> None:
    text = "mississippi"
    pattern = "s"
    result = matcher.search(text, pattern)

    assert result.matches == [2, 3, 5, 6]


def test_bm_functional_interface() -> None:
    result = boyer_moore_search("abracadabra", "cad")
    assert result.matches == [4]


def test_bm_type_errors(matcher: BoyerMooreMatcher) -> None:
    with pytest.raises(TypeError):
        matcher.search(42, "pattern")  # type: ignore

    with pytest.raises(TypeError):
        matcher.search("text", 3.14)  # type: ignore
