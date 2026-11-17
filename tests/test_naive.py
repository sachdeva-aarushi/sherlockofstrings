"""Tests for Naïve (Brute-Force) String Matching Algorithm."""

import pytest
from algorithms.naive import NaiveStringMatcher, naive_search


@pytest.fixture
def matcher() -> NaiveStringMatcher:
    return NaiveStringMatcher()


def test_naive_basic_match(matcher: NaiveStringMatcher) -> None:
    text = "hello world"
    pattern = "world"
    result = matcher.search(text, pattern)

    assert result.algorithm_name == "Naive"
    assert result.matches == [6]
    assert result.preprocessing_comparisons == 0
    assert result.search_comparisons > 0
    assert result.total_comparisons == result.search_comparisons


def test_naive_multiple_matches(matcher: NaiveStringMatcher) -> None:
    text = "abc123abc456abc"
    pattern = "abc"
    result = matcher.search(text, pattern)

    assert result.matches == [0, 6, 12]


def test_naive_overlapping_matches(matcher: NaiveStringMatcher) -> None:
    text = "AAAA"
    pattern = "AA"
    result = matcher.search(text, pattern)

    # In "AAAA", "AA" appears at indices 0, 1, 2
    assert result.matches == [0, 1, 2]


def test_naive_no_match(matcher: NaiveStringMatcher) -> None:
    text = "abcdef"
    pattern = "xyz"
    result = matcher.search(text, pattern)

    assert result.matches == []
    assert result.search_comparisons > 0


def test_naive_empty_pattern(matcher: NaiveStringMatcher) -> None:
    text = "abc"
    result = matcher.search(text, "")

    assert result.matches == [0, 1, 2, 3]
    assert result.preprocessing_comparisons == 0
    assert result.search_comparisons == 0
    assert result.total_comparisons == 0


def test_naive_empty_text_and_empty_pattern(matcher: NaiveStringMatcher) -> None:
    result = matcher.search("", "")

    assert result.matches == [0]
    assert result.total_comparisons == 0


def test_naive_pattern_longer_than_text(matcher: NaiveStringMatcher) -> None:
    text = "cat"
    pattern = "caterpillar"
    result = matcher.search(text, pattern)

    assert result.matches == []
    assert result.search_comparisons == 0


def test_naive_empty_text_non_empty_pattern(matcher: NaiveStringMatcher) -> None:
    result = matcher.search("", "a")

    assert result.matches == []
    assert result.search_comparisons == 0


def test_naive_single_character_matches(matcher: NaiveStringMatcher) -> None:
    text = "banana"
    pattern = "a"
    result = matcher.search(text, pattern)

    assert result.matches == [1, 3, 5]
    assert result.search_comparisons == 6


def test_naive_identical_text_and_pattern(matcher: NaiveStringMatcher) -> None:
    text = "exactmatch"
    pattern = "exactmatch"
    result = matcher.search(text, pattern)

    assert result.matches == [0]
    assert result.search_comparisons == len(text)


def test_naive_worst_case_comparisons(matcher: NaiveStringMatcher) -> None:
    # Text: "AAAA", Pattern: "AAAB"
    # Shift 0: compares A, A, A, B against A, A, A, A -> 4 comparisons
    text = "AAAA"
    pattern = "AAAB"
    result = matcher.search(text, pattern)

    assert result.matches == []
    assert result.search_comparisons == 4


def test_naive_functional_interface() -> None:
    result = naive_search("banana", "an")
    assert result.matches == [1, 3]


def test_naive_type_errors(matcher: NaiveStringMatcher) -> None:
    with pytest.raises(TypeError):
        matcher.search(123, "pattern")  # type: ignore

    with pytest.raises(TypeError):
        matcher.search("text", None)  # type: ignore
