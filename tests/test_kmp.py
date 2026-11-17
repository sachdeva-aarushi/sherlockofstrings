"""Tests for Knuth-Morris-Pratt (KMP) String Matching Algorithm."""

import pytest
from algorithms.kmp import KMPMatcher, kmp_search


@pytest.fixture
def matcher() -> KMPMatcher:
    return KMPMatcher()


def test_kmp_lps_table_examples(matcher: KMPMatcher) -> None:
    # Basic repetitive
    lps, comps = matcher.compute_lps_table("AAAA")
    assert lps == [0, 1, 2, 3]

    # No repeating proper prefix-suffix
    lps, comps = matcher.compute_lps_table("ABCDE")
    assert lps == [0, 0, 0, 0, 0]

    # Classic textbook example
    lps, comps = matcher.compute_lps_table("AABAACAABAA")
    assert lps == [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]

    # Empty pattern
    lps, comps = matcher.compute_lps_table("")
    assert lps == []
    assert comps == 0

    # Single character
    lps, comps = matcher.compute_lps_table("A")
    assert lps == [0]
    assert comps == 0


def test_kmp_basic_match(matcher: KMPMatcher) -> None:
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    result = matcher.search(text, pattern)

    assert result.algorithm_name == "KMP"
    assert result.matches == [10]
    assert result.preprocessing_comparisons > 0
    assert result.search_comparisons > 0
    assert result.total_comparisons == result.preprocessing_comparisons + result.search_comparisons


def test_kmp_multiple_matches(matcher: KMPMatcher) -> None:
    text = "needle in a needle haystack with another needle"
    pattern = "needle"
    result = matcher.search(text, pattern)

    assert result.matches == [0, 12, 41]


def test_kmp_overlapping_matches(matcher: KMPMatcher) -> None:
    text = "AAAA"
    pattern = "AA"
    result = matcher.search(text, pattern)

    assert result.matches == [0, 1, 2]


def test_kmp_no_match(matcher: KMPMatcher) -> None:
    text = "ABCDEFGHIJ"
    pattern = "XYZ"
    result = matcher.search(text, pattern)

    assert result.matches == []


def test_kmp_empty_pattern(matcher: KMPMatcher) -> None:
    text = "hello"
    result = matcher.search(text, "")

    assert result.matches == [0, 1, 2, 3, 4, 5]
    assert result.total_comparisons == 0


def test_kmp_empty_text_and_empty_pattern(matcher: KMPMatcher) -> None:
    result = matcher.search("", "")
    assert result.matches == [0]


def test_kmp_pattern_longer_than_text(matcher: KMPMatcher) -> None:
    text = "short"
    pattern = "very long pattern exceeding text"
    result = matcher.search(text, pattern)

    assert result.matches == []
    # Preprocessing is still performed for the pattern
    assert result.preprocessing_comparisons > 0
    assert result.search_comparisons == 0


def test_kmp_empty_text_non_empty_pattern(matcher: KMPMatcher) -> None:
    result = matcher.search("", "abc")
    assert result.matches == []
    assert result.preprocessing_comparisons > 0
    assert result.search_comparisons == 0


def test_kmp_linear_search_bound(matcher: KMPMatcher) -> None:
    # For any text and pattern, KMP search comparisons <= 2 * n
    text = "A" * 100
    pattern = ("A" * 9) + "B"
    result = matcher.search(text, pattern)

    assert result.search_comparisons <= 2 * len(text)


def test_kmp_functional_interface() -> None:
    result = kmp_search("coconuts for coco", "coco")
    assert result.matches == [0, 13]


def test_kmp_type_errors(matcher: KMPMatcher) -> None:
    with pytest.raises(TypeError):
        matcher.search(["invalid"], "pattern")  # type: ignore

    with pytest.raises(TypeError):
        matcher.search("text", {"invalid": 1})  # type: ignore
