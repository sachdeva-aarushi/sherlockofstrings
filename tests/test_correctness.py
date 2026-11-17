"""Comprehensive cross-validation tests for all pattern matching algorithms.

Validates Naïve, KMP, and Boyer-Moore against Python's built-in reference behavior
across edge cases, repetitive structures, dense overlaps, adversarial inputs,
and extensive randomized fuzzing.
"""

import random
from typing import List
import pytest

from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from generators.input_generator import (
    Alphabet,
    generate_best_case_boyer_moore,
    generate_binary_sequence,
    generate_dna_sequence,
    generate_embedded_matches,
    generate_overlapping_case,
    generate_periodic_string,
    generate_random_string,
    generate_repetitive_string,
    generate_worst_case_naive,
)


def reference_find_all(text: str, pattern: str) -> List[int]:
    """Reference ground truth implementation using Python built-in slice equality."""
    if not pattern:
        return list(range(len(text) + 1))
    n = len(text)
    m = len(pattern)
    if m > n:
        return []
    return [i for i in range(n - m + 1) if text[i : i + m] == pattern]


@pytest.fixture(scope="module")
def algorithms():
    return {
        "Naive": NaiveStringMatcher(),
        "KMP": KMPMatcher(),
        "Boyer-Moore": BoyerMooreMatcher(),
    }


def assert_all_match_reference(algorithms, text: str, pattern: str) -> None:
    """Helper verifying that all algorithms produce results identical to reference."""
    ref_matches = reference_find_all(text, pattern)

    for name, matcher in algorithms.items():
        result = matcher.search(text, pattern)
        assert result.matches == ref_matches, (
            f"Algorithm {name} mismatch on text={text!r}, pattern={pattern!r}. "
            f"Expected {ref_matches}, got {result.matches}"
        )


# =====================================================================
# 1. Edge Case Suites
# =====================================================================

@pytest.mark.parametrize(
    "text,pattern",
    [
        ("", ""),
        ("", "a"),
        ("", "abc"),
        ("a", ""),
        ("abc", ""),
        ("a", "a"),
        ("a", "b"),
        ("ab", "ab"),
        ("ab", "ba"),
        ("short", "longer_pattern_than_text"),
        ("exact", "exact"),
        ("mississippi", "iss"),
        ("mississippi", "p"),
        ("mississippi", "i"),
        ("mississippi", "not_present"),
    ],
)
def test_edge_cases(algorithms, text: str, pattern: str) -> None:
    assert_all_match_reference(algorithms, text, pattern)


# =====================================================================
# 2. Overlapping and Repetitive Cases
# =====================================================================

@pytest.mark.parametrize(
    "text_len,pat_len",
    [
        (10, 2),
        (20, 5),
        (50, 10),
        (100, 1),
        (100, 25),
        (200, 50),
    ],
)
def test_dense_overlapping_homogeneous(algorithms, text_len: int, pat_len: int) -> None:
    text, pattern = generate_overlapping_case(text_len, pat_len, char="A")
    assert_all_match_reference(algorithms, text, pattern)


@pytest.mark.parametrize(
    "period,repeats,pat",
    [
        ("AB", 30, "AB"),
        ("AB", 30, "ABA"),
        ("AB", 30, "ABAB"),
        ("ABC", 20, "BCA"),
        ("ABC", 20, "BCAB"),
        ("AABA", 25, "AABAA"),
    ],
)
def test_periodic_inputs(algorithms, period: str, repeats: int, pat: str) -> None:
    text = generate_periodic_string(period, len(period) * repeats)
    assert_all_match_reference(algorithms, text, pat)


# =====================================================================
# 3. Adversarial / Boundary Inputs
# =====================================================================

def test_worst_case_naive_adversarial(algorithms) -> None:
    text, pattern = generate_worst_case_naive(text_length=300, pattern_length=15)
    assert_all_match_reference(algorithms, text, pattern)


def test_best_case_boyer_moore_adversarial(algorithms) -> None:
    text, pattern = generate_best_case_boyer_moore(text_length=300, pattern_length=15)
    assert_all_match_reference(algorithms, text, pattern)


def test_embedded_exact_locations(algorithms) -> None:
    text, pattern, inserted = generate_embedded_matches(
        text_length=200, pattern="FINDME", num_matches=4, seed=99
    )
    assert_all_match_reference(algorithms, text, pattern)


# =====================================================================
# 4. Randomized Corpus Fuzz Testing
# =====================================================================

def test_randomized_binary_fuzz(algorithms) -> None:
    rng = random.Random(101)
    for _ in range(30):
        n = rng.randint(20, 200)
        m = rng.randint(1, min(20, n))
        text = generate_binary_sequence(n, seed=rng.randint(0, 10000))
        pattern = generate_binary_sequence(m, seed=rng.randint(0, 10000))
        assert_all_match_reference(algorithms, text, pattern)


def test_randomized_dna_fuzz(algorithms) -> None:
    rng = random.Random(202)
    for _ in range(30):
        n = rng.randint(30, 300)
        m = rng.randint(1, min(30, n))
        text = generate_dna_sequence(n, seed=rng.randint(0, 10000))
        pattern = generate_dna_sequence(m, seed=rng.randint(0, 10000))
        assert_all_match_reference(algorithms, text, pattern)


def test_randomized_ascii_fuzz(algorithms) -> None:
    rng = random.Random(303)
    for _ in range(40):
        n = rng.randint(50, 400)
        m = rng.randint(1, min(40, n))
        text = generate_random_string(n, Alphabet.LOWERCASE, seed=rng.randint(0, 10000))
        # 50% chance pattern is extracted from text (guaranteed match), 50% independent
        if rng.random() < 0.5:
            start = rng.randint(0, n - m)
            pattern = text[start : start + m]
        else:
            pattern = generate_random_string(m, Alphabet.LOWERCASE, seed=rng.randint(0, 10000))
        assert_all_match_reference(algorithms, text, pattern)
