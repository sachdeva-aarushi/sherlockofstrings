"""Tests for dataset and input generators."""

import pytest
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


def test_random_string_determinism() -> None:
    s1 = generate_random_string(50, Alphabet.LOWERCASE, seed=123)
    s2 = generate_random_string(50, Alphabet.LOWERCASE, seed=123)
    s3 = generate_random_string(50, Alphabet.LOWERCASE, seed=456)

    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 50
    assert all(c in Alphabet.LOWERCASE for c in s1)


def test_dna_and_binary_generators() -> None:
    dna = generate_dna_sequence(40, seed=1)
    assert len(dna) == 40
    assert all(c in "ACGT" for c in dna)

    binary = generate_binary_sequence(40, seed=1)
    assert len(binary) == 40
    assert all(c in "01" for c in binary)


def test_repetitive_and_periodic() -> None:
    rep = generate_repetitive_string("X", 15)
    assert rep == "XXXXXXXXXXXXXXX"

    periodic = generate_periodic_string("ABC", 10)
    assert periodic == "ABCABCABCA"


def test_worst_case_naive_structure() -> None:
    text, pat = generate_worst_case_naive(text_length=100, pattern_length=10)
    assert len(text) == 100
    assert len(pat) == 10
    assert pat == ("A" * 9) + "B"
    assert text == "A" * 100


def test_best_case_boyer_moore_structure() -> None:
    text, pat = generate_best_case_boyer_moore(text_length=100, pattern_length=10)
    assert len(text) == 100
    assert len(pat) == 10
    assert text == "A" * 100
    assert pat == "B" * 10


def test_overlapping_case_structure() -> None:
    text, pat = generate_overlapping_case(text_length=50, pattern_length=5, char="Z")
    assert text == "Z" * 50
    assert pat == "Z" * 5


def test_embedded_matches() -> None:
    text, pat, indices = generate_embedded_matches(
        text_length=120, pattern="TARGET", num_matches=3, seed=42
    )
    assert len(indices) == 3
    for idx in indices:
        assert text[idx : idx + len(pat)] == "TARGET"


def test_generator_validation_errors() -> None:
    with pytest.raises(ValueError):
        generate_random_string(-5)

    with pytest.raises(ValueError):
        generate_random_string(10, alphabet="")

    with pytest.raises(ValueError):
        generate_repetitive_string("multiple", 10)

    with pytest.raises(ValueError):
        generate_worst_case_naive(10, 20)  # Pattern > text
