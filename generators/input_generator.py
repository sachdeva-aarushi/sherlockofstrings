# Synthetic input generators for string matching experiments across 11 distributions.

from dataclasses import dataclass
from enum import Enum
import random
import string
from typing import List, Optional, Tuple


class Alphabet:
    BINARY = "01"
    DNA = "ACGT"
    LOWERCASE = string.ascii_lowercase
    UPPERCASE = string.ascii_uppercase
    ALPHANUMERIC = string.ascii_letters + string.digits
    PRINTABLE = string.printable.strip()


class InputDistribution(str, Enum):
    RANDOM = "random"
    RANDOM_WITH_MATCH = "random_with_match"
    REPETITIVE = "repetitive"
    PERIODIC = "periodic"
    BEST_CASE = "best_case"
    ADVERSARIAL_PREFIX = "adversarial_prefix"
    ABSENT = "absent"
    MATCH_AT_START = "match_at_start"
    MATCH_AT_MIDDLE = "match_at_middle"
    MATCH_AT_END = "match_at_end"
    OVERLAPPING = "overlapping"


def get_alphabet_by_size(size: int) -> str:
    if size <= 0:
        raise ValueError(f"Alphabet size must be positive, got {size}")

    if size == 2:
        return Alphabet.BINARY
    if size == 4:
        return Alphabet.DNA
    if size <= 26:
        return string.ascii_lowercase[:size]
    if size <= 52:
        return (string.ascii_lowercase + string.ascii_uppercase)[:size]
    if size <= 62:
        return (string.ascii_letters + string.digits)[:size]

    chars = [chr(c) for c in range(33, 127) if chr(c) not in string.whitespace]
    if size <= len(chars):
        return "".join(chars[:size])

    for code in range(161, 255):
        if len(chars) >= size:
            break
        chars.append(chr(code))

    if len(chars) < size:
        raise ValueError(f"Requested alphabet size {size} exceeds supported pool ({len(chars)})")

    return "".join(chars[:size])


@dataclass(frozen=True)
class GeneratedInput:
    text: str
    pattern: str
    distribution: InputDistribution
    text_length: int
    pattern_length: int
    alphabet_size: int
    seed: Optional[int]


def generate_random_string(
    length: int,
    alphabet: str = Alphabet.LOWERCASE,
    seed: Optional[int] = None,
) -> str:
    if length < 0:
        raise ValueError("length must be non-negative")
    if not alphabet:
        raise ValueError("alphabet cannot be empty")

    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_dna_sequence(length: int, seed: Optional[int] = None) -> str:
    return generate_random_string(length, Alphabet.DNA, seed)


def generate_binary_sequence(length: int, seed: Optional[int] = None) -> str:
    return generate_random_string(length, Alphabet.BINARY, seed)


def generate_repetitive_string(char: str, length: int) -> str:
    if len(char) != 1:
        raise ValueError("char must be a single character")
    if length < 0:
        raise ValueError("length must be non-negative")
    return char * length


def generate_periodic_string(period: str, total_length: int) -> str:
    if not period:
        raise ValueError("period cannot be empty")
    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    repeats = (total_length // len(period)) + 1
    return (period * repeats)[:total_length]


def generate_worst_case_naive(
    text_length: int, pattern_length: int
) -> Tuple[str, str]:
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    pattern = ("A" * (pattern_length - 1)) + "B"
    text = "A" * text_length
    return text, pattern


def generate_best_case_boyer_moore(
    text_length: int, pattern_length: int
) -> Tuple[str, str]:
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    pattern = "B" * pattern_length
    text = "A" * text_length
    return text, pattern


def generate_overlapping_case(
    text_length: int, pattern_length: int, char: str = "A"
) -> Tuple[str, str]:
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    text = char * text_length
    pattern = char * pattern_length
    return text, pattern


def generate_embedded_matches(
    text_length: int,
    pattern: str,
    num_matches: int,
    seed: Optional[int] = None,
    alphabet: str = Alphabet.LOWERCASE,
) -> Tuple[str, str, List[int]]:
    m = len(pattern)
    if m == 0:
        raise ValueError("pattern cannot be empty")
    if num_matches * m > text_length:
        raise ValueError("num_matches * pattern_length exceeds text_length")

    rng = random.Random(seed)
    available_slots = text_length - (num_matches * m)
    gaps = [rng.randint(0, available_slots) for _ in range(num_matches + 1)]
    scale = available_slots / max(1, sum(gaps))
    gap_lengths = [int(g * scale) for g in gaps]
    diff = available_slots - sum(gap_lengths)
    gap_lengths[0] += diff

    text_parts = []
    inserted_indices = []
    current_idx = 0

    for i in range(num_matches):
        bg = "".join(rng.choice(alphabet) for _ in range(gap_lengths[i]))
        text_parts.append(bg)
        current_idx += len(bg)
        inserted_indices.append(current_idx)
        text_parts.append(pattern)
        current_idx += m

    bg_tail = "".join(rng.choice(alphabet) for _ in range(gap_lengths[-1]))
    text_parts.append(bg_tail)
    text = "".join(text_parts)

    return text, pattern, inserted_indices


def generate_benchmark_input(
    distribution: InputDistribution,
    text_length: int,
    pattern_length: int,
    alphabet_size: int = 26,
    seed: Optional[int] = None,
) -> GeneratedInput:
    if text_length < 0 or pattern_length < 0:
        raise ValueError("Lengths must be non-negative")

    rng = random.Random(seed)
    alphabet = get_alphabet_by_size(alphabet_size)

    if distribution == InputDistribution.RANDOM:
        text = "".join(rng.choice(alphabet) for _ in range(text_length))
        pat_seed = rng.randint(0, 1_000_000)
        rng_pat = random.Random(pat_seed)
        pattern = "".join(rng_pat.choice(alphabet) for _ in range(pattern_length))

    elif distribution == InputDistribution.RANDOM_WITH_MATCH:
        text = "".join(rng.choice(alphabet) for _ in range(text_length))
        if pattern_length == 0:
            pattern = ""
        elif pattern_length <= text_length:
            start = rng.randint(0, text_length - pattern_length)
            pattern = text[start : start + pattern_length]
        else:
            pattern = "".join(rng.choice(alphabet) for _ in range(pattern_length))

    elif distribution == InputDistribution.REPETITIVE:
        char = alphabet[0]
        text = char * text_length
        pattern = char * pattern_length

    elif distribution == InputDistribution.PERIODIC:
        if alphabet_size >= 2:
            period = alphabet[0] + alphabet[1]
        else:
            period = alphabet[0]
        text = generate_periodic_string(period, text_length)
        pattern = generate_periodic_string(period, pattern_length)

    elif distribution == InputDistribution.BEST_CASE:
        if alphabet_size >= 2:
            text_char = alphabet[0]
            pat_char = alphabet[1]
        else:
            text_char = "A"
            pat_char = "B"
        text = text_char * text_length
        pattern = pat_char * pattern_length

    elif distribution == InputDistribution.ADVERSARIAL_PREFIX:
        if alphabet_size >= 2:
            a_char = alphabet[0]
            b_char = alphabet[1]
        else:
            a_char = "A"
            b_char = "B"
        text = a_char * text_length
        if pattern_length <= 1:
            pattern = b_char * pattern_length
        else:
            pattern = (a_char * (pattern_length - 1)) + b_char

    elif distribution == InputDistribution.ABSENT:
        if alphabet_size >= 2:
            half = max(1, alphabet_size // 2)
            text_alpha = alphabet[:half]
            pat_alpha = alphabet[half:]
            text = "".join(rng.choice(text_alpha) for _ in range(text_length))
            pattern = "".join(rng.choice(pat_alpha) for _ in range(pattern_length))
        else:
            text = "A" * text_length
            pattern = "B" * pattern_length

    elif distribution == InputDistribution.MATCH_AT_START:
        if alphabet_size >= 2:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            tail = "".join(rng.choice(text_alpha) for _ in range(max(0, text_length - pattern_length)))
            text = (pattern + tail)[:text_length]
        else:
            pattern = "A" * pattern_length
            text = "A" * text_length

    elif distribution == InputDistribution.MATCH_AT_MIDDLE:
        if alphabet_size >= 2 and text_length >= pattern_length:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            mid_start = (text_length - pattern_length) // 2
            prefix = "".join(rng.choice(text_alpha) for _ in range(mid_start))
            suffix_len = text_length - mid_start - pattern_length
            suffix = "".join(rng.choice(text_alpha) for _ in range(max(0, suffix_len)))
            text = prefix + pattern + suffix
        else:
            text = "A" * text_length
            pattern = "A" * pattern_length

    elif distribution == InputDistribution.MATCH_AT_END:
        if alphabet_size >= 2 and text_length >= pattern_length:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            prefix_len = text_length - pattern_length
            prefix = "".join(rng.choice(text_alpha) for _ in range(prefix_len))
            text = prefix + pattern
        else:
            text = "A" * text_length
            pattern = "A" * pattern_length

    elif distribution == InputDistribution.OVERLAPPING:
        char = alphabet[0]
        text = char * text_length
        pattern = char * pattern_length

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return GeneratedInput(
        text=text,
        pattern=pattern,
        distribution=distribution,
        text_length=len(text),
        pattern_length=len(pattern),
        alphabet_size=alphabet_size,
        seed=seed,
    )
