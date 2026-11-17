"""MatchLab Algorithms Package.

Exposes standard pattern matching algorithms implementing the unified BaseMatcher interface.
"""

from algorithms.base import BaseMatcher, MatchResult
from algorithms.naive import NaiveStringMatcher, naive_search
from algorithms.kmp import KMPMatcher, kmp_search
from algorithms.boyer_moore import BoyerMooreMatcher, boyer_moore_search

__all__ = [
    "BaseMatcher",
    "MatchResult",
    "NaiveStringMatcher",
    "naive_search",
    "KMPMatcher",
    "kmp_search",
    "BoyerMooreMatcher",
    "boyer_moore_search",
]
