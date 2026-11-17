# Naive brute-force string matching algorithm.

import time
from typing import List

from algorithms.base import BaseMatcher, MatchResult


class NaiveStringMatcher(BaseMatcher):
    @property
    def name(self) -> str:
        return "Naive"

    def search(self, text: str, pattern: str) -> MatchResult:
        self._validate_inputs(text, pattern)
        start_time = time.perf_counter()

        n = len(text)
        m = len(pattern)

        if m == 0:
            return self._handle_empty_pattern(text, start_time)

        if m > n:
            return self._handle_pattern_longer_than_text(text, pattern, start_time, 0)

        matches: List[int] = []
        search_comparisons = 0

        for i in range(n - m + 1):
            match = True
            for j in range(m):
                search_comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                matches.append(i)

        elapsed = time.perf_counter() - start_time

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=matches,
            preprocessing_comparisons=0,
            search_comparisons=search_comparisons,
            total_comparisons=search_comparisons,
            execution_time_seconds=elapsed,
            preprocessing_time_seconds=0.0,
            search_time_seconds=elapsed,
            metadata={
                "shifts_evaluated": n - m + 1,
            },
        )


def naive_search(text: str, pattern: str) -> MatchResult:
    return NaiveStringMatcher().search(text, pattern)
