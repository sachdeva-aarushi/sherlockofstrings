# Knuth-Morris-Pratt (KMP) string matching algorithm.

import time
from typing import List, Tuple

from algorithms.base import BaseMatcher, MatchResult


class KMPMatcher(BaseMatcher):
    @property
    def name(self) -> str:
        return "KMP"

    def compute_lps_table(self, pattern: str) -> Tuple[List[int], int]:
        m = len(pattern)
        if m == 0:
            return [], 0

        lps = [0] * m
        length = 0
        i = 1
        preprocessing_comparisons = 0

        while i < m:
            preprocessing_comparisons += 1
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

        return lps, preprocessing_comparisons

    def preprocess(self, pattern: str) -> Tuple[List[int], int]:
        return self.compute_lps_table(pattern)

    def search_with_preprocessed(
        self, text: str, pattern: str, preprocessed: Tuple[List[int], int]
    ) -> MatchResult:
        self._validate_inputs(text, pattern)
        start_time = time.perf_counter()

        n = len(text)
        m = len(pattern)
        lps, prep_comparisons = preprocessed

        if m == 0:
            return self._handle_empty_pattern(text, start_time)

        if m > n:
            return self._handle_pattern_longer_than_text(
                text, pattern, start_time, prep_comparisons
            )

        matches: List[int] = []
        search_comparisons = 0
        i = 0
        j = 0

        while i < n:
            search_comparisons += 1
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == m:
                    matches.append(i - j)
                    j = lps[j - 1]
            elif j != 0:
                j = lps[j - 1]
            else:
                i += 1

        elapsed = time.perf_counter() - start_time

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=matches,
            preprocessing_comparisons=prep_comparisons,
            search_comparisons=search_comparisons,
            total_comparisons=prep_comparisons + search_comparisons,
            execution_time_seconds=elapsed,
            preprocessing_time_seconds=0.0,
            search_time_seconds=elapsed,
            metadata={
                "lps_table": lps,
            },
        )

    def search(self, text: str, pattern: str) -> MatchResult:
        self._validate_inputs(text, pattern)
        t_prep_start = time.perf_counter()
        preprocessed = self.preprocess(pattern)
        prep_elapsed = time.perf_counter() - t_prep_start

        t_search_start = time.perf_counter()
        result = self.search_with_preprocessed(text, pattern, preprocessed)
        search_elapsed = time.perf_counter() - t_search_start

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=result.matches,
            preprocessing_comparisons=result.preprocessing_comparisons,
            search_comparisons=result.search_comparisons,
            total_comparisons=result.total_comparisons,
            execution_time_seconds=prep_elapsed + search_elapsed,
            preprocessing_time_seconds=prep_elapsed,
            search_time_seconds=search_elapsed,
            metadata=result.metadata,
        )


def kmp_search(text: str, pattern: str) -> MatchResult:
    return KMPMatcher().search(text, pattern)
