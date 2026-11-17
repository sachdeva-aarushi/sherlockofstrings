# Boyer-Moore string matching algorithm with Bad Character and Strong Good Suffix heuristics.

import time
from typing import Dict, List, Tuple

from algorithms.base import BaseMatcher, MatchResult


class BoyerMooreMatcher(BaseMatcher):
    @property
    def name(self) -> str:
        return "Boyer-Moore"

    def compute_bad_character_table(self, pattern: str) -> Dict[str, int]:
        table: Dict[str, int] = {}
        for index, char in enumerate(pattern):
            table[char] = index
        return table

    def compute_z_array(self, string: str) -> Tuple[List[int], int]:
        n = len(string)
        if n == 0:
            return [], 0

        z = [0] * n
        z[0] = n
        left = 0
        right = 0
        comparisons = 0

        for i in range(1, n):
            if i <= right:
                k = i - left
                remaining = right - i + 1
                if z[k] < remaining:
                    z[i] = z[k]
                    continue
                else:
                    match_len = remaining
            else:
                match_len = 0

            while (i + match_len < n) and (match_len < n):
                comparisons += 1
                if string[i + match_len] == string[match_len]:
                    match_len += 1
                else:
                    break

            z[i] = match_len
            if i + match_len - 1 > right:
                left = i
                right = i + match_len - 1

        return z, comparisons

    def compute_good_suffix_table(
        self, pattern: str
    ) -> Tuple[List[int], int, int]:
        m = len(pattern)
        if m == 0:
            return [], 1, 0

        reversed_pattern = pattern[::-1]
        z_array, prep_comparisons = self.compute_z_array(reversed_pattern)

        n_array = [0] * m
        for j in range(m - 1):
            n_array[j] = z_array[m - 1 - j]
        n_array[m - 1] = m

        max_prefix_len = [0] * (m + 1)
        longest_proper_border = 0

        for l in range(1, m):
            if n_array[l - 1] == l:
                max_prefix_len[l] = l
                longest_proper_border = l

        for length in range(1, m + 1):
            if max_prefix_len[length] == 0:
                max_prefix_len[length] = max_prefix_len[length - 1]

        good_suffix = [1] * m
        for j in range(m - 1):
            k = m - 1 - j
            prefix_len = max_prefix_len[k]
            good_suffix[j] = m - prefix_len

        for j in range(m - 1):
            k = n_array[j]
            if k > 0 and k < m:
                mismatch_pos = m - 1 - k
                good_suffix[mismatch_pos] = (m - 1) - j

        full_match_shift = m - longest_proper_border
        if full_match_shift <= 0:
            full_match_shift = 1

        return good_suffix, full_match_shift, prep_comparisons

    def preprocess(
        self, pattern: str
    ) -> Tuple[Dict[str, int], List[int], int, int]:
        bad_char = self.compute_bad_character_table(pattern)
        good_suffix, full_match_shift, prep_comparisons = self.compute_good_suffix_table(pattern)
        return bad_char, good_suffix, full_match_shift, prep_comparisons

    def search_with_preprocessed(
        self,
        text: str,
        pattern: str,
        preprocessed: Tuple[Dict[str, int], List[int], int, int],
    ) -> MatchResult:
        self._validate_inputs(text, pattern)
        start_time = time.perf_counter()

        n = len(text)
        m = len(pattern)
        bad_char, good_suffix, full_match_shift, prep_comparisons = preprocessed

        if m == 0:
            return self._handle_empty_pattern(text, start_time)

        if m > n:
            return self._handle_pattern_longer_than_text(
                text, pattern, start_time, prep_comparisons
            )

        matches: List[int] = []
        search_comparisons = 0
        s = 0

        while s <= n - m:
            j = m - 1

            while j >= 0:
                search_comparisons += 1
                if pattern[j] != text[s + j]:
                    break
                j -= 1

            if j < 0:
                matches.append(s)
                s += full_match_shift
            else:
                mismatched_char = text[s + j]
                last_char_pos = bad_char.get(mismatched_char, -1)
                bc_shift = max(1, j - last_char_pos)
                gs_shift = good_suffix[j]
                s += max(bc_shift, gs_shift)

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
                "bad_char_table_size": len(bad_char),
                "good_suffix_table": good_suffix,
                "full_match_shift": full_match_shift,
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


def boyer_moore_search(text: str, pattern: str) -> MatchResult:
    return BoyerMooreMatcher().search(text, pattern)
