# Base contracts and data structures for string matching algorithms.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List


@dataclass(frozen=True)
class MatchResult:
    algorithm_name: str
    pattern: str
    text: str
    matches: List[int]
    preprocessing_comparisons: int
    search_comparisons: int
    total_comparisons: int
    execution_time_seconds: float
    preprocessing_time_seconds: float = 0.0
    search_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_comparisons != (self.preprocessing_comparisons + self.search_comparisons):
            object.__setattr__(
                self,
                "total_comparisons",
                self.preprocessing_comparisons + self.search_comparisons,
            )


class BaseMatcher(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def search(self, text: str, pattern: str) -> MatchResult:
        pass

    def preprocess(self, pattern: str) -> Any:
        return None

    def search_with_preprocessed(
        self, text: str, pattern: str, preprocessed: Any
    ) -> MatchResult:
        return self.search(text, pattern)

    def _validate_inputs(self, text: str, pattern: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got {type(pattern).__name__}")

    def _handle_empty_pattern(self, text: str, start_time: float) -> MatchResult:
        elapsed = time.perf_counter() - start_time
        return MatchResult(
            algorithm_name=self.name,
            pattern="",
            text=text,
            matches=list(range(len(text) + 1)),
            preprocessing_comparisons=0,
            search_comparisons=0,
            total_comparisons=0,
            execution_time_seconds=elapsed,
            metadata={"note": "empty_pattern"},
        )

    def _handle_pattern_longer_than_text(
        self, text: str, pattern: str, start_time: float, preprocessing_comparisons: int = 0
    ) -> MatchResult:
        elapsed = time.perf_counter() - start_time
        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=[],
            preprocessing_comparisons=preprocessing_comparisons,
            search_comparisons=0,
            total_comparisons=preprocessing_comparisons,
            execution_time_seconds=elapsed,
            metadata={"note": "pattern_longer_than_text"},
        )
