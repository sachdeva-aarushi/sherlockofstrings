# Metrics data models for raw and aggregated benchmark measurements.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class RawMeasurement:
    experiment_id: str
    timestamp: str
    algorithm_name: str
    distribution: str
    text_length: int
    pattern_length: int
    alphabet_size: int
    seed: Optional[int]
    repetition_index: int
    preprocessing_time_ns: int
    search_time_ns: int
    total_time_ns: int
    preprocessing_time_seconds: float
    search_time_seconds: float
    total_time_seconds: float
    preprocessing_comparisons: int
    search_comparisons: int
    total_comparisons: int
    match_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregatedMetric:
    experiment_id: str
    timestamp: str
    algorithm_name: str
    distribution: str
    text_length: int
    pattern_length: int
    alphabet_size: int
    seed: Optional[int]
    repetition_count: int

    mean_preprocessing_time_seconds: float
    mean_search_time_seconds: float
    median_search_time_seconds: float
    std_search_time_seconds: float
    min_search_time_seconds: float
    max_search_time_seconds: float

    mean_total_time_seconds: float
    median_total_time_seconds: float
    std_total_time_seconds: float
    min_total_time_seconds: float
    max_total_time_seconds: float

    mean_preprocessing_comparisons: float
    median_preprocessing_comparisons: float
    mean_search_comparisons: float
    median_search_comparisons: float
    mean_total_comparisons: float
    median_total_comparisons: float
    std_total_comparisons: float

    mean_throughput_chars_per_sec: float
    match_count: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def raw_measurements_to_dataframe(measurements: List[RawMeasurement]) -> pd.DataFrame:
    if not measurements:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in measurements])


def aggregated_metrics_to_dataframe(metrics: List[AggregatedMetric]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in metrics])
