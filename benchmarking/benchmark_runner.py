# Centralized benchmark runner measuring runtime and comparison operations.

from datetime import datetime, timezone
import statistics as py_stats
import time
from typing import Dict, List, Optional

from algorithms.base import BaseMatcher
from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.experiment_manager import ExperimentManager
from benchmarking.metrics import AggregatedMetric, RawMeasurement
from generators.input_generator import (
    GeneratedInput,
    InputDistribution,
    generate_benchmark_input,
)


class BenchmarkRunner:
    DEFAULT_MATCHERS: Dict[str, BaseMatcher] = {
        "Naive": NaiveStringMatcher(),
        "KMP": KMPMatcher(),
        "Boyer-Moore": BoyerMooreMatcher(),
    }

    def __init__(self, matchers: Optional[Dict[str, BaseMatcher]] = None) -> None:
        self.matchers = matchers or dict(self.DEFAULT_MATCHERS)

    def measure_single_repetition(
        self,
        matcher: BaseMatcher,
        input_data: GeneratedInput,
        experiment_id: str,
        repetition_index: int,
    ) -> RawMeasurement:
        text = input_data.text
        pattern = input_data.pattern

        t_prep_start = time.perf_counter_ns()
        preprocessed = matcher.preprocess(pattern)
        t_prep_end = time.perf_counter_ns()
        prep_time_ns = t_prep_end - t_prep_start

        t_search_start = time.perf_counter_ns()
        match_result = matcher.search_with_preprocessed(text, pattern, preprocessed)
        t_search_end = time.perf_counter_ns()
        search_time_ns = t_search_end - t_search_start

        total_time_ns = prep_time_ns + search_time_ns
        timestamp = datetime.now(timezone.utc).isoformat()

        return RawMeasurement(
            experiment_id=experiment_id,
            timestamp=timestamp,
            algorithm_name=matcher.name,
            distribution=str(input_data.distribution.value),
            text_length=input_data.text_length,
            pattern_length=input_data.pattern_length,
            alphabet_size=input_data.alphabet_size,
            seed=input_data.seed,
            repetition_index=repetition_index,
            preprocessing_time_ns=prep_time_ns,
            search_time_ns=search_time_ns,
            total_time_ns=total_time_ns,
            preprocessing_time_seconds=prep_time_ns / 1_000_000_000.0,
            search_time_seconds=search_time_ns / 1_000_000_000.0,
            total_time_seconds=total_time_ns / 1_000_000_000.0,
            preprocessing_comparisons=match_result.preprocessing_comparisons,
            search_comparisons=match_result.search_comparisons,
            total_comparisons=match_result.total_comparisons,
            match_count=len(match_result.matches),
        )

    def aggregate_measurements(
        self,
        measurements: List[RawMeasurement],
    ) -> AggregatedMetric:
        if not measurements:
            raise ValueError("Cannot aggregate empty measurements list")

        first = measurements[0]
        k = len(measurements)

        search_times = [m.search_time_seconds for m in measurements]
        total_times = [m.total_time_seconds for m in measurements]
        prep_times = [m.preprocessing_time_seconds for m in measurements]

        prep_comps = [m.preprocessing_comparisons for m in measurements]
        search_comps = [m.search_comparisons for m in measurements]
        total_comps = [m.total_comparisons for m in measurements]

        mean_search_time = py_stats.mean(search_times)
        std_search_time = py_stats.stdev(search_times) if k > 1 else 0.0
        median_search_time = py_stats.median(search_times)
        min_search_time = min(search_times)
        max_search_time = max(search_times)

        mean_total_time = py_stats.mean(total_times)
        std_total_time = py_stats.stdev(total_times) if k > 1 else 0.0
        median_total_time = py_stats.median(total_times)
        min_total_time = min(total_times)
        max_total_time = max(total_times)

        mean_throughput = (
            first.text_length / mean_search_time if mean_search_time > 0 else 0.0
        )

        return AggregatedMetric(
            experiment_id=first.experiment_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            algorithm_name=first.algorithm_name,
            distribution=first.distribution,
            text_length=first.text_length,
            pattern_length=first.pattern_length,
            alphabet_size=first.alphabet_size,
            seed=first.seed,
            repetition_count=k,
            mean_preprocessing_time_seconds=py_stats.mean(prep_times),
            mean_search_time_seconds=mean_search_time,
            median_search_time_seconds=median_search_time,
            std_search_time_seconds=std_search_time,
            min_search_time_seconds=min_search_time,
            max_search_time_seconds=max_search_time,
            mean_total_time_seconds=mean_total_time,
            median_total_time_seconds=median_total_time,
            std_total_time_seconds=std_total_time,
            min_total_time_seconds=min_total_time,
            max_total_time_seconds=max_total_time,
            mean_preprocessing_comparisons=py_stats.mean(prep_comps),
            median_preprocessing_comparisons=py_stats.median(prep_comps),
            mean_search_comparisons=py_stats.mean(search_comps),
            median_search_comparisons=py_stats.median(search_comps),
            mean_total_comparisons=py_stats.mean(total_comps),
            median_total_comparisons=py_stats.median(total_comps),
            std_total_comparisons=py_stats.stdev(total_comps) if k > 1 else 0.0,
            mean_throughput_chars_per_sec=mean_throughput,
            match_count=first.match_count,
        )

    def run_experiment(self, config: ExperimentConfig) -> ExperimentManager:
        raw_list: List[RawMeasurement] = []
        agg_list: List[AggregatedMetric] = []

        active_matchers = [
            self.matchers[name]
            for name in config.algorithms
            if name in self.matchers
        ]
        if not active_matchers:
            raise ValueError(f"No recognized algorithms in {config.algorithms}")

        for dist in config.distributions:
            for alphabet_size in config.alphabet_sizes:
                for m in config.pattern_lengths:
                    for n in config.text_lengths:
                        input_data = generate_benchmark_input(
                            distribution=dist,
                            text_length=n,
                            pattern_length=m,
                            alphabet_size=alphabet_size,
                            seed=config.seed,
                        )

                        for matcher in active_matchers:
                            for _ in range(config.warmup_runs):
                                prep = matcher.preprocess(input_data.pattern)
                                matcher.search_with_preprocessed(
                                    input_data.text, input_data.pattern, prep
                                )

                            point_measurements: List[RawMeasurement] = []
                            for rep in range(config.repetitions):
                                raw = self.measure_single_repetition(
                                    matcher=matcher,
                                    input_data=input_data,
                                    experiment_id=config.experiment_id,
                                    repetition_index=rep,
                                )
                                point_measurements.append(raw)
                                raw_list.append(raw)

                            agg = self.aggregate_measurements(point_measurements)
                            agg_list.append(agg)

        return ExperimentManager(
            config=config,
            raw_measurements=raw_list,
            aggregated_metrics=agg_list,
        )
