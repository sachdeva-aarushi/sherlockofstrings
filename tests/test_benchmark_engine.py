"""Comprehensive tests for MatchLab Benchmarking and Empirical Analysis layers."""

import json
from pathlib import Path
import tempfile
import pytest
import pandas as pd

from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from analytics.complexity_estimator import ComplexityEstimator, ComplexityFit
from analytics.statistics import (
    calculate_comparison_statistics,
    calculate_runtime_statistics,
    compute_summary_stats,
)
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.experiment_manager import ExperimentManager
from benchmarking.metrics import (
    AggregatedMetric,
    RawMeasurement,
    aggregated_metrics_to_dataframe,
    raw_measurements_to_dataframe,
)
from generators.input_generator import (
    Alphabet,
    GeneratedInput,
    InputDistribution,
    generate_benchmark_input,
    get_alphabet_by_size,
)


# =====================================================================
# 1. Input Generation & Reproducibility Tests
# =====================================================================

def test_input_generation_reproducibility() -> None:
    """Verify that identical configuration and seed yield exact byte-for-byte identical inputs."""
    for dist in InputDistribution:
        inp1 = generate_benchmark_input(
            distribution=dist,
            text_length=150,
            pattern_length=12,
            alphabet_size=10,
            seed=4242,
        )
        inp2 = generate_benchmark_input(
            distribution=dist,
            text_length=150,
            pattern_length=12,
            alphabet_size=10,
            seed=4242,
        )
        assert inp1.text == inp2.text, f"Distribution {dist} text not reproducible"
        assert inp1.pattern == inp2.pattern, f"Distribution {dist} pattern not reproducible"
        assert inp1.text_length == 150
        assert inp1.pattern_length == 12


def test_input_generation_seed_diversity() -> None:
    """Verify that different seeds generate distinct texts for random distributions."""
    inp1 = generate_benchmark_input(InputDistribution.RANDOM, 100, 10, seed=1)
    inp2 = generate_benchmark_input(InputDistribution.RANDOM, 100, 10, seed=2)
    assert inp1.text != inp2.text


def test_configurable_alphabet_sizes() -> None:
    """Verify alphabet cardinality is strictly respected."""
    for size in [2, 4, 10, 26, 52, 60]:
        alpha = get_alphabet_by_size(size)
        assert len(alpha) == size
        assert len(set(alpha)) == size  # All unique

    with pytest.raises(ValueError):
        get_alphabet_by_size(0)


def test_distribution_semantics() -> None:
    """Verify structural properties of specialized distributions."""
    # Absent: pattern chars disjoint from text
    inp_absent = generate_benchmark_input(InputDistribution.ABSENT, 100, 10, alphabet_size=20, seed=1)
    assert not set(inp_absent.pattern).intersection(set(inp_absent.text))

    # Match at start
    inp_start = generate_benchmark_input(InputDistribution.MATCH_AT_START, 100, 10, alphabet_size=20, seed=1)
    assert inp_start.text.startswith(inp_start.pattern)

    # Match at end
    inp_end = generate_benchmark_input(InputDistribution.MATCH_AT_END, 100, 10, alphabet_size=20, seed=1)
    assert inp_end.text.endswith(inp_end.pattern)

    # Repetitive
    inp_rep = generate_benchmark_input(InputDistribution.REPETITIVE, 80, 8, alphabet_size=20, seed=1)
    assert len(set(inp_rep.text)) == 1
    assert len(set(inp_rep.pattern)) == 1
    assert inp_rep.text[0] == inp_rep.pattern[0]


# =====================================================================
# 2. Experiment Configuration & Metric Collection Tests
# =====================================================================

def test_experiment_config_defaults() -> None:
    config = ExperimentConfig()
    assert config.repetitions == 30  # Default 30 repetitions required
    assert config.warmup_runs >= 1
    assert len(config.algorithms) == 3
    assert config.created_at is not None
    assert config.seed == 42


def test_benchmark_runner_metric_collection() -> None:
    """Verify that warmups, timing, comparisons, and match counts are collected properly."""
    config = ExperimentConfig(
        distributions=[InputDistribution.RANDOM],
        text_lengths=[100],
        pattern_lengths=[8],
        alphabet_sizes=[26],
        repetitions=5,
        warmup_runs=2,
        seed=10,
    )
    runner = BenchmarkRunner()
    manager = runner.run_experiment(config)

    assert len(manager.raw_measurements) == 3 * 5  # 3 algorithms * 5 repetitions
    assert len(manager.aggregated_metrics) == 3    # 3 algorithms aggregated

    for raw in manager.raw_measurements:
        assert raw.preprocessing_time_ns >= 0
        assert raw.search_time_ns >= 0
        assert raw.total_time_ns == raw.preprocessing_time_ns + raw.search_time_ns
        assert raw.total_comparisons == raw.preprocessing_comparisons + raw.search_comparisons
        assert raw.text_length == 100
        assert raw.pattern_length == 8

    # Verify input was shared across algorithms
    matches_by_alg = {m.algorithm_name: m.match_count for m in manager.aggregated_metrics}
    assert matches_by_alg["Naive"] == matches_by_alg["KMP"] == matches_by_alg["Boyer-Moore"]


# =====================================================================
# 3. Statistical Analysis Tests
# =====================================================================

def test_statistical_aggregation_correctness() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_summary_stats(values)

    assert stats["count"] == 5.0
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["stdev"] == pytest.approx(15.811388, rel=1e-4)

    runtime_stats = calculate_runtime_statistics(values)
    assert runtime_stats["mean_runtime"] == 30.0
    assert runtime_stats["median_runtime"] == 30.0

    comp_stats = calculate_comparison_statistics(values)
    assert comp_stats["mean_comparisons"] == 30.0
    assert comp_stats["median_comparisons"] == 30.0


# =====================================================================
# 4. Tabular Processing & Serialization Tests
# =====================================================================

def test_dataframe_conversion_and_export() -> None:
    config = ExperimentConfig(
        distributions=[InputDistribution.RANDOM],
        text_lengths=[60],
        pattern_lengths=[5],
        repetitions=3,
        warmup_runs=1,
    )
    runner = BenchmarkRunner()
    manager = runner.run_experiment(config)

    df_raw = manager.get_raw_dataframe()
    df_agg = manager.get_aggregated_dataframe()

    assert isinstance(df_raw, pd.DataFrame)
    assert isinstance(df_agg, pd.DataFrame)
    assert len(df_raw) == 9  # 3 algs * 3 reps
    assert len(df_agg) == 3  # 3 algs

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = manager.export_all(tmp_dir)

        # Check CSV files
        raw_csv = pd.read_csv(paths["raw_csv"])
        assert len(raw_csv) == 9
        agg_csv = pd.read_csv(paths["aggregated_csv"])
        assert len(agg_csv) == 3

        # Check JSON files
        with open(paths["raw_json"], "r", encoding="utf-8") as f:
            raw_json_data = json.load(f)
            assert len(raw_json_data) == 9

        with open(paths["aggregated_json"], "r", encoding="utf-8") as f:
            agg_json_data = json.load(f)
            assert len(agg_json_data) == 3


# =====================================================================
# 5. Empirical Complexity Estimation Tests & Safeguards
# =====================================================================

def test_complexity_linear_fit() -> None:
    # Synthetic linear growth: y = 3 * x
    xs = [100, 200, 400, 800, 1600]
    ys = [3 * x for x in xs]
    fit = ComplexityEstimator.fit_log_log(xs, ys)

    assert fit.valid is True
    assert fit.exponent == pytest.approx(1.0, abs=0.01)
    assert fit.r_squared == pytest.approx(1.0, abs=0.001)
    assert "Linear" in fit.empirical_classification
    assert "asymptotic" in fit.asymptotic_distinction.lower()


def test_complexity_quadratic_fit() -> None:
    # Synthetic quadratic growth: y = 0.5 * x^2
    xs = [50, 100, 200, 400]
    ys = [0.5 * (x ** 2) for x in xs]
    fit = ComplexityEstimator.fit_log_log(xs, ys)

    assert fit.valid is True
    assert fit.exponent == pytest.approx(2.0, abs=0.01)
    assert "Quadratic" in fit.empirical_classification


def test_complexity_safeguards_non_positive_and_insufficient() -> None:
    # Zeros and negative values should be safely filtered
    xs = [100, 200, 0, -50]
    ys = [10, 20, 0, -5]
    fit = ComplexityEstimator.fit_log_log(xs, ys)
    assert fit.valid is True
    assert fit.num_points == 2
    assert fit.exponent == pytest.approx(1.0, abs=0.01)

    # Insufficient data points (< 2)
    fit_insufficient = ComplexityEstimator.fit_log_log([100], [10])
    assert fit_insufficient.valid is False
    assert fit_insufficient.num_points == 1
    assert "Insufficient" in fit_insufficient.empirical_classification


def test_complexity_safeguards_degenerate_domain() -> None:
    # Zero variance in x (all n values identical)
    xs = [100, 100, 100]
    ys = [10, 12, 14]
    fit = ComplexityEstimator.fit_log_log(xs, ys)
    assert fit.valid is False
    assert "Degenerate" in fit.empirical_classification


# =====================================================================
# 6. Cross-Algorithm Consistency on Benchmark Inputs
# =====================================================================

@pytest.mark.parametrize(
    "dist",
    [
        InputDistribution.RANDOM,
        InputDistribution.RANDOM_WITH_MATCH,
        InputDistribution.REPETITIVE,
        InputDistribution.PERIODIC,
        InputDistribution.ADVERSARIAL_PREFIX,
        InputDistribution.BEST_CASE,
        InputDistribution.ABSENT,
        InputDistribution.MATCH_AT_START,
        InputDistribution.MATCH_AT_MIDDLE,
        InputDistribution.MATCH_AT_END,
        InputDistribution.OVERLAPPING,
    ],
)
def test_cross_algorithm_consistency_across_distributions(dist: InputDistribution) -> None:
    """Verify that all three algorithms find the exact same matches on all distributions."""
    inp = generate_benchmark_input(
        distribution=dist,
        text_length=120,
        pattern_length=8,
        alphabet_size=8,
        seed=777,
    )
    naive = NaiveStringMatcher().search(inp.text, inp.pattern)
    kmp = KMPMatcher().search(inp.text, inp.pattern)
    bm = BoyerMooreMatcher().search(inp.text, inp.pattern)

    assert naive.matches == kmp.matches == bm.matches, (
        f"Mismatch on distribution {dist.value}: "
        f"Naive={naive.matches}, KMP={kmp.matches}, BM={bm.matches}"
    )
