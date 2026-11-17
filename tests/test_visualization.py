"""Tests for MatchLab visualization utilities and chart builders."""

from pathlib import Path
import tempfile
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from analytics.complexity_estimator import ComplexityEstimator
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import ExperimentConfig
from generators.input_generator import InputDistribution
from visualization.plots import (
    build_comparisons_vs_size_chart,
    build_distribution_comparison_chart,
    build_log_log_regression_chart,
    build_pattern_length_scaling_chart,
    build_phase_breakdown_chart,
    build_runtime_vs_size_chart,
    plot_algorithm_comparison_bar,
    plot_comparisons_vs_text_length,
    plot_execution_time_vs_text_length,
)


@pytest.fixture(scope="module")
def benchmark_data():
    config = ExperimentConfig(
        distributions=[InputDistribution.RANDOM, InputDistribution.ADVERSARIAL_PREFIX],
        text_lengths=[100, 200],
        pattern_lengths=[5, 10],
        repetitions=2,
        warmup_runs=1,
        seed=999,
    )
    runner = BenchmarkRunner()
    manager = runner.run_experiment(config)
    return manager


def test_build_runtime_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    chart = build_runtime_vs_size_chart(df)
    assert isinstance(chart, alt.Chart)


def test_build_comparisons_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    chart = build_comparisons_vs_size_chart(df)
    assert isinstance(chart, alt.Chart)


def test_build_phase_breakdown_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    chart = build_phase_breakdown_chart(df, text_length=200, pattern_length=5)
    # Grouped column charts return an Altair Chart or FacetedChart
    assert hasattr(chart, "to_dict")


def test_build_distribution_comparison_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    chart = build_distribution_comparison_chart(df, text_length=200, pattern_length=5)
    assert hasattr(chart, "to_dict")


def test_build_pattern_length_scaling_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    chart = build_pattern_length_scaling_chart(df, text_length=200)
    assert isinstance(chart, alt.Chart)


def test_build_log_log_regression_chart(benchmark_data) -> None:
    df = benchmark_data.get_aggregated_dataframe()
    fits = {}
    for alg in ["Naive", "KMP", "Boyer-Moore"]:
        alg_df = df[df["algorithm_name"] == alg]
        fit = ComplexityEstimator.fit_log_log(
            alg_df["text_length"].tolist(),
            alg_df["mean_search_comparisons"].tolist(),
        )
        fits[alg] = fit

    chart = build_log_log_regression_chart(df, fits)
    assert hasattr(chart, "to_dict")


def test_matplotlib_plots_and_save(benchmark_data) -> None:
    metrics = benchmark_data.aggregated_metrics

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        fig1 = plot_comparisons_vs_text_length(
            metrics, pattern_length=5, save_path=str(tmp_path / "fig1.png")
        )
        assert isinstance(fig1, plt.Figure)
        assert (tmp_path / "fig1.png").exists()

        fig2 = plot_execution_time_vs_text_length(
            metrics, pattern_length=5, save_path=str(tmp_path / "fig2.png")
        )
        assert isinstance(fig2, plt.Figure)
        assert (tmp_path / "fig2.png").exists()

        fig3 = plot_algorithm_comparison_bar(
            metrics, text_length=100, pattern_length=5, save_path=str(tmp_path / "fig3.png")
        )
        assert isinstance(fig3, plt.Figure)
        assert (tmp_path / "fig3.png").exists()

        plt.close("all")
