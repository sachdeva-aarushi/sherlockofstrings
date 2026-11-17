"""MatchLab Visualization Package."""

from visualization.plots import (
    ALGORITHM_COLORS,
    ALGORITHM_COLOR_SCALE,
    build_runtime_vs_size_chart,
    build_comparisons_vs_size_chart,
    build_phase_breakdown_chart,
    build_distribution_comparison_chart,
    build_pattern_length_scaling_chart,
    build_log_log_regression_chart,
    plot_comparisons_vs_text_length,
    plot_execution_time_vs_text_length,
    plot_algorithm_comparison_bar,
)

__all__ = [
    "ALGORITHM_COLORS",
    "ALGORITHM_COLOR_SCALE",
    "build_runtime_vs_size_chart",
    "build_comparisons_vs_size_chart",
    "build_phase_breakdown_chart",
    "build_distribution_comparison_chart",
    "build_pattern_length_scaling_chart",
    "build_log_log_regression_chart",
    "plot_comparisons_vs_text_length",
    "plot_execution_time_vs_text_length",
    "plot_algorithm_comparison_bar",
]
