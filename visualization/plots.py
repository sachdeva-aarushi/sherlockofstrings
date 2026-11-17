# Visualization utilities and chart builders using Altair and Matplotlib.

from typing import Dict, List, Optional
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics.complexity_estimator import ComplexityFit
from benchmarking.metrics import AggregatedMetric

ALGORITHM_COLORS: Dict[str, str] = {
    "Naive": "#dc2626",
    "KMP": "#2563eb",
    "Boyer-Moore": "#059669",
}

ALGORITHM_COLOR_SCALE = alt.Scale(
    domain=["Naive", "KMP", "Boyer-Moore"],
    range=["#dc2626", "#2563eb", "#059669"],
)


def build_runtime_vs_size_chart(df: pd.DataFrame) -> alt.Chart:
    plot_df = df.copy()
    if "mean_search_time_ms" not in plot_df.columns and "mean_search_time_seconds" in plot_df.columns:
        plot_df["mean_search_time_ms"] = plot_df["mean_search_time_seconds"] * 1000.0

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60))
        .encode(
            x=alt.X(
                "text_length:Q",
                title="Corpus Text Length (n)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            y=alt.Y(
                "mean_search_time_ms:Q",
                title="Search Runtime (ms)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            color=alt.Color(
                "algorithm_name:N",
                title="Algorithm",
                scale=ALGORITHM_COLOR_SCALE,
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("distribution:N", title="Distribution"),
                alt.Tooltip("text_length:Q", title="Text Length (n)"),
                alt.Tooltip("pattern_length:Q", title="Pattern Length (m)"),
                alt.Tooltip("mean_search_time_ms:Q", title="Search Time (ms)", format=".3f"),
                alt.Tooltip("mean_search_comparisons:Q", title="Search Comps", format=",.0f"),
                alt.Tooltip("mean_throughput_chars_per_sec:Q", title="Throughput (c/s)", format=",.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Search Runtime vs Corpus Text Length",
                subtitle="Measured wall-clock search time across varying corpus sizes (pattern length fixed)",
                fontSize=14,
                fontWeight=600,
                color="#0f172a",
            ),
            height=360,
        )
        .interactive()
    )
    return chart


def build_comparisons_vs_size_chart(df: pd.DataFrame) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60))
        .encode(
            x=alt.X(
                "text_length:Q",
                title="Corpus Text Length (n)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            y=alt.Y(
                "mean_search_comparisons:Q",
                title="Search Character Comparisons",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            color=alt.Color(
                "algorithm_name:N",
                title="Algorithm",
                scale=ALGORITHM_COLOR_SCALE,
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("text_length:Q", title="Text Length (n)"),
                alt.Tooltip("pattern_length:Q", title="Pattern Length (m)"),
                alt.Tooltip("mean_search_comparisons:Q", title="Search Comps", format=",.0f"),
                alt.Tooltip("mean_preprocessing_comparisons:Q", title="Prep Comps", format=",.0f"),
                alt.Tooltip("mean_total_comparisons:Q", title="Total Comps", format=",.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Character Comparisons vs Corpus Text Length",
                subtitle="Deterministic comparison counts isolating algorithmic operations from machine noise",
                fontSize=14,
                fontWeight=600,
                color="#0f172a",
            ),
            height=360,
        )
        .interactive()
    )
    return chart


def build_phase_breakdown_chart(df: pd.DataFrame, text_length: int, pattern_length: int) -> alt.Chart:
    target_df = df[
        (df["text_length"] == text_length) & (df["pattern_length"] == pattern_length)
    ].copy()

    if target_df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    melted = target_df.melt(
        id_vars=["algorithm_name"],
        value_vars=["mean_preprocessing_comparisons", "mean_search_comparisons"],
        var_name="Phase",
        value_name="Comparisons",
    )
    melted["Phase"] = melted["Phase"].map({
        "mean_preprocessing_comparisons": "Preprocessing",
        "mean_search_comparisons": "Search",
    })

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Phase:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Comparisons:Q", title="Character Comparisons", axis=alt.Axis(grid=True, gridColor="#e2e8f0")),
            color=alt.Color(
                "Phase:N",
                scale=alt.Scale(domain=["Preprocessing", "Search"], range=["#f59e0b", "#3b82f6"]),
                legend=alt.Legend(title="Execution Phase"),
            ),
            column=alt.Column(
                "algorithm_name:N",
                title="Algorithm",
                header=alt.Header(labelFontSize=12, labelFontWeight="bold"),
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("Phase:N", title="Phase"),
                alt.Tooltip("Comparisons:Q", title="Comparisons", format=",.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text=f"Phase Comparison Breakdown (n={text_length}, m={pattern_length})",
                fontSize=14,
                fontWeight=600,
            ),
            height=280,
        )
    )
    return chart


def build_distribution_comparison_chart(df: pd.DataFrame, text_length: int, pattern_length: int) -> alt.Chart:
    target_df = df[
        (df["text_length"] == text_length) & (df["pattern_length"] == pattern_length)
    ].copy()

    if target_df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    if "mean_search_time_ms" not in target_df.columns:
        target_df["mean_search_time_ms"] = target_df["mean_search_time_seconds"] * 1000.0

    chart = (
        alt.Chart(target_df)
        .mark_bar()
        .encode(
            x=alt.X("algorithm_name:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_search_time_ms:Q", title="Search Time (ms)", axis=alt.Axis(grid=True, gridColor="#e2e8f0")),
            color=alt.Color("algorithm_name:N", scale=ALGORITHM_COLOR_SCALE, legend=None),
            column=alt.Column(
                "distribution:N",
                title="Input Distribution",
                header=alt.Header(labelFontSize=11, labelFontWeight="bold"),
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("distribution:N", title="Distribution"),
                alt.Tooltip("mean_search_time_ms:Q", title="Search Time (ms)", format=".3f"),
                alt.Tooltip("mean_search_comparisons:Q", title="Search Comps", format=",.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text=f"Runtime Across Distributions (n={text_length}, m={pattern_length})",
                fontSize=14,
                fontWeight=600,
            ),
            height=280,
        )
    )
    return chart


def build_pattern_length_scaling_chart(df: pd.DataFrame, text_length: int) -> alt.Chart:
    target_df = df[df["text_length"] == text_length].copy()
    if target_df.empty:
        return alt.Chart(pd.DataFrame()).mark_line()

    if "mean_search_time_ms" not in target_df.columns:
        target_df["mean_search_time_ms"] = target_df["mean_search_time_seconds"] * 1000.0

    chart = (
        alt.Chart(target_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=60))
        .encode(
            x=alt.X(
                "pattern_length:Q",
                title="Pattern Length (m)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            y=alt.Y(
                "mean_search_time_ms:Q",
                title="Search Runtime (ms)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            color=alt.Color(
                "algorithm_name:N",
                title="Algorithm",
                scale=ALGORITHM_COLOR_SCALE,
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("pattern_length:Q", title="Pattern Length (m)"),
                alt.Tooltip("mean_search_time_ms:Q", title="Search Time (ms)", format=".3f"),
                alt.Tooltip("mean_search_comparisons:Q", title="Search Comps", format=",.0f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text=f"Runtime vs Pattern Length (Text Length n={text_length})",
                subtitle="Illustrates sublinear advantage of Boyer-Moore as pattern length m increases",
                fontSize=14,
                fontWeight=600,
            ),
            height=320,
        )
        .interactive()
    )
    return chart


def build_log_log_regression_chart(
    df: pd.DataFrame,
    fits: Dict[str, ComplexityFit],
) -> alt.Chart:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["text_length"] > 0]
    plot_df = plot_df[plot_df["mean_search_comparisons"] > 0]

    if plot_df.empty:
        return alt.Chart(pd.DataFrame()).mark_point()

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=80, opacity=0.85)
        .encode(
            x=alt.X(
                "text_length:Q",
                scale=alt.Scale(type="log"),
                title="log(Corpus Text Length n)",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            y=alt.Y(
                "mean_search_comparisons:Q",
                scale=alt.Scale(type="log"),
                title="log(Search Comparisons C(n))",
                axis=alt.Axis(grid=True, gridColor="#e2e8f0"),
            ),
            color=alt.Color(
                "algorithm_name:N",
                title="Algorithm",
                scale=ALGORITHM_COLOR_SCALE,
            ),
            tooltip=[
                alt.Tooltip("algorithm_name:N", title="Algorithm"),
                alt.Tooltip("text_length:Q", title="n"),
                alt.Tooltip("mean_search_comparisons:Q", title="Comparisons", format=",.0f"),
            ],
        )
    )

    line = (
        alt.Chart(plot_df)
        .transform_regression(
            "text_length",
            "mean_search_comparisons",
            groupby=["algorithm_name"],
            method="pow",
        )
        .mark_line(strokeWidth=2, strokeDash=[4, 4])
        .encode(
            x=alt.X("text_length:Q", scale=alt.Scale(type="log")),
            y=alt.Y("mean_search_comparisons:Q", scale=alt.Scale(type="log")),
            color=alt.Color("algorithm_name:N", scale=ALGORITHM_COLOR_SCALE),
        )
    )

    return (
        (scatter + line)
        .properties(
            title=alt.TitleParams(
                text="Empirical Scaling Log-Log Regression: log(C(n)) vs log(n)",
                subtitle="Slope of the linear trend in log-log space estimates empirical scaling exponent k",
                fontSize=14,
                fontWeight=600,
            ),
            height=360,
        )
        .interactive()
    )


def plot_comparisons_vs_text_length(
    metrics: List[AggregatedMetric],
    pattern_length: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    filtered = metrics
    if pattern_length is not None:
        filtered = [m for m in metrics if m.pattern_length == pattern_length]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    algorithms = sorted(list(set(m.algorithm_name for m in filtered)))
    styles = {
        "Naive": {"color": "#dc2626", "marker": "o", "linestyle": "--"},
        "KMP": {"color": "#2563eb", "marker": "s", "linestyle": "-"},
        "Boyer-Moore": {"color": "#059669", "marker": "^", "linestyle": "-."},
    }

    for alg in algorithms:
        alg_metrics = sorted(
            [m for m in filtered if m.algorithm_name == alg], key=lambda x: x.text_length
        )
        x = [m.text_length for m in alg_metrics]
        y = [m.mean_total_comparisons for m in alg_metrics]
        style = styles.get(alg, {"color": "black", "marker": "x", "linestyle": "-"})
        ax.plot(x, y, label=alg, **style)

    ax.set_title(
        f"Total Character Comparisons vs Text Length"
        + (f" (m={pattern_length})" if pattern_length else ""),
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )
    ax.set_xlabel("Text Length (n)", fontsize=10, color="#334155")
    ax.set_ylabel("Mean Total Comparisons", fontsize=10, color="#334155")
    ax.grid(True, linestyle=":", alpha=0.5, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_execution_time_vs_text_length(
    metrics: List[AggregatedMetric],
    pattern_length: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    filtered = metrics
    if pattern_length is not None:
        filtered = [m for m in metrics if m.pattern_length == pattern_length]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    algorithms = sorted(list(set(m.algorithm_name for m in filtered)))
    styles = {
        "Naive": {"color": "#dc2626", "marker": "o", "linestyle": "--"},
        "KMP": {"color": "#2563eb", "marker": "s", "linestyle": "-"},
        "Boyer-Moore": {"color": "#059669", "marker": "^", "linestyle": "-."},
    }

    for alg in algorithms:
        alg_metrics = sorted(
            [m for m in filtered if m.algorithm_name == alg], key=lambda x: x.text_length
        )
        x = [m.text_length for m in alg_metrics]
        y = [m.mean_search_time_seconds * 1000.0 for m in alg_metrics]
        style = styles.get(alg, {"color": "black", "marker": "x", "linestyle": "-"})
        ax.plot(x, y, label=alg, **style)

    ax.set_title(
        "Search Runtime (ms) vs Text Length",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )
    ax.set_xlabel("Text Length (n)", fontsize=10, color="#334155")
    ax.set_ylabel("Search Runtime (ms)", fontsize=10, color="#334155")
    ax.grid(True, linestyle=":", alpha=0.5, color="#94a3b8")
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_algorithm_comparison_bar(
    metrics: List[AggregatedMetric],
    text_length: int,
    pattern_length: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    target_metrics = [
        m for m in metrics
        if m.text_length == text_length and m.pattern_length == pattern_length
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    names = [m.algorithm_name for m in target_metrics]
    prep = [m.mean_preprocessing_comparisons for m in target_metrics]
    search = [m.mean_search_comparisons for m in target_metrics]

    x = range(len(names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], prep, width, label="Preprocessing Phase", color="#f59e0b")
    ax.bar([i + width / 2 for i in x], search, width, label="Search Phase", color="#3b82f6")

    ax.set_title(
        f"Phase Comparison Breakdown (n={text_length}, m={pattern_length})",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Character Comparisons", fontsize=10, color="#334155")
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color="#94a3b8")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig
