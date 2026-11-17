# Streamlit analytical dashboard for MatchLab algorithm benchmarking.

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import streamlit as st
except ImportError:
    st = None

from analytics.complexity_estimator import ComplexityEstimator, ComplexityFit
from analytics.statistics import compute_summary_stats
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import ExperimentConfig
from benchmarking.experiment_manager import ExperimentManager
from benchmarking.metrics import AggregatedMetric, RawMeasurement
from generators.input_generator import InputDistribution
from visualization.plots import (
    ALGORITHM_COLORS,
    build_comparisons_vs_size_chart,
    build_distribution_comparison_chart,
    build_log_log_regression_chart,
    build_pattern_length_scaling_chart,
    build_phase_breakdown_chart,
    build_runtime_vs_size_chart,
)

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.matchlab-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.note-box {
    font-size: 0.8rem;
    color: #cbd5e1;
    background: #1e293b;
    border: 1px solid #334155;
    padding: 12px 16px;
    border-radius: 4px;
    line-height: 1.5;
}

@media (prefers-color-scheme: light) {
    .note-box {
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
}

@media (prefers-color-scheme: dark) {
    .matchlab-panel {
        background: #0f172a;
        border: 1px solid #1e293b;
    }
}

.metric-box {
    border-left: 3px solid #0f172a;
    padding-left: 12px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 2px;
}
.metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    color: #0f172a;
    line-height: 1.2;
}
.metric-sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

@media (prefers-color-scheme: dark) {
    .metric-box { border-left-color: #94a3b8; }
    .metric-value { color: #f8fafc; }
    .metric-label { color: #94a3b8; }
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b;
    border-radius: 0px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2px solid #0f172a !important;
    font-weight: 600;
}

@media (prefers-color-scheme: dark) {
    .stTabs [data-baseweb="tab-list"] { border-bottom-color: #334155; }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom-color: #38bdf8 !important;
    }
}
</style>
"""


def initialize_session_state() -> None:
    if "experiment_manager" not in st.session_state:
        config = ExperimentConfig(
            experiment_id="baseline_init",
            name="initial_baseline",
            distributions=[
                InputDistribution.RANDOM,
                InputDistribution.ADVERSARIAL_PREFIX,
            ],
            text_lengths=[500, 1000, 2000],
            pattern_lengths=[10, 25],
            alphabet_sizes=[26],
            algorithms=["Naive", "KMP", "Boyer-Moore"],
            repetitions=5,
            warmup_runs=1,
            seed=42,
        )
        runner = BenchmarkRunner()
        manager = runner.run_experiment(config)
        st.session_state["experiment_manager"] = manager


def render_header() -> None:
    st.markdown(
        """
        <div style="border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 20px;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                MatchLab Analysis Platform
            </div>
            <div style="font-size: 0.85rem; font-weight: 500; color: #94a3b8; margin-top: 4px;">
                Empirical Evaluation and Complexity Analysis: Naïve, Knuth-Morris-Pratt, and Boyer-Moore
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(manager: ExperimentManager) -> None:
    df_agg = manager.get_aggregated_dataframe()
    config = manager.config

    if df_agg.empty:
        st.info("No benchmark results recorded yet. Configure and execute an experiment in Benchmark Lab.")
        return

    st.markdown("#### Experiment Overview & Key Performance Indicators")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 16px;'>"
        "Statistical summary derived from empirical executions across the current parameter grid. "
        "Performance rankings depend on alphabet entropy, pattern length, and structural redundancy."
        "</div>",
        unsafe_allow_html=True,
    )

    alg_avg_time = df_agg.groupby("algorithm_name")["mean_search_time_seconds"].mean()
    fastest_alg = alg_avg_time.idxmin()
    fastest_time_ms = alg_avg_time.min() * 1000.0

    median_runtime_ms = df_agg["median_search_time_seconds"].median() * 1000.0
    total_comps = df_agg["mean_total_comparisons"].sum()
    total_prep_comps = df_agg["mean_preprocessing_comparisons"].sum()
    prep_overhead_pct = (total_prep_comps / total_comps * 100.0) if total_comps > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Fastest in Configuration</div>
                <div class="metric-value">{fastest_alg}</div>
                <div class="metric-sub">{fastest_time_ms:.3f} ms avg search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Median Search Runtime</div>
                <div class="metric-value">{median_runtime_ms:.3f} ms</div>
                <div class="metric-sub">Across all measured points</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Comparisons</div>
                <div class="metric-value">{total_comps:,.0f}</div>
                <div class="metric-sub">Across grid evaluations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Preprocessing Overhead</div>
                <div class="metric-value">{prep_overhead_pct:.1f}%</div>
                <div class="metric-sub">{total_prep_comps:,.0f} prep comparisons</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### Execution Provenance")
    prov_col1, prov_col2, prov_col3, prov_col4 = st.columns(4)
    with prov_col1:
        st.caption(f"**Experiment ID:** `{config.experiment_id}`")
        st.caption(f"**Timestamp (UTC):** `{config.created_at[:19]}`")
    with prov_col2:
        st.caption(f"**Random Seed:** `{config.seed}`")
        st.caption(f"**Repetitions / Point:** `{config.repetitions}`")
    with prov_col3:
        st.caption(f"**Evaluated Algorithms:** `{', '.join(config.algorithms)}`")
        st.caption(f"**Distributions:** `{', '.join(d.value if hasattr(d, 'value') else str(d) for d in config.distributions)}`")
    with prov_col4:
        st.caption(f"**Text Lengths (n):** `{config.text_lengths}`")
        st.caption(f"**Pattern Lengths (m):** `{config.pattern_lengths}`")

    st.markdown("---")
    st.markdown(
        """
        <div class="note-box">
            <strong style="color: #ffffff;">Methodological Note on Algorithmic Dominance:</strong><br>
            Exact string matching exhibits no universal absolute winner. Boyer-Moore excels in average-case search over large alphabets (e.g. English, ASCII) by executing sublinear skips of size <em>m</em>. Knuth-Morris-Pratt guarantees strict <em>O(n)</em> worst-case search regardless of input degeneration. Naïve matching avoids all preprocessing overhead, making it competitive on very short patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benchmark_lab() -> None:
    st.markdown("#### Benchmark Lab: Parameter Grid Configuration")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 20px;'>"
        "Configure systematic empirical parameter sweeps. Measurements isolate preprocessing from search, "
        "and execute warm-up iterations prior to high-precision nanosecond timing."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("benchmark_config_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Algorithmic & Input Scope")
            selected_algs = st.multiselect(
                "Algorithms to Evaluate",
                ["Naive", "KMP", "Boyer-Moore"],
                default=["Naive", "KMP", "Boyer-Moore"],
                help="Select target algorithms to benchmark.",
            )

            all_dists = [d.value for d in InputDistribution]
            selected_dists = st.multiselect(
                "Input Distributions",
                all_dists,
                default=["random", "adversarial_prefix"],
                help="Generative distributions controlling alphabet entropy, repetition, and alignment.",
            )

            alphabet_sizes = st.multiselect(
                "Alphabet Cardinalities (|Σ|)",
                [2, 4, 10, 26, 52],
                default=[26],
                help="Alphabet cardinality (2: Binary, 4: DNA, 26: Lowercase ASCII, 52: Mixed Letters).",
            )

        with col_right:
            st.markdown("##### Dimension Sweeps & Statistical Rigor")
            text_lengths_str = st.text_input(
                "Text Lengths n (comma-separated)",
                value="500, 1000, 2000, 4000",
                help="List of corpus lengths to evaluate.",
            )

            pattern_lengths_str = st.text_input(
                "Pattern Lengths m (comma-separated)",
                value="10, 25",
                help="List of search pattern lengths.",
            )

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                repetitions = st.number_input(
                    "Repetitions per Point",
                    min_value=3,
                    max_value=50,
                    value=30,
                    step=1,
                    help="Default 30 repetitions for central limit theorem statistical validity.",
                )
            with col_sub2:
                seed = st.number_input(
                    "Deterministic Random Seed",
                    min_value=0,
                    max_value=999999,
                    value=42,
                    step=1,
                    help="Ensures exact dataset reproducibility.",
                )

        st.markdown("<br>", unsafe_allow_html=True)
        execute_button = st.form_submit_button(
            "Execute Benchmark Experiment",
            type="primary",
            width="stretch",
        )

    if execute_button:
        try:
            n_list = [int(x.strip()) for x in text_lengths_str.split(",") if x.strip()]
            m_list = [int(x.strip()) for x in pattern_lengths_str.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid text or pattern length input. Please provide comma-separated integers.")
            return

        if not selected_algs:
            st.error("Select at least one algorithm to benchmark.")
            return
        if not selected_dists:
            st.error("Select at least one input distribution.")
            return

        dist_enums = [InputDistribution(d) for d in selected_dists]

        new_config = ExperimentConfig(
            name="custom_benchmark",
            distributions=dist_enums,
            text_lengths=n_list,
            pattern_lengths=m_list,
            alphabet_sizes=alphabet_sizes if alphabet_sizes else [26],
            algorithms=selected_algs,
            repetitions=repetitions,
            warmup_runs=3,
            seed=seed,
        )

        with st.spinner(f"Executing benchmark sweep across {len(selected_algs) * len(dist_enums) * len(n_list) * len(m_list)} parameter configurations ({repetitions} reps each)..."):
            runner = BenchmarkRunner()
            manager = runner.run_experiment(new_config)
            st.session_state["experiment_manager"] = manager

        st.success(
            f"Experiment `{manager.config.experiment_id}` completed successfully. "
            f"Collected {len(manager.raw_measurements)} raw measurement records."
        )


def render_comparison_analysis(manager: ExperimentManager) -> None:
    df_agg = manager.get_aggregated_dataframe()
    if df_agg.empty:
        st.info("No benchmark results recorded yet.")
        return

    st.markdown("#### Algorithm Performance & Comparison Analysis")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 16px;'>"
        "Interactive comparative analysis across runtime, comparison steps, phase overhead, and distribution sensitivity."
        "</div>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        available_dists = sorted(df_agg["distribution"].unique())
        selected_dist = st.selectbox("Filter Distribution", available_dists, index=0)
    with col_f2:
        available_m = sorted(df_agg["pattern_length"].unique())
        selected_m = st.selectbox("Filter Pattern Length (m)", available_m, index=0)
    with col_f3:
        available_n = sorted(df_agg["text_length"].unique())
        selected_n = st.selectbox("Select Text Length for Breakdown (n)", available_n, index=len(available_n) - 1)

    filtered_df = df_agg[
        (df_agg["distribution"] == selected_dist) & (df_agg["pattern_length"] == selected_m)
    ].copy()

    st.markdown("##### Empirical Scaling with Text Length (n)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if not filtered_df.empty:
            chart_runtime = build_runtime_vs_size_chart(filtered_df)
            st.altair_chart(chart_runtime, width="stretch")
        else:
            st.caption("No data for current filter selection.")
    with col_c2:
        if not filtered_df.empty:
            chart_comps = build_comparisons_vs_size_chart(filtered_df)
            st.altair_chart(chart_comps, width="stretch")
        else:
            st.caption("No data for current filter selection.")

    st.markdown("---")

    st.markdown("##### Phase Breakdown & Distribution Sensitivity")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        chart_phase = build_phase_breakdown_chart(df_agg, selected_n, selected_m)
        st.altair_chart(chart_phase, width="stretch")
    with col_c4:
        chart_dist = build_distribution_comparison_chart(df_agg, selected_n, selected_m)
        st.altair_chart(chart_dist, width="stretch")

    st.markdown("---")

    st.markdown("##### Runtime Sensitivity to Pattern Length (m)")
    chart_pattern = build_pattern_length_scaling_chart(
        df_agg[df_agg["distribution"] == selected_dist], selected_n
    )
    st.altair_chart(chart_pattern, width="stretch")


def render_complexity_analysis(manager: ExperimentManager) -> None:
    df_agg = manager.get_aggregated_dataframe()
    if df_agg.empty:
        st.info("No benchmark results recorded yet.")
        return

    st.markdown("#### Empirical Scaling vs. Theoretical Asymptotic Complexity")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 20px;'>"
        "Estimates observed empirical growth rate <em>k</em> using ordinary least squares regression in log-log space: "
        "<code>log(T(n)) = log(c) + k · log(n)</code>. "
        "Distinguishes concrete experimental growth from asymptotic Big-O limits."
        "</div>",
        unsafe_allow_html=True,
    )

    fits: Dict[str, ComplexityFit] = {}
    fit_rows = []

    algorithms = sorted(df_agg["algorithm_name"].unique())
    dists = sorted(df_agg["distribution"].unique())
    chosen_dist = st.selectbox("Distribution for Scaling Regression", dists, index=0)
    subset = df_agg[df_agg["distribution"] == chosen_dist]

    for alg in algorithms:
        alg_data = subset[subset["algorithm_name"] == alg]
        if len(alg_data) >= 2:
            xs = alg_data["text_length"].tolist()
            ys = alg_data["mean_search_comparisons"].tolist()
            fit = ComplexityEstimator.fit_log_log(xs, ys)
            fits[alg] = fit

            fit_time = ComplexityEstimator.fit_log_log(
                xs, alg_data["mean_search_time_seconds"].tolist()
            )

            fit_rows.append({
                "Algorithm": alg,
                "Comparison Exponent (k)": f"{fit.exponent:.3f}",
                "Comparison R²": f"{fit.r_squared:.4f}",
                "Runtime Exponent (k)": f"{fit_time.exponent:.3f}",
                "Runtime R²": f"{fit_time.r_squared:.4f}",
                "Empirical Classification": fit.empirical_classification,
                "Theoretical Worst Case": "O((n - m + 1) * m)" if alg == "Naive" else ("O(n)" if alg == "KMP" else "O(n * m)"),
                "Theoretical Best Case": "O(n)" if alg in ["Naive", "KMP"] else "O(n / m)",
            })

    col_table, col_plot = st.columns([1, 1])

    with col_table:
        st.markdown("##### Empirical Exponents & Theoretical Baselines")
        if fit_rows:
            st.dataframe(pd.DataFrame(fit_rows), width="stretch", hide_index=True)
        else:
            st.caption("Insufficient parameter points (need at least 2 distinct n values).")

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #475569; margin-top: 12px; line-height: 1.5;">
                <strong>Interpretation Guidelines:</strong><br>
                • <code>k ≈ 1.0</code>: Linear scaling with corpus length <em>n</em>.<br>
                • <code>k < 1.0</code>: Sublinear scaling (characteristic of Boyer-Moore on large alphabets).<br>
                • <code>k ≈ 2.0</code>: Quadratic scaling (characteristic of Naïve on adversarial repetitive prefixes).<br>
                • <em>R² ≈ 1.000</em> indicates strong power-law scaling across the tested domain.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_plot:
        st.markdown("##### Log-Log Regression Trajectories")
        chart_log = build_log_log_regression_chart(subset, fits)
        st.altair_chart(chart_log, width="stretch")

    st.markdown("---")
    st.markdown("##### Detailed Asymptotic Distinction Note")
    st.markdown(
        """
        <div class="note-box">
            <strong style="color: #ffffff;">Rigorous Engineering Distinction: Empirical Scaling vs Asymptotic Complexity</strong><br>
            • <strong>Theoretical Asymptotic Complexity</strong> characterizes algorithmic limiting behavior as <code>n → ∞</code> on idealized Turing or RAM models, abstracting away constant factors and hardware memory architecture.<br>
            • <strong>Empirical Complexity Estimation</strong> evaluates real-world execution across a practical parameter interval <code>[n_min, n_max]</code>. Empirical measurements reflect real CPU cache line residency, memory bandwidth saturation, and Python runtime overhead. Consequently, an algorithm with optimal Big-O bounds may exhibit higher constant overhead on small inputs than simpler algorithms.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_explorer(manager: ExperimentManager) -> None:
    df_agg = manager.get_aggregated_dataframe()
    df_raw = manager.get_raw_dataframe()

    if df_agg.empty:
        st.info("No benchmark results recorded yet.")
        return

    st.markdown("#### Results Explorer & Dataset Export")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 16px;'>"
        "Inspect, filter, and export both aggregated summary statistics and raw trial repetitions."
        "</div>",
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        algs = ["All"] + sorted(df_agg["algorithm_name"].unique().tolist())
        sel_alg = st.selectbox("Algorithm", algs)
    with col_f2:
        dists = ["All"] + sorted(df_agg["distribution"].unique().tolist())
        sel_dist = st.selectbox("Dataset Distribution", dists)
    with col_f3:
        ns = ["All"] + [str(x) for x in sorted(df_agg["text_length"].unique())]
        sel_n = st.selectbox("Text Length (n)", ns)
    with col_f4:
        ms = ["All"] + [str(x) for x in sorted(df_agg["pattern_length"].unique())]
        sel_m = st.selectbox("Pattern Length (m)", ms)

    filtered_agg = df_agg.copy()
    if sel_alg != "All":
        filtered_agg = filtered_agg[filtered_agg["algorithm_name"] == sel_alg]
    if sel_dist != "All":
        filtered_agg = filtered_agg[filtered_agg["distribution"] == sel_dist]
    if sel_n != "All":
        filtered_agg = filtered_agg[filtered_agg["text_length"] == int(sel_n)]
    if sel_m != "All":
        filtered_agg = filtered_agg[filtered_agg["pattern_length"] == int(sel_m)]

    tab_agg, tab_raw = st.tabs(["Aggregated Summary Metrics", "Raw Individual Repetitions"])

    with tab_agg:
        st.dataframe(filtered_agg, width="stretch")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_agg = filtered_agg.to_csv(index=False)
            st.download_button(
                label="Download Aggregated Metrics (CSV)",
                data=csv_agg,
                file_name=f"matchlab_aggregated_{manager.config.experiment_id}.csv",
                mime="text/csv",
            )
        with col_exp2:
            json_agg = filtered_agg.to_json(orient="records", indent=2)
            st.download_button(
                label="Download Aggregated Metrics (JSON)",
                data=json_agg,
                file_name=f"matchlab_aggregated_{manager.config.experiment_id}.json",
                mime="application/json",
            )

    with tab_raw:
        filtered_raw = df_raw.copy()
        if sel_alg != "All":
            filtered_raw = filtered_raw[filtered_raw["algorithm_name"] == sel_alg]
        if sel_dist != "All":
            filtered_raw = filtered_raw[filtered_raw["distribution"] == sel_dist]
        if sel_n != "All":
            filtered_raw = filtered_raw[filtered_raw["text_length"] == int(sel_n)]
        if sel_m != "All":
            filtered_raw = filtered_raw[filtered_raw["pattern_length"] == int(sel_m)]

        st.dataframe(filtered_raw, width="stretch")

        col_exp3, col_exp4 = st.columns(2)
        with col_exp3:
            csv_raw = filtered_raw.to_csv(index=False)
            st.download_button(
                label="Download Raw Measurements (CSV)",
                data=csv_raw,
                file_name=f"matchlab_raw_{manager.config.experiment_id}.csv",
                mime="text/csv",
            )
        with col_exp4:
            json_raw = filtered_raw.to_json(orient="records", indent=2)
            st.download_button(
                label="Download Raw Measurements (JSON)",
                data=json_raw,
                file_name=f"matchlab_raw_{manager.config.experiment_id}.json",
                mime="application/json",
            )


def main() -> None:
    if st is None:
        print("Streamlit is required. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="MatchLab | Empirical Pattern-Matching Analyzer",
        page_icon="M",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialize_session_state()
    render_header()

    manager = st.session_state["experiment_manager"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Benchmark Lab",
            "Algorithm Comparison",
            "Complexity Analysis",
            "Results Explorer",
        ]
    )

    with tab1:
        render_overview(manager)

    with tab2:
        render_benchmark_lab()

    with tab3:
        render_comparison_analysis(manager)

    with tab4:
        render_complexity_analysis(manager)

    with tab5:
        render_results_explorer(manager)


if __name__ == "__main__":
    main()
