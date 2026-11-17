# CLI entry point for algorithm demonstration and benchmark sweeps.

import argparse
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import ExperimentConfig
from analytics.complexity_estimator import ComplexityEstimator
from generators.input_generator import (
    InputDistribution,
    generate_worst_case_naive,
    generate_best_case_boyer_moore,
    generate_overlapping_case,
)


def run_demonstration() -> None:
    matchers = [
        NaiveStringMatcher(),
        KMPMatcher(),
        BoyerMooreMatcher(),
    ]

    scenarios = [
        (
            "Natural DNA Sequence",
            "GCATATCGCATATAGCATATCGGCATATCGCATATCGCATATA",
            "GCATATC",
        ),
        (
            "Dense Overlapping Matches ('A'*30, 'A'*4)",
            generate_overlapping_case(30, 4)[0],
            generate_overlapping_case(30, 4)[1],
        ),
        (
            "Naïve Adversarial Worst-Case ('A'*50, 'A'*4 + 'B')",
            generate_worst_case_naive(50, 5)[0],
            generate_worst_case_naive(50, 5)[1],
        ),
        (
            "Boyer-Moore Best-Case Sublinear ('A'*50, 'B'*5)",
            generate_best_case_boyer_moore(50, 5)[0],
            generate_best_case_boyer_moore(50, 5)[1],
        ),
    ]

    print("\n" + "=" * 80)
    print("MatchLab: Empirical Algorithm Comparison Demo".center(80))
    print("=" * 80)

    for title, text, pattern in scenarios:
        print(f"\nScenario: {title}")
        print(f"  Text (n={len(text)}):    {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Pattern (m={len(pattern)}): {pattern}")
        print("-" * 80)
        print(
            f"{'Algorithm':<15} | {'Matches':<8} | {'Prep Comps':<10} | {'Search Comps':<12} | "
            f"{'Total Comps':<11} | {'Time (ms)':<9}"
        )
        print("-" * 80)

        for matcher in matchers:
            res = matcher.search(text, pattern)
            time_ms = res.execution_time_seconds * 1000.0
            print(
                f"{res.algorithm_name:<15} | {len(res.matches):<8} | "
                f"{res.preprocessing_comparisons:<10} | {res.search_comparisons:<12} | "
                f"{res.total_comparisons:<11} | {time_ms:<9.3f}"
            )
        print("-" * 80)


def run_benchmark_cli(
    distribution_name: str = "random",
    repetitions: int = 30,
    export_dir: str = "results",
) -> None:
    dist = InputDistribution(distribution_name)
    print(f"\nExecuting Benchmark Sweep on Distribution: {dist.value}")
    print(f"Repetitions per configuration point: {repetitions}")

    config = ExperimentConfig(
        name=f"benchmark_{dist.value}",
        distributions=[dist],
        text_lengths=[500, 1000, 2000, 4000],
        pattern_lengths=[10],
        alphabet_sizes=[26],
        repetitions=repetitions,
        warmup_runs=3,
        seed=42,
    )
    runner = BenchmarkRunner()
    manager = runner.run_experiment(config)

    df_agg = manager.get_aggregated_dataframe()
    cols_to_show = [
        "algorithm_name",
        "text_length",
        "pattern_length",
        "mean_search_comparisons",
        "mean_preprocessing_comparisons",
        "mean_total_comparisons",
        "mean_search_time_seconds",
        "mean_throughput_chars_per_sec",
    ]
    formatted_df = df_agg[cols_to_show].copy()
    formatted_df["mean_search_time_ms"] = (
        formatted_df["mean_search_time_seconds"] * 1000.0
    ).round(4)
    formatted_df.drop(columns=["mean_search_time_seconds"], inplace=True)

    print("\nAggregated Benchmark Results:")
    print(formatted_df.to_string(index=False))

    export_paths = manager.export_all(export_dir)
    print("\nExported Datasets:")
    for key, path in export_paths.items():
        print(f"  {key:<18}: {path}")

    print("\nEmpirical Complexity Scaling Analysis (log(T(n)) = log(c) + k * log(n)):")
    for alg in config.algorithms:
        alg_rows = df_agg[df_agg["algorithm_name"] == alg]
        if len(alg_rows) >= 2:
            xs = alg_rows["text_length"].tolist()
            fit_comps = ComplexityEstimator.fit_log_log(
                xs, alg_rows["mean_search_comparisons"].tolist()
            )
            fit_time = ComplexityEstimator.fit_log_log(
                xs, alg_rows["mean_search_time_seconds"].tolist()
            )
            print(f"\nAlgorithm: {alg}")
            print(
                f"  Comparison Exponent k = {fit_comps.exponent:.3f} (R² = {fit_comps.r_squared:.4f}) "
                f"-> {fit_comps.empirical_classification}"
            )
            print(
                f"  Runtime Exponent    k = {fit_time.exponent:.3f} (R² = {fit_time.r_squared:.4f}) "
                f"-> {fit_time.empirical_classification}"
            )
            print(f"  Distinction: {fit_comps.asymptotic_distinction}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatchLab: Empirical Pattern-Matching Algorithm Analyzer"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run algorithm comparison demo"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run benchmark sweep for distribution",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
        help="Number of independent repetitions per configuration point (default: 30)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="results",
        help="Directory to save CSV and JSON exports",
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_cli(
            distribution_name=args.benchmark,
            repetitions=args.repetitions,
            export_dir=args.export_dir,
        )
    else:
        run_demonstration()


if __name__ == "__main__":
    main()
