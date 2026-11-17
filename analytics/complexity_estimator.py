# Empirical power-law complexity estimator using log-log OLS regression.

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ComplexityFit:
    exponent: float
    constant: float
    r_squared: float
    standard_error: float
    num_points: int
    empirical_classification: str
    asymptotic_distinction: str
    valid: bool
    warning: Optional[str] = None


class ComplexityEstimator:
    @staticmethod
    def fit_log_log(
        n_values: Sequence[float | int],
        metric_values: Sequence[float | int],
    ) -> ComplexityFit:
        if len(n_values) != len(metric_values):
            raise ValueError("n_values and metric_values must have identical lengths")

        filtered_points: List[Tuple[float, float]] = [
            (float(n), float(m))
            for n, m in zip(n_values, metric_values)
            if n is not None and m is not None and n > 0 and m > 0
        ]

        if len(filtered_points) < 2:
            return ComplexityFit(
                exponent=0.0,
                constant=0.0,
                r_squared=0.0,
                standard_error=0.0,
                num_points=len(filtered_points),
                empirical_classification="Insufficient Data",
                asymptotic_distinction=(
                    "Empirical scaling cannot be evaluated with fewer than 2 positive points. "
                    "Theoretical asymptotic complexity remains unchanged."
                ),
                valid=False,
                warning="At least 2 positive data pairs required for log-log regression",
            )

        log_x = [math.log(p[0]) for p in filtered_points]
        log_y = [math.log(p[1]) for p in filtered_points]
        k_pts = len(log_x)

        mean_lx = sum(log_x) / k_pts
        mean_ly = sum(log_y) / k_pts

        var_lx = sum((lx - mean_lx) ** 2 for lx in log_x)

        if var_lx < 1e-12:
            return ComplexityFit(
                exponent=0.0,
                constant=math.exp(mean_ly),
                r_squared=0.0,
                standard_error=0.0,
                num_points=k_pts,
                empirical_classification="Degenerate Domain (n is constant)",
                asymptotic_distinction=(
                    "Input text length n showed zero variance. "
                    "Empirical scaling requires varying input lengths."
                ),
                valid=False,
                warning="Zero variance in input lengths; cannot estimate growth rate",
            )

        cov_xy = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
        exponent = cov_xy / var_lx
        log_c = mean_ly - exponent * mean_lx
        constant = math.exp(log_c)

        predicted_ly = [log_c + exponent * lx for lx in log_x]
        ss_tot = sum((ly - mean_ly) ** 2 for ly in log_y)
        ss_res = sum((ly - py) ** 2 for ly, py in zip(log_y, predicted_ly))

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        r_squared = max(0.0, min(1.0, r_squared))

        if k_pts > 2:
            s_res = math.sqrt(ss_res / (k_pts - 2))
            std_error = s_res / math.sqrt(var_lx)
        else:
            std_error = 0.0

        if exponent < 0.2:
            classification = f"Empirical Sublinear / Constant (k ≈ {exponent:.2f})"
        elif 0.8 <= exponent <= 1.2:
            classification = f"Empirical Linear (k ≈ {exponent:.2f})"
        elif 1.2 < exponent <= 1.4:
            classification = f"Empirical Superlinear (k ≈ {exponent:.2f})"
        elif 1.8 <= exponent <= 2.2:
            classification = f"Empirical Quadratic (k ≈ {exponent:.2f})"
        else:
            classification = f"Empirical Polynomial (k ≈ {exponent:.2f})"

        distinction = (
            f"Observed empirical scaling exponent k={exponent:.3f} over finite range "
            f"n in [{int(filtered_points[0][0])}, {int(filtered_points[-1][0])}]. "
            "Note: Empirical scaling measures concrete wall-clock or step growth over a practical domain "
            "subject to hardware cache, interpreter overhead, and constant factors. "
            "It must not be conflated with asymptotic Big-O bounds as n -> infinity."
        )

        return ComplexityFit(
            exponent=exponent,
            constant=constant,
            r_squared=r_squared,
            standard_error=std_error,
            num_points=k_pts,
            empirical_classification=classification,
            asymptotic_distinction=distinction,
            valid=True,
        )

    @classmethod
    def fit_power_law(
        cls, x_values: Sequence[float | int], y_values: Sequence[float | int]
    ) -> ComplexityFit:
        return cls.fit_log_log(x_values, y_values)
