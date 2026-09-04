"""
Drift Analysis
================

Standard navigation metrics measuring how far the estimated trajectory
deviates from the true (ground-truth) trajectory after GPS is turned off.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DriftMetrics:
    label: str
    position_error_m: np.ndarray   # instantaneous position error over time
    rmse_m: float
    final_error_m: float
    cep50_m: float                  # Circular Error Probable (median error)
    max_error_m: float


def compute_drift(t: np.ndarray, est_x: np.ndarray, est_y: np.ndarray,
                   true_x: np.ndarray, true_y: np.ndarray, label: str,
                   gps_off_idx: int = 0) -> DriftMetrics:
    """
    gps_off_idx: index at which GPS was turned off. Metrics are computed from
    this point onward (before that point GPS corrections are already active,
    so "drift" is only meaningful after GPS is off).
    """
    err = np.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2)
    err_after_off = err[gps_off_idx:]

    return DriftMetrics(
        label=label,
        position_error_m=err,
        rmse_m=float(np.sqrt(np.mean(err_after_off ** 2))),
        final_error_m=float(err_after_off[-1]),
        cep50_m=float(np.median(err_after_off)),
        max_error_m=float(np.max(err_after_off)),
    )


def summarize(metrics_list):
    lines = []
    lines.append(f"{'Configuration':<30}{'RMSE (m)':>12}{'CEP50 (m)':>12}{'Final Err (m)':>14}{'Max (m)':>12}")
    lines.append("-" * 80)
    for m in metrics_list:
        lines.append(f"{m.label:<30}{m.rmse_m:>12.1f}{m.cep50_m:>12.1f}{m.final_error_m:>14.1f}{m.max_error_m:>12.1f}")
    return "\n".join(lines)
