"""
Result Export (CSV / JSON)
==============================

Two lightweight, stdlib-only export functions used by both the CLI demo
and the web UI:

  - export_trajectories_csv : per-timestep positions (ground truth + all
                               three configurations, in local meters and
                               lat/lon) as a plain CSV file.
  - export_benchmark_json   : the numeric comparison (RMSE/CEP50/final
                               error per configuration) plus run metadata,
                               as a JSON file.

Implemented with the standard library only (`csv`, `json`) — no pandas
dependency needed just to write these two files.
"""

import csv
import json
from datetime import datetime, timezone

from qnav.simulation.trajectory import local_xy_to_latlon


def export_trajectories_csv(result, path: str) -> None:
    """Writes one row per simulation timestep to a CSV file."""
    gt = result.ground_truth
    gt_lat, gt_lon = local_xy_to_latlon(gt.x, gt.y, result.lat0, result.lon0)
    cls_lat, cls_lon = local_xy_to_latlon(result.hist_classical[:, 0], result.hist_classical[:, 1],
                                           result.lat0, result.lon0)
    q_lat, q_lon = local_xy_to_latlon(result.hist_quantum[:, 0], result.hist_quantum[:, 1],
                                       result.lat0, result.lon0)
    qn_lat, qn_lon = local_xy_to_latlon(result.hist_qnav[:, 0], result.hist_qnav[:, 1],
                                         result.lat0, result.lon0)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s", "gps_on",
            "gt_x_m", "gt_y_m", "gt_lat", "gt_lon",
            "classical_x_m", "classical_y_m", "classical_lat", "classical_lon",
            "quantum_x_m", "quantum_y_m", "quantum_lat", "quantum_lon",
            "qnav_x_m", "qnav_y_m", "qnav_lat", "qnav_lon",
        ])
        for i in range(len(gt.t)):
            writer.writerow([
                f"{gt.t[i]:.3f}", int(gt.t[i] < result.gps_off_time),
                f"{gt.x[i]:.3f}", f"{gt.y[i]:.3f}", f"{gt_lat[i]:.7f}", f"{gt_lon[i]:.7f}",
                f"{result.hist_classical[i,0]:.3f}", f"{result.hist_classical[i,1]:.3f}",
                f"{cls_lat[i]:.7f}", f"{cls_lon[i]:.7f}",
                f"{result.hist_quantum[i,0]:.3f}", f"{result.hist_quantum[i,1]:.3f}",
                f"{q_lat[i]:.7f}", f"{q_lon[i]:.7f}",
                f"{result.hist_qnav[i,0]:.3f}", f"{result.hist_qnav[i,1]:.3f}",
                f"{qn_lat[i]:.7f}", f"{qn_lon[i]:.7f}",
            ])


def _metrics_dict(m):
    return {
        "label": m.label,
        "rmse_m": round(m.rmse_m, 3),
        "cep50_m": round(m.cep50_m, 3),
        "final_error_m": round(m.final_error_m, 3),
        "max_error_m": round(m.max_error_m, 3),
    }


def export_benchmark_json(result, path: str) -> None:
    """Writes the numeric comparison + run metadata to a JSON file."""
    improvement_q = (1 - result.metrics_quantum.rmse_m / result.metrics_classical.rmse_m) * 100
    improvement_qnav = (1 - result.metrics_qnav.rmse_m / result.metrics_classical.rmse_m) * 100

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "location": {
            "name": result.location_name,
            "lat": result.lat0,
            "lon": result.lon0,
        },
        "scenario": {
            "duration_s": result.duration_s,
            "gps_off_time_s": result.gps_off_time,
            "dt_s": result.dt,
            "route_source": result.route_source,
            "total_distance_m": round(result.total_distance_m, 1),
            "gravity_matching_enabled": result.gravity_matching_enabled,
            "gravity_matches_applied": result.n_matches_applied,
        },
        "metrics": {
            "classical_imu": _metrics_dict(result.metrics_classical),
            "quantum_enhanced_imu": _metrics_dict(result.metrics_quantum),
            "qnav_full_system": _metrics_dict(result.metrics_qnav),
        },
        "improvement_vs_classical_pct": {
            "quantum_enhanced": round(improvement_q, 1),
            "qnav_full_system": round(improvement_qnav, 1),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
