import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.simulation.runner import run_simulation
from qnav.comparison.export import export_trajectories_csv, export_benchmark_json


def test_run_simulation_offline_produces_consistent_result():
    result = run_simulation(lat=37.7749, lon=-122.4194, duration_s=120, gps_off_time=40,
                             use_road_network=False, gravity_matching=True, seed=1)
    n = len(result.ground_truth.t)
    assert result.hist_classical.shape == (n, 2)
    assert result.hist_quantum.shape == (n, 2)
    assert result.hist_qnav.shape == (n, 2)
    assert result.route_source == "synthetic"
    assert result.metrics_qnav.rmse_m >= 0


def test_gravity_matching_toggle_affects_qnav_only():
    on = run_simulation(lat=37.7749, lon=-122.4194, duration_s=900, gps_off_time=300,
                         use_road_network=False, gravity_matching=True, seed=7)
    off = run_simulation(lat=37.7749, lon=-122.4194, duration_s=900, gps_off_time=300,
                          use_road_network=False, gravity_matching=False, seed=7)

    assert on.n_matches_applied > 0
    assert off.n_matches_applied == 0
    # Classical and Quantum-only configurations must be unaffected by the toggle.
    assert on.metrics_classical.rmse_m == off.metrics_classical.rmse_m
    assert on.metrics_quantum.rmse_m == off.metrics_quantum.rmse_m
    # With matching off, Q-Nav must degrade to exactly the quantum-only result.
    assert off.metrics_qnav.rmse_m == off.metrics_quantum.rmse_m


def test_configurable_duration_and_gps_cutoff_are_respected():
    result = run_simulation(lat=0.0, lon=0.0, duration_s=60, gps_off_time=20,
                             use_road_network=False, seed=2)
    assert result.duration_s == 60
    assert result.gps_off_time == 20
    assert abs(result.ground_truth.t[-1] - 60) < 1.0


def test_export_trajectories_csv_writes_expected_columns(tmp_path=None):
    result = run_simulation(lat=37.7749, lon=-122.4194, duration_s=30, gps_off_time=10,
                             use_road_network=False, seed=1)
    out_path = "/tmp/_qnav_test_trajectories.csv"
    export_trajectories_csv(result, out_path)

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert "gt_lat" in header and "qnav_lon" in header
    assert len(rows) == len(result.ground_truth.t)
    os.remove(out_path)


def test_export_benchmark_json_writes_expected_structure():
    result = run_simulation(lat=37.7749, lon=-122.4194, duration_s=30, gps_off_time=10,
                             use_road_network=False, seed=1)
    out_path = "/tmp/_qnav_test_benchmark.json"
    export_benchmark_json(result, out_path)

    with open(out_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["location"]["name"]
    assert "classical_imu" in payload["metrics"]
    assert "qnav_full_system" in payload["metrics"]
    assert "quantum_enhanced" in payload["improvement_vs_classical_pct"]
    os.remove(out_path)


if __name__ == "__main__":
    test_run_simulation_offline_produces_consistent_result()
    test_gravity_matching_toggle_affects_qnav_only()
    test_configurable_duration_and_gps_cutoff_are_respected()
    test_export_trajectories_csv_writes_expected_columns()
    test_export_benchmark_json_writes_expected_structure()
    print("All runner/export tests passed.")
