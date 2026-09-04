"""
Q-Nav Demo: comparing three navigation configurations on a route starting
at a configurable worldwide location (default: San Francisco, USA), with
GPS turned off partway through.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --location "Tokyo, Japan"
    python examples/run_demo.py --lat 48.8566 --lon 2.3522
    python examples/run_demo.py --duration 600 --gps-cutoff 180
    python examples/run_demo.py --no-gravity-matching
    python examples/run_demo.py --no-road-network

Location resolution (see qnav.simulation.location for details):
    --lat/--lon   : fully offline, exact, no network access required.
    --location    : optional convenience — resolved to lat/lon via the
                    free OpenStreetMap Nominatim API (requires network).
                    Falls back to the default location if geocoding fails.
    (neither)     : defaults to San Francisco, USA.

Route source (see qnav.simulation.trajectory / road_route for details):
    By default the ground-truth route is fetched from the real
    OpenStreetMap road network (via OSRM), so the simulated vehicle
    follows actual drivable roads and never crosses buildings, parks, or
    water. If that's unavailable (no network, no route found), Q-Nav
    automatically falls back to an offline analytic route generator.
    Pass --no-road-network to skip the network attempt entirely.

None of this affects the underlying physics: sensors, INS mechanization,
EKF fusion, and gravity map matching all operate in a location-agnostic
local tangent-plane frame and are entirely unchanged by any of these
options.

Configurations
----------------
1. Classical IMU      : MEMS accelerometer + gyro only; pure dead-reckoning once GPS is off
2. Quantum-Enhanced   : atom-interferometer based accel + gyro; dead-reckoning once GPS is off
3. Q-Nav (Full System): quantum IMU + quantum gravimeter + gravity map matching corrections
                        (disable with --no-gravity-matching to see the quantum IMU alone)

Outputs (written to output/):
    trajectory_comparison.png   - trajectory comparison plot
    drift_comparison.png        - drift over time
    interactive_map.html        - interactive route map on real OpenStreetMap tiles, centered on the chosen location
    summary.txt                 - numeric summary table
    trajectories.csv            - per-timestep positions (local meters + lat/lon) for every configuration
    benchmark.json              - numeric comparison + run metadata
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.simulation.runner import run_simulation
from qnav.comparison.drift_analysis import summarize
from qnav.comparison.export import export_trajectories_csv, export_benchmark_json
from qnav.viz.map_view import plot_trajectories, plot_drift, export_leaflet_map


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-Nav: quantum-inertial GPS-denied navigation demo (worldwide)."
    )
    parser.add_argument("--location", type=str, default=None,
                         help="Place name to geocode via OpenStreetMap Nominatim, "
                              "e.g. 'Tokyo, Japan' (requires network; optional).")
    parser.add_argument("--lat", type=float, default=None,
                         help="Latitude of the start point (offline, no network needed).")
    parser.add_argument("--lon", type=float, default=None,
                         help="Longitude of the start point (offline, no network needed).")
    parser.add_argument("--duration", type=float, default=900.0,
                         help="Total route duration in seconds (default: 900).")
    parser.add_argument("--gps-cutoff", type=float, default=300.0,
                         help="Time in seconds at which GPS is turned off (default: 300).")
    parser.add_argument("--no-gravity-matching", action="store_true",
                         help="Disable gravity-map-matching corrections for Q-Nav "
                              "(it then behaves like the Quantum-Enhanced configuration).")
    parser.add_argument("--no-road-network", action="store_true",
                         help="Skip real road-network routing (OSRM) and use the offline "
                              "synthetic route generator directly.")
    return parser.parse_args()


def run(location: str = None, lat: float = None, lon: float = None,
        duration: float = 900.0, gps_cutoff: float = 300.0,
        gravity_matching: bool = True, use_road_network: bool = True):
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Q-NAV: Quantum-Inertial Navigation Prototype")
    print("=" * 70)

    result = run_simulation(
        location=location, lat=lat, lon=lon,
        duration_s=duration, gps_off_time=gps_cutoff,
        gravity_matching=gravity_matching, use_road_network=use_road_network,
        verbose=True,
    )

    print(f"\nGravity map matching applied {result.n_matches_applied} times.")

    summary_text = summarize([result.metrics_classical, result.metrics_quantum, result.metrics_qnav])
    print("\n" + summary_text)

    improvement_q = (1 - result.metrics_quantum.rmse_m / result.metrics_classical.rmse_m) * 100
    improvement_qnav = (1 - result.metrics_qnav.rmse_m / result.metrics_classical.rmse_m) * 100
    print(f"\nQuantum-Enhanced reduced RMSE by {improvement_q:.1f}% vs. classical IMU.")
    print(f"Q-Nav reduced RMSE by {improvement_qnav:.1f}% vs. classical IMU.")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Q-NAV SIMULATION SUMMARY\n")
        f.write(f"Location: {result.location_name} (lat={result.lat0:.5f}, lon={result.lon0:.5f})\n")
        f.write(f"Route source: {result.route_source}\n")
        f.write(f"Route length: {result.total_distance_m/1000:.2f} km, duration {result.duration_s:.0f}s, "
                f"GPS OFF @ {result.gps_off_time:.0f}s\n")
        f.write(f"Gravity map matching: {'ON' if result.gravity_matching_enabled else 'OFF'}\n\n")
        f.write(summary_text + "\n\n")
        f.write(f"Quantum-Enhanced RMSE improvement vs classical: {improvement_q:.1f}%\n")
        f.write(f"Q-Nav RMSE improvement vs classical: {improvement_qnav:.1f}%\n")
        f.write(f"Gravity map matching corrections applied: {result.n_matches_applied}\n")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    gt = result.ground_truth
    tracks = {
        "Classical IMU (Dead Reckoning)": (result.hist_classical[:, 0], result.hist_classical[:, 1]),
        "Quantum-Enhanced IMU": (result.hist_quantum[:, 0], result.hist_quantum[:, 1]),
        result.metrics_qnav.label: (result.hist_qnav[:, 0], result.hist_qnav[:, 1]),
    }
    gps_off_idx = int(min(result.gps_off_time / result.dt, len(gt.t) - 1))
    gps_off_xy = (gt.x[gps_off_idx], gt.y[gps_off_idx])

    plot_trajectories(gt, tracks, gps_off_xy=gps_off_xy, zoom_from_idx=gps_off_idx,
                       start_label=f"Start ({result.location_name})",
                       save_path=os.path.join(OUTPUT_DIR, "trajectory_comparison.png"))
    plot_drift(gt.t, [result.metrics_classical, result.metrics_quantum, result.metrics_qnav],
               gps_off_time=result.gps_off_time,
               save_path=os.path.join(OUTPUT_DIR, "drift_comparison.png"))
    export_leaflet_map(gt, tracks, save_path=os.path.join(OUTPUT_DIR, "interactive_map.html"),
                        lat0=result.lat0, lon0=result.lon0, location_name=result.location_name)

    # ------------------------------------------------------------------
    # CSV / JSON export
    # ------------------------------------------------------------------
    export_trajectories_csv(result, os.path.join(OUTPUT_DIR, "trajectories.csv"))
    export_benchmark_json(result, os.path.join(OUTPUT_DIR, "benchmark.json"))

    print(f"\nOutputs written to '{OUTPUT_DIR}'.")
    print(f"Total runtime: {time.time() - t_start:.1f} s")
    return result


if __name__ == "__main__":
    args = parse_args()
    run(location=args.location, lat=args.lat, lon=args.lon,
        duration=args.duration, gps_cutoff=args.gps_cutoff,
        gravity_matching=not args.no_gravity_matching,
        use_road_network=not args.no_road_network)
