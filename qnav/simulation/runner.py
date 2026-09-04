"""
Simulation Runner
====================

The single place where the Q-Nav simulation loop lives. Both the CLI demo
(`examples/run_demo.py`) and the local web UI (`qnav/web/server.py`) call
`run_simulation()` so there is exactly one implementation of the physics
loop — sensors, EKF fusion, and gravity map matching are never duplicated
or reimplemented for the web UI.

This module only wires together and configures the unchanged building
blocks from `qnav.sensors`, `qnav.fusion`, and `qnav.gravity`; it does not
alter their physics.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple

import numpy as np

from qnav.sensors.imu_classical import ClassicalIMU, IMUSpec
from qnav.sensors.quantum_accel import QuantumAccelerometer, QuantumAccelSpec
from qnav.sensors.quantum_gyro import QuantumGyroscope, QuantumGyroSpec
from qnav.sensors.quantum_gravimeter import QuantumGravimeter, QuantumGravimeterSpec

from qnav.fusion.state import init_state, init_covariance, IDX_X, IDX_Y
from qnav.fusion.ekf import EKF

from qnav.gravity.gravity_map import GravityMap
from qnav.gravity import map_matching

from qnav.simulation.trajectory import generate_route, GroundTruth, DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME
from qnav.simulation.gps_simulator import GPSSimulator, GPSSpec
from qnav.simulation.location import resolve_location

from qnav.comparison.drift_analysis import compute_drift, DriftMetrics


@dataclass
class SimulationResult:
    ground_truth: GroundTruth
    hist_classical: np.ndarray   # (N, 2)
    hist_quantum: np.ndarray     # (N, 2)
    hist_qnav: np.ndarray        # (N, 2)
    metrics_classical: DriftMetrics
    metrics_quantum: DriftMetrics
    metrics_qnav: DriftMetrics
    gravity_match_events: List[Tuple[float, float, float]]  # (t, x, y) where a fix was applied
    n_matches_applied: int
    location_name: str
    lat0: float
    lon0: float
    dt: float
    duration_s: float
    gps_off_time: float
    gravity_matching_enabled: bool
    route_source: str            # "road" or "synthetic"
    total_distance_m: float


def run_simulation(location: str = None, lat: float = None, lon: float = None,
                    duration_s: float = 900.0, gps_off_time: float = 300.0,
                    dt: float = 0.05, avg_speed: float = 13.0,
                    gravity_matching: bool = True, use_road_network: bool = True,
                    seed: int = 7, verbose: bool = False) -> SimulationResult:
    """
    Runs the full Q-Nav scenario once and returns every array/metric needed
    by both the CLI plots/exports and the web UI's JSON API.

    gravity_matching=False disables the gravity-map-matching correction for
    the "Q-Nav" configuration only; in that case its EKF receives the exact
    same quantum-IMU inputs as the "Quantum-Enhanced" configuration and the
    two should track each other closely — a useful way to see, directly,
    how much of Q-Nav's improvement comes from the gravity map.
    """
    lat0, lon0, location_name = resolve_location(location=location, lat=lat, lon=lon)

    if verbose:
        print(f"Start: {location_name}  (lat={lat0:.5f}, lon={lon0:.5f})  |  GPS OFF @ t = {gps_off_time} s")

    # ------------------------------------------------------------------
    # Gravity anomaly map + ground-truth route (road network by default,
    # falls back to the offline synthetic generator automatically)
    # ------------------------------------------------------------------
    gravity_map = GravityMap(extent_m=20000.0, resolution_m=50.0, seed=42)
    gt = generate_route(duration_s=duration_s, dt=dt, avg_speed=avg_speed, seed=seed,
                         gravity_map=gravity_map, lat0=lat0, lon0=lon0,
                         use_road_network=use_road_network)
    n = len(gt.t)
    gps_off_idx = int(np.clip(gps_off_time / dt, 0, max(n - 1, 0)))
    total_distance = float(np.sum(np.hypot(np.diff(gt.x), np.diff(gt.y))))

    if verbose:
        print(f"Route length: {total_distance/1000:.2f} km, {n} steps, dt={dt}s, "
              f"source={gt.route_source}")

    # ------------------------------------------------------------------
    # Sensors (unchanged models; only instantiated here)
    # ------------------------------------------------------------------
    classical_imu = ClassicalIMU(IMUSpec(), dt=dt, seed=1)
    quantum_accel = QuantumAccelerometer(QuantumAccelSpec(cycle_time=0.1), seed=2)
    quantum_gyro = QuantumGyroscope(QuantumGyroSpec(cycle_time=0.1), seed=3)
    quantum_gravimeter = QuantumGravimeter(QuantumGravimeterSpec(cycle_time=1.0), seed=4)

    gps_sim = GPSSimulator(GPSSpec(noise_std_m=3.0, update_rate_hz=1.0),
                            gps_off_time=gps_off_time, seed=5)

    # ------------------------------------------------------------------
    # EKFs (three configurations, identical fusion architecture — unchanged)
    # ------------------------------------------------------------------
    x0 = init_state(gt.x[0], gt.y[0], gt.vx[0], gt.vy[0], gt.yaw[0])
    P0 = init_covariance(pos_std=0.5, vel_std=0.3, yaw_std=0.02)

    ekf_classical = EKF(x0.copy(), P0.copy())
    ekf_quantum = EKF(x0.copy(), P0.copy())
    ekf_qnav = EKF(x0.copy(), P0.copy())

    H_pos = np.zeros((2, 8))
    H_pos[0, IDX_X] = 1.0
    H_pos[1, IDX_Y] = 1.0
    R_gps = np.diag([gps_sim.spec.noise_std_m ** 2] * 2)

    # ------------------------------------------------------------------
    # Gravity map matching buffer (Q-Nav only, and only if enabled)
    # ------------------------------------------------------------------
    gravity_buffer = deque(maxlen=40)   # (t, qnav_x, qnav_y, measured_g)
    last_match_time = -np.inf
    match_interval_s = 40.0
    n_matches_applied = 0
    gravity_match_events: List[Tuple[float, float, float]] = []

    hist_classical = np.zeros((n, 2))
    hist_quantum = np.zeros((n, 2))
    hist_qnav = np.zeros((n, 2))

    classical_accel_std_nominal = classical_imu._accel_vrw / np.sqrt(dt)
    classical_gyro_std_nominal = classical_imu._gyro_arw / np.sqrt(dt)

    # ------------------------------------------------------------------
    # Main simulation loop (unchanged sensor/EKF/gravity-matching logic)
    # ------------------------------------------------------------------
    for i in range(n):
        t = gt.t[i]
        true_accel = gt.accel_body[i]
        true_yaw_rate = gt.yaw_rate[i]
        true_x, true_y = gt.x[i], gt.y[i]
        true_g = gt.gravity_true[i]

        # --- Classical IMU ---
        meas_a_c, meas_g_c = classical_imu.measure(true_accel, true_yaw_rate)
        ekf_classical.predict(meas_a_c, meas_g_c, dt,
                               classical_accel_std_nominal, classical_gyro_std_nominal)

        # --- Quantum IMU (shared by both "quantum-enhanced" and "Q-Nav" configs) ---
        meas_a_q, is_new_a = quantum_accel.measure(true_accel, dt)
        meas_g_q, is_new_g = quantum_gyro.measure(true_yaw_rate, dt)
        noise_a = quantum_accel.accel_noise_std if is_new_a else 1e-9
        noise_g = quantum_gyro.gyro_noise_std if is_new_g else 1e-9

        ekf_quantum.predict(meas_a_q, meas_g_q, dt, noise_a, noise_g,
                             ba_rw_std=1e-8, bg_rw_std=1e-9)
        ekf_qnav.predict(meas_a_q, meas_g_q, dt, noise_a, noise_g,
                          ba_rw_std=1e-8, bg_rw_std=1e-9)

        # --- GPS update (if available, applied identically to all three EKFs) ---
        fix = gps_sim.maybe_get_fix(t, true_x, true_y)
        if fix is not None:
            ekf_classical.update(fix, H_pos, R_gps)
            ekf_quantum.update(fix, H_pos, R_gps)
            ekf_qnav.update(fix, H_pos, R_gps)

        # --- Quantum Gravimeter + Gravity Map Matching (Q-Nav only, optional) ---
        if gravity_matching:
            g_meas, is_new_gravity = quantum_gravimeter.measure(true_g, dt)
            if is_new_gravity:
                qx, qy = ekf_qnav.position()
                gravity_buffer.append((t, qx, qy, g_meas))

                if (t >= gps_off_time and t - last_match_time >= match_interval_s
                        and len(gravity_buffer) >= 20):
                    last_match_time = t
                    cur_x, cur_y = ekf_qnav.position()
                    offsets = [(bx - cur_x, by - cur_y) for (_, bx, by, _) in gravity_buffer]
                    measured_seq = [bg for (_, _, _, bg) in gravity_buffer]

                    pos_std = float(np.sqrt(max(ekf_qnav.P[IDX_X, IDX_X], ekf_qnav.P[IDX_Y, IDX_Y])))
                    search_radius = float(np.clip(3.0 * pos_std, 150.0, 900.0))

                    result = map_matching.match(
                        gravity_map, measured_seq, offsets,
                        search_center_xy=(cur_x, cur_y),
                        search_radius_m=search_radius, search_step_m=50.0,
                    )
                    if result.matched:
                        z = np.array([result.x, result.y])
                        R_grav = np.diag([result.confidence_std_m ** 2] * 2)
                        ekf_qnav.update(z, H_pos, R_grav)
                        n_matches_applied += 1
                        gravity_match_events.append((float(t), float(result.x), float(result.y)))

        hist_classical[i] = ekf_classical.position()
        hist_quantum[i] = ekf_quantum.position()
        hist_qnav[i] = ekf_qnav.position()

        if verbose and i % 4000 == 0:
            print(f"  t={t:6.1f}s  |  classical_err={np.hypot(hist_classical[i,0]-true_x, hist_classical[i,1]-true_y):7.1f}m"
                  f"  quantum_err={np.hypot(hist_quantum[i,0]-true_x, hist_quantum[i,1]-true_y):7.1f}m"
                  f"  qnav_err={np.hypot(hist_qnav[i,0]-true_x, hist_qnav[i,1]-true_y):7.1f}m")

    # ------------------------------------------------------------------
    # Metrics (unchanged)
    # ------------------------------------------------------------------
    m_classical = compute_drift(gt.t, hist_classical[:, 0], hist_classical[:, 1],
                                 gt.x, gt.y, "Classical IMU (Dead Reckoning)", gps_off_idx)
    m_quantum = compute_drift(gt.t, hist_quantum[:, 0], hist_quantum[:, 1],
                               gt.x, gt.y, "Quantum-Enhanced IMU", gps_off_idx)
    qnav_label = "Q-Nav (Quantum + Gravity Map)" if gravity_matching else "Q-Nav (Gravity Matching OFF)"
    m_qnav = compute_drift(gt.t, hist_qnav[:, 0], hist_qnav[:, 1],
                            gt.x, gt.y, qnav_label, gps_off_idx)

    return SimulationResult(
        ground_truth=gt,
        hist_classical=hist_classical,
        hist_quantum=hist_quantum,
        hist_qnav=hist_qnav,
        metrics_classical=m_classical,
        metrics_quantum=m_quantum,
        metrics_qnav=m_qnav,
        gravity_match_events=gravity_match_events,
        n_matches_applied=n_matches_applied,
        location_name=location_name,
        lat0=lat0,
        lon0=lon0,
        dt=dt,
        duration_s=duration_s,
        gps_off_time=gps_off_time,
        gravity_matching_enabled=gravity_matching,
        route_source=gt.route_source,
        total_distance_m=total_distance,
    )
