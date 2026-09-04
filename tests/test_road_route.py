import json
import sys, os
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.simulation.road_route import fetch_road_route, RoadRoutingError
from qnav.simulation import trajectory


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _osrm_ok_payload():
    # A tiny fake "road" geometry: a short zigzag near San Francisco.
    coords = [
        [-122.4194, 37.7749],
        [-122.4180, 37.7755],
        [-122.4170, 37.7760],
        [-122.4194, 37.7749],
    ]
    return {"code": "Ok", "routes": [{"geometry": {"coordinates": coords}}]}


def test_fetch_road_route_parses_coordinates_as_lat_lon():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_osrm_ok_payload())):
        route = fetch_road_route(37.7749, -122.4194, target_distance_m=2000, seed=1)

    assert len(route) == 4
    # coords were [lon, lat]; fetch_road_route must return (lat, lon)
    lat0, lon0 = route[0]
    assert abs(lat0 - 37.7749) < 1e-6
    assert abs(lon0 - (-122.4194)) < 1e-6


def test_fetch_road_route_raises_on_no_route():
    payload = {"code": "NoRoute", "routes": []}
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        try:
            fetch_road_route(37.7749, -122.4194)
            assert False, "expected RoadRoutingError"
        except RoadRoutingError:
            pass


def test_fetch_road_route_raises_on_network_failure():
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        try:
            fetch_road_route(37.7749, -122.4194)
            assert False, "expected RoadRoutingError"
        except RoadRoutingError:
            pass


def test_generate_route_falls_back_to_synthetic_when_routing_fails():
    """generate_route() must never crash if OSRM is unreachable — it should
    transparently fall back to the offline synthetic generator."""
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        gt = trajectory.generate_route(duration_s=30, dt=0.1, use_road_network=True, seed=1)
    assert gt.route_source == "synthetic"
    assert len(gt.t) > 0


def test_generate_route_from_path_resamples_a_simple_square_loop():
    """Offline test of the arc-length resampling logic using a synthetic
    polyline (no network involved) — validates the road-route conversion
    pipeline independent of OSRM availability."""
    # A 400m x 400m square loop, in local ENU meters. duration_s is generous
    # enough for more than one full lap even with curvature-limited
    # cornering speed (real vehicles slow for the four 90-degree corners).
    path_xy = [(0, 0), (400, 0), (400, 400), (0, 400), (0, 0)]
    gt = trajectory.generate_route_from_path(path_xy, duration_s=500, dt=0.5,
                                              avg_speed=8.0, seed=3, route_source="road")

    assert gt.route_source == "road"
    # Should stay within (and reasonably near) the bounding box of the path.
    assert gt.x.min() >= -50 and gt.x.max() <= 450
    assert gt.y.min() >= -50 and gt.y.max() <= 450
    # Tiling (lap repetition) must actually be happening: the vehicle should
    # cover well over one full loop perimeter (1600m) in 500s.
    distance_traveled = np.sum(np.hypot(np.diff(gt.x), np.diff(gt.y)))
    assert distance_traveled > 1600.0


def test_generate_route_from_path_has_physically_plausible_kinematics():
    """
    Regression test for a real bug: real road polylines have sharp,
    near-instantaneous corners at intersections. Differentiating the raw
    arc-length-resampled path without smoothing produced unphysical
    yaw-rate/acceleration spikes (tens of rad/s, tens of g) at every corner,
    which blew up the sensor models and made the EKF diverge (RMSE in the
    thousands of meters instead of tens). generate_route_from_path must
    keep derived kinematics within plausible vehicle limits even for sharp
    corners.
    """
    # A deliberately harsh double 90-degree-turn corner (worse than any real
    # OSRM intersection geometry) as a stress test.
    path_xy = [(0, 0), (500, 0), (500, 1), (1000, 1), (1000, 500)]
    gt = trajectory.generate_route_from_path(path_xy, duration_s=100, dt=0.05,
                                              avg_speed=13.0, seed=1)

    assert np.max(np.abs(gt.yaw_rate)) < 3.0, "yaw rate spike at a sharp corner is unrealistic"
    assert np.max(np.abs(gt.accel_body)) < 20.0, "acceleration spike at a sharp corner is unrealistic"


def test_latlon_local_roundtrip():
    lat0, lon0 = 51.5074, -0.1278  # London
    x, y = trajectory.latlon_to_local(51.51, -0.12, lat0, lon0)
    lat_back, lon_back = trajectory.local_xy_to_latlon(x, y, lat0, lon0)
    assert abs(lat_back - 51.51) < 1e-9
    assert abs(lon_back - (-0.12)) < 1e-9


def test_low_speed_heading_is_held_not_noisy():
    """
    Regression test for a real bug: heading computed as atan2(vy, vx) is
    numerically unstable whenever speed is very small (curvature-limited
    slowdowns, or braking to a stop), producing spurious ~90-degree jumps
    that corrupted yaw_rate and made zero-noise dead reckoning drift by
    hundreds of meters even with perfect sensors. Re-integrating yaw_rate
    must reproduce the reported yaw closely everywhere.
    """
    # A path with a very tight corner that forces the curvature-limited
    # speed profile down to a near-crawl for a stretch.
    path_xy = [(0, 0), (300, 0), (300, 2), (0, 2), (0, 0)]
    gt = trajectory.generate_route_from_path(path_xy, duration_s=200, dt=0.05,
                                              avg_speed=13.0, seed=1, route_source="road")
    reintegrated_yaw = gt.yaw[0] + np.concatenate(([0.0], np.cumsum(gt.yaw_rate[:-1] * 0.05)))
    assert np.max(np.abs(reintegrated_yaw - gt.yaw)) < 0.5


def test_full_pipeline_on_a_closed_loop_road_route_matches_expected_physics():
    """
    End-to-end regression test for the real-world bug reported from the web
    UI: on an actual OSRM road route (always a closed loop — see
    road_route._loop_waypoints), drift must land in a physically sane range
    (tens of meters, not thousands), and the quantum-enhanced configuration
    must never be worse than classical, since it only ever receives
    strictly lower-noise measurements through the identical EKF.
    """
    from qnav.simulation.runner import run_simulation

    pts = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for dx, dy in [(600, 0), (0, 600), (-600, 0), (0, -600)]:  # closed rectangle
        steps = max(int(np.hypot(dx, dy) / 10), 2)
        for i in range(1, steps + 1):
            pts.append((x + dx * i / steps, y + dy * i / steps))
        x, y = x + dx, y + dy
    pts = np.array(pts)

    lat0, lon0 = 37.7749, -122.4194
    lat, lon = trajectory.local_xy_to_latlon(pts[:, 0], pts[:, 1], lat0, lon0)
    fake_route = list(zip(lat.tolist(), lon.tolist()))

    with patch("qnav.simulation.road_route.fetch_road_route", return_value=fake_route):
        result = run_simulation(lat=lat0, lon=lon0, duration_s=900, gps_off_time=300,
                                 use_road_network=True, gravity_matching=True, seed=7)

    assert result.route_source == "road"
    assert result.metrics_classical.rmse_m < 100.0, "classical drift is unrealistically large"
    assert result.metrics_quantum.rmse_m <= result.metrics_classical.rmse_m, \
        "quantum sensors must never do worse than classical through the same EKF"
    assert result.metrics_qnav.rmse_m <= result.metrics_quantum.rmse_m * 1.05, \
        "gravity map matching must not make Q-Nav meaningfully worse than quantum-only"


if __name__ == "__main__":
    test_fetch_road_route_parses_coordinates_as_lat_lon()
    test_fetch_road_route_raises_on_no_route()
    test_fetch_road_route_raises_on_network_failure()
    test_generate_route_falls_back_to_synthetic_when_routing_fails()
    test_generate_route_from_path_resamples_a_simple_square_loop()
    test_generate_route_from_path_has_physically_plausible_kinematics()
    test_latlon_local_roundtrip()
    test_low_speed_heading_is_held_not_noisy()
    test_full_pipeline_on_a_closed_loop_road_route_matches_expected_physics()
    print("All road-route tests passed.")
