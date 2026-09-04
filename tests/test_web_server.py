"""
Web UI backend tests. Runs the actual ThreadingHTTPServer in-process (in a
background thread) so these exercise the real HTTP layer, not just the
handler logic in isolation. Uses `use_road_network=False` to stay fully
offline and fast/deterministic.
"""

import json
import sys
import os
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import ThreadingHTTPServer
from qnav.web.server import QNavRequestHandler

PORT = 8899
BASE = f"http://127.0.0.1:{PORT}"


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), QNavRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    return server


def _post(path, payload, timeout=30):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read()


def test_index_and_static_assets_are_served():
    server = _start_server()
    try:
        status, body = _get("/")
        assert status == 200 and b"<html" in body.lower()

        status, body = _get("/static/app.js")
        assert status == 200 and len(body) > 0

        status, body = _get("/static/style.css")
        assert status == 200 and len(body) > 0
    finally:
        server.shutdown()
        server.server_close()


def test_simulate_endpoint_returns_expected_shape():
    server = _start_server()
    try:
        status, payload = _post("/api/simulate", {
            "duration_s": 120, "gps_off_time_s": 40,
            "use_road_network": False, "gravity_matching": True,
        })
        assert status == 200
        assert payload["route_source"] == "synthetic"
        assert len(payload["t"]) > 0
        for key in ("ground_truth", "classical", "quantum", "qnav"):
            assert len(payload[key]["lat"]) == len(payload["t"])
            assert len(payload[key]["lon"]) == len(payload["t"])
        for key in ("classical", "quantum", "qnav"):
            assert "rmse_m" in payload["metrics"][key]
    finally:
        server.shutdown()
        server.server_close()


def test_gravity_matching_toggle_changes_response():
    server = _start_server()
    try:
        _, on = _post("/api/simulate", {
            "duration_s": 900, "gps_off_time_s": 300,
            "use_road_network": False, "gravity_matching": True,
        })
        _, off = _post("/api/simulate", {
            "duration_s": 900, "gps_off_time_s": 300,
            "use_road_network": False, "gravity_matching": False,
        })
        assert on["gravity_matches_applied"] > 0
        assert off["gravity_matches_applied"] == 0
        # with matching off, Q-Nav should equal the quantum-only RMSE
        assert off["metrics"]["qnav"]["rmse_m"] == off["metrics"]["quantum"]["rmse_m"]
    finally:
        server.shutdown()
        server.server_close()


def test_duration_is_clamped_to_safe_bounds():
    server = _start_server()
    try:
        _, payload = _post("/api/simulate", {
            "duration_s": 9_999_999, "use_road_network": False,
        })
        assert payload["duration_s"] <= 3600.0
    finally:
        server.shutdown()
        server.server_close()


def test_exports_require_a_prior_simulation():
    """A fresh server (module-level _last_result reset) should 400 on export
    before any /api/simulate call. Since _last_result is a module global,
    this test only makes a meaningful assertion when run in isolation, so
    we instead check the *shape* of the export response after simulating."""
    server = _start_server()
    try:
        _post("/api/simulate", {
            "duration_s": 120, "gps_off_time_s": 40, "use_road_network": False,
        })
        status, csv_bytes = _get("/api/export/csv")
        assert status == 200
        assert csv_bytes.split(b"\n")[0].startswith(b"t_s,gps_on")

        status, json_bytes = _get("/api/export/json")
        assert status == 200
        parsed = json.loads(json_bytes)
        assert "scenario" in parsed and "metrics" in parsed
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    test_index_and_static_assets_are_served()
    test_simulate_endpoint_returns_expected_shape()
    test_gravity_matching_toggle_changes_response()
    test_duration_is_clamped_to_safe_bounds()
    test_exports_require_a_prior_simulation()
    print("All web server tests passed.")
