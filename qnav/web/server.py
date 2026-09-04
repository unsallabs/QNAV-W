"""
Q-Nav Local Web UI — Backend
================================

A minimal, dependency-free local web server (Python's built-in
`http.server`, no Flask/Django) that serves the single-page web UI and a
small JSON API. The API calls the exact same `run_simulation()` used by
the CLI demo (`qnav.simulation.runner`) — there is only one implementation
of the physics; this module is purely presentation/transport.

Routes
--------
GET  /                    -> the single-page app (qnav/web/static/index.html)
GET  /static/<file>       -> static assets (app.js, style.css)
POST /api/simulate        -> runs a simulation from a JSON config, returns
                              a JSON payload with (decimated) trajectories,
                              metrics, and gravity-match events for the UI
GET  /api/export/csv      -> downloads the last run's trajectories.csv
GET  /api/export/json     -> downloads the last run's benchmark.json

Launch via `python app.py` (see the project root).
"""

import json
import os
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from qnav.simulation.runner import run_simulation
from qnav.simulation.trajectory import local_xy_to_latlon
from qnav.comparison.export import export_trajectories_csv, export_benchmark_json

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Guards the single "last simulation result", used only to serve CSV/JSON
# downloads for whatever was most recently run in the UI. Simulations run
# synchronously per request — this is a local single-user tool, not a
# multi-tenant service, so a simple lock is sufficient.
_state_lock = threading.Lock()
_last_result = None


def _decimate_indices(n, max_points=600):
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points, dtype=int))


def _track_payload(x, y, lat0, lon0, idx):
    lat, lon = local_xy_to_latlon(x[idx], y[idx], lat0, lon0)
    return {"lat": np.round(lat, 7).tolist(), "lon": np.round(lon, 7).tolist()}


def _metrics_payload(m):
    return {
        "label": m.label,
        "rmse_m": round(m.rmse_m, 2),
        "cep50_m": round(m.cep50_m, 2),
        "final_error_m": round(m.final_error_m, 2),
        "max_error_m": round(m.max_error_m, 2),
    }


def build_simulate_response(result, max_points: int = 600) -> dict:
    """Turns a full-resolution SimulationResult into a compact JSON-ready dict for the browser."""
    gt = result.ground_truth
    idx = _decimate_indices(len(gt.t), max_points)
    t_dec = gt.t[idx]

    improvement_q = (1 - result.metrics_quantum.rmse_m / result.metrics_classical.rmse_m) * 100
    improvement_qnav = (1 - result.metrics_qnav.rmse_m / result.metrics_classical.rmse_m) * 100

    gravity_events = []
    for (t, x, y) in result.gravity_match_events:
        lat, lon = local_xy_to_latlon(x, y, result.lat0, result.lon0)
        gravity_events.append({"t": round(t, 1), "lat": round(float(lat), 7), "lon": round(float(lon), 7)})

    return {
        "location": {"name": result.location_name, "lat": result.lat0, "lon": result.lon0},
        "route_source": result.route_source,
        "duration_s": result.duration_s,
        "gps_off_time_s": result.gps_off_time,
        "gravity_matching_enabled": result.gravity_matching_enabled,
        "gravity_matches_applied": result.n_matches_applied,
        "total_distance_m": round(result.total_distance_m, 1),
        "t": np.round(t_dec, 2).tolist(),
        "ground_truth": _track_payload(gt.x, gt.y, result.lat0, result.lon0, idx),
        "classical": _track_payload(result.hist_classical[:, 0], result.hist_classical[:, 1],
                                     result.lat0, result.lon0, idx),
        "quantum": _track_payload(result.hist_quantum[:, 0], result.hist_quantum[:, 1],
                                   result.lat0, result.lon0, idx),
        "qnav": _track_payload(result.hist_qnav[:, 0], result.hist_qnav[:, 1],
                                result.lat0, result.lon0, idx),
        "drift": {
            "classical": np.round(result.metrics_classical.position_error_m[idx], 2).tolist(),
            "quantum": np.round(result.metrics_quantum.position_error_m[idx], 2).tolist(),
            "qnav": np.round(result.metrics_qnav.position_error_m[idx], 2).tolist(),
        },
        "gravity_match_events": gravity_events,
        "metrics": {
            "classical": _metrics_payload(result.metrics_classical),
            "quantum": _metrics_payload(result.metrics_quantum),
            "qnav": _metrics_payload(result.metrics_qnav),
        },
        "improvement_vs_classical_pct": {
            "quantum": round(improvement_q, 1),
            "qnav": round(improvement_qnav, 1),
        },
    }


class QNavRequestHandler(BaseHTTPRequestHandler):
    server_version = "QNavWebUI/1.0"

    def log_message(self, fmt, *args):
        pass  # keep the console output of examples/app.py clean

    # -- helpers ----------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str, download_name: str = None):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, status=404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(body)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        global _last_result

        if self.path == "/" or self.path == "":
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return

        if self.path.startswith("/static/"):
            filename = os.path.basename(self.path)
            content_type = {
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".html": "text/html; charset=utf-8",
            }.get(os.path.splitext(filename)[1], "application/octet-stream")
            self._send_file(os.path.join(STATIC_DIR, filename), content_type)
            return

        if self.path == "/api/export/csv":
            with _state_lock:
                result = _last_result
            if result is None:
                self._send_json({"error": "run a simulation first"}, status=400)
                return
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                export_trajectories_csv(result, tmp_path)
                self._send_file(tmp_path, "text/csv", download_name="qnav_trajectories.csv")
            finally:
                os.remove(tmp_path)
            return

        if self.path == "/api/export/json":
            with _state_lock:
                result = _last_result
            if result is None:
                self._send_json({"error": "run a simulation first"}, status=400)
                return
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                export_benchmark_json(result, tmp_path)
                self._send_file(tmp_path, "application/json", download_name="qnav_benchmark.json")
            finally:
                os.remove(tmp_path)
            return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        global _last_result

        if self.path != "/api/simulate":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            cfg = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON request body"}, status=400)
            return

        try:
            duration = float(cfg.get("duration_s", 900.0))
            gps_cutoff = float(cfg.get("gps_off_time_s", 300.0))
            duration = float(np.clip(duration, 30.0, 3600.0))
            gps_cutoff = float(np.clip(gps_cutoff, 0.0, duration))

            lat = cfg.get("lat")
            lon = cfg.get("lon")
            result = run_simulation(
                location=cfg.get("location") or None,
                lat=float(lat) if lat is not None else None,
                lon=float(lon) if lon is not None else None,
                duration_s=duration,
                gps_off_time=gps_cutoff,
                gravity_matching=bool(cfg.get("gravity_matching", True)),
                use_road_network=bool(cfg.get("use_road_network", True)),
            )
        except Exception as exc:  # noqa: BLE001 - report back to the UI instead of a bare 500
            self._send_json({"error": f"simulation failed: {exc}"}, status=500)
            return

        with _state_lock:
            _last_result = result

        self._send_json(build_simulate_response(result))


def start_server(port: int = 8765, open_browser: bool = True):
    server = ThreadingHTTPServer(("127.0.0.1", port), QNavRequestHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Q-Nav web UI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Q-Nav web UI.")
    finally:
        server.server_close()
