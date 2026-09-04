"""
Q-Nav Web UI launcher.

Usage:
    python app.py                # serves on http://127.0.0.1:8765 and opens a browser tab
    python app.py --port 9000    # use a different port
    python app.py --no-browser   # don't auto-open a browser tab

The web UI lets you configure location, GPS cutoff, duration, and
gravity-map matching (on/off), then shows the reference route plus the
Classical IMU, Quantum-Enhanced, and Q-Nav trajectories on a live map with
a lightweight playback animation, a drift chart, and CSV/JSON export.

It calls the exact same simulation code as `examples/run_demo.py`
(`qnav.simulation.runner.run_simulation`) — there is only one
implementation of the sensors/EKF/gravity-matching physics.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qnav.web.server import start_server


def main():
    parser = argparse.ArgumentParser(description="Launch the Q-Nav local web UI.")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765).")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab.")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
