"""
GPS Simulator
================

Produces GPS fixes from ground-truth position using realistic
consumer-grade noise (~2-5 m 1-sigma, 1 Hz update). After the configured
`gps_off_time`, no fixes are produced at all -> the "GPS turned off"
scenario.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class GPSSpec:
    noise_std_m: float = 3.0
    update_rate_hz: float = 1.0


class GPSSimulator:
    def __init__(self, spec: GPSSpec = None, gps_off_time: float = None, seed: int = None):
        self.spec = spec or GPSSpec()
        self.gps_off_time = gps_off_time
        self.rng = np.random.default_rng(seed)
        self._last_fix_time = -np.inf

    def maybe_get_fix(self, t: float, true_x: float, true_y: float):
        """Checks whether a GPS fix is available at time t. Returns (x_meas, y_meas) if so, else None."""
        if self.gps_off_time is not None and t >= self.gps_off_time:
            return None

        if t - self._last_fix_time < 1.0 / self.spec.update_rate_hz:
            return None

        self._last_fix_time = t
        noise = self.rng.normal(0, self.spec.noise_std_m, size=2)
        return np.array([true_x, true_y]) + noise
