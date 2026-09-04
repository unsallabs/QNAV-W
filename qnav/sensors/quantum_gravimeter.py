"""
Quantum Gravimeter Model (Atom Interferometer)
====================================================================

Physical principle
--------------------
A quantum gravimeter uses the exact same dphi = k_eff * g * T^2 equation as
the quantum accelerometer; the only difference is that the measured
acceleration is gravity acting on freely-falling atoms, typically along the
vertical axis, with a much longer T and higher atom number, targeting
micro-Gal precision (1 uGal = 1e-8 m/s^2).

This precision level is enough to detect small gravity anomalies caused by
Earth's local mass distribution (geology, topography, water bodies) — which
is the basis of "gravity map matching" navigation: the measured g(t) signal
is correlated against a pre-known gravity-anomaly map to derive an absolute
position fix.

A study published on August 26, 2026 demonstrated this approach in the real
world using a mobile quantum gravimeter + classical IMU over an 83 km
GNSS-free marine route (see project README).

References: Peters, Chung & Chu (2001) absolute atom gravimeter;
Bidel et al. (2018) mobile marine/airborne quantum gravimetry.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QuantumGravimeterSpec:
    atom_number: float = 1e7           # N, gravimeters typically use a higher atom number
    pulse_separation_T: float = 0.3     # s, longer free-fall -> higher sensitivity
    effective_wavevector: float = 1.61e7  # rad/m
    cycle_time: float = 1.0             # s, measurement cycle (gravimeters run slower)
    measurement_noise_floor: float = 3e-8  # m/s^2 (~3 microGal), electronics/vibration noise floor


class QuantumGravimeter:
    """Atom-interferometer simulator that measures absolute vertical gravity."""

    def __init__(self, spec: QuantumGravimeterSpec = None, seed: int = None):
        self.spec = spec or QuantumGravimeterSpec()
        self.rng = np.random.default_rng(seed)

        self.phase_noise_std = 1.0 / np.sqrt(self.spec.atom_number)
        shot_noise_limited_std = self.phase_noise_std / (
            self.spec.effective_wavevector * self.spec.pulse_separation_T ** 2
        )
        # In real systems, the noise floor (vibration isolation, etc.) is usually
        # dominant over shot noise; combine them in quadrature.
        self.total_noise_std = float(
            np.sqrt(shot_noise_limited_std ** 2 + self.spec.measurement_noise_floor ** 2)
        )
        self._time_since_last_sample = 0.0

    def measure(self, true_gravity: float, dt: float):
        """
        Samples the true local gravity magnitude (m/s^2, absolute value ~9.78-9.83).
        Returns: (measured_gravity, is_new_sample: bool)
        """
        self._time_since_last_sample += dt

        if self._time_since_last_sample < self.spec.cycle_time:
            return getattr(self, "_last_measurement", true_gravity), False

        self._time_since_last_sample = 0.0
        measured = true_gravity + self.rng.normal(0, self.total_noise_std)
        self._last_measurement = measured
        return measured, True
