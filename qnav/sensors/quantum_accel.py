"""
Quantum Accelerometer Model (Atom Interferometer / Mach-Zehnder)
====================================================================

Physical principle
--------------------
A cloud of laser-cooled atoms (typically Rb-87) is split into two paths
by a sequence of Raman laser pulses. Under acceleration, the two paths
accumulate different quantum phases. A third pulse recombines the paths,
and the resulting interference pattern (the fraction of atoms in each
output port) depends on the accumulated phase difference:

    dphi = k_eff * a * T^2

    k_eff : effective laser wavevector (rad/m), for a two-photon Raman
            transition: k_eff = 4*pi / lambda_eff
    a     : measured acceleration (m/s^2)
    T     : free-fall time between pulses (s)

This equation is the fundamental working principle of atom-interferometer
based accelerometers (e.g. AOSense, Muquans, Imperial College London
AI-sensors).

Quantum-limited noise (Standard Quantum Limit)
-------------------------------------------------
With N independent (uncorrelated) atoms, the phase estimation precision
is limited by atomic shot noise:

    dphi_min = 1 / sqrt(N)

Which translates into the acceleration measurement noise:

    da = dphi_min / (k_eff * T^2) = 1 / (sqrt(N) * k_eff * T^2)

Key difference vs classical MEMS
------------------------------------
The "zero point" of an atom interferometer is set by the internal atomic
transition frequency (an atomic-clock reference) — NOT a mechanical
structure that drifts with temperature/aging like a MEMS sensor. As a
result, a quantum accelerometer has **near-negligible long-term bias
drift**; the real constraint instead comes from the low bandwidth and
dead-time caused by the measurement cycle (the atom cloud has to be
re-prepared/re-cooled after every measurement).

This module simulates these two real physical characteristics:
  1. Shot-noise limited white noise (high precision, low bias drift)
  2. Low bandwidth / update rate (T_cycle ~ 0.1-1 s typical)

References: Kasevich & Chu (1991) Mach-Zehnder atom interferometer;
Peters, Chung & Chu (2001) absolute gravimeter; Imperial College London
quantum accelerometer research.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QuantumAccelSpec:
    atom_number: float = 1e6          # N, number of atoms per measurement cloud
    pulse_separation_T: float = 0.1    # s, time between Raman pulses (free-fall time)
    effective_wavevector: float = 1.61e7  # rad/m, k_eff = 4*pi/lambda (Rb-87, lambda=780nm, 2-photon)
    cycle_time: float = 0.1            # s, measurement cycle (atom prep+measure), sets bandwidth
    residual_bias: float = 1e-6        # m/s^2, remaining systematic error (AC Stark shift, etc.)
    bias_corr_time: float = 3600.0     # s, very slow residual drift (~10-100x slower than classical IMU)


class QuantumAccelerometer:
    """
    Atom-interferometer based accelerometer simulator.
    Physical equation: dphi = k_eff * a * T^2  ->  a_measured = (dphi_true + dphi_shot) / (k_eff * T^2)
    """

    def __init__(self, spec: QuantumAccelSpec = None, seed: int = None):
        self.spec = spec or QuantumAccelSpec()
        self.rng = np.random.default_rng(seed)

        # Standard Quantum Limit (shot-noise limited) phase precision
        self.phase_noise_std = 1.0 / np.sqrt(self.spec.atom_number)

        # How phase noise translates into acceleration noise: da = dphi / (k_eff * T^2)
        self.accel_noise_std = self.phase_noise_std / (
            self.spec.effective_wavevector * self.spec.pulse_separation_T ** 2
        )

        # Very slow residual systematic drift (2 axes: x, y)
        self.bias = self.rng.normal(0, self.spec.residual_bias, size=2)

        self._time_since_last_sample = 0.0

    def bandwidth_hz(self) -> float:
        """Effective bandwidth resulting from the measurement cycle."""
        return 1.0 / self.spec.cycle_time

    def _propagate_bias(self, dt: float):
        beta = np.exp(-dt / self.spec.bias_corr_time)
        sigma = self.spec.residual_bias * np.sqrt(max(1 - beta ** 2, 0))
        self.bias = self.bias * beta + self.rng.normal(0, sigma, size=2)

    def measure(self, true_accel_xy: np.ndarray, dt: float):
        """
        Samples the true acceleration (m/s^2). If called more often than the
        cycle_time, holds the last valid measurement (real dead-time behavior
        of atom-interferometer hardware).

        Returns: (measured_accel[2], is_new_sample: bool)
        """
        self._time_since_last_sample += dt

        if self._time_since_last_sample < self.spec.cycle_time:
            return getattr(self, "_last_measurement", np.asarray(true_accel_xy, dtype=float)), False

        self._time_since_last_sample = 0.0
        self._propagate_bias(self.spec.cycle_time)

        # Simulate the shot-noise-limited phase measurement via the true phase
        true_phase = self.spec.effective_wavevector * np.asarray(true_accel_xy, dtype=float) * self.spec.pulse_separation_T ** 2
        measured_phase = true_phase + self.rng.normal(0, self.phase_noise_std, size=2)

        measured_accel = measured_phase / (self.spec.effective_wavevector * self.spec.pulse_separation_T ** 2)
        measured_accel = measured_accel + self.bias

        self._last_measurement = measured_accel
        return measured_accel, True
