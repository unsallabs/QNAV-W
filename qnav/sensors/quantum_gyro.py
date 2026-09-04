"""
Quantum Gyroscope Model (Atom Interferometer / Sagnac Effect)
====================================================================

Physical principle
--------------------
When an atom cloud is guided around a closed-loop interferometer path in a
rotating reference frame (angular rate Omega), the two paths accumulate an
extra phase difference — the atomic Sagnac effect:

    dphi_sagnac = (2 * m_atom / hbar) * A_enclosed * Omega

    m_atom      : atomic mass (kg)
    hbar        : reduced Planck constant
    A_enclosed  : area enclosed by the interferometer (m^2)
    Omega       : measured angular rate (rad/s)

Atom-Sagnac interferometers have a much higher intrinsic sensitivity than
classical (fiber-optic) Sagnac gyroscopes due to the mass/wavelength ratio
(m_atom*c^2 >> photon energy), which is why atom gyros can in principle
reach much lower bias drift.

This module follows the same approach as the quantum accelerometer module:
shot-noise limited precision + very slow residual systematic drift + low
bandwidth (cycle time).

References: Gustavson, Bouyer & Kasevich (1997) atom interferometer
gyroscope; Imperial College London / Southampton quantum inertial sensing
research.
"""

from dataclasses import dataclass
import numpy as np

HBAR = 1.054571817e-34  # J*s
RB87_MASS = 1.4432e-25   # kg, Rubidium-87 atomic mass


@dataclass
class QuantumGyroSpec:
    atom_number: float = 1e6           # N, number of atoms per measurement cloud
    enclosed_area: float = 4e-4         # m^2, interferometer loop area (typical lab-scale ~ cm^2)
    cycle_time: float = 0.1             # s, measurement cycle
    residual_bias: float = 1e-7         # rad/s, remaining systematic error
    bias_corr_time: float = 3600.0      # s, very slow residual drift


class QuantumGyroscope:
    """Atom-Sagnac interferometer based gyroscope simulator."""

    def __init__(self, spec: QuantumGyroSpec = None, seed: int = None):
        self.spec = spec or QuantumGyroSpec()
        self.rng = np.random.default_rng(seed)

        self.phase_noise_std = 1.0 / np.sqrt(self.spec.atom_number)

        # Sagnac scale factor: dphi = scale_factor * Omega
        self.scale_factor = (2 * RB87_MASS / HBAR) * self.spec.enclosed_area

        # How phase noise translates into angular rate noise
        self.gyro_noise_std = self.phase_noise_std / self.scale_factor

        self.bias = self.rng.normal(0, self.spec.residual_bias)
        self._time_since_last_sample = 0.0

    def _propagate_bias(self, dt: float):
        beta = np.exp(-dt / self.spec.bias_corr_time)
        sigma = self.spec.residual_bias * np.sqrt(max(1 - beta ** 2, 0))
        self.bias = self.bias * beta + self.rng.normal(0, sigma)

    def measure(self, true_yaw_rate: float, dt: float):
        """
        Samples the true angular rate (rad/s).
        Returns: (measured_yaw_rate, is_new_sample: bool)
        """
        self._time_since_last_sample += dt

        if self._time_since_last_sample < self.spec.cycle_time:
            return getattr(self, "_last_measurement", true_yaw_rate), False

        self._time_since_last_sample = 0.0
        self._propagate_bias(self.spec.cycle_time)

        true_phase = self.scale_factor * true_yaw_rate
        measured_phase = true_phase + self.rng.normal(0, self.phase_noise_std)
        measured_yaw_rate = measured_phase / self.scale_factor + self.bias

        self._last_measurement = measured_yaw_rate
        return measured_yaw_rate, True
