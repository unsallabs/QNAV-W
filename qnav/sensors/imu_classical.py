"""
Classical MEMS IMU Model
==========================

Models realistic MEMS accelerometer/gyroscope behavior using Allan-variance
based noise components:

  1. Velocity/Angle Random Walk (VRW/ARW) -> white noise, grows with sqrt(dt)
  2. Bias Instability                     -> slow, correlated (Gauss-Markov) drift
  3. The bias itself also random-walks over time

These components match the parameters given in real IMU datasheets
(e.g. ADIS16495, STIM300) for "in-run bias stability" and
"velocity random walk". Reference: IEEE Std 952-2020 (Inertial Sensor
Terminology), Allan (1966).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IMUSpec:
    """Datasheet-style noise parameters (SI units)."""
    accel_bias_instability: float = 5e-4        # m/s^2  (~50 micro-g, typical consumer MEMS)
    accel_velocity_random_walk: float = 0.06     # m/s/sqrt(hr)
    accel_bias_corr_time: float = 300.0          # s, Gauss-Markov correlation time

    gyro_bias_instability: float = 5e-5          # rad/s (~10 deg/hr)
    gyro_angle_random_walk: float = 0.003        # rad/sqrt(hr)
    gyro_bias_corr_time: float = 300.0           # s


class ClassicalIMU:
    """
    Strapdown IMU simulator: given true (ground-truth) acceleration/angular
    rate, produces noisy + biased measurement output.
    """

    def __init__(self, spec: IMUSpec = None, dt: float = 0.01, seed: int = None):
        self.spec = spec or IMUSpec()
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Convert random walk densities from m/(s*sqrt(hr)) -> m/(s*sqrt(s))
        self._accel_vrw = self.spec.accel_velocity_random_walk / 60.0
        self._gyro_arw = self.spec.gyro_angle_random_walk / 60.0

        self.accel_bias = self.rng.normal(0, self.spec.accel_bias_instability, size=2)
        self.gyro_bias = self.rng.normal(0, self.spec.gyro_bias_instability)

    def _propagate_bias(self):
        """1st order Gauss-Markov process: b_{k+1} = b_k * exp(-dt/tau) + w_k"""
        beta_a = np.exp(-self.dt / self.spec.accel_bias_corr_time)
        beta_g = np.exp(-self.dt / self.spec.gyro_bias_corr_time)

        sigma_a = self.spec.accel_bias_instability * np.sqrt(max(1 - beta_a ** 2, 0))
        sigma_g = self.spec.gyro_bias_instability * np.sqrt(max(1 - beta_g ** 2, 0))

        self.accel_bias = self.accel_bias * beta_a + self.rng.normal(0, sigma_a, size=2)
        self.gyro_bias = self.gyro_bias * beta_g + self.rng.normal(0, sigma_g)

    def measure(self, true_accel_xy: np.ndarray, true_yaw_rate: float):
        """
        Given true (ground-truth) acceleration [ax, ay] (m/s^2) and yaw rate
        (rad/s), returns the noisy/biased sensor measurement:
        (measured_accel[2], measured_yaw_rate)
        """
        self._propagate_bias()

        accel_noise = self.rng.normal(0, self._accel_vrw / np.sqrt(self.dt), size=2)
        gyro_noise = self.rng.normal(0, self._gyro_arw / np.sqrt(self.dt))

        measured_accel = np.asarray(true_accel_xy, dtype=float) + self.accel_bias + accel_noise
        measured_yaw_rate = true_yaw_rate + self.gyro_bias + gyro_noise

        return measured_accel, measured_yaw_rate
