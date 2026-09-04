"""
Extended Kalman Filter (EKF) Core
=====================================

Standard EKF predict/update loop. Both the "classical IMU" and the
"quantum-enhanced IMU" configurations of Q-Nav use the EXACT SAME EKF
class — the only difference is the sensor noise/bias characteristics fed
into it. This keeps the comparison fair and the fusion architecture
sensor-agnostic.

Predict:
    x_k|k-1 = f(x_k-1, u_k)
    P_k|k-1 = F P_{k-1} F^T + Q

Update (for any linear/linearized measurement, e.g. GPS position or a
gravity-map-matching position fix):
    y = z - H x_k|k-1                      (innovation)
    S = H P_k|k-1 H^T + R
    K = P_k|k-1 H^T S^-1
    x_k = x_k|k-1 + K y
    P_k = (I - K H) P_k|k-1
"""

import numpy as np
from qnav.fusion.state import STATE_DIM
from qnav.fusion.process_models import propagate_state, state_transition_jacobian, process_noise_matrix


class EKF:
    def __init__(self, state: np.ndarray, covariance: np.ndarray):
        self.x = state.copy()
        self.P = covariance.copy()
        self.history = [self.x.copy()]

    def predict(self, accel_meas: np.ndarray, gyro_meas: float, dt: float,
                accel_noise_std: float, gyro_noise_std: float,
                ba_rw_std: float = 1e-6, bg_rw_std: float = 1e-7):
        F = state_transition_jacobian(self.x, accel_meas, dt)
        Q = process_noise_matrix(dt, accel_noise_std, gyro_noise_std, ba_rw_std, bg_rw_std)

        self.x = propagate_state(self.x, accel_meas, gyro_meas, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Generic linear measurement update (GPS fix, gravity-map fix, etc.)."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(STATE_DIM)
        self.P = (I - K @ H) @ self.P

    def record(self):
        self.history.append(self.x.copy())

    def position(self):
        return self.x[0], self.x[1]
