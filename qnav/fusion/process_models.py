"""
Strapdown INS Mechanization Equations
========================================

Acceleration measured in the body frame is rotated into the nav frame
(local tangent plane, ENU) using the estimated yaw angle, and the state is
propagated forward:

    R(yaw) = [[cos(yaw), -sin(yaw)],
              [sin(yaw),  cos(yaw)]]

    a_nav = R(yaw) @ (a_body_meas - bias_a)
    yaw_{k+1} = yaw_k + (gyro_meas - bias_g) * dt

    v_{k+1} = v_k + a_nav * dt
    p_{k+1} = p_k + v_k * dt + 0.5 * a_nav * dt^2

Biases are modeled as a random walk: b_{k+1} = b_k + w  (E[w]=0)

This is the standard 2D simplification of strapdown INS mechanization
(sufficient for flat/short-range ground vehicle navigation, as opposed to a
full 3D/quaternion mechanization).
"""

import numpy as np
from qnav.fusion.state import (
    IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_YAW, IDX_BAX, IDX_BAY, IDX_BG, STATE_DIM
)


def rotation_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def propagate_state(state: np.ndarray, accel_meas: np.ndarray, gyro_meas: float, dt: float) -> np.ndarray:
    """Nonlinear state transition f(x, u)."""
    x, y, vx, vy, yaw, bax, bay, bg = state

    accel_corrected = accel_meas - np.array([bax, bay])
    R = rotation_matrix(yaw)
    a_nav = R @ accel_corrected

    new_yaw = yaw + (gyro_meas - bg) * dt
    new_vx = vx + a_nav[0] * dt
    new_vy = vy + a_nav[1] * dt
    new_x = x + vx * dt + 0.5 * a_nav[0] * dt ** 2
    new_y = y + vy * dt + 0.5 * a_nav[1] * dt ** 2

    new_state = state.copy()
    new_state[IDX_X] = new_x
    new_state[IDX_Y] = new_y
    new_state[IDX_VX] = new_vx
    new_state[IDX_VY] = new_vy
    new_state[IDX_YAW] = new_yaw
    # biases are a random walk: nominal value is retained, uncertainty grows via process noise
    return new_state


def state_transition_jacobian(state: np.ndarray, accel_meas: np.ndarray, dt: float) -> np.ndarray:
    """Analytic Jacobian F = df/dx, for EKF covariance propagation."""
    yaw = state[IDX_YAW]
    bax, bay = state[IDX_BAX], state[IDX_BAY]

    ax_c = accel_meas[0] - bax
    ay_c = accel_meas[1] - bay
    c, s = np.cos(yaw), np.sin(yaw)

    # a_nav = [c*ax_c - s*ay_c, s*ax_c + c*ay_c]
    # d(a_nav)/d(yaw) = [-s*ax_c - c*ay_c, c*ax_c - s*ay_c]
    da_nav_dyaw = np.array([-s * ax_c - c * ay_c, c * ax_c - s * ay_c])
    # d(a_nav)/d(bax) = [-c, -s] ; d(a_nav)/d(bay) = [s, -c]
    da_nav_dbax = np.array([-c, -s])
    da_nav_dbay = np.array([s, -c])

    F = np.eye(STATE_DIM)

    # dx/dvx, dy/dvy
    F[IDX_X, IDX_VX] = dt
    F[IDX_Y, IDX_VY] = dt

    # dx/dyaw, dy/dyaw  (from 0.5*a_nav*dt^2 term)
    F[IDX_X, IDX_YAW] = 0.5 * dt ** 2 * da_nav_dyaw[0]
    F[IDX_Y, IDX_YAW] = 0.5 * dt ** 2 * da_nav_dyaw[1]

    # dx/dbax, dx/dbay, dy/dbax, dy/dbay
    F[IDX_X, IDX_BAX] = 0.5 * dt ** 2 * da_nav_dbax[0]
    F[IDX_X, IDX_BAY] = 0.5 * dt ** 2 * da_nav_dbay[0]
    F[IDX_Y, IDX_BAX] = 0.5 * dt ** 2 * da_nav_dbax[1]
    F[IDX_Y, IDX_BAY] = 0.5 * dt ** 2 * da_nav_dbay[1]

    # dvx/dyaw, dvy/dyaw
    F[IDX_VX, IDX_YAW] = dt * da_nav_dyaw[0]
    F[IDX_VY, IDX_YAW] = dt * da_nav_dyaw[1]

    # dvx/dbax, dvx/dbay, dvy/dbax, dvy/dbay
    F[IDX_VX, IDX_BAX] = dt * da_nav_dbax[0]
    F[IDX_VX, IDX_BAY] = dt * da_nav_dbay[0]
    F[IDX_VY, IDX_BAX] = dt * da_nav_dbax[1]
    F[IDX_VY, IDX_BAY] = dt * da_nav_dbay[1]

    # dyaw/dbg
    F[IDX_YAW, IDX_BG] = -dt

    return F


def process_noise_matrix(dt, accel_noise_std, gyro_noise_std,
                          ba_random_walk_std=1e-6, bg_random_walk_std=1e-7):
    """
    Q matrix: how sensor noise contributes to state uncertainty growth.
    Simple approach - inject accel/gyro noise directly into the vx, vy, yaw
    channels, and bias random-walk noise into the bias channels.
    """
    Q = np.zeros((STATE_DIM, STATE_DIM))
    var_v = (accel_noise_std * dt) ** 2
    var_p = (0.5 * accel_noise_std * dt ** 2) ** 2
    var_yaw = (gyro_noise_std * dt) ** 2

    Q[IDX_X, IDX_X] = var_p
    Q[IDX_Y, IDX_Y] = var_p
    Q[IDX_VX, IDX_VX] = var_v
    Q[IDX_VY, IDX_VY] = var_v
    Q[IDX_YAW, IDX_YAW] = var_yaw
    Q[IDX_BAX, IDX_BAX] = (ba_random_walk_std * np.sqrt(dt)) ** 2
    Q[IDX_BAY, IDX_BAY] = (ba_random_walk_std * np.sqrt(dt)) ** 2
    Q[IDX_BG, IDX_BG] = (bg_random_walk_std * np.sqrt(dt)) ** 2
    return Q
