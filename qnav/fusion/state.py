"""
EKF State Vector Definition
==============================

An 8-dimensional state vector, in local tangent-plane (ENU, meters)
coordinates, for a 2D strapdown INS:

    x = [ x, y, vx, vy, yaw, ba_x, ba_y, bg ]^T

    x, y     : position (m), East-North relative to the start point
    vx, vy   : velocity (m/s)
    yaw      : heading angle (rad)
    ba_x,ba_y: accelerometer bias estimate (m/s^2)
    bg       : gyroscope bias estimate (rad/s)

Estimating biases as part of the state vector is standard practice in real
strapdown INS/EKF architectures (see Titterton & Weston, "Strapdown
Inertial Navigation Technology").
"""

import numpy as np

IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_YAW = 4
IDX_BAX = 5
IDX_BAY = 6
IDX_BG = 7

STATE_DIM = 8


def init_state(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0, yaw0=0.0):
    s = np.zeros(STATE_DIM)
    s[IDX_X] = x0
    s[IDX_Y] = y0
    s[IDX_VX] = vx0
    s[IDX_VY] = vy0
    s[IDX_YAW] = yaw0
    return s


def init_covariance(pos_std=1.0, vel_std=0.5, yaw_std=0.05, ba_std=5e-4, bg_std=5e-5):
    P = np.diag([
        pos_std ** 2, pos_std ** 2,
        vel_std ** 2, vel_std ** 2,
        yaw_std ** 2,
        ba_std ** 2, ba_std ** 2,
        bg_std ** 2,
    ])
    return P
