import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.fusion.state import init_state, init_covariance, IDX_X, IDX_Y
from qnav.fusion.ekf import EKF


def test_predict_moves_state_forward_with_constant_accel():
    x0 = init_state()
    P0 = init_covariance()
    ekf = EKF(x0, P0)

    dt = 0.1
    accel = np.array([1.0, 0.0])  # body-frame forward acceleration
    for _ in range(10):
        ekf.predict(accel, 0.0, dt, accel_noise_std=1e-6, gyro_noise_std=1e-6)

    # with yaw=0, forward acceleration should map directly onto the x axis
    assert ekf.x[IDX_X] > 0
    assert np.isclose(ekf.x[IDX_Y], 0.0, atol=1e-6)


def test_gps_update_pulls_state_towards_measurement():
    x0 = init_state()
    P0 = init_covariance(pos_std=50.0)
    ekf = EKF(x0, P0)

    H = np.zeros((2, 8))
    H[0, IDX_X] = 1.0
    H[1, IDX_Y] = 1.0
    R = np.diag([3.0 ** 2, 3.0 ** 2])

    z = np.array([100.0, 50.0])
    ekf.update(z, H, R)

    assert abs(ekf.x[IDX_X] - 100.0) < abs(x0[IDX_X] - 100.0)
    assert abs(ekf.x[IDX_Y] - 50.0) < abs(x0[IDX_Y] - 50.0)


def test_covariance_shrinks_after_update():
    x0 = init_state()
    P0 = init_covariance(pos_std=50.0)
    ekf = EKF(x0, P0)

    H = np.zeros((2, 8))
    H[0, IDX_X] = 1.0
    H[1, IDX_Y] = 1.0
    R = np.diag([3.0 ** 2, 3.0 ** 2])

    trace_before = np.trace(ekf.P)
    ekf.update(np.array([0.0, 0.0]), H, R)
    trace_after = np.trace(ekf.P)

    assert trace_after < trace_before


if __name__ == "__main__":
    test_predict_moves_state_forward_with_constant_accel()
    test_gps_update_pulls_state_towards_measurement()
    test_covariance_shrinks_after_update()
    print("All EKF tests passed.")
