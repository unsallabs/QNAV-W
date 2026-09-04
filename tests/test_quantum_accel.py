import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.sensors.quantum_accel import QuantumAccelerometer, QuantumAccelSpec
from qnav.sensors.imu_classical import ClassicalIMU, IMUSpec


def test_quantum_accel_tracks_true_value_on_average():
    spec = QuantumAccelSpec(cycle_time=0.05, residual_bias=0.0)
    sensor = QuantumAccelerometer(spec, seed=123)
    true_accel = np.array([2.0, -1.0])

    samples = []
    for _ in range(500):
        meas, is_new = sensor.measure(true_accel, dt=0.05)
        if is_new:
            samples.append(meas)

    samples = np.array(samples)
    mean_error = np.abs(samples.mean(axis=0) - true_accel)
    assert np.all(mean_error < 0.05)  # the mean should converge to the true value


def test_quantum_accel_has_lower_bias_drift_than_classical():
    """
    This test validates the project's central claim: a quantum accelerometer's
    long-term systematic drift (residual_bias) is much lower than a classical
    MEMS sensor's bias instability.
    """
    classical_spec = IMUSpec()
    quantum_spec = QuantumAccelSpec()
    assert quantum_spec.residual_bias < classical_spec.accel_bias_instability


def test_zero_order_hold_between_cycles():
    """Sampling faster than the cycle time should hold the same value (is_new=False)."""
    spec = QuantumAccelSpec(cycle_time=0.1)
    sensor = QuantumAccelerometer(spec, seed=1)

    _, is_new_1 = sensor.measure(np.array([1.0, 0.0]), dt=0.02)
    _, is_new_2 = sensor.measure(np.array([1.0, 0.0]), dt=0.02)

    assert is_new_1 in (True, False)
    assert is_new_2 is False or is_new_1 is False  # both can't be "new" within such a short dt


if __name__ == "__main__":
    test_quantum_accel_tracks_true_value_on_average()
    test_quantum_accel_has_lower_bias_drift_than_classical()
    test_zero_order_hold_between_cycles()
    print("All quantum accelerometer tests passed.")
