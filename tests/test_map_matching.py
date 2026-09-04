import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.gravity.gravity_map import GravityMap
from qnav.gravity import map_matching


def test_map_matching_recovers_true_position_with_clean_signal():
    """
    In a noiseless (ideal) scenario: given gravity values read directly from
    the map along a known route, map matching should be able to recover the
    true position (within the search-grid resolution).
    """
    gmap = GravityMap(extent_m=10000.0, resolution_m=50.0, seed=1)

    true_x, true_y = 1200.0, -800.0
    offsets = [(-100.0, 0.0), (-50.0, 0.0), (0.0, 0.0)]
    measured = [gmap.gravity_at(true_x + ox, true_y + oy) for ox, oy in offsets]

    # deliberately offset the search center from the true value
    result = map_matching.match(
        gmap, measured, offsets,
        search_center_xy=(true_x + 150.0, true_y - 150.0),
        search_radius_m=300.0, search_step_m=50.0,
    )

    assert result.matched
    error = np.hypot(result.x - true_x, result.y - true_y)
    assert error <= 75.0  # error on the order of the grid resolution is acceptable


def test_map_matching_refuses_with_too_few_samples():
    gmap = GravityMap(extent_m=5000.0, resolution_m=50.0, seed=2)
    result = map_matching.match(
        gmap, measured_gravity_sequence=[9.8, 9.8], ins_relative_offsets=[(0, 0), (1, 1)],
        search_center_xy=(0, 0), search_radius_m=200.0,
    )
    assert result.matched is False


if __name__ == "__main__":
    test_map_matching_recovers_true_position_with_clean_signal()
    test_map_matching_refuses_with_too_few_samples()
    print("All gravity map matching tests passed.")
