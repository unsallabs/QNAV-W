import json
import sys, os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.simulation.geocode import geocode_location, GeocodingError


class _FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_geocode_success_parses_lat_lon_and_display_name():
    payload = [{"lat": "35.6762", "lon": "139.6503", "display_name": "Tokyo, Japan"}]
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        lat, lon, name = geocode_location("Tokyo, Japan")

    assert abs(lat - 35.6762) < 1e-6
    assert abs(lon - 139.6503) < 1e-6
    assert name == "Tokyo, Japan"


def test_geocode_raises_on_no_results():
    with patch("urllib.request.urlopen", return_value=_FakeResponse([])):
        try:
            geocode_location("Nonexistent Place XYZ 12345")
            assert False, "expected GeocodingError"
        except GeocodingError:
            pass


def test_geocode_raises_on_network_failure():
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        try:
            geocode_location("Paris, France")
            assert False, "expected GeocodingError"
        except GeocodingError:
            pass


def test_geocode_raises_on_malformed_result():
    payload = [{"display_name": "Somewhere"}]  # missing lat/lon
    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        try:
            geocode_location("Somewhere")
            assert False, "expected GeocodingError"
        except GeocodingError:
            pass


if __name__ == "__main__":
    test_geocode_success_parses_lat_lon_and_display_name()
    test_geocode_raises_on_no_results()
    test_geocode_raises_on_network_failure()
    test_geocode_raises_on_malformed_result()
    print("All geocoding tests passed.")
