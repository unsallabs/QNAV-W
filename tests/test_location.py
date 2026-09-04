import sys, os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnav.simulation.location import resolve_location
from qnav.simulation.trajectory import DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME
from qnav.simulation.geocode import GeocodingError


def test_lat_lon_override_is_offline_and_exact():
    """--lat/--lon must work with zero network access and no approximation."""
    lat0, lon0, name = resolve_location(lat=48.8566, lon=2.3522)
    assert lat0 == 48.8566
    assert lon0 == 2.3522
    assert "48.8566" in name  # falls back to coordinate label when no --location given


def test_lat_lon_override_uses_location_as_label_only():
    lat0, lon0, name = resolve_location(location="Paris, France", lat=48.8566, lon=2.3522)
    assert lat0 == 48.8566
    assert lon0 == 2.3522
    assert name == "Paris, France"


def test_default_location_is_san_francisco():
    lat0, lon0, name = resolve_location()
    assert lat0 == DEFAULT_LAT
    assert lon0 == DEFAULT_LON
    assert name == DEFAULT_LOCATION_NAME


def test_zero_latitude_and_longitude_are_respected():
    """A location on the equator/prime meridian (lat or lon == 0.0) must not
    be mistaken for 'not provided' (a classic falsy-value bug)."""
    lat0, lon0, name = resolve_location(lat=0.0, lon=0.0)
    assert lat0 == 0.0
    assert lon0 == 0.0


def test_location_name_uses_geocoding_when_available():
    with patch("qnav.simulation.location.geocode_location",
               return_value=(35.6762, 139.6503, "Tokyo, Japan")):
        lat0, lon0, name = resolve_location(location="Tokyo, Japan")

    assert abs(lat0 - 35.6762) < 1e-6
    assert abs(lon0 - 139.6503) < 1e-6
    assert name == "Tokyo, Japan"


def test_geocoding_failure_falls_back_to_default_location():
    with patch("qnav.simulation.location.geocode_location",
               side_effect=GeocodingError("no network")):
        lat0, lon0, name = resolve_location(location="Nowhereville")

    assert lat0 == DEFAULT_LAT
    assert lon0 == DEFAULT_LON
    assert name == DEFAULT_LOCATION_NAME


if __name__ == "__main__":
    test_lat_lon_override_is_offline_and_exact()
    test_lat_lon_override_uses_location_as_label_only()
    test_default_location_is_san_francisco()
    test_zero_latitude_and_longitude_are_respected()
    test_location_name_uses_geocoding_when_available()
    test_geocoding_failure_falls_back_to_default_location()
    print("All location resolution tests passed.")
