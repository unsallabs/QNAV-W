"""
Location Resolution
======================

Resolves the demo's real-world start location from CLI-style inputs, with
a strict priority order that keeps the simulation offline-first:

    1. Explicit --lat/--lon        -> always available, fully offline, exact
    2. --location "place name"     -> optional, resolved via Nominatim
                                       geocoding, requires network
    3. Neither given                -> falls back to the default location
                                       (San Francisco, USA)

If geocoding is requested but fails for any reason (no network, no
results, service error), we fall back to the default location and print a
warning rather than crashing — geocoding is a convenience, not a
dependency of the core simulation.

Nothing in this module touches sensors, the EKF, or gravity-map matching:
the physics is entirely local-tangent-plane and location-agnostic. This
module only decides where on Earth (0, 0) is for the purposes of map
display.
"""

from qnav.simulation.trajectory import DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME
from qnav.simulation.geocode import geocode_location, GeocodingError


def resolve_location(location: str = None, lat: float = None, lon: float = None):
    """
    Returns (lat0, lon0, location_name) for the simulation's start point.

    - If both `lat` and `lon` are given, they are used directly (offline,
      no network call). `location`, if also given, is used only as a
      display label.
    - Else if `location` is given, it is geocoded via Nominatim. On
      failure, falls back to the default location with a warning.
    - Else, returns the default location (San Francisco, USA).
    """
    if lat is not None and lon is not None:
        name = location if location else f"({float(lat):.4f}, {float(lon):.4f})"
        return float(lat), float(lon), name

    if location:
        try:
            lat0, lon0, display_name = geocode_location(location)
            return lat0, lon0, display_name
        except GeocodingError as exc:
            print(f"[Q-Nav] Geocoding failed for '{location}' ({exc}); "
                  f"falling back to default location: {DEFAULT_LOCATION_NAME}.")
            return DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME

    return DEFAULT_LAT, DEFAULT_LON, DEFAULT_LOCATION_NAME
