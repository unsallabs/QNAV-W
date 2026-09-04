"""
Nominatim Geocoding (optional)
================================

Resolves a free-text place name (e.g. "Tokyo, Japan") to (lat, lon) using
the free OpenStreetMap Nominatim API. This is OPTIONAL and only used for
the convenience of the `--location` flag; it requires network access and
issues a single HTTP GET request.

The core simulation (sensors, EKF, gravity map matching, drift analysis)
never depends on this module and runs fully offline via `--lat/--lon`
(see qnav.simulation.location).

Implemented with the Python standard library only (urllib + json) to keep
the project lightweight — no extra dependency (e.g. `requests` or
`geopy`) is required just to resolve a place name.

Nominatim's usage policy requires a descriptive User-Agent header and a
low request rate; both are respected here for a single, on-demand lookup.
"""

import json
import urllib.parse
import urllib.request

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "Q-Nav-Prototype/1.0 (educational quantum-inertial navigation demo)"


class GeocodingError(Exception):
    """Raised when a location name cannot be resolved (network issue, no results, etc.)."""


def geocode_location(query: str, timeout: float = 5.0):
    """
    Resolves a free-text location query to (lat, lon, display_name).

    Raises GeocodingError on any failure: no network access, request
    timeout, or no matching results. Callers should catch this and fall
    back to a default/offline location rather than crashing the whole
    simulation over an optional convenience feature.
    """
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
    })
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:  # network errors, timeouts, DNS failures, etc.
        raise GeocodingError(f"could not reach Nominatim geocoding service ({exc})") from exc

    try:
        results = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeocodingError(f"unexpected response from geocoding service ({exc})") from exc

    if not results:
        raise GeocodingError(f"no results found for location: '{query}'")

    result = results[0]
    try:
        lat = float(result["lat"])
        lon = float(result["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingError(f"malformed geocoding result for '{query}' ({exc})") from exc

    display_name = result.get("display_name", query)
    return lat, lon, display_name
