"""
Road-Network Route Fetching (optional, via OSRM)
====================================================================

Fetches a real, drivable route along the actual OpenStreetMap road network
using the public OSRM (Open Source Routing Machine) demo server. Because
the returned route follows real roads, it inherently never crosses
buildings, parks, or water — those simply aren't part of the drivable road
graph OSRM routes on.

This is OPTIONAL and requires network access to router.project-osrm.org.
If it's unavailable (no network, service down, no route found), callers
should fall back to the analytic synthetic route generator
(`qnav.simulation.trajectory.generate_synthetic_route`) — the rest of the
pipeline (sensors, EKF, gravity map matching) doesn't care which one
produced the ground-truth path. `qnav.simulation.trajectory.generate_route`
already does this fallback automatically.

Implemented with the Python standard library only (urllib + json + math),
the same approach as `qnav.simulation.geocode`, to keep the project
lightweight — no extra dependency (e.g. `osmnx`, `requests`) is required
just to fetch a road-snapped route.
"""

import json
import math
import random
import urllib.parse
import urllib.request

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/"
USER_AGENT = "Q-Nav-Prototype/1.0 (educational quantum-inertial navigation demo)"
EARTH_RADIUS_M = 6371000.0


class RoadRoutingError(Exception):
    """Raised when a road-network route cannot be fetched or found."""


def _destination_point(lat, lon, bearing_deg, distance_m):
    """Great-circle destination point given a start, bearing (deg), and distance (m)."""
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    ang = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _loop_waypoints(lat0, lon0, target_distance_m, seed):
    """
    Picks a handful of waypoints roughly forming a loop around the start
    point, spaced so the resulting road route is in the neighborhood of
    target_distance_m. OSRM then finds the real drivable route through
    them (start -> waypoints -> back near start).
    """
    rng = random.Random(seed)
    n_legs = 4
    leg_distance = max(target_distance_m / n_legs, 400.0)
    bearing = rng.uniform(0, 360)

    points = [(lat0, lon0)]
    lat, lon = lat0, lon0
    for _ in range(n_legs - 1):
        # Keep turning in roughly the same rotational direction so the
        # waypoints trace out a loop rather than a zigzag.
        bearing += rng.uniform(60, 130)
        lat, lon = _destination_point(lat, lon, bearing, leg_distance)
        points.append((lat, lon))
    points.append((lat0, lon0))  # close the loop back at the start
    return points


def fetch_road_route(lat0, lon0, target_distance_m=12000.0, seed=7, timeout=10.0):
    """
    Fetches a real, drivable, road-snapped route (a rough loop starting and
    ending near (lat0, lon0), roughly target_distance_m long) via the OSRM
    public routing API.

    Returns a list of (lat, lon) tuples tracing the actual road geometry,
    in travel order. Raises RoadRoutingError on any failure (no network,
    request timeout, no route found, malformed response, etc.) — callers
    are expected to catch this and fall back to an offline route generator.
    """
    waypoints = _loop_waypoints(lat0, lon0, target_distance_m, seed)
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    url = OSRM_ROUTE_URL + coord_str + "?" + urllib.parse.urlencode({
        "overview": "full",
        "geometries": "geojson",
    })
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:  # network errors, timeouts, DNS failures, etc.
        raise RoadRoutingError(f"could not reach OSRM routing service ({exc})") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RoadRoutingError(f"unexpected response from routing service ({exc})") from exc

    if data.get("code") != "Ok" or not data.get("routes"):
        raise RoadRoutingError(f"no drivable route found ({data.get('message', data.get('code'))})")

    try:
        coords = data["routes"][0]["geometry"]["coordinates"]  # [[lon, lat], ...]
    except (KeyError, IndexError, TypeError) as exc:
        raise RoadRoutingError(f"malformed routing response ({exc})") from exc

    if len(coords) < 2:
        raise RoadRoutingError("routing service returned an empty route")

    return [(lat, lon) for lon, lat in coords]
