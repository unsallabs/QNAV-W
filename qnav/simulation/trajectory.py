"""
Ground-Truth Route Generator
================================

Produces the ground-truth vehicle trajectory the simulation treats as
"reality". Two ways to get one:

1. **Road-network route** (default): fetched from the real OpenStreetMap
   road graph via OSRM (`qnav.simulation.road_route`). Because it follows
   actual drivable roads, it never crosses buildings, parks, or water —
   those simply aren't part of the road network OSRM routes on. Requires
   network access.

2. **Synthetic analytic route** (fallback, or via `use_road_network=False`):
   an analytically generated curvy path (sum of low-frequency sinusoids for
   heading/speed). Fully offline, no network required. Used automatically
   whenever road-network routing is unavailable (no network, no route
   found, service error), so the simulation never hard-depends on the
   internet.

Either way, the result is a `GroundTruth` with the SAME fields — the rest
of the pipeline (sensors, EKF, gravity map matching) is completely
agnostic to which one produced it.

The real-world location (default: San Francisco, USA) only decides "where
on Earth is local (0, 0)" for map display and road-network queries; the
underlying physics operates entirely in local tangent-plane meters. See
`qnav.simulation.location` for worldwide location resolution.
"""

from dataclasses import dataclass
import numpy as np

# Default demo location: San Francisco, USA (approximate city-center coordinates).
# Any other worldwide location can be used instead via --location or --lat/--lon
# (see qnav.simulation.location.resolve_location) without touching this module.
DEFAULT_LAT = 37.7749
DEFAULT_LON = -122.4194
DEFAULT_LOCATION_NAME = "San Francisco, USA"

EARTH_RADIUS_M = 6371000.0


@dataclass
class GroundTruth:
    t: np.ndarray
    x: np.ndarray             # local ENU east (m)
    y: np.ndarray             # local ENU north (m)
    vx: np.ndarray
    vy: np.ndarray
    yaw: np.ndarray           # rad, 0 = facing east (standard math convention)
    yaw_rate: np.ndarray      # rad/s
    accel_body: np.ndarray    # (N, 2), [forward, lateral] acceleration, m/s^2
    gravity_true: np.ndarray  # "true" local g at each step (consistent with the map)
    route_source: str = "synthetic"  # "road" (OSRM) or "synthetic" (analytic fallback)


def rotation_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def local_xy_to_latlon(x, y, lat0=DEFAULT_LAT, lon0=DEFAULT_LON):
    """Simple equirectangular projection (accurate enough for short distances)."""
    lat = lat0 + (np.asarray(y) / EARTH_RADIUS_M) * (180.0 / np.pi)
    lon = lon0 + (np.asarray(x) / (EARTH_RADIUS_M * np.cos(np.radians(lat0)))) * (180.0 / np.pi)
    return lat, lon


def latlon_to_local(lat, lon, lat0=DEFAULT_LAT, lon0=DEFAULT_LON):
    """Inverse of local_xy_to_latlon: (lat, lon) -> local ENU (x, y) meters."""
    y = (np.asarray(lat) - lat0) * (np.pi / 180.0) * EARTH_RADIUS_M
    x = (np.asarray(lon) - lon0) * (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    return x, y


def _speed_profile(t, avg_speed, rng):
    """Smooth speed variation around avg_speed (mixed urban/highway character)."""
    speed = avg_speed + 3.0 * np.sin(2 * np.pi * t / 180.0 + 0.4) \
        + 1.5 * np.sin(2 * np.pi * t / 55.0 + 1.1)
    return np.clip(speed, 2.0, None)  # prevent the vehicle from fully stopping


def generate_synthetic_route(duration_s: float = 900.0, dt: float = 0.02,
                              avg_speed: float = 13.0, seed: int = 7,
                              gravity_map=None) -> GroundTruth:
    """
    Analytically generates a realistic ground vehicle route: near-constant
    speed with smooth turns, fully offline. heading(t) and speed(t) are the
    sum of a few low-frequency sinusoids (roughly resembling successive
    straight/curved road sections). Position is the integral of the
    velocity vector; acceleration (body-frame) is the derivative of
    velocity rotated into the vehicle frame — all ground-truth kinematic
    quantities are mutually consistent, analytically derived.

    duration_s : total route duration (s). Default 900s * 13m/s ~= 11.7 km.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt)
    n = len(t)

    n_components = 4
    heading = np.zeros(n)
    for _ in range(n_components):
        period = rng.uniform(60.0, 240.0)      # s, turn period
        amp = rng.uniform(0.08, 0.35)           # rad, turn intensity
        phase = rng.uniform(0, 2 * np.pi)
        heading += amp * np.sin(2 * np.pi * t / period + phase)

    speed = _speed_profile(t, avg_speed, rng)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x = np.concatenate(([0.0], np.cumsum((vx[:-1] + vx[1:]) / 2 * dt)))
    y = np.concatenate(([0.0], np.cumsum((vy[:-1] + vy[1:]) / 2 * dt)))

    yaw = heading
    yaw_rate = np.gradient(yaw, t)

    ax_nav = np.gradient(vx, t)
    ay_nav = np.gradient(vy, t)

    accel_body = np.zeros((n, 2))
    for i in range(n):
        R_T = rotation_matrix(yaw[i]).T  # nav->body: R(yaw)^-1 = R(yaw)^T
        accel_body[i] = R_T @ np.array([ax_nav[i], ay_nav[i]])

    if gravity_map is not None:
        gravity_true = np.array([gravity_map.gravity_at(x[i], y[i]) for i in range(n)])
    else:
        gravity_true = np.full(n, 9.80665)

    return GroundTruth(t=t, x=x, y=y, vx=vx, vy=vy, yaw=yaw, yaw_rate=yaw_rate,
                        accel_body=accel_body, gravity_true=gravity_true,
                        route_source="synthetic")


def _resample_uniform_arclength(path_xy, spacing_m=2.0):
    """Resamples a polyline at uniform arc-length spacing (no corner rounding —
    see _round_corners, applied separately, once, to the full multi-lap path)."""
    seg_vectors = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(seg_vectors[:, 0], seg_vectors[:, 1])
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = s[-1]
    n_pts = max(int(total_length / spacing_m) + 1, 2)
    s_uniform = np.linspace(0.0, total_length, n_pts)
    x_uniform = np.interp(s_uniform, s, path_xy[:, 0])
    y_uniform = np.interp(s_uniform, s, path_xy[:, 1])
    return np.column_stack([x_uniform, y_uniform]), total_length


def _round_corners(path_uniform, spacing_m=2.0, corner_rounding_m=20.0):
    """
    Geometrically rounds sharp corners with a moving-average filter in the
    arc-length domain (a plain average can't overshoot, unlike higher-order
    polynomial smoothing — important because real road polylines, and the
    multi-lap paths built from them, are jagged/zigzagged, not smooth).

    This matters independently of speed: a raw polyline vertex has
    mathematically infinite curvature (two straight segments meeting at a
    point), so no amount of slowing down makes it "safe" to differentiate —
    a real vehicle doesn't pivot in place at intersections (or at a route's
    turn-around point), it follows a rounded arc with some finite turn
    radius. Rounding the geometry first is what gives the curvature-based
    speed profile (computed next) something physically meaningful to work
    with. Must be applied to the FULL multi-lap path (including any
    turn-around points), not just a single lap — otherwise the turn-around
    itself becomes an unrounded, infinitely-sharp 180-degree corner.
    """
    n_pts = len(path_uniform)
    window = int(round(corner_rounding_m / spacing_m))
    if window % 2 == 0:
        window += 1
    window = max(window, 3)
    if n_pts <= window:
        return path_uniform

    kernel = np.ones(window) / window
    pad = window // 2
    x_padded = np.pad(path_uniform[:, 0], pad, mode="edge")
    y_padded = np.pad(path_uniform[:, 1], pad, mode="edge")
    x_rounded = np.convolve(x_padded, kernel, mode="valid")
    y_rounded = np.convolve(y_padded, kernel, mode="valid")
    return np.column_stack([x_rounded, y_rounded])


def _is_closed_loop(path_uniform):
    return np.hypot(*(path_uniform[0] - path_uniform[-1])) < 3.0 * (
        np.hypot(*(path_uniform[1] - path_uniform[0])) if len(path_uniform) > 1 else 1.0
    )


def _build_multilap_path(path_uniform, target_length_m, max_laps=200):
    """
    Extends a single-lap uniform path to at least `target_length_m` of total
    arc length, so there's enough road to drive for the requested duration
    even if the real route itself is short.

    Only CLOSED loops (end point ~= start point, as built by
    `road_route._loop_waypoints` — always true for real OSRM routes, which
    are requested as loops) are extended, by direct repetition: the end of
    one lap already coincides with the start of the next, so this
    introduces no discontinuity or curvature artifact.

    Open paths (a straight point-to-point route — only really arises for
    synthetic/test inputs, since real routes are always closed loops) are
    returned as-is, without any reversal/"ping-pong" tiling: turning a
    vehicle around 180 degrees to retrace the exact same path is a much
    tighter maneuver than any real intersection, and coordinate-space
    smoothing of a there-and-back shape can pull points close together and
    inflate curvature rather than reduce it. If the path is shorter (in
    time) than the requested duration, the vehicle brakes to a stop at the
    end and holds — see the `is_closed_loop` boundary condition in
    `_curvature_limited_speed_profile`.
    """
    if not _is_closed_loop(path_uniform):
        return path_uniform

    laps = [path_uniform]
    total_length = np.sum(np.hypot(*np.diff(path_uniform, axis=0).T))
    n_laps = 1
    while total_length < target_length_m and n_laps < max_laps:
        next_lap = path_uniform[1:]  # drop the duplicate shared point
        laps.append(next_lap)
        total_length += np.sum(np.hypot(*np.diff(np.concatenate([laps[-2][-1:], next_lap]), axis=0).T))
        n_laps += 1

    return np.concatenate(laps, axis=0)


def _curvature_limited_speed_profile(path_uniform, avg_speed,
                                      lat_accel_max=2.0, lon_accel_max=1.5, lon_decel_max=2.5,
                                      min_speed=2.0, is_closed_loop=False):
    """
    Standard curvature-based speed planning, as used in autonomous-vehicle and
    racing-line trajectory planners (e.g. Heilmeier et al. 2020, "Minimum
    Curvature Trajectory Planning and Control for an Autonomous Race Car"):

      1. At each point, cap speed so lateral (centripetal) acceleration stays
         under `lat_accel_max`: v_curve = sqrt(lat_accel_max / curvature).
      2. Run a forward pass limiting how fast speed can *increase* between
         points to `lon_accel_max`, then a backward pass limiting how fast it
         can *decrease* to `lon_decel_max` — a "forward-backward solver" that
         makes sure the resulting speed profile is actually achievable (you
         can't instantly jump to a higher speed, and you must start braking
         before a sharp corner, not at it).

    This keeps a simulated vehicle from doing what a plain arc-length
    resample does: driving through 90-degree intersections at full cruising
    speed. Real roads have frequent sharp turns; real drivers slow down for
    them. Limits default to comfortable-driving values from the literature
    (~0.12-0.2 g lateral, ~1-2.5 m/s^2 longitudinal; see e.g. ISO 22179 and
    passenger-comfort studies), not race-car limits.

    Returns v(s), one speed per point in `path_uniform`.
    """
    diffs = np.diff(path_uniform, axis=0)
    seg_len = np.maximum(np.hypot(diffs[:, 0], diffs[:, 1]), 1e-6)
    n = len(path_uniform)

    if n < 4:
        return np.full(n, avg_speed)

    heading_seg = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
    dtheta = np.diff(heading_seg)  # heading change between consecutive segments, length n-2
    ds_between = np.maximum((seg_len[:-1] + seg_len[1:]) / 2.0, 1e-6)
    kappa_interior = np.abs(dtheta) / ds_between

    kappa = np.zeros(n)
    kappa[1:-1] = kappa_interior
    kappa[0] = kappa_interior[0]
    kappa[-1] = kappa_interior[-1]
    if n >= 5:  # light smoothing: real digitization noise can spike curvature at single points
        kernel = np.ones(5) / 5.0
        kappa = np.convolve(kappa, kernel, mode="same")

    v_cruise_max = avg_speed * 1.3
    with np.errstate(divide="ignore"):
        v_curve = np.sqrt(lat_accel_max / np.maximum(kappa, 1e-9))
    v_curve = np.clip(v_curve, min_speed, v_cruise_max)

    if not is_closed_loop:
        # An open path has to end somewhere — without this, the vehicle
        # would still be at cruising speed the instant the path runs out,
        # and then "hold position" (see _build_multilap_path) would force
        # an instantaneous, unphysical stop. Pin the endpoints near zero so
        # the forward/backward solver below naturally brakes into a real
        # stop (and eases away from a start) at a realistic deceleration.
        v_curve[0] = min(v_curve[0], min_speed)
        v_curve[-1] = 0.05

    # Forward pass: speed can only increase as fast as lon_accel_max allows.
    v_fwd = v_curve.copy()
    for i in range(1, n):
        v_fwd[i] = min(v_fwd[i], np.sqrt(v_fwd[i - 1] ** 2 + 2 * lon_accel_max * seg_len[i - 1]))

    # Backward pass: speed must have started decreasing early enough to
    # respect lon_decel_max braking into a slow corner (or the route's end).
    v_final = v_fwd.copy()
    for i in range(n - 2, -1, -1):
        v_final[i] = min(v_final[i], np.sqrt(v_final[i + 1] ** 2 + 2 * lon_decel_max * seg_len[i]))

    if not is_closed_loop:
        v_final[-1] = 0.05  # avoid exact zero (division safety in _integrate_time)

    return v_final


def _integrate_time(path_uniform, speed_profile):
    """Cumulative time at each point of an already-final (rounded, single
    direction of travel) path, given the speed at each point."""
    diffs = np.diff(path_uniform, axis=0)
    seg_len = np.maximum(np.hypot(diffs[:, 0], diffs[:, 1]), 1e-6)
    avg_leg_speed = np.maximum((speed_profile[:-1] + speed_profile[1:]) / 2.0, 1e-6)
    leg_dt = seg_len / avg_leg_speed
    return np.concatenate(([0.0], np.cumsum(leg_dt)))


def generate_route_from_path(path_xy, duration_s: float = 900.0, dt: float = 0.02,
                              avg_speed: float = 13.0, seed: int = 7,
                              gravity_map=None, route_source: str = "road") -> GroundTruth:
    """
    Turns an arbitrary polyline (e.g. a real road-network route, already in
    local ENU meters) into a time-parameterized GroundTruth using
    curvature-based speed planning: the vehicle slows down for sharp turns
    and accelerates on straights, exactly like a real driver, using the same
    curvature-limited-speed + forward-backward-solver approach used in
    autonomous-vehicle and racing-line trajectory planning (e.g. Heilmeier
    et al. 2020, "Minimum Curvature Trajectory Planning and Control for an
    Autonomous Race Car"). This is what keeps derived accelerations
    physically plausible on real road polylines, which — unlike the smooth
    analytic synthetic route — have sharp, near-instantaneous corners at
    every intersection (and would otherwise also get a sharp corner at the
    route's turn-around point, if the road is shorter than the requested
    duration).

    path_xy : array-like of shape (M, 2), local (x, y) meters, in travel order.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        raise ValueError("path_xy must contain at least 2 points")

    base_path, base_length = _resample_uniform_arclength(path_xy, spacing_m=2.0)
    if base_length < 10.0:
        raise ValueError(f"path is too short to simulate ({base_length:.1f} m)")

    # Build enough road (tiling closed loops; open paths are used as-is and
    # brake to a stop at the end — see _curvature_limited_speed_profile) to
    # cover the requested duration, THEN round corners once over the whole
    # thing, so nothing — not the original route's intersections, not a
    # closed loop's seam — is left as an unphysical infinite-curvature point.
    is_closed = _is_closed_loop(base_path)
    target_length_m = avg_speed * duration_s * 3.0
    multilap_path = _build_multilap_path(base_path, target_length_m)
    multilap_path = _round_corners(multilap_path, spacing_m=2.0, corner_rounding_m=20.0)

    speed_profile = _curvature_limited_speed_profile(multilap_path, avg_speed, is_closed_loop=is_closed)
    t_multilap = _integrate_time(multilap_path, speed_profile)

    # Rare edge case: curvature-limited speed dropped the effective average
    # speed enough that even the generous 3x buffer wasn't long enough.
    # Extend further rather than silently truncating the requested duration.
    # (Only applies to closed loops — an open path's length is fixed.)
    retries = 0
    while is_closed and t_multilap[-1] < duration_s and retries < 4:
        target_length_m *= 2.0
        multilap_path = _round_corners(_build_multilap_path(base_path, target_length_m),
                                        spacing_m=2.0, corner_rounding_m=20.0)
        speed_profile = _curvature_limited_speed_profile(multilap_path, avg_speed, is_closed_loop=is_closed)
        t_multilap = _integrate_time(multilap_path, speed_profile)
        retries += 1

    t = np.arange(0.0, duration_s, dt)
    n = len(t)
    x = np.interp(t, t_multilap, multilap_path[:, 0])
    y = np.interp(t, t_multilap, multilap_path[:, 1])

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    # Heading from atan2(vy, vx) is numerically unstable whenever speed gets
    # very small (dividing near-zero vy,vx by each other is dominated by
    # noise) — exactly what happens whenever the curvature-based speed
    # profile slows the vehicle for a sharp turn, or it brakes to a stop at
    # an open path's end. A real vehicle's heading doesn't actually swing
    # wildly at low speed; heading is simply unobservable from velocity
    # there. Standard practice (as in real navigation systems): hold the
    # last reliable heading estimate while speed is below a small threshold,
    # rather than feed spurious heading noise into yaw_rate.
    speed = np.hypot(vx, vy)
    raw_yaw = np.arctan2(vy, vx)
    low_speed = speed < 0.5
    if np.any(low_speed) and not np.all(low_speed):
        valid_idx = np.flatnonzero(~low_speed)
        fill_idx = np.searchsorted(valid_idx, np.arange(n), side="right") - 1
        fill_idx = np.clip(fill_idx, 0, len(valid_idx) - 1)
        raw_yaw = np.where(low_speed, raw_yaw[valid_idx[fill_idx]], raw_yaw)

    yaw = np.unwrap(raw_yaw)
    yaw_rate = np.gradient(yaw, t)
    yaw_rate[low_speed] = 0.0  # no reliable turning information at a standstill

    ax_nav = np.gradient(vx, t)
    ay_nav = np.gradient(vy, t)

    accel_body = np.zeros((n, 2))
    for i in range(n):
        R_T = rotation_matrix(yaw[i]).T
        accel_body[i] = R_T @ np.array([ax_nav[i], ay_nav[i]])

    if gravity_map is not None:
        gravity_true = np.array([gravity_map.gravity_at(x[i], y[i]) for i in range(n)])
    else:
        gravity_true = np.full(n, 9.80665)

    return GroundTruth(t=t, x=x, y=y, vx=vx, vy=vy, yaw=yaw, yaw_rate=yaw_rate,
                        accel_body=accel_body, gravity_true=gravity_true,
                        route_source=route_source)


def generate_route(duration_s: float = 900.0, dt: float = 0.02,
                    avg_speed: float = 13.0, seed: int = 7,
                    gravity_map=None, lat0: float = DEFAULT_LAT, lon0: float = DEFAULT_LON,
                    use_road_network: bool = True) -> GroundTruth:
    """
    Generates the ground-truth route for the simulation. By default,
    attempts to fetch a real, drivable route from the OpenStreetMap road
    network (via OSRM) centered on (lat0, lon0), so the simulated vehicle
    follows actual roads and never crosses buildings, parks, or water. If
    that's unavailable for any reason (no network, no route found, service
    error), automatically falls back to the offline synthetic route
    generator and prints a one-line notice — the simulation never hard-
    depends on network access.

    Set `use_road_network=False` to skip the network attempt entirely and
    go straight to the synthetic generator (useful for fully offline runs
    or fast/deterministic testing).
    """
    if use_road_network:
        from qnav.simulation.road_route import fetch_road_route, RoadRoutingError
        try:
            target_distance_m = avg_speed * duration_s
            latlon_path = fetch_road_route(lat0, lon0, target_distance_m=target_distance_m, seed=seed)
            x, y = latlon_to_local(np.array([p[0] for p in latlon_path]),
                                    np.array([p[1] for p in latlon_path]), lat0, lon0)
            path_xy = np.column_stack([x, y])
            return generate_route_from_path(path_xy, duration_s=duration_s, dt=dt,
                                             avg_speed=avg_speed, seed=seed,
                                             gravity_map=gravity_map, route_source="road")
        except RoadRoutingError as exc:
            print(f"[Q-Nav] Road-network routing unavailable ({exc}); "
                  f"falling back to the offline synthetic route generator.")

    return generate_synthetic_route(duration_s=duration_s, dt=dt, avg_speed=avg_speed,
                                     seed=seed, gravity_map=gravity_map)
