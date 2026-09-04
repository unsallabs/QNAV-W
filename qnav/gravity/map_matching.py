"""
Gravity Map Matching
======================

Idea (same principle as real TERCOM / gravity-aided navigation systems)
---------------------------------------------------------------------------
Over a short time window, the INS produces a "shape" — the relative
position offsets along the trajectory — which we trust (short-term INS
drift is negligible). The quantum gravimeter collects a sequence of
gravity measurements (g_1, g_2, ..., g_N) along that same trajectory. We
compare this measured sequence against the sequence that WOULD have been
measured for every candidate global offset on the reference gravity map,
and pick the offset with the lowest error / best correlation.

This is the only mechanism that provides an ABSOLUTE position correction
without GPS (the IMU alone only gives relative/dead-reckoning position; it
has no absolute reference).

This module performs a coarse grid-search within the EKF's current
uncertainty radius (from the P matrix); real systems typically use a
particle filter or correlation optimization instead — the algorithmic
principle is the same.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MapMatchResult:
    x: float
    y: float
    residual: float          # RMS error of the best match (m/s^2)
    confidence_std_m: float  # position uncertainty derived from match sharpness (m)
    matched: bool


def match(gravity_map, measured_gravity_sequence, ins_relative_offsets,
          search_center_xy, search_radius_m, search_step_m=100.0):
    """
    measured_gravity_sequence : sequence of measured g values (m/s^2), last N samples
    ins_relative_offsets      : the (dx, dy) relative positions from the INS for the
                                 same N samples (the last sample's offset is (0,0),
                                 representing "now")
    search_center_xy          : current EKF position estimate (x, y), search center
    search_radius_m           : search radius (derived from EKF position uncertainty)

    Returns: MapMatchResult -- the best-matching absolute (x, y) and its
    associated confidence.
    """
    cx, cy = search_center_xy
    offsets = np.asarray(ins_relative_offsets)
    measured = np.asarray(measured_gravity_sequence)

    if len(measured) < 3:
        return MapMatchResult(cx, cy, np.inf, search_radius_m, matched=False)

    candidates = np.arange(-search_radius_m, search_radius_m + 1e-9, search_step_m)
    best_residual = np.inf
    best_xy = (cx, cy)
    residual_grid = []

    for dx in candidates:
        for dy in candidates:
            cand_x, cand_y = cx + dx, cy + dy
            predicted = np.array([
                gravity_map.gravity_at(cand_x + ox, cand_y + oy) for ox, oy in offsets
            ])
            # de-mean both sequences before comparing (bias-free correlation)
            residual = np.sqrt(np.mean(((measured - measured.mean()) -
                                         (predicted - predicted.mean())) ** 2))
            residual_grid.append(residual)
            if residual < best_residual:
                best_residual = residual
                best_xy = (cand_x, cand_y)

    residual_grid = np.asarray(residual_grid)
    # match sharpness: how distinctive the best residual is relative to the
    # average residual -> a sharper peak means a more trustworthy fix (lower
    # confidence_std)
    sharpness = residual_grid.mean() / (best_residual + 1e-12)
    confidence_std_m = float(np.clip(search_step_m * 3.0 / np.sqrt(sharpness), 10.0, search_radius_m))

    return MapMatchResult(best_xy[0], best_xy[1], float(best_residual), confidence_std_m, matched=True)
