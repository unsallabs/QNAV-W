"""
Visualization
================

- plot_trajectories : plots ground truth + all configurations' trajectories
  on the same axes (with a zoom inset on the final drift region), and marks
  the GPS-off point.
- plot_drift         : compares position error over time (drift).
- export_leaflet_map : produces a standalone HTML file showing the
  trajectories on a real-world map (Leaflet.js, OpenStreetMap tiles).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from qnav.simulation.trajectory import local_xy_to_latlon


def plot_trajectories(ground_truth, tracks: dict, gps_off_xy=None, save_path=None,
                       zoom_from_idx=None, start_label="Start"):
    """
    tracks: {label: (x_array, y_array), ...}
    zoom_from_idx: if given, adds a zoom inset around the trajectory endpoint
                   (e.g. from the GPS-off index onward), since meter-scale
                   drift differences are invisible at full-route scale.
    start_label: legend label for the starting point marker (e.g.
                 "Start (Tokyo, Japan)"); location-agnostic by default.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    ax.plot(ground_truth.x, ground_truth.y, color="black", linewidth=2.5,
            label="Ground Truth", zorder=5)
    for i, (label, (x, y)) in enumerate(tracks.items()):
        ax.plot(x, y, linewidth=1.6, label=label, color=colors[i % len(colors)], alpha=0.9)

    if gps_off_xy is not None:
        ax.scatter(*gps_off_xy, color="black", marker="x", s=140, zorder=6, label="GPS OFF")
    ax.scatter(ground_truth.x[0], ground_truth.y[0], color="green", marker="o",
               s=100, zorder=6, label=start_label)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Q-Nav: Trajectory Comparison (GPS-Denied Navigation)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    if zoom_from_idx is not None:
        # Zoom in on the FINAL point only (drift accumulates most there);
        # showing the whole post-GPS-off leg would hide meter-scale differences.
        end_x = ground_truth.x[-1]
        end_y = ground_truth.y[-1]

        all_end_x = [end_x]
        all_end_y = [end_y]
        for (x, y) in tracks.values():
            all_end_x.append(np.asarray(x)[-1])
            all_end_y.append(np.asarray(y)[-1])

        cx, cy = np.mean(all_end_x), np.mean(all_end_y)
        half_span = max(np.max(np.abs(np.array(all_end_x) - cx)),
                         np.max(np.abs(np.array(all_end_y) - cy)), 10.0) * 1.6
        x0, x1 = cx - half_span, cx + half_span
        y0, y1 = cy - half_span, cy + half_span

        # crop the tail of the route to match this window
        tail = max(zoom_from_idx, len(ground_truth.x) - 400)
        gx = ground_truth.x[tail:]
        gy = ground_truth.y[tail:]

        axins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=2)
        axins.plot(gx, gy, color="black", linewidth=2.5, zorder=5)
        for i, (label, (x, y)) in enumerate(tracks.items()):
            axins.plot(np.asarray(x)[tail:], np.asarray(y)[tail:],
                       linewidth=2.0, color=colors[i % len(colors)], alpha=0.95)
            axins.scatter(np.asarray(x)[-1], np.asarray(y)[-1],
                          color=colors[i % len(colors)], s=30, zorder=6)
        axins.scatter(end_x, end_y, color="black", marker="*", s=90, zorder=7)
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_aspect("equal", adjustable="box")
        axins.set_title("Zoom: drift at route end", fontsize=9)
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drift(t, metrics_list, gps_off_time=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i, m in enumerate(metrics_list):
        ax.plot(t, m.position_error_m, label=f"{m.label} (RMSE={m.rmse_m:.1f} m)",
                color=colors[i % len(colors)], linewidth=1.8)

    if gps_off_time is not None:
        ax.axvline(gps_off_time, color="black", linestyle="--", alpha=0.6, label="GPS OFF")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Drift Comparison Over Time")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_leaflet_map(ground_truth, tracks: dict, save_path: str,
                        lat0=None, lon0=None, decimate=25, location_name="Start",
                        zoom_start=13):
    """
    Produces a standalone HTML file showing the trajectories on a real
    OpenStreetMap, centered on (lat0, lon0) — i.e. wherever the demo's
    start location resolved to (see qnav.simulation.location). Works for
    any location worldwide; there is nothing location-specific baked in
    beyond the passed-in coordinates.
    """
    from qnav.simulation.trajectory import DEFAULT_LAT, DEFAULT_LON
    # Use explicit None-checks (not `or`) so a valid lat/lon of 0.0
    # (e.g. a location on the equator or prime meridian) isn't discarded.
    lat0 = DEFAULT_LAT if lat0 is None else lat0
    lon0 = DEFAULT_LON if lon0 is None else lon0

    def to_latlon_list(x, y):
        x = np.asarray(x)[::decimate]
        y = np.asarray(y)[::decimate]
        lat, lon = local_xy_to_latlon(x, y, lat0, lon0)
        return list(zip(lat.tolist(), lon.tolist()))

    layers = {"Ground Truth": to_latlon_list(ground_truth.x, ground_truth.y)}
    for label, (x, y) in tracks.items():
        layers[label] = to_latlon_list(x, y)

    colors = {
        "Ground Truth": "#000000",
        "Classical IMU (Dead Reckoning)": "#d62728",
        "Quantum-Enhanced IMU": "#1f77b4",
        "Q-Nav (Quantum + Gravity Map)": "#2ca02c",
    }

    safe_location_name = location_name.replace("'", "\\'")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Q-Nav — {location_name} Route Comparison</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  .legend {{ background: white; padding: 8px 12px; border-radius: 6px; font-family: sans-serif; font-size: 13px; }}
  .legend div {{ margin: 2px 0; }}
  .legend span {{ display:inline-block; width:14px; height:3px; margin-right:6px; vertical-align:middle; }}
</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const layers = {json.dumps(layers)};
  const colors = {json.dumps(colors)};

  const map = L.map('map').setView([{lat0}, {lon0}], {zoom_start});
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  const legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
    const div = L.DomUtil.create('div', 'legend');
    let html = '<b>Q-Nav Routes</b><br/>';
    for (const label in layers) {{
      const c = colors[label] || '#888';
      html += `<div><span style="background:${{c}}"></span>${{label}}</div>`;
    }}
    div.innerHTML = html;
    return div;
  }};
  legend.addTo(map);

  for (const label in layers) {{
    const latlngs = layers[label];
    const color = colors[label] || '#888';
    const weight = (label === 'Ground Truth') ? 5 : 3;
    L.polyline(latlngs, {{color: color, weight: weight, opacity: 0.85}}).addTo(map);
  }}

  L.marker(layers['Ground Truth'][0]).addTo(map).bindPopup('Start: {safe_location_name}').openPopup();
</script>
</body>
</html>
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
