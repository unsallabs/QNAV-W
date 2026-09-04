// Q-Nav Web UI — vanilla JS, no build step, no charting library.
// Talks to the local Python backend (qnav/web/server.py) via a tiny JSON API.

const COLORS = { gt: "#111111", classical: "#d62728", quantum: "#1f77b4", qnav: "#2ca02c" };
const PLAYBACK_MS = 14000; // real-time duration of one full animation playthrough

let map, layerGroup;
let markers = {};
let gpsOffMarkerObj = null;
let data = null;          // last /api/simulate response
let gpsOffIndex = 0;
let currentIndex = 0;
let playing = false;
let animStartTime = 0;
let animStartIndex = 0;
let animFrameId = null;

function initMap() {
  map = L.map("map").setView([37.7749, -122.4194], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  layerGroup = L.layerGroup().addTo(map);
}

function setStatus(msg, isError) {
  const el = document.getElementById("status-line");
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

function closestIndex(tArray, target) {
  let best = 0, bestDiff = Infinity;
  for (let i = 0; i < tArray.length; i++) {
    const diff = Math.abs(tArray[i] - target);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  }
  return best;
}

function readConfig() {
  const loc = document.getElementById("location").value.trim();
  const latVal = document.getElementById("lat").value;
  const lonVal = document.getElementById("lon").value;
  return {
    location: loc || null,
    lat: latVal === "" ? null : parseFloat(latVal),
    lon: lonVal === "" ? null : parseFloat(lonVal),
    duration_s: parseFloat(document.getElementById("duration").value) || 900,
    gps_off_time_s: parseFloat(document.getElementById("gps-cutoff").value) || 300,
    gravity_matching: document.getElementById("gravity-matching").checked,
    use_road_network: true,
  };
}

async function runSimulation(e) {
  e.preventDefault();
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  setStatus("Running simulation… (road-network lookup may take a few seconds)");
  stopAnimation();

  try {
    const resp = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readConfig()),
    });
    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);

    data = payload;
    setStatus(`Done — route source: ${data.route_source}, ` +
               `${(data.total_distance_m / 1000).toFixed(2)} km.`);
    renderResult(data);
  } catch (err) {
    setStatus("Error: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function renderResult(d) {
  layerGroup.clearLayers();
  markers = {};
  gpsOffMarkerObj = null;

  const gtLatLngs = d.ground_truth.lat.map((la, i) => [la, d.ground_truth.lon[i]]);
  L.polyline(gtLatLngs, { color: COLORS.gt, weight: 4, opacity: 0.9 }).addTo(layerGroup);

  ["classical", "quantum", "qnav"].forEach((key) => {
    const latlngs = d[key].lat.map((la, i) => [la, d[key].lon[i]]);
    L.polyline(latlngs, { color: COLORS[key], weight: 2.5, opacity: 0.9 }).addTo(layerGroup);
    markers[key] = L.circleMarker(latlngs[0], { radius: 5, color: COLORS[key], fillOpacity: 1 }).addTo(layerGroup);
  });
  markers.gt = L.circleMarker(gtLatLngs[0], { radius: 5, color: COLORS.gt, fillOpacity: 1 }).addTo(layerGroup);

  map.fitBounds(L.polyline(gtLatLngs).getBounds(), { padding: [20, 20] });

  gpsOffIndex = closestIndex(d.t, d.gps_off_time_s);
  const gpsOffLatLng = gtLatLngs[gpsOffIndex];
  gpsOffMarkerObj = L.circleMarker(gpsOffLatLng, {
    radius: 8, color: "#000", weight: 2, fillColor: "#fff", fillOpacity: 1,
  }).bindPopup(`GPS OFF at t=${d.gps_off_time_s}s`).addTo(layerGroup);

  d.gravity_match_events.forEach((ev) => {
    L.circleMarker([ev.lat, ev.lon], {
      radius: 4, color: COLORS.qnav, fillOpacity: 0.6, weight: 1,
    }).bindPopup(`Gravity-map fix at t=${ev.t}s`).addTo(layerGroup);
  });

  renderMetricsTable(d);

  document.getElementById("results-panel").classList.remove("hidden");
  document.getElementById("export-csv").href = "/api/export/csv";
  document.getElementById("export-json").href = "/api/export/json";

  const slider = document.getElementById("progress");
  slider.min = 0;
  slider.max = d.t.length - 1;
  slider.value = 0;
  slider.disabled = false;
  document.getElementById("play-btn").disabled = false;

  currentIndex = 0;
  updateFrame(0);
}

function renderMetricsTable(d) {
  const rows = [
    ["classical", COLORS.classical, "Classical IMU"],
    ["quantum", COLORS.quantum, "Quantum-Enhanced"],
    ["qnav", COLORS.qnav, "Q-Nav"],
  ];
  let html = "<tr><th>Config</th><th>RMSE</th><th>Final</th></tr>";
  rows.forEach(([key, color, label]) => {
    const m = d.metrics[key];
    html += `<tr><td><span class="swatch" style="background:${color}"></span>${label}</td>` +
            `<td>${m.rmse_m} m</td><td>${m.final_error_m} m</td></tr>`;
  });
  document.getElementById("metrics-table").innerHTML = html;

  document.getElementById("improvement-line").textContent =
    `Quantum-Enhanced: ${d.improvement_vs_classical_pct.quantum}% lower RMSE than classical. ` +
    `Q-Nav: ${d.improvement_vs_classical_pct.qnav}% lower RMSE than classical. ` +
    `Gravity-map fixes applied: ${d.gravity_matches_applied}.`;
}

function updateFrame(index) {
  if (!data) return;
  currentIndex = index;
  const t = data.t[index];

  markers.gt.setLatLng([data.ground_truth.lat[index], data.ground_truth.lon[index]]);
  ["classical", "quantum", "qnav"].forEach((key) => {
    markers[key].setLatLng([data[key].lat[index], data[key].lon[index]]);
  });

  document.getElementById("time-label").textContent = `t = ${t.toFixed(0)} s`;
  document.getElementById("progress").value = index;

  const banner = document.getElementById("gps-banner");
  if (index >= gpsOffIndex) banner.classList.remove("hidden");
  else banner.classList.add("hidden");

  drawDriftChart(data, index);
}

function stopAnimation() {
  playing = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = null;
  document.getElementById("play-btn").textContent = "▶ Play";
}

function animationStep(timestamp) {
  if (!playing || !data) return;
  const n = data.t.length;
  const elapsed = timestamp - animStartTime;
  const progress = Math.min(elapsed / PLAYBACK_MS, 1);
  const index = Math.min(n - 1, Math.round(animStartIndex + progress * (n - 1 - animStartIndex)));
  updateFrame(index);

  if (progress >= 1) {
    stopAnimation();
    return;
  }
  animFrameId = requestAnimationFrame(animationStep);
}

function togglePlay() {
  if (!data) return;
  if (playing) {
    stopAnimation();
    return;
  }
  playing = true;
  document.getElementById("play-btn").textContent = "⏸ Pause";
  const n = data.t.length;
  animStartIndex = currentIndex >= n - 1 ? 0 : currentIndex;
  animStartTime = performance.now();
  animFrameId = requestAnimationFrame(animationStep);
}

function onScrub() {
  stopAnimation();
  const index = parseInt(document.getElementById("progress").value, 10);
  updateFrame(index);
}

function drawDriftChart(d, playheadIndex) {
  const canvas = document.getElementById("drift-chart");
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth);
  const h = (canvas.height = canvas.clientHeight || 180);
  ctx.clearRect(0, 0, w, h);

  const padL = 42, padR = 10, padT = 10, padB = 20;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const tMax = d.duration_s;
  const allErr = [...d.drift.classical, ...d.drift.quantum, ...d.drift.qnav];
  const yMax = Math.max(...allErr, 1) * 1.1;

  const xScale = (t) => padL + (t / tMax) * plotW;
  const yScale = (v) => padT + plotH - (v / yMax) * plotH;

  // axes
  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.fillStyle = "#666";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(yMax.toFixed(0) + " m", padL - 4, padT + 8);
  ctx.fillText("0", padL - 4, padT + plotH);
  ctx.textAlign = "center";
  ctx.fillText("0", padL, padT + plotH + 12);
  ctx.fillText(tMax.toFixed(0) + " s", padL + plotW, padT + plotH + 12);

  // GPS-off dashed line
  const gpsX = xScale(d.gps_off_time_s);
  ctx.strokeStyle = "#999";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(gpsX, padT);
  ctx.lineTo(gpsX, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // series
  const drawSeries = (values, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = xScale(d.t[i]), y = yScale(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries(d.drift.classical, COLORS.classical);
  drawSeries(d.drift.quantum, COLORS.quantum);
  drawSeries(d.drift.qnav, COLORS.qnav);

  // playhead
  if (playheadIndex != null) {
    const px = xScale(d.t[playheadIndex]);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
    ctx.stroke();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  document.getElementById("config-form").addEventListener("submit", runSimulation);
  document.getElementById("play-btn").addEventListener("click", togglePlay);
  document.getElementById("progress").addEventListener("input", onScrub);
  window.addEventListener("resize", () => { if (data) drawDriftChart(data, currentIndex); });
});
