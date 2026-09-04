"""
Gravity Anomaly Map
======================

In the real world this map would be compiled from satellite gravimetry
(GRACE/GOCE) or field surveys (e.g. EGM2008, WGM12-style geodetic models —
see NGA/NOAA). For this MVP, we generate a SYNTHETIC anomaly field that has
a realistic "character" (multiple spatial frequencies, non-repeating): the
sum of many random-amplitude/phase/frequency 2D sinusoids plus a few
localized Gaussian peaks/troughs. This roughly resembles the rough,
location-specific anomaly pattern produced by real geological structures
(mass concentrations, density contrasts).

Anomaly magnitudes are kept in a realistic range: typical land-area gravity
anomalies are on the order of tens of mGal (1 mGal = 1e-5 m/s^2).
"""

from dataclasses import dataclass, field
import numpy as np


G0 = 9.80665  # m/s^2, standard gravity reference


@dataclass
class GravityMap:
    extent_m: float = 20000.0      # map spans [-extent/2, +extent/2] in both x and y (m)
    resolution_m: float = 50.0      # grid resolution (m)
    anomaly_amplitude_mgal: float = 40.0  # typical anomaly amplitude (mGal)
    n_components: int = 12
    seed: int = 42
    grid: np.ndarray = field(init=False, repr=False)
    xs: np.ndarray = field(init=False, repr=False)
    ys: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n = int(self.extent_m / self.resolution_m)
        self.xs = np.linspace(-self.extent_m / 2, self.extent_m / 2, n)
        self.ys = np.linspace(-self.extent_m / 2, self.extent_m / 2, n)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")

        field_mgal = np.zeros_like(X)
        for _ in range(self.n_components):
            kx = rng.uniform(0.5, 6.0) * 2 * np.pi / self.extent_m
            ky = rng.uniform(0.5, 6.0) * 2 * np.pi / self.extent_m
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.3, 1.0)
            field_mgal += amp * np.sin(kx * X + ky * Y + phase)

        # Add a few localized mass-anomaly peaks/troughs (mimicking geological features)
        for _ in range(5):
            cx = rng.uniform(-self.extent_m / 2, self.extent_m / 2)
            cy = rng.uniform(-self.extent_m / 2, self.extent_m / 2)
            sigma = rng.uniform(self.extent_m * 0.03, self.extent_m * 0.12)
            amp = rng.uniform(-1.2, 1.2)
            field_mgal += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

        # normalize and scale to the desired amplitude
        field_mgal = field_mgal / np.max(np.abs(field_mgal)) * self.anomaly_amplitude_mgal
        self.grid = field_mgal * 1e-5  # mGal -> m/s^2

    def anomaly_at(self, x: float, y: float) -> float:
        """Returns the (bilinearly interpolated) anomaly at (x, y), in m/s^2."""
        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1]):
            return 0.0

        ix = np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2)
        iy = np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2)

        x0, x1 = self.xs[ix], self.xs[ix + 1]
        y0, y1 = self.ys[iy], self.ys[iy + 1]
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)

        v00 = self.grid[ix, iy]
        v10 = self.grid[ix + 1, iy]
        v01 = self.grid[ix, iy + 1]
        v11 = self.grid[ix + 1, iy + 1]

        v0 = v00 * (1 - tx) + v10 * tx
        v1 = v01 * (1 - tx) + v11 * tx
        return float(v0 * (1 - ty) + v1 * ty)

    def gravity_at(self, x: float, y: float) -> float:
        return G0 + self.anomaly_at(x, y)
