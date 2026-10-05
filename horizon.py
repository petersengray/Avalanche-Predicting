"""
horizon.py
----------
Precomputes, for a set of terrain cells, the height of the skyline in every
compass direction (the "horizon angle"). With that stored, deciding whether
a cell is shaded by surrounding terrain at any moment is a lookup:

    shaded  <=>  sun elevation < horizon angle in the sun's direction

This replaces running the full ray-march (solar.terrain_shadow_mask) for
every timestep, so the daily forecast job doesn't need the elevation model
at all -- only the small file of per-cell values written here.
"""

import numpy as np
from scipy.ndimage import map_coordinates

N_AZIMUTHS = 72  # every 5 degrees


def horizon_angles(elevation, cell_m, rows, cols, max_dist_m=10000, step_m=None,
                   n_azimuths=N_AZIMUTHS):
    """
    elevation : full DEM (m), projected grid with north up
    rows, cols : integer arrays of the cells to evaluate
    Returns (len(rows), n_azimuths) array of horizon angles in degrees
    (0 = flat horizon; can be negative where the land falls away).
    Azimuth k is k * 360 / n_azimuths degrees (0 = N, 90 = E).
    """
    step_m = step_m or cell_m
    elev = np.nan_to_num(elevation, nan=np.nanmin(elevation))
    z0 = elev[rows, cols]
    r0, c0 = rows.astype(float), cols.astype(float)
    out = np.full((len(rows), n_azimuths), -90.0, dtype=np.float32)
    dists = np.arange(step_m, max_dist_m + step_m, step_m)
    # Use coarser sampling further out (distant terrain is smoothly varying).
    dists = np.unique(np.concatenate([dists[dists <= 1000],
                                      dists[(dists > 1000)][::3]]))
    for k in range(n_azimuths):
        az = np.radians(k * 360.0 / n_azimuths)
        d_col, d_row = np.sin(az), -np.cos(az)
        best = np.full(len(rows), -90.0)
        for d in dists:
            rr = r0 + d_row * d / cell_m
            cc = c0 + d_col * d / cell_m
            z = map_coordinates(elev, [rr, cc], order=1, mode="nearest")
            best = np.maximum(best, np.degrees(np.arctan2(z - z0, d)))
        out[:, k] = best
    return out


def is_shaded(horizon, sun_azimuth_deg, sun_elevation_deg):
    """horizon: (n_cells, n_az). Returns bool (n_cells,) for one instant."""
    n_az = horizon.shape[1]
    pos = (sun_azimuth_deg % 360) / (360.0 / n_az)
    k0 = int(np.floor(pos)) % n_az
    k1 = (k0 + 1) % n_az
    w = pos - np.floor(pos)
    h = (1 - w) * horizon[:, k0] + w * horizon[:, k1]
    return sun_elevation_deg <= h
