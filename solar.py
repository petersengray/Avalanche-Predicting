"""
solar.py
--------
Sun position (via pvlib's NREL Solar Position Algorithm) and the geometry
that converts "sun is over here in the sky" + "slope faces this way at this
angle" into how much direct sunlight actually hits that slope.
"""

import numpy as np
import pandas as pd
import pvlib
from scipy.ndimage import map_coordinates


def sun_position(lat, lon, times, tz="America/Denver", altitude_m=3000):
    """
    Get solar azimuth + elevation for a location across a set of times.

    lat, lon   : degrees
    times      : list/array of naive datetimes, or a pandas DatetimeIndex
    tz         : timezone of the times given
    altitude_m : site elevation, improves accuracy slightly (refraction/
                 atmosphere thickness)

    Returns a DataFrame indexed by time with columns:
      elevation  -- degrees above horizon (negative = sun below horizon)
      azimuth    -- degrees, 0=N, 90=E, 180=S, 270=W
      apparent_zenith
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    solpos = pvlib.solarposition.get_solarposition(
        idx, lat, lon, altitude=altitude_m
    )
    return solpos


def clear_sky_dni(lat, lon, times, altitude_m=3000, tz="America/Denver"):
    """
    Estimate clear-sky Direct Normal Irradiance (DNI, W/m^2) -- the power
    per square meter hitting a surface held perpendicular to the sun's rays.
    Uses pvlib's Ineichen clear-sky model with default turbidity.

    This is a CLEAR SKY estimate. If you have real cloud-cover forecasts
    (e.g. from api.weather.gov), multiply DNI by a clearness factor
    (roughly 1 - cloud_fraction, though real cloud attenuation of direct
    beam is closer to (1 - cloud_fraction)^3 for anything but thin cirrus).
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    solpos = sun_position(lat, lon, idx, tz=tz, altitude_m=altitude_m)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(idx, lat, lon)
    airmass_rel = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass_rel, altitude_m)
    cs = pvlib.clearsky.ineichen(
        solpos["apparent_zenith"], airmass_abs, linke_turbidity, altitude=altitude_m
    )
    return cs["dni"], solpos


def angle_of_incidence(slope_deg, aspect_deg, sun_elevation_deg, sun_azimuth_deg):
    """
    Angle between the sun's rays and the slope's normal vector (0 deg =
    sun directly perpendicular to the slope face = max possible heating;
    90+ deg = sun is edge-on or slope is in its own shadow).

    slope_deg, aspect_deg : arrays (rows, cols), from terrain.slope_aspect
    sun_elevation_deg, sun_azimuth_deg : scalars, from sun_position() for
        one instant in time

    Returns cos(incidence angle) clipped to >= 0 (negative means the sun is
    behind the slope -- self-shadowed -- so it contributes zero direct flux).
    This is the standard formula used in solar-radiation-on-terrain models.
    """
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    sun_el_rad = np.radians(sun_elevation_deg)
    sun_az_rad = np.radians(sun_azimuth_deg)

    cos_incidence = (
        np.sin(sun_el_rad) * np.cos(slope_rad)
        + np.cos(sun_el_rad) * np.sin(slope_rad) * np.cos(sun_az_rad - aspect_rad)
    )
    return np.clip(cos_incidence, 0, None)


def terrain_shadow_mask(elevation, cell_size_m, sun_azimuth_deg, sun_elevation_deg,
                         max_distance_m=None, step_m=None):
    """
    Determine which DEM cells are shadowed by OTHER terrain -- a
    neighboring ridge or peak blocking the sun -- as opposed to
    self-shadowing from the cell's own slope/aspect (that part is handled
    separately by `angle_of_incidence`).

    Method: for every cell, march outward toward the sun's azimuth and,
    at each sampled distance, compute the elevation angle that would be
    required for something at that spot to just graze the horizon from
    the cell's point of view. If the terrain anywhere along that ray
    requires a steeper angle than the sun's actual elevation, the cell is
    shadowed. This is the standard horizon-based shadow algorithm used in
    terrain solar-radiation models (e.g. GRASS r.sun's shadowing mode).

    elevation, cell_size_m : from terrain.slope_aspect / terrain.synthetic_peak
    sun_azimuth_deg, sun_elevation_deg : scalars, one instant in time
    max_distance_m : how far to search for blocking terrain. Defaults to
        ~60% of the DEM's extent. Real ridgelines can shadow slopes from
        many km away -- if your DEM tile is small, distant peaks OUTSIDE
        the loaded tile won't be accounted for. Widen the DEM extent if
        you know a specific tall peak sits just off-tile.
    step_m : distance between samples along the ray. Defaults to one
        cell width; finer steps are more accurate but slower.

    Returns
    -------
    shadow_mask : (rows, cols) boolean array, True = terrain-shadowed
    """
    rows, cols = elevation.shape
    if sun_elevation_deg <= 0:
        return np.ones((rows, cols), dtype=bool)

    if max_distance_m is None:
        max_distance_m = 0.6 * max(rows, cols) * cell_size_m
    if step_m is None:
        step_m = cell_size_m

    az_rad = np.radians(sun_azimuth_deg)
    # Direction TOWARD the sun, in (row, col) space. Row increases
    # southward, col increases eastward (matches terrain.slope_aspect's
    # convention: aspect 0=N is decreasing row, 90=E is increasing col).
    d_col = np.sin(az_rad)
    d_row = -np.cos(az_rad)

    row_idx, col_idx = np.indices((rows, cols)).astype(float)
    max_required_angle = np.full((rows, cols), -np.inf)

    n_steps = max(1, int(max_distance_m / step_m))
    for k in range(1, n_steps + 1):
        dist = k * step_m
        sample_row = row_idx + d_row * dist / cell_size_m
        sample_col = col_idx + d_col * dist / cell_size_m
        sampled_elev = map_coordinates(
            elevation, [sample_row, sample_col], order=1, mode="nearest"
        )
        angle_needed = np.degrees(np.arctan2(sampled_elev - elevation, dist))
        max_required_angle = np.maximum(max_required_angle, angle_needed)

    return max_required_angle > sun_elevation_deg


def direct_flux_on_slope(dni, slope_deg, aspect_deg, sun_elevation_deg, sun_azimuth_deg,
                          shadow_mask=None):
    """
    Direct solar flux actually landing on the slope surface (W/m^2),
    combining DNI with the incidence-angle geometry. Zero if the sun is
    below the horizon, the slope is self-shadowed (its own aspect faces
    away from the sun), or `shadow_mask` marks the cell as blocked by
    neighboring terrain (see `terrain_shadow_mask`).
    """
    if sun_elevation_deg <= 0:
        return np.zeros_like(slope_deg)
    cos_i = angle_of_incidence(slope_deg, aspect_deg, sun_elevation_deg, sun_azimuth_deg)
    flux = dni * cos_i
    if shadow_mask is not None:
        flux = np.where(shadow_mask, 0.0, flux)
    return flux
