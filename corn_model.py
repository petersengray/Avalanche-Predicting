"""
corn_model.py
-------------
Daily melt-freeze ("corn cycle") timing for every terrain cell.

Model: an enhanced temperature-index melt model (Pellicciotti et al. 2005,
J. Glaciology 51(175)), which estimates surface melt from both sunshine and
air warmth:

    melt (mm water / hour) = max(0, TF * T + SRF * (1 - albedo) * G)   when T > T_THRESH
                           = 0                                           otherwise

    (Pellicciotti applies the temperature term only above a threshold near
    0C; here T may be negative so that cold air slows -- but does not stop --
    sun-driven softening, which matches how east faces corn up on cold,
    clear mornings.)

    T  air temperature at the cell (deg C), from the NWS forecast adjusted
       to the cell's elevation with a standard lapse rate
    G  sunlight reaching the slope (W/m^2): direct beam (zero when the cell
       is in terrain shadow or facing away from the sun) + diffuse sky light,
       both reduced by forecast cloud cover

Melt is summed from early morning. Each cell then moves through three states:

    FROZEN  -> SOFT (corn)  once cumulative melt >= soften_mm
    SOFT    -> WET          once cumulative melt >= wet_mm

soften_mm grows with how cold the previous night was (a colder night builds
a thicker refrozen crust that takes more energy to thaw). If the night never
dropped below freezing, there is no refreeze: the snow starts the day soft
and the slope is flagged.

!! The thresholds below are PLACEHOLDERS chosen to give plausible timing.
!! They must be calibrated against real observations (see calibrate.py)
!! before anyone relies on them. This is a planning aid, not an avalanche
!! forecast -- always read the Sierra Avalanche Center forecast.
"""

from dataclasses import dataclass, asdict

import numpy as np

import horizon as hz

FROZEN, SOFT, WET = 0, 1, 2


@dataclass
class CornParams:
    tf: float = 0.05            # mm / h / degC        (Pellicciotti 2005)
    srf: float = 0.0094         # mm m^2 / (W h)       (Pellicciotti 2005)
    albedo: float = 0.60        # aged spring snow
    t_thresh_c: float = -8.0    # no melt below this air temp       PLACEHOLDER
    soften_base_mm: float = 1.5 # melt needed to soften after a ~0C night  PLACEHOLDER
    soften_per_deg_mm: float = 0.4  # extra per degC of overnight cold   PLACEHOLDER
    wet_extra_mm: float = 12.0  # extra melt beyond softening until too wet  PLACEHOLDER
    lapse_c_per_m: float = 0.0065

    def as_dict(self):
        return asdict(self)


def cell_temperature(temp_grid_c, grid_elev_m, cell_elev_m, p: CornParams):
    """(n_times,) forecast temps at the NWS grid point -> (n_times, n_cells)."""
    return temp_grid_c[:, None] - p.lapse_c_per_m * (cell_elev_m[None, :] - grid_elev_m)


def sunlight_on_cells(solpos, dni, dhi, cloud_fraction, cells):
    """
    Sunlight (W/m^2) on each cell at each time.
    solpos: DataFrame with apparent_elevation, azimuth; dni/dhi: arrays (n_times,)
    cloud_fraction: (n_times,) 0-1;  cells: dict from slope_cells.npz
    Returns (n_times, n_cells).
    """
    slope = np.radians(cells["slope_deg"])
    aspect = np.radians(cells["aspect_deg"])
    horizon = cells["horizon_deg"].astype(np.float32) / 2.0
    sky_view = (1 + np.cos(slope)) / 2
    out = np.zeros((len(dni), len(slope)), dtype=np.float32)
    for i, (el, az) in enumerate(zip(solpos["apparent_elevation"], solpos["azimuth"])):
        if el <= 0:
            continue
        e, a = np.radians(el), np.radians(az)
        cos_i = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect)
        direct = dni[i] * np.clip(cos_i, 0, None)
        direct[hz.is_shaded(horizon, az, el)] = 0.0
        c = cloud_fraction[i]
        direct *= np.clip((1 - c) ** 3, 0, 1)     # clouds block the direct beam hard
        diffuse = dhi[i] * sky_view * (1 - 0.5 * c)  # crude: thick cloud dims diffuse too
        out[i] = direct + diffuse
    return out


def run_day(times, temp_cells_c, sun_wm2, overnight_low_c, p: CornParams):
    """
    times: DatetimeIndex (n_times) at regular spacing
    temp_cells_c, sun_wm2: (n_times, n_cells)
    overnight_low_c: (n_cells,) previous night's minimum air temp at each cell
    Returns dict with:
      state (n_times, n_cells) int8, soften_hour / wet_hour (n_cells) decimal
      local hour or NaN, no_refreeze (n_cells) bool, melt_mm (n_cells) total
    """
    dt_h = (times[1] - times[0]).total_seconds() / 3600
    # Sun drives surface melt even when the air is a little below freezing
    # (that's how east faces corn up on a cold, clear morning). Air colder
    # than 0C subtracts (heat lost to the air); below t_thresh_c nothing melts.
    melt_rate = np.where(
        temp_cells_c > p.t_thresh_c,
        np.clip(p.tf * temp_cells_c + p.srf * (1 - p.albedo) * sun_wm2, 0, None),
        0.0,
    )
    cum = np.cumsum(melt_rate * dt_h, axis=0)

    no_refreeze = overnight_low_c > 0.0
    soften_mm = p.soften_base_mm + p.soften_per_deg_mm * np.clip(-overnight_low_c, 0, None)
    wet_mm = soften_mm + p.wet_extra_mm
    # No refreeze: starts the day soft; a smaller push makes it wet.
    soften_mm = np.where(no_refreeze, 0.0, soften_mm)
    wet_mm = np.where(no_refreeze, p.wet_extra_mm * 0.5, wet_mm)

    state = np.full(cum.shape, FROZEN, dtype=np.int8)
    state[cum >= soften_mm[None, :]] = SOFT
    state[cum >= wet_mm[None, :]] = WET
    state[:, no_refreeze] = np.maximum(state[:, no_refreeze], SOFT)

    hours = np.array([t.hour + t.minute / 60 for t in times])

    def first_hour(mask):
        any_ = mask.any(axis=0)
        idx = mask.argmax(axis=0)
        return np.where(any_, hours[idx], np.nan)

    return dict(
        state=state,
        soften_hour=first_hour(state >= SOFT),
        wet_hour=first_hour(state >= WET),
        no_refreeze=no_refreeze,
        melt_mm=cum[-1],
    )
