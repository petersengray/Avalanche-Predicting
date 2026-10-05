"""
calibrate.py
------------
Fits the heuristic's two tunable knobs -- energy_threshold_whm2 and
temp_threshold_c -- against real field observations of when a slope
actually went isothermal / wet-loose-active, by grid-searching for the
combination that minimizes average prediction error.

HOW TO USE
----------
Replace PLACEHOLDER_OBSERVATIONS below with your own logged field
observations. Each entry is a single point you (or a report) actually
observed going isothermal/wet/active -- not a whole terrain grid. You
need, per observation: date, lat/lon, site altitude, the slope angle +
aspect of that specific spot, that day's low/high temps, and the local
clock time you observed the change. The more observations across
different aspects/dates, the better-constrained the fit.

Good sources: your own field notes, avalanche center observations
(e.g. CAIC/utahavalanchecenter.org public obs), or your own trip reports
if you noted timing.
"""

import numpy as np
import pandas as pd

import solar
import risk_model


PLACEHOLDER_OBSERVATIONS = [
    # EXAMPLE ROWS ONLY -- these numbers are illustrative, not real
    # observations. Delete/replace with your own logged data before
    # trusting calibrate()'s output.
    # observed_hour is local 24h decimal time (e.g. 11.5 = 11:30am)
    dict(date="2026-03-20", lat=39.62, lon=-105.90, altitude_m=3600,
         slope_deg=35, aspect_deg=135, t_min_c=-6, t_max_c=8,
         observed_hour=10.5),
    dict(date="2026-03-20", lat=39.62, lon=-105.90, altitude_m=3600,
         slope_deg=30, aspect_deg=270, t_min_c=-6, t_max_c=8,
         observed_hour=13.5),
    dict(date="2026-04-02", lat=39.62, lon=-105.90, altitude_m=3600,
         slope_deg=32, aspect_deg=90, t_min_c=-3, t_max_c=10,
         observed_hour=9.75),
]


def predict_unsafe_hour(obs, energy_threshold_whm2, temp_threshold_c,
                         tz="America/Denver"):
    """Run the point-scale model for one observation's location/date/
    slope/aspect and return the predicted crossing hour (or NaN if it
    never crosses that day)."""
    times = pd.date_range(f"{obs['date']} 05:00", f"{obs['date']} 20:00",
                           freq="15min", tz=tz)
    dni, solpos = solar.clear_sky_dni(obs["lat"], obs["lon"], times,
                                       altitude_m=obs["altitude_m"], tz=tz)
    air_temp_c = risk_model.diurnal_temperature(times, obs["t_min_c"], obs["t_max_c"])

    flux = np.array([
        solar.direct_flux_on_slope(
            dni.iloc[i], obs["slope_deg"], obs["aspect_deg"],
            solpos["apparent_elevation"].iloc[i], solpos["azimuth"].iloc[i])
        for i in range(len(times))
    ])
    # compute_unsafe_time expects (n_times, rows, cols); treat this single
    # point as a 1x1 "grid"
    flux_stack = flux.reshape(-1, 1, 1)

    unsafe_hour, _ = risk_model.compute_unsafe_time(
        flux_stack, times, air_temp_c,
        energy_threshold_whm2=energy_threshold_whm2,
        temp_threshold_c=temp_threshold_c,
    )
    return unsafe_hour[0, 0]


def calibrate(observations=PLACEHOLDER_OBSERVATIONS,
              energy_range=np.arange(150, 601, 25),
              temp_range=np.arange(-6, 2.1, 1.0),
              never_crossed_penalty_hours=6.0):
    """
    Grid search over (energy_threshold, temp_threshold) combinations,
    scoring each by mean absolute error (hours) against observed timing.

    Returns
    -------
    best   : dict, the best-fit combo and its error
    table  : DataFrame of every combo tried, sorted best-to-worst -- worth
             inspecting for how sharply peaked (well-constrained) vs. flat
             (under-determined, need more/more-varied observations) the
             fit is
    """
    results = []
    for e_thresh in energy_range:
        for t_thresh in temp_range:
            errors = []
            for obs in observations:
                pred = predict_unsafe_hour(obs, e_thresh, t_thresh)
                if np.isnan(pred):
                    errors.append(never_crossed_penalty_hours)
                else:
                    errors.append(abs(pred - obs["observed_hour"]))
            results.append({
                "energy_threshold_whm2": e_thresh,
                "temp_threshold_c": t_thresh,
                "mean_abs_error_hours": np.mean(errors),
            })
    table = pd.DataFrame(results).sort_values("mean_abs_error_hours").reset_index(drop=True)
    return table.iloc[0].to_dict(), table


if __name__ == "__main__":
    n_obs = len(PLACEHOLDER_OBSERVATIONS)
    print(f"Calibrating against {n_obs} observation(s) "
          f"-- PLACEHOLDER data, replace with real field observations before trusting this.\n")

    best, table = calibrate()
    print("Best fit:")
    print(f"  energy_threshold_whm2 = {best['energy_threshold_whm2']:.0f}")
    print(f"  temp_threshold_c      = {best['temp_threshold_c']:.1f}")
    print(f"  mean_abs_error_hours  = {best['mean_abs_error_hours']:.2f}")
    print("\nTop 10 combos (check whether the fit is sharply peaked or flat):")
    print(table.head(10).to_string(index=False))
