"""
risk_model.py
-------------
The "when does it become unsafe" heuristic.

This is deliberately simple (as chosen): we track cumulative solar energy
absorbed per slope cell since sunrise (Wh/m^2), and flag a cell as
"sun-affected" once BOTH:
  1. cumulative energy exceeds a threshold (proxy for enough heating to
     start melting/weakening surface bonds), AND
  2. air temperature is at/above a freezing-adjacent threshold (because if
     it's -15C out, a lot of sun can hit a slope without meaningfully
     warming the snow surface)

This mirrors how forecasters talk about it qualitatively ("give it until
mid-morning on the sunny aspects once temps are above freezing"), but it is
NOT a substitute for an energy-balance snow model or real avalanche
forecasting -- see the caveats at the bottom of this file.
"""

import numpy as np


def diurnal_temperature(times, t_min_c, t_max_c, t_min_hour=6, t_max_hour=15):
    """
    Simple sinusoidal model of air temperature through the day, since we're
    not wiring up a live weather API in this prototype. Swap this out for
    real hourly forecast data (e.g. api.weather.gov) when you productionize.

    times : pandas DatetimeIndex
    t_min_c, t_max_c : forecast low/high for the day (Celsius)
    t_min_hour, t_max_hour : local hour of day the low/high occur
    """
    hours = np.array([t.hour + t.minute / 60 for t in times])
    # Cosine centered so trough sits at t_min_hour, peak at t_max_hour
    phase = 2 * np.pi * (hours - t_min_hour) / (2 * (t_max_hour - t_min_hour))
    temp = t_min_c + (t_max_c - t_min_c) * (1 - np.cos(phase)) / 2
    # Clamp the shape outside the min/max window so it doesn't oscillate
    # unrealistically overnight
    temp = np.clip(temp, t_min_c, t_max_c)
    return temp


def compute_unsafe_time(
    flux_stack, times, air_temp_c,
    energy_threshold_whm2=350.0,
    temp_threshold_c=-2.0,
    dt_hours=None,
):
    """
    For every terrain cell, find the first time of day it becomes
    "sun-affected / potentially unsafe".

    flux_stack : (n_times, rows, cols) array of direct flux (W/m^2),
                 one slice per timestep -- from solar.direct_flux_on_slope
    times      : list of the timestamps corresponding to flux_stack's
                 first axis
    air_temp_c : array, same length as times -- air temp at each timestep
    energy_threshold_whm2 : cumulative energy (Wh/m^2) needed before a
                 slope is flagged. This is a tunable knob, not a physical
                 constant -- calibrate against your own field observations.
    temp_threshold_c : air temp must be at/above this for heating to
                 "count" toward risk
    dt_hours   : timestep length in hours (auto-detected from `times` if
                 not given)

    Returns
    -------
    unsafe_hour : (rows, cols) array, decimal hour of day (local) each cell
                 first crosses the threshold, or np.nan if it never does
                 that day
    cumulative_energy : (n_times, rows, cols) running Wh/m^2 per cell, in
                 case you want to plot the buildup over the day
    """
    n_times, rows, cols = flux_stack.shape

    if dt_hours is None:
        dt_hours = (times[1] - times[0]).total_seconds() / 3600.0

    cumulative_energy = np.zeros_like(flux_stack)
    unsafe_hour = np.full((rows, cols), np.nan)
    running = np.zeros((rows, cols))

    for i in range(n_times):
        temp_ok = air_temp_c[i] >= temp_threshold_c
        # Only accumulate energy toward the threshold while temp condition
        # is met -- cold, sunny snow still gets radiation but stays stable
        if temp_ok:
            running = running + flux_stack[i] * dt_hours
        cumulative_energy[i] = running

        newly_unsafe = (running >= energy_threshold_whm2) & np.isnan(unsafe_hour)
        hour_of_day = times[i].hour + times[i].minute / 60
        unsafe_hour[newly_unsafe] = hour_of_day

    return unsafe_hour, cumulative_energy
