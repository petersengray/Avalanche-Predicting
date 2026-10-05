"""
weather.py
----------
Real hourly temperature + cloud cover from the National Weather Service
API (api.weather.gov, free, no key required, US locations only).

NOTE ON NETWORK ACCESS: this module needs outbound internet access to
api.weather.gov. If that's unavailable (e.g. a sandboxed environment, or
you're offline, or the point is outside NWS coverage), `get_forecast()`
catches the failure and falls back to the synthetic sinusoidal model from
risk_model.py so the rest of the pipeline still runs -- it just prints a
warning so you know you're not looking at a real forecast.
"""

import numpy as np
import pandas as pd
import requests

import risk_model

NWS_USER_AGENT = "sunsafe-prototype (research/personal use)"


def _fetch_grid_data(lat, lon, timeout=15):
    """Two-step NWS lookup: find the forecast gridpoint for this lat/lon,
    then pull its raw gridded time-series data (temperature, sky cover
    at native NWS resolution, in metric units)."""
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    r = requests.get(points_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    grid_data_url = r.json()["properties"]["forecastGridData"]

    r2 = requests.get(grid_data_url, headers=headers, timeout=timeout)
    r2.raise_for_status()
    return r2.json()["properties"]


def _expand_nws_time_series(entries):
    """
    NWS gridData packs values as [{"validTime": "2026-04-15T13:00:00+00:00/PT3H",
    "value": 4.4}, ...] -- an ISO8601 start time + duration, meaning
    "value holds from validTime for the given duration". Expand this into
    a per-hour pandas Series.
    """
    rows = []
    for entry in entries:
        start_str, duration_str = entry["validTime"].split("/")
        start = pd.Timestamp(start_str)
        hours = pd.Timedelta(duration_str).total_seconds() / 3600
        for h in range(int(round(hours))):
            rows.append((start + pd.Timedelta(hours=h), entry["value"]))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def get_forecast(lat, lon, times, t_min_c=-8, t_max_c=6, tz="America/Denver"):
    """
    Get hourly air temperature (C) and cloud fraction (0-1) for each
    timestamp in `times`. Tries the real NWS forecast first; on any
    failure (network, coverage, parsing), falls back to the synthetic
    diurnal model and returns cloud_fraction=0 (clear sky) so upstream
    code keeps working.

    Returns a DataFrame indexed by `times` with columns temp_c, cloud_fraction,
    and a `source` attribute ("nws" or "synthetic_fallback") you can check.
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize(tz)

    try:
        grid = _fetch_grid_data(lat, lon)
        temp_series = _expand_nws_time_series(grid["temperature"]["values"])
        sky_series = _expand_nws_time_series(grid["skyCover"]["values"])

        temp_c = temp_series.reindex(idx, method="nearest", tolerance=pd.Timedelta("2h"))
        cloud_pct = sky_series.reindex(idx, method="nearest", tolerance=pd.Timedelta("2h"))

        if temp_c.isna().any() or cloud_pct.isna().any():
            raise ValueError("NWS series didn't cover the full requested time range")

        df = pd.DataFrame({
            "temp_c": temp_c.values,
            "cloud_fraction": (cloud_pct.values / 100.0),
        }, index=idx)
        df.attrs["source"] = "nws"
        return df

    except Exception as e:
        print(f"[weather] Live NWS forecast unavailable ({type(e).__name__}: {e}). "
              f"Falling back to synthetic diurnal temperature model "
              f"(t_min={t_min_c}C, t_max={t_max_c}C, clear sky assumed).")
        temp_c = risk_model.diurnal_temperature(idx, t_min_c, t_max_c)
        df = pd.DataFrame({
            "temp_c": temp_c,
            "cloud_fraction": np.zeros(len(idx)),
        }, index=idx)
        df.attrs["source"] = "synthetic_fallback"
        return df


def cloud_attenuation_factor(cloud_fraction):
    """
    Rough attenuation of DIRECT beam radiation from cloud cover. Direct
    beam is much more sensitive to clouds than diffuse/total irradiance --
    even partial cloud cover blocks the direct solar disk -- so this uses
    a steeper falloff than a simple linear (1 - cloud_fraction). This is a
    commonly used approximation, not a radiative-transfer calculation.
    """
    return np.clip((1 - cloud_fraction) ** 3, 0, 1)


def get_hourly_grid(lat, lon, tz="America/Los_Angeles", timeout=20):
    """
    Full NWS hourly forecast (about 7 days) for the grid cell containing
    (lat, lon). Returns (DataFrame indexed by local time with temp_c and
    cloud_fraction, grid_elevation_m, grid_id). Raises on failure -- the
    caller decides what to fall back to.
    """
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    r = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                     headers=headers, timeout=timeout)
    r.raise_for_status()
    props = r.json()["properties"]
    grid_id = f"{props['gridId']}/{props['gridX']},{props['gridY']}"
    r2 = requests.get(props["forecastGridData"], headers=headers, timeout=timeout)
    r2.raise_for_status()
    g = r2.json()["properties"]
    temp = _expand_nws_time_series(g["temperature"]["values"])
    sky = _expand_nws_time_series(g["skyCover"]["values"])
    df = pd.concat({"temp_c": temp, "cloud_fraction": sky / 100.0}, axis=1).dropna()
    df.index = df.index.tz_convert(tz)
    elev_m = float(g.get("elevation", {}).get("value") or np.nan)
    return df, elev_m, grid_id


def synthetic_hourly(start, days, t_min_c, t_max_c, tz="America/Los_Angeles"):
    """Clear-sky sinusoidal stand-in for get_hourly_grid (testing / offline)."""
    idx = pd.date_range(pd.Timestamp(start).normalize() - pd.Timedelta(days=1),
                        periods=24 * (days + 2), freq="h", tz=tz)
    temp = risk_model.diurnal_temperature(idx, t_min_c, t_max_c)
    return pd.DataFrame({"temp_c": temp, "cloud_fraction": 0.0}, index=idx)
