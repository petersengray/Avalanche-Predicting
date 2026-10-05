"""
daily_forecast.py
-----------------
The daily job. For each tour and ski area, and for today plus the next two
days, works out when the snow softens to corn and when it gets too wet,
using:

  data/slope_cells.npz   per-cell terrain + skyline (from build_slopes.py)
  data/slopes.geojson    ski-area outlines and names (from build_slopes.py)
  NWS hourly forecast    temperature + cloud cover, one grid point per tour

and writes docs/data/forecast.json for the webpage.

Usage:
    python daily_forecast.py                       # live NWS forecast
    python daily_forecast.py --date 2027-04-15 --synthetic -4 7
        # offline test: clear sky, overnight low -4C / high 7C at the
        # NWS grid elevation (assumed 2,400 m)
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

import corn_model as cm
import solar
import weather

OUT_PATH = os.path.join("docs", "data", "forecast.json")
DAYS = 3
DAY_START, DAY_END, STEP = "05:00", "19:00", "15min"
SYNTH_GRID_ELEV_M = 2400.0


def hhmm(h):
    if h is None or not np.isfinite(h):
        return None
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


def time_at_fraction(frac, hours, level):
    """First time the area fraction reaches `level`, or None."""
    idx = np.nonzero(frac >= level)[0]
    return hhmm(hours[idx[0]]) if len(idx) else None


def refreeze_label(low_c):
    if not np.isfinite(low_c):
        return "unknown"
    if low_c > 0:
        return "none"
    if low_c > -3:
        return "weak"
    return "solid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="first day (YYYY-MM-DD), default today (local)")
    ap.add_argument("--synthetic", nargs=2, type=float, metavar=("TMIN", "TMAX"),
                    help="skip NWS; use a clear-sky sine-wave day with this low/high (C)")
    ap.add_argument("--out", default=OUT_PATH)
    a = ap.parse_args()

    cfg = yaml.safe_load(open("tours.yaml"))
    tz = cfg["region"]["timezone"]
    cells_npz = np.load(os.path.join("data", "slope_cells.npz"))
    cells = {k: cells_npz[k] for k in cells_npz.files}
    slope_ids = list(cells["slope_ids"])
    areas = {f["properties"]["slope_id"]: f["properties"]
             for f in json.load(open(os.path.join("data", "slopes.geojson")))["features"]}
    params = cm.CornParams()

    first_day = pd.Timestamp(a.date) if a.date else pd.Timestamp.now(tz=tz).normalize().tz_localize(None)
    days = [first_day + pd.Timedelta(days=i) for i in range(DAYS)]

    # Sun position and clear-sky light: one calculation for the whole region.
    w, s, e, n = cfg["region"]["bbox"]
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    day_times = {d: pd.date_range(f"{d:%Y-%m-%d} {DAY_START}", f"{d:%Y-%m-%d} {DAY_END}",
                                  freq=STEP, tz=tz) for d in days}
    sun = {}
    for d, t in day_times.items():
        dni, solpos = solar.clear_sky_dni(lat0, lon0, t, altitude_m=2500, tz=tz)
        dhi = _clear_sky_dhi(lat0, lon0, t, solpos)
        sun[d] = (solpos, dni.values, dhi)

    # Weather: one NWS grid point per tour summit (cached by grid id).
    wx_cache, out_tours = {}, []
    for tour in cfg["tours"]:
        lat, lon = tour["summit"]["lat"], tour["summit"]["lon"]
        if a.synthetic:
            df = weather.synthetic_hourly(first_day, DAYS, *a.synthetic, tz=tz)
            gelev, gid, source = SYNTH_GRID_ELEV_M, "synthetic", "synthetic"
        else:
            try:
                df, gelev, gid = weather.get_hourly_grid(lat, lon, tz=tz)
                if gid in wx_cache:
                    df, gelev = wx_cache[gid]
                wx_cache[gid] = (df, gelev)
                source = "nws"
            except Exception as ex:  # keep the page alive; say so loudly
                print(f"[weather] NWS failed for {tour['name']}: {type(ex).__name__}: {ex}")
                df, gelev, gid, source = None, np.nan, None, "unavailable"

        out_slopes = []
        for spec in tour["slopes"]:
            sid = spec["id"]
            if sid not in slope_ids:
                continue
            sel = cells["slope_index"] == slope_ids.index(sid)
            sub = {k: (v[sel] if hasattr(v, "shape") and v.shape[:1] == sel.shape else v)
                   for k, v in cells.items()}
            props = areas.get(sid, {})
            slope_out = dict(
                id=sid, name=spec["name"],
                area_ha=props.get("area_ha"), mean_slope_deg=props.get("mean_slope_deg"),
                mean_aspect_deg=props.get("mean_aspect_deg"),
                elev_min_ft=props.get("elev_min_ft"), elev_max_ft=props.get("elev_max_ft"),
                days=[],
            )
            for d in days:
                slope_out["days"].append(
                    forecast_slope_day(d, day_times[d], sun[d], df, gelev, sub, params))
            out_slopes.append(slope_out)

        out_tours.append(dict(id=tour["id"], name=tour["name"], summit=tour["summit"],
                              weather_source=source, nws_grid=gid,
                              grid_elev_ft=None if not np.isfinite(gelev) else round(gelev * 3.28084),
                              slopes=out_slopes))

    result = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
        timezone=tz, region=cfg["region"]["name"],
        avalanche_center=cfg["region"].get("avalanche_center"),
        days=[f"{d:%Y-%m-%d}" for d in days],
        times=[f"{t:%H:%M}" for t in day_times[days[0]]],
        model=dict(name="enhanced temperature-index melt (Pellicciotti et al. 2005)",
                   params=params.as_dict(), calibrated=False),
        disclaimer=("Experimental planning aid with uncalibrated thresholds. Not an avalanche "
                    "forecast. Always read the Sierra Avalanche Center forecast."),
        tours=out_tours,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"wrote {a.out}")
    print_summary(result)


def _clear_sky_dhi(lat, lon, times, solpos):
    import pvlib
    lt = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
    am = pvlib.atmosphere.get_absolute_airmass(
        pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"]), 2500)
    cs = pvlib.clearsky.ineichen(solpos["apparent_zenith"], am, lt, altitude=2500)
    return cs["dhi"].fillna(0).values


def forecast_slope_day(d, times, sun, wx, grid_elev_m, cells, p):
    solpos, dni, dhi = sun
    hours = np.array([t.hour + t.minute / 60 for t in times])
    base = dict(date=f"{d:%Y-%m-%d}")
    if wx is None:
        return dict(base, available=False)

    day_wx = wx.reindex(times, method="nearest", tolerance=pd.Timedelta("90min"))
    if day_wx.isna().any().any():
        return dict(base, available=False)
    night = wx.loc[(wx.index >= times[0] - pd.Timedelta(hours=11)) &
                   (wx.index <= times[0] + pd.Timedelta(hours=4))]

    t_cells = cm.cell_temperature(day_wx["temp_c"].values, grid_elev_m, cells["elev_m"], p)
    low_grid = night["temp_c"].min() if len(night) else np.nan
    low_cells = low_grid - p.lapse_c_per_m * (cells["elev_m"] - grid_elev_m)
    if not np.isfinite(low_grid):
        low_cells = np.full(len(cells["elev_m"]), -3.0)  # assume a typical night

    g = cm.sunlight_on_cells(solpos, dni, dhi, day_wx["cloud_fraction"].values, cells)
    r = cm.run_day(times, t_cells, g, low_cells, p)

    frac_soft = (r["state"] >= cm.SOFT).mean(axis=1)
    frac_wet = (r["state"] >= cm.WET).mean(axis=1)
    mean_elev = float(cells["elev_m"].mean())
    lapse = lambda tc: tc - p.lapse_c_per_m * (mean_elev - grid_elev_m)
    daytime = (hours >= 9) & (hours <= 16)
    return dict(
        base, available=True,
        overnight_low_c=None if not np.isfinite(low_grid) else round(float(lapse(low_grid)), 1),
        refreeze=refreeze_label(lapse(low_grid)),
        high_c=round(float(lapse(day_wx["temp_c"].max())), 1),
        cloud_pct=round(float(day_wx["cloud_fraction"].values[daytime].mean() * 100)),
        first_soft=time_at_fraction(frac_soft, hours, 0.1),
        soft=time_at_fraction(frac_soft, hours, 0.5),
        first_wet=time_at_fraction(frac_wet, hours, 0.1),
        wet=time_at_fraction(frac_wet, hours, 0.5),
        frac_soft=[round(float(x), 2) for x in frac_soft],
        frac_wet=[round(float(x), 2) for x in frac_wet],
    )


def print_summary(res):
    for d_i, d in enumerate(res["days"]):
        print(f"\n{d}")
        for t in res["tours"]:
            for s in t["slopes"]:
                x = s["days"][d_i]
                if not x.get("available"):
                    print(f"  {s['id']:<22} (no weather)")
                    continue
                print(f"  {s['id']:<22} refreeze {x['refreeze']:<6} "
                      f"low {x['overnight_low_c']:>5}C high {x['high_c']:>5}C  "
                      f"soft {x['first_soft'] or '--':>5}/{x['soft'] or '--':>5}  "
                      f"wet {x['first_wet'] or '--':>5}/{x['wet'] or '--':>5}")


if __name__ == "__main__":
    main()
