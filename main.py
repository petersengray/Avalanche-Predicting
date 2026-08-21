# %% [markdown]
# # SunSafe prototype: solar-timing wet-slide risk model
#
# Estimates, for every point on a slope, roughly what time of day it
# crosses from "sun hasn't done much yet" into "sun-affected / worth
# watching," based on slope angle, aspect, sun position, terrain
# shadowing, and air temperature.
#
# This build adds: real DEM fetching (USGS 3DEP), real weather fetching
# (NWS), terrain-on-terrain shadow casting, and cloud attenuation of
# direct radiation. Real-data fetches gracefully fall back to synthetic
# models if network access isn't available (see terrain.py / weather.py).
#
# Still a HEURISTIC PROTOTYPE, not a validated avalanche forecasting tool
# -- see caveats at the bottom.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LightSource

import terrain
import solar
import weather
import risk_model

# %% [markdown]
# ## 1. Config

# %%
LAT, LON = 39.62, -105.90          # example: Loveland Pass area, CO
ALT_M = 3600
TZ = "America/Denver"
DATE = "2026-04-15"                 # spring corn-cycle season

USE_REAL_DEM = True                 # tries USGS 3DEP, falls back to synthetic
USE_REAL_WEATHER = True             # tries NWS, falls back to synthetic diurnal model
DEM_RADIUS_KM = 1.5
DEM_RESOLUTION_M = 10

# Calibrated (or placeholder -- see calibrate.py) heuristic thresholds
ENERGY_THRESHOLD_WHM2 = 350.0
TEMP_THRESHOLD_C = -2.0

# Fallback synthetic-weather params, used only if USE_REAL_WEATHER fails
FALLBACK_T_MIN_C, FALLBACK_T_MAX_C = -8, 6

# %% [markdown]
# ## 2. Terrain: real DEM if available, else synthetic ridge + neighbor
# (the neighbor peak exists so terrain shadowing has something to do)

# %%
if USE_REAL_DEM:
    elevation, cell_size_m = terrain.get_dem(
        LAT, LON, radius_km=DEM_RADIUS_KM, resolution_m=DEM_RESOLUTION_M,
        fallback_fn=terrain.synthetic_ridge_with_neighbor,
        synthetic_fallback_kwargs=dict(size=200, cell_size_m=15,
                                        main_peak_height_m=500,
                                        neighbor_height_m=650),
    )
else:
    elevation, cell_size_m = terrain.synthetic_ridge_with_neighbor(
        size=200, cell_size_m=15, main_peak_height_m=500, neighbor_height_m=650,
    )

slope_deg, aspect_deg = terrain.slope_aspect(elevation, cell_size_m)
print(f"Elevation range: {elevation.min():.0f}-{elevation.max():.0f} m, "
      f"cell size: {cell_size_m:.1f} m")
print(f"Slope range: {slope_deg.min():.0f}-{slope_deg.max():.0f} deg")

# %% [markdown]
# ## 3. Time series, sun position, clear-sky DNI

# %%
times = pd.date_range(f"{DATE} 06:00", f"{DATE} 19:00", freq="15min", tz=TZ)
dni, solpos = solar.clear_sky_dni(LAT, LON, times, altitude_m=ALT_M, tz=TZ)

# %% [markdown]
# ## 4. Weather: real NWS forecast if available, else synthetic diurnal model

# %%
if USE_REAL_WEATHER:
    fc = weather.get_forecast(LAT, LON, times, t_min_c=FALLBACK_T_MIN_C,
                               t_max_c=FALLBACK_T_MAX_C, tz=TZ)
else:
    fc = pd.DataFrame({
        "temp_c": risk_model.diurnal_temperature(times, FALLBACK_T_MIN_C, FALLBACK_T_MAX_C),
        "cloud_fraction": np.zeros(len(times)),
    }, index=times)
    fc.attrs["source"] = "synthetic_fallback"

air_temp_c = fc["temp_c"].values
cloud_fraction = fc["cloud_fraction"].values
print(f"Weather source: {fc.attrs.get('source')}")

# Attenuate DNI by cloud cover (no-op if cloud_fraction is all zero)
dni_effective = dni.values * weather.cloud_attenuation_factor(cloud_fraction)

# %% [markdown]
# ## 5. Direct flux on every slope cell, every timestep -- WITH terrain shadowing

# %%
flux_stack = np.zeros((len(times), *slope_deg.shape))
for i, t in enumerate(times):
    sun_el = solpos["apparent_elevation"].iloc[i]
    sun_az = solpos["azimuth"].iloc[i]
    shadow_mask = solar.terrain_shadow_mask(elevation, cell_size_m, sun_az, sun_el)
    flux_stack[i] = solar.direct_flux_on_slope(
        dni_effective[i], slope_deg, aspect_deg, sun_el, sun_az,
        shadow_mask=shadow_mask,
    )

# %% [markdown]
# ## 6. Apply the heuristic risk model

# %%
unsafe_hour, cumulative_energy = risk_model.compute_unsafe_time(
    flux_stack, times, air_temp_c,
    energy_threshold_whm2=ENERGY_THRESHOLD_WHM2,
    temp_threshold_c=TEMP_THRESHOLD_C,
)

# %% [markdown]
# ## 7. Visualize: terrain, aspect, slope, and risk timing

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elevation, vert_exag=1.5, dx=cell_size_m, dy=cell_size_m)

ax = axes[0, 0]
ax.imshow(hillshade, cmap="gray")
ax.set_title("Terrain (hillshade)")
ax.axis("off")

ax = axes[0, 1]
im = ax.imshow(aspect_deg, cmap="hsv", vmin=0, vmax=360)
ax.set_title("Aspect (compass direction slope faces)")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, label="degrees (0=N, 90=E, 180=S, 270=W)")

ax = axes[1, 0]
im = ax.imshow(slope_deg, cmap="viridis", vmin=0, vmax=60)
ax.set_title("Slope angle")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, label="degrees")

ax = axes[1, 1]
masked = np.ma.masked_invalid(unsafe_hour)
im = ax.imshow(masked, cmap="RdYlBu_r", vmin=8, vmax=17)
ax.set_title("Hour of day slope becomes sun-affected\n(gray = doesn't cross threshold)")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, label="hour (local)")

plt.tight_layout()
plt.savefig("/home/claude/sunsafe/terrain_and_risk.png", dpi=140)
print("Saved terrain_and_risk.png")

# %% [markdown]
# ## 8. Shadow effect, before/after
# Pick one mid-morning timestep (when shadowing from the neighbor peak
# matters most) and show flux WITH vs WITHOUT terrain shadowing side by
# side, so the effect of step #3 is visible on its own.

# %%
demo_idx = len(times) // 4  # a mid-morning timestep
sun_el = solpos["apparent_elevation"].iloc[demo_idx]
sun_az = solpos["azimuth"].iloc[demo_idx]
shadow_mask_demo = solar.terrain_shadow_mask(elevation, cell_size_m, sun_az, sun_el)

flux_no_shadow = solar.direct_flux_on_slope(
    dni_effective[demo_idx], slope_deg, aspect_deg, sun_el, sun_az, shadow_mask=None)
flux_with_shadow = solar.direct_flux_on_slope(
    dni_effective[demo_idx], slope_deg, aspect_deg, sun_el, sun_az, shadow_mask=shadow_mask_demo)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
vmax = max(flux_no_shadow.max(), 1)

im = axes[0].imshow(flux_no_shadow, cmap="inferno", vmin=0, vmax=vmax)
axes[0].set_title(f"Direct flux, self-shadow only\n{times[demo_idx].strftime('%H:%M')}")
axes[0].axis("off")
plt.colorbar(im, ax=axes[0], fraction=0.046, label="W/m\u00b2")

im = axes[1].imshow(flux_with_shadow, cmap="inferno", vmin=0, vmax=vmax)
axes[1].set_title("Direct flux, WITH terrain shadowing")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, label="W/m\u00b2")

diff_pct = 100 * (flux_no_shadow - flux_with_shadow) / np.maximum(flux_no_shadow, 1e-6)
im = axes[2].imshow(diff_pct, cmap="Reds", vmin=0, vmax=100)
axes[2].set_title("% flux lost to terrain shadowing")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, label="%")

plt.tight_layout()
plt.savefig("/home/claude/sunsafe/shadow_comparison.png", dpi=140)
print("Saved shadow_comparison.png "
      f"(sun at {sun_el:.1f}\u00b0 elevation, {sun_az:.0f}\u00b0 azimuth; "
      f"{shadow_mask_demo.mean()*100:.1f}% of cells terrain-shadowed at this moment)")

# %% [markdown]
# ## 9. Timing curves for a few representative aspects

# %%
# Find a representative steep-enough cell closest to each cardinal aspect,
# rather than trusting fixed grid coordinates -- robust to whatever
# terrain (synthetic or real DEM) actually loaded.
MIN_SLOPE_FOR_DEMO = 15.0
cardinal_targets = {"North face": 0, "East face": 90, "South face": 180, "West face": 270}

steep_mask = slope_deg >= MIN_SLOPE_FOR_DEMO
targets = {}
if steep_mask.any():
    steep_rows, steep_cols = np.where(steep_mask)
    steep_aspects = aspect_deg[steep_mask]
    for label, target_aspect in cardinal_targets.items():
        angular_diff = np.abs((steep_aspects - target_aspect + 180) % 360 - 180)
        best = np.argmin(angular_diff)
        targets[label] = (steep_rows[best], steep_cols[best])

fig, ax = plt.subplots(figsize=(10, 6))
for label, (r, c) in targets.items():
    energy_curve = cumulative_energy[:, r, c]
    ax.plot(times, energy_curve,
            label=f"{label} (slope={slope_deg[r,c]:.0f}\u00b0, "
                  f"aspect={aspect_deg[r,c]:.0f}\u00b0)")

ax.axhline(ENERGY_THRESHOLD_WHM2, color="red", linestyle="--", alpha=0.6,
           label="risk threshold")
ax.set_xlabel("Time of day")
ax.set_ylabel("Cumulative solar energy (Wh/m\u00b2)")
ax.set_title(f"Sun exposure buildup by aspect \u2014 {DATE} "
             f"({fc.attrs.get('source')} weather)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("/home/claude/sunsafe/aspect_timing_curves.png", dpi=140)
print("Saved aspect_timing_curves.png")

# %% [markdown]
# ## Caveats / what this prototype still does NOT do
#
# - **DEM/weather network access**: fetching real USGS 3DEP DEM tiles and
#   real NWS forecasts requires outbound internet access to
#   elevation.nationalmap.gov / api.weather.gov. If unavailable, both
#   fall back automatically to synthetic terrain / a synthetic diurnal
#   temperature curve -- check the printed `[terrain]` / `[weather]`
#   messages and `fc.attrs['source']` to see which path was actually used.
# - **Terrain shadowing is limited to the loaded DEM tile's extent**: a
#   real peak just outside your downloaded tile won't shadow anything.
#   Widen `DEM_RADIUS_KM` if a known tall neighbor sits nearby.
# - **Cloud attenuation is a rough empirical curve**, not a real
#   radiative-transfer / cloud-optical-depth model.
# - **No snowpack physics**: this tracks incoming energy, not actual snow
#   surface temperature or wetness. A real energy-balance model would add
#   longwave radiation, sensible/latent heat exchange (wind-dependent),
#   and snow albedo (fresh vs. old snow absorb very differently).
# - **Thresholds need real calibration**: run `calibrate.py` against your
#   own field observations (see that file's docstring) before trusting
#   absolute predicted times -- the defaults here are placeholders.
# - **This does not replace avalanche forecasting or field observations.**
#   Cornice fall, wind loading, persistent weak layers, and other hazards
#   this tool says nothing about can matter far more than solar timing.
