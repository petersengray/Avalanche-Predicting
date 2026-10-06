# Snow Softening Forecast

A planning aid for backcountry skiers on Lake Tahoe's West Shore: for each
tour and ski area, when the snow should soften to corn and when it gets too
wet, from terrain, sun angle and the NWS forecast.

> Experimental. Thresholds are uncalibrated placeholders. This is not an
> avalanche forecast. Always read the
> [Sierra Avalanche Center](https://www.sierraavalanchecenter.org/forecasts) forecast.

## How it fits together

| Step | Runs | What it does |
|---|---|---|
| `build_slopes.py` | on your computer, when tours change | Downloads USGS 3DEP elevation, builds each ski area (from `ski_areas.geojson`, or detected from terrain), and writes `data/slopes.geojson` plus `data/slope_cells.npz` (per-cell steepness, facing direction and skyline). |
| `daily_forecast.py` | every morning on GitHub Actions | Pulls the NWS hourly forecast for each tour, runs the melt model (`corn_model.py`) for today + 2 days, writes `docs/data/forecast.json`. |
| `docs/` | GitHub Pages | The webpage that reads `forecast.json`. |

Tours and ski areas are defined in `tours.yaml` and `ski_areas.geojson`.
Ski-area outlines were traced from guidebook route maps.

## Running locally

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_slopes.py              # only when tours/areas change
python daily_forecast.py            # live NWS forecast
python daily_forecast.py --date 2027-04-15 --synthetic -4 7   # offline test day
```
