"""
build_slopes.py
---------------
One-time (re-run whenever tours.yaml changes) preprocessing step:

  1. Downloads one USGS 3DEP DEM covering the whole region, in UTM so that
     pixel size is true ground meters (slope angles are correct).
  2. Snaps each summit in tours.yaml to the actual DEM high point nearby.
  3. Turns each slope definition (aspect range + steepness + elevation band
     + distance from summit) into a polygon of real terrain.
  4. Writes data/slopes.geojson (polygons + stats, WGS84 lat/lon) and
     data/slopes_preview.png (map to eyeball the result).

Usage:
    python build_slopes.py                 # download DEM (cached in data/)
    python build_slopes.py --dem my.tif    # use a GeoTIFF you already have
"""

import argparse
import json
import math
import os

import numpy as np
import yaml

import terrain

FT_PER_M = 3.28084
DATA_DIR = "data"


# ---------------------------------------------------------------- DEM ----

def load_region_dem(region, dem_path=None):
    """Return (elevation_m, transform, crs) for the region bbox."""
    import rasterio

    if dem_path is None:
        dem_path = os.path.join(DATA_DIR, "region_dem.tif")
        if not os.path.exists(dem_path):
            w, s, e, n = region["bbox"]
            epsg = terrain.utm_epsg((w + e) / 2, (s + n) / 2)
            print(f"[dem] downloading 3DEP for bbox {region['bbox']} in EPSG:{epsg} ...")
            terrain.fetch_3dep_bbox(region["bbox"], epsg,
                                    resolution_m=region.get("dem_resolution_m", 10),
                                    out_path=dem_path)
    with rasterio.open(dem_path) as src:
        elev = src.read(1, masked=True).astype(float).filled(np.nan)
        if not src.crs or not src.crs.is_projected:
            raise ValueError(f"{dem_path} must be in a projected (metric) CRS, e.g. UTM.")
        print(f"[dem] {dem_path}: {elev.shape[1]}x{elev.shape[0]} px @ "
              f"{abs(src.transform.a):.1f} m, {src.crs}")
        return elev, src.transform, src.crs


# ---------------------------------------------------------- geometry ----

def to_pixel(transform, crs, lat, lon):
    from rasterio.warp import transform as warp
    xs, ys = warp("EPSG:4326", crs, [lon], [lat])
    col, row = ~transform * (xs[0], ys[0])
    return int(round(row)), int(round(col))


def to_lonlat(transform, crs, row, col):
    from rasterio.warp import transform as warp
    x, y = transform * (col + 0.5, row + 0.5)
    lons, lats = warp(crs, "EPSG:4326", [x], [y])
    return lats[0], lons[0]


def snap_summit(elev, transform, crs, cell_m, summit, radius_m):
    r0, c0 = to_pixel(transform, crs, summit["lat"], summit["lon"])
    rad = int(math.ceil(radius_m / cell_m))
    rows, cols = elev.shape
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        raise ValueError(f"summit {summit} is outside the DEM — widen region.bbox")
    r_lo, r_hi = max(0, r0 - rad), min(rows, r0 + rad + 1)
    c_lo, c_hi = max(0, c0 - rad), min(cols, c0 + rad + 1)
    win = elev[r_lo:r_hi, c_lo:c_hi]
    rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    win = np.where((rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2, win, np.nan)
    i = np.nanargmax(win)
    r, c = np.unravel_index(i, win.shape)
    r, c = r + r_lo, c + c_lo
    moved = math.hypot(r - r0, c - c0) * cell_m
    return r, c, elev[r, c], moved


def aspect_in_range(aspect, lo, hi):
    lo, hi = lo % 360, hi % 360
    if lo <= hi:
        return (aspect >= lo) & (aspect <= hi)
    return (aspect >= lo) | (aspect <= hi)  # wraps through north


def circular_mean_deg(a):
    r = np.radians(a)
    return (math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) + 360) % 360


def slope_mask(elev, slope_deg, aspect_deg, cell_m, sr, sc, spec, other_summits=(),
               owner=None):
    """Cells matching the slope spec, kept only if connected to terrain
    near the summit (so we get the faces that drop off THIS peak, not a
    random gully 1 km away). Cells closer to another tour's summit are
    excluded so neighboring peaks (Rubicon / Hidden / Jakes) don't
    claim each other's faces."""
    from scipy import ndimage

    rows, cols = elev.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    dist_m = np.hypot(rr - sr, cc - sc) * cell_m
    # Which summit "owns" each cell (nearest summit wins).
    orow0, ocol0 = owner if owner else (sr, sc)
    d_owner = np.hypot(rr - orow0, cc - ocol0) * cell_m
    own = np.ones(elev.shape, dtype=bool)
    for (orow, ocol) in other_summits:
        own &= d_owner <= np.hypot(rr - orow, cc - ocol) * cell_m

    in_area = (
        own
        & (elev >= spec["min_elev_ft"] / FT_PER_M)
        & (dist_m <= spec["max_dist_m"])
        & np.isfinite(elev)
    )

    # 1) The FACE: everything facing the right way that is connected to the
    #    summit area, regardless of steepness. Faces often start as a gentle
    #    shoulder and only steepen lower down, so steepness must not break
    #    the connection to the summit.
    face = in_area & aspect_in_range(aspect_deg, *spec["aspect_range"])
    face = ndimage.binary_closing(face, structure=np.ones((3, 3)), iterations=2) & in_area
    labels, n = ndimage.label(face)
    if n == 0:
        return face
    near = dist_m <= max(250.0, 0.25 * spec["max_dist_m"])
    keep_ids = np.unique(labels[near & (labels > 0)])
    if keep_ids.size == 0:  # nothing touches the summit area: keep the biggest
        sizes = ndimage.sum(face, labels, range(1, n + 1))
        keep_ids = [int(np.argmax(sizes)) + 1]
    face = np.isin(labels, keep_ids)

    # 2) The SKI TERRAIN: the part of that face inside the steepness band.
    m = face & (slope_deg >= spec["min_slope_deg"]) & (slope_deg <= spec["max_slope_deg"])
    m = ndimage.binary_closing(m, structure=np.ones((3, 3))) & face
    m = ndimage.binary_opening(m, structure=np.ones((2, 2)))

    if spec.get("whole_face"):
        # Treat the whole face as skiable: smooth it into one region.
        m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=4) & face
        m = ndimage.binary_fill_holes(m)

    # Drop specks under ~0.5 ha.
    labels2, n2 = ndimage.label(m)
    sizes = ndimage.sum(m, labels2, range(1, n2 + 1))
    min_px = 5000 / cell_m ** 2
    return np.isin(labels2, [i + 1 for i, s in enumerate(sizes) if s >= min_px])


def mask_to_geojson(mask, transform, crs):
    from rasterio import features
    from rasterio.warp import transform_geom
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    polys = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask,
                                                    transform=transform) if v == 1]
    if not polys:
        return None
    geom = unary_union(polys).simplify(5)  # 5 m tolerance
    return transform_geom(crs, "EPSG:4326", mapping(geom), precision=6)


def load_drawn_areas(path):
    """slope_id -> GeoJSON geometry (WGS84) from a hand-traced FeatureCollection."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        fc = json.load(f)
    return {ft["properties"]["slope_id"]: ft["geometry"] for ft in fc["features"]}


def polygon_mask(geom_ll, shape, transform, crs):
    from rasterio import features
    from rasterio.warp import transform_geom
    g = transform_geom("EPSG:4326", crs, geom_ll)
    return features.geometry_mask([g], out_shape=shape, transform=transform, invert=True)


# -------------------------------------------------------------- main ----

def build(config_path="tours.yaml", dem_path=None, preview=True):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    region = cfg["region"]
    os.makedirs(DATA_DIR, exist_ok=True)

    elev, transform, crs = load_region_dem(region, dem_path)
    cell_m = abs(transform.a)
    slope_deg, aspect_deg = terrain.slope_aspect(np.nan_to_num(elev, nan=np.nanmin(elev)), cell_m)

    drawn = load_drawn_areas(cfg.get("ski_areas", "ski_areas.geojson"))
    features_out, report, overlay = [], [], np.zeros(elev.shape, dtype=int)
    summits = []
    snapped = [snap_summit(elev, transform, crs, cell_m, t["summit"],
                           t.get("snap_radius_m", 300)) for t in cfg["tours"]]
    distinct = {(r, c) for r, c, _, _ in snapped}
    # Slopes may name their own sub-summit (e.g. Maggies North); those count
    # as summits too when splitting terrain between neighboring peaks.
    sub_summits = {}
    for t in cfg["tours"]:
        for sp in t["slopes"]:
            if "summit" in sp:
                r, c, _, _ = snap_summit(elev, transform, crs, cell_m, sp["summit"],
                                         sp["summit"].get("snap_radius_m", 200))
                sub_summits[sp["id"]] = (r, c)
                distinct.add((r, c))
    for t_i, tour in enumerate(cfg["tours"], start=1):
        s = tour["summit"]
        sr, sc, s_elev, moved = snapped[t_i - 1]
        others = [p for p in distinct if p != (sr, sc)]
        lat, lon = to_lonlat(transform, crs, sr, sc)
        dz_ft = s_elev * FT_PER_M - s["elev_ft"]
        flag = "  <-- CHECK" if abs(dz_ft) > 100 or moved > 250 else ""
        report.append(f"{tour['name']:<32} snapped {moved:4.0f} m -> {lat:.5f}, {lon:.5f}  "
                      f"DEM {s_elev*FT_PER_M:6.0f} ft (listed {s['elev_ft']}, {dz_ft:+.0f}){flag}")
        summits.append((sr, sc, tour["name"]))

        for spec in tour["slopes"]:
            # A slope can set its own `start` (top of the run) when the skied
            # face begins below a flat shoulder rather than at the summit.
            owner = sub_summits.get(spec["id"], (sr, sc))
            ar, ac = (to_pixel(transform, crs, spec["start"]["lat"], spec["start"]["lon"])
                      if "start" in spec else owner)
            others = [p for p in distinct if p != owner]
            if spec["id"] in drawn:
                # Hand-traced ski area (ski_areas.geojson) wins over the
                # automatic aspect/steepness detection.
                geom = drawn[spec["id"]]
                m = polygon_mask(geom, elev.shape, transform, crs)
                source = "traced"
            else:
                m = slope_mask(elev, slope_deg, aspect_deg, cell_m, ar, ac, spec, others,
                               owner=owner)
                geom = mask_to_geojson(m, transform, crs)
                source = "derived"
            overlay[m] = len(features_out) + 1
            if not m.any():
                geom = None
            if geom is None:
                report.append(f"    {spec['id']:<22} NO TERRAIN MATCHED — loosen the spec")
                continue
            props = {
                "tour_id": tour["id"], "tour_name": tour["name"],
                "slope_id": spec["id"], "slope_name": spec["name"],
                "summit_lat": round(lat, 6), "summit_lon": round(lon, 6),
                "area_ha": round(m.sum() * cell_m ** 2 / 1e4, 1),
                "mean_slope_deg": round(float(slope_deg[m].mean()), 1),
                "p90_slope_deg": round(float(np.percentile(slope_deg[m], 90)), 1),
                "mean_aspect_deg": round(circular_mean_deg(aspect_deg[m]), 0),
                "elev_min_ft": int(np.nanmin(elev[m]) * FT_PER_M),
                "elev_max_ft": int(np.nanmax(elev[m]) * FT_PER_M),
                "pct_30_45_deg": round(float(((slope_deg[m] >= 30) & (slope_deg[m] <= 45)).mean() * 100), 0),
                "source": source,
                "spec": spec,
            }
            features_out.append({"type": "Feature", "geometry": geom, "properties": props})
            report.append(f"    {spec['id']:<22} [{source}] {props['area_ha']:6.1f} ha  "
                          f"slope {props['mean_slope_deg']:4.1f}° (p90 {props['p90_slope_deg']})  "
                          f"aspect {props['mean_aspect_deg']:3.0f}°  "
                          f"{props['elev_min_ft']}-{props['elev_max_ft']} ft")

    out = os.path.join(DATA_DIR, "slopes.geojson")
    with open(out, "w") as f:
        json.dump({"type": "FeatureCollection", "region": region, "features": features_out}, f)
    print("\n".join(report))
    print(f"\nwrote {out} ({len(features_out)} slopes)")

    if preview:
        render_preview(elev, cell_m, overlay, summits, cfg, os.path.join(DATA_DIR, "slopes_preview.png"))
    return features_out


def render_preview(elev, cell_m, overlay, summits, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(elev, nan=np.nanmin(elev)), vert_exag=1.5, dx=cell_m, dy=cell_m)
    fig, ax = plt.subplots(figsize=(10, 11), dpi=130)
    ax.imshow(hs, cmap="gray")
    cs = ax.contour(elev * FT_PER_M, levels=np.arange(6400, 10200, 400), colors="k",
                    linewidths=0.3, alpha=0.5)
    ax.clabel(cs, fmt="%d", fontsize=5)
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
              "#f032e6", "#9a6324", "#469990", "#800000", "#808000", "#000075"]
    for i in range(1, overlay.max() + 1):
        ov = np.ma.masked_where(overlay != i, overlay)
        ax.imshow(ov, cmap=matplotlib.colors.ListedColormap([colors[(i - 1) % len(colors)]]),
                  alpha=0.6, interpolation="nearest")
    labelled = {}
    for (r, c, name) in summits:
        labelled.setdefault((r, c), []).append(name)
    summits = [(r, c, " / ".join(dict.fromkeys(n.split(" — ")[0] for n in names)))
               for (r, c), names in labelled.items()]
    for (r, c, name) in summits:
        ax.plot(c, r, "k^", ms=6)
        ax.annotate(name, (c, r), xytext=(5, 5), textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set_title(f"{cfg['region']['name']} — ski areas (north up)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tours.yaml")
    ap.add_argument("--dem", default=None, help="existing GeoTIFF in a projected CRS")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    build(a.config, a.dem, preview=not a.no_preview)
