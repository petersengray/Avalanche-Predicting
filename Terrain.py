"""
terrain.py
----------
Turns a Digital Elevation Model (DEM) into slope angle + aspect at every
grid cell. This is the "topography" half of the problem.

For the prototype we generate a synthetic peak so the physics can be tested
without needing real DEM data. When you're ready to plug in a real mountain,
`load_dem_geotiff()` shows how to swap in a GeoTIFF via rasterio (e.g. from
USGS 3DEP: https://apps.nationalmap.gov/downloader/).
"""

import numpy as np


def synthetic_peak(size=200, cell_size_m=10, peak_height_m=600,
                    ridge=False):
    """
    Generate a synthetic DEM (elevation grid) for testing.

    size          : grid is size x size cells
    cell_size_m   : real-world meters per grid cell (controls slope steepness)
    peak_height_m : height of the peak above the base
    ridge         : if True, makes an elongated ridge instead of a cone,
                    which gives you a nice mix of aspects (N/S/E/W faces)
                    all in one small area -- good for testing.

    Returns
    -------
    elevation : (size, size) ndarray, meters
    cell_size_m : float, meters per pixel (needed later for slope calc)
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    if ridge:
        # An elongated ridge (like a real alpine ridge) running N-S, with
        # a bowl carved into the east side to mimic a couloir/cirque.
        elevation = peak_height_m * np.exp(-(X**2) / 0.15)
        bowl = 0.4 * peak_height_m * np.exp(-((X - 0.4)**2 + Y**2) / 0.1)
        elevation = elevation - bowl
        elevation -= elevation.min()
    else:
        R = np.sqrt(X**2 + Y**2)
        elevation = peak_height_m * np.exp(-(R**2) / 0.3)

    return elevation.astype(float), cell_size_m


def synthetic_ridge_with_neighbor(size=200, cell_size_m=15,
                                   main_peak_height_m=500,
                                   neighbor_height_m=650,
                                   neighbor_offset=(-0.45, 0.0)):
    """
    Like `synthetic_peak(ridge=True)`, but adds a second, taller peak
    nearby -- e.g. an adjacent summit -- specifically so there's terrain
    tall enough to cast a real shadow onto the main ridge at low sun
    angles. Useful for exercising/demoing `solar.terrain_shadow_mask`,
    which a lone smooth ridge mostly doesn't trigger.

    neighbor_offset : (dx, dy) in normalized [-1, 1] plot coordinates,
        same frame as the main ridge/bowl. Negative dx = to the west.
    """
    elevation, cell_size_m = synthetic_peak(
        size=size, cell_size_m=cell_size_m, peak_height_m=main_peak_height_m,
        ridge=True,
    )

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    ox, oy = neighbor_offset
    neighbor = neighbor_height_m * np.exp(-(((X - ox) ** 2) + (Y - oy) ** 2) / 0.04)
    elevation = elevation + neighbor
    elevation -= elevation.min()

    return elevation.astype(float), cell_size_m


def load_dem_geotiff(path):
    """
    Load a real DEM from a GeoTIFF (e.g. USGS 3DEP 1/3 arc-second data).
    Requires `rasterio` (pip install rasterio --break-system-packages).

    Returns elevation array (meters) and cell_size_m (approx, assumes
    projected/metric CRS -- reproject to UTM first if your DEM is in
    lat/lon degrees).
    """
    import rasterio
    with rasterio.open(path) as src:
        elevation = src.read(1).astype(float)
        cell_size_m = abs(src.transform.a)  # pixel width in CRS units
    return elevation, cell_size_m


USGS_3DEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)


def fetch_3dep_dem(lat, lon, radius_km=2.0, resolution_m=10, out_path="dem.tif",
                    timeout=60):
    """
    Download a real DEM tile centered on (lat, lon) from USGS 3DEP via
    their public ImageServer REST API -- no API key needed. US coverage
    only.

    NOTE ON NETWORK ACCESS: requires outbound internet access to
    elevation.nationalmap.gov. This will fail in network-sandboxed
    environments (it does in the one this code was developed in) but
    works fine from a normal machine with internet access.

    radius_km    : half-width of the square tile to download
    resolution_m : target pixel size in meters (output is reprojected to
                   Web Mercator (EPSG:3857) so cell_size_m is ~meters;
                   accuracy degrades somewhat at high latitudes due to
                   Mercator distortion -- fine for most US ski terrain,
                   worth reprojecting to a local UTM zone instead if you
                   need survey-grade precision)

    Returns the path to the downloaded GeoTIFF; load it with
    `load_dem_geotiff()`.
    """
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    side_px = max(64, int(2 * radius_km * 1000 / resolution_m))

    params = {
        "bbox": bbox,
        "bboxSR": 4326,
        "size": f"{side_px},{side_px}",
        "imageSR": 3857,
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    import requests
    resp = requests.get(USGS_3DEP_EXPORT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def get_dem(lat, lon, radius_km=2.0, resolution_m=10, cache_path="dem.tif",
            synthetic_fallback_kwargs=None, fallback_fn=None):
    """
    Convenience wrapper: try to fetch+load a real DEM for (lat, lon); if
    that fails for any reason (no network, outside US coverage, etc.),
    fall back to synthetic terrain so the rest of the pipeline still runs.
    Prints which path was taken.

    fallback_fn : which synthetic generator to use if the real fetch
        fails. Defaults to `synthetic_peak`; pass
        `synthetic_ridge_with_neighbor` if you want the fallback terrain
        to include a shadow-casting neighbor peak.
    """
    fallback_fn = fallback_fn or synthetic_peak
    try:
        path = fetch_3dep_dem(lat, lon, radius_km=radius_km,
                               resolution_m=resolution_m, out_path=cache_path)
        elevation, cell_size_m = load_dem_geotiff(path)
        print(f"[terrain] Loaded real USGS 3DEP DEM for ({lat}, {lon}), "
              f"{elevation.shape[0]}x{elevation.shape[1]} cells "
              f"@ ~{cell_size_m:.1f}m/cell.")
        return elevation, cell_size_m
    except Exception as e:
        kwargs = synthetic_fallback_kwargs or {}
        print(f"[terrain] Real DEM fetch failed ({type(e).__name__}: {e}). "
              f"Falling back to synthetic terrain ({fallback_fn.__name__}).")
        return fallback_fn(**kwargs)


def slope_aspect(elevation, cell_size_m):
    """
    Compute slope angle and aspect at every cell using the standard
    Horn (1981) method (same algorithm used by ArcGIS/QGIS/GDAL).

    Returns
    -------
    slope_deg  : (rows, cols) ndarray, 0 = flat, 90 = vertical
    aspect_deg : (rows, cols) ndarray, compass bearing the slope FACES,
                 0/360 = North, 90 = East, 180 = South, 270 = West
                 (this is the direction meltwater would run downhill,
                 i.e. the direction the slope is "looking")
    """
    # Gradient in the y (row/north-south) and x (col/east-west) directions
    dz_dy, dz_dx = np.gradient(elevation, cell_size_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: direction of steepest descent.
    # np.gradient's dz_dy increases with increasing row index (south in a
    # standard array-as-image layout), so we flip sign to get "north-up"
    # compass convention.
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = (450 - np.degrees(aspect_rad)) % 360  # north-referenced

    return slope_deg, aspect_deg
