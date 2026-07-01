"""
One-off preprocessing step: builds the WorldPop "women 15-49" population
density overlay used by build_interactive_map.py.

This is kept separate from build_interactive_map.py because the source data
is heavy: WorldPop's server advertises `Accept-Ranges: bytes` but does not
actually honour HTTP Range requests (confirmed by testing -- it returns a
full 200 response regardless), so GDAL/rasterio's usual /vsicurl/ windowed
read doesn't work here. Each of the 7 needed 5-year age-band rasters must be
downloaded in full (~60 MB each, ~420 MB total) to read the small Kano FUA
window out of them. Re-running that on every map build would be slow, so
this script does it once and writes a small cached PNG + bounds file that
build_interactive_map.py loads directly.

Source: WorldPop "Age and sex structures" (Constrained, building-footprint
based), Nigeria, 2020, 100 m -- the same 2020/building-footprint WorldPop
product the report's methodology section describes as the model's demand
input. Bands f_15 .. f_45 (5-year groups) are summed to give women aged
15-49 (childbearing age).

Run this manually, then commit the two output files it writes:
    worldpop_women_15_49_2020.png
    worldpop_women_15_49_2020.json
"""

import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

AGE_BANDS = [15, 20, 25, 30, 35, 40, 45]  # -> women 15-49
BASE_URL = (
    "https://data.worldpop.org/GIS/AgeSex_structures/"
    "Global_2000_2020_Constrained/2020/NGA/nga_f_{band}_2020_constrained.tif"
)
OUT_PNG = "worldpop_women_15_49_2020.png"
OUT_JSON = "worldpop_women_15_49_2020.json"

# Kano FUA bounding box, same definition used in build_interactive_map.py
# (recomputed here so this stays consistent if the IDEAMAPS grid ever shifts)
print("Fetching IDEAMAPS grid extent for the Kano FUA bounding box...")
emoc_url = (
    "https://raw.githubusercontent.com/urbanbigdatacentre/"
    "ideamaps-models/dev/models/emergency-maternal-care/kano/model-outputs.csv"
)
emoc = pd.read_csv(emoc_url, usecols=["longitude", "latitude"])
lon_lo, lon_hi = emoc["longitude"].min(), emoc["longitude"].max()
lat_lo, lat_hi = emoc["latitude"].min(), emoc["latitude"].max()

cache_dir = Path(tempfile.gettempdir()) / "worldpop_nga_agesex_cache"
cache_dir.mkdir(exist_ok=True)

total = None
window_geo_bounds = None  # (left, bottom, right, top); identical across bands (same source grid)
for band in AGE_BANDS:
    local_path = cache_dir / f"nga_f_{band}_2020_constrained.tif"
    if not local_path.exists():
        print(f"Downloading age band f_{band} (~60 MB)...")
        r = requests.get(BASE_URL.format(band=band), timeout=180)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    else:
        print(f"Using cached f_{band} raster ({local_path})")

    with rasterio.open(local_path) as src:
        win = from_bounds(lon_lo, lat_lo, lon_hi, lat_hi, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        arr = np.where((arr == src.nodata) | (arr < 0), 0, arr)
        total = arr.copy() if total is None else total + arr
        window_geo_bounds = window_bounds(win, src.transform)

print(f"Window shape: {total.shape}, total women 15-49 in FUA bbox: {total.sum():,.0f}")

# ---------------------------------------------------------------------------
# Colorize: sqrt scale (population is heavily right-skewed: a few dense
# cells, many near-zero), sequential purple ramp so it's visually distinct
# from the green/orange/red deprivation categories and blue/orange facility
# markers. Fully transparent where population is ~0 (unsettled cells).
# ---------------------------------------------------------------------------
vmax = np.percentile(total[total > 0], 99)  # clip the top 1% of hot cells so the ramp isn't washed out
scaled = np.clip(np.sqrt(total) / np.sqrt(vmax), 0, 1)

cmap = matplotlib.colormaps["Purples"]
rgba = (cmap(scaled) * 255).astype(np.uint8)
alpha = np.where(total <= 0.5, 0, np.clip(scaled * 255 * 1.4, 60, 230)).astype(np.uint8)
rgba[..., 3] = alpha

Image.fromarray(rgba, mode="RGBA").save(OUT_PNG)
left, bottom, right, top = window_geo_bounds
with open(OUT_JSON, "w") as f:
    json.dump(
        {
            "bounds": [[bottom, left], [top, right]],
            "max_women_15_49_per_cell": float(total.max()),
            "total_women_15_49_in_bbox": float(total.sum()),
        },
        f,
        indent=2,
    )

print(f"Wrote {OUT_PNG} and {OUT_JSON}")
