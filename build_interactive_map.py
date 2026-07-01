"""
Interactive exploration map for the Kano EmOC access-deprivation analysis.

Pulls the same three live sources used in kano_emoc_deprivation.qmd
(IDEAMAPS grid model, GRID3 NGA Health Facilities, Macharia et al. 2023
verified EmOC masterlist) and renders them as togglable Leaflet layers:

  - EmOC access deprivation grid (Low / Medium / High), rasterised from the
    123k 100 m grid cells for fast rendering
  - Community-validation "focus" cells (transition zones between deprivation
    levels, per IDEAMAPS dataset-metadata.json) - loaded in the .qmd but
    never surfaced there
  - GRID3 confirmed hospital-level facilities (General / Specialized /
    Teaching-Tertiary), with the whitespace bug in facility_level_option
    fixed so Aminu Kano Teaching Hospital is no longer silently dropped
  - All other GRID3 facilities, clustered, for context
  - Macharia et al. (2023) verified comprehensive-EmOC facilities
  - Murtala Mohammed GH / Standard Hospital highlighted with their 2 km
    analysis radius, plus the city-centre zoom and FUA bounding-box extents
    used in the report

Output: kano_emoc_interactive_map.html (self-contained aside from CDN
tile/JS references, same as folium's default).
"""

import io
import json

import folium
import numpy as np
import pandas as pd
import requests
from folium.plugins import Fullscreen, MarkerCluster, MeasureControl, MiniMap
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Load data (same sources as the .qmd)
# ---------------------------------------------------------------------------

print("Downloading IDEAMAPS grid model output...")
emoc_url = (
    "https://raw.githubusercontent.com/urbanbigdatacentre/"
    "ideamaps-models/dev/models/emergency-maternal-care/kano/model-outputs.csv"
)
emoc = pd.read_csv(emoc_url)
emoc["deprivation"] = emoc["result"].map({0: "Low", 1: "Medium", 2: "High"})
emoc["focused"] = emoc["focused"].astype(bool)

lon_lo_fua, lon_hi_fua = emoc["longitude"].min(), emoc["longitude"].max()
lat_lo_fua, lat_hi_fua = emoc["latitude"].min(), emoc["latitude"].max()

print("Downloading GRID3 NGA Health Facilities (Kano)...")
grid3_url = (
    "https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/"
    "GRID3_NGA_health_facilities_v2_0/FeatureServer/0/query"
    "?where=state%3D%27Kano%27"
    "&outFields=facility_name,facility_level,facility_level_option,"
    "ownership,ownership_type,latitude,longitude"
    "&resultRecordCount=2000&f=json"
)
grid3_json = requests.get(grid3_url, timeout=60).json()
if grid3_json.get("exceededTransferLimit"):
    print("WARNING: GRID3 query hit the server record limit; results are truncated.")
grid3 = pd.DataFrame(f["attributes"] for f in grid3_json["features"])

# The API returns "Teaching/Tertiary\xa0Hospital" (non-breaking space), which
# does NOT match a plain-space string comparison -- this silently dropped
# Aminu Kano Teaching Hospital from `confirmed_hospitals` in the .qmd.
grid3["facility_level_option_clean"] = (
    grid3["facility_level_option"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
)

grid3_fua = grid3[
    grid3["longitude"].between(lon_lo_fua, lon_hi_fua)
    & grid3["latitude"].between(lat_lo_fua, lat_hi_fua)
    & grid3["latitude"].notna()
    & grid3["longitude"].notna()
].copy()

HOSPITAL_LEVELS = {"General Hospital", "Teaching/Tertiary Hospital", "Specialized Hospital"}
confirmed = grid3_fua[grid3_fua["facility_level_option_clean"].isin(HOSPITAL_LEVELS)].copy()
other_facilities = grid3_fua[~grid3_fua["facility_level_option_clean"].isin(HOSPITAL_LEVELS)].copy()

print("Downloading Macharia et al. (2023) verified EmOC masterlist...")
macharia = pd.read_csv("https://ndownloader.figshare.com/files/41570451", low_memory=False)
kano_emoc = macharia[macharia["state"] == 9].copy()  # state 9 = Kano (LGAs: Dala, Nasarawa, Fagge...)
kano_emoc_fua = kano_emoc[
    kano_emoc["longitude"].between(lon_lo_fua, lon_hi_fua)
    & kano_emoc["latitude"].between(lat_lo_fua, lat_hi_fua)
].copy()
kano_emoc_fua["owner_label"] = np.where(kano_emoc_fua["owner"] == 1, "Public", "Private")
kano_emoc_fua["level_label"] = kano_emoc_fua["facility_level"].map(
    {0: "Primary", 1: "Secondary", 2: "Tertiary"}
)  # left as NaN (not defaulted to "Primary") when unknown, unlike the .qmd's map chunk

murtala = confirmed[confirmed["facility_name"].str.contains("Murtala", na=False)].iloc[0]
standard = confirmed[confirmed["facility_name"].str.contains("Standard", na=False)].iloc[0]

# ---------------------------------------------------------------------------
# 2. Rasterise the 123k-cell grid into a PNG overlay (fast to render/pan/zoom
#    vs. one Leaflet marker per cell)
# ---------------------------------------------------------------------------

print("Rasterising deprivation grid...")
COLORS = {
    "Low": (46, 204, 113, 235),
    "Medium": (243, 156, 18, 235),
    "High": (231, 76, 60, 235),
}
cell_w = float(emoc["lon_max"].sub(emoc["lon_min"]).mean())
cell_h = float(emoc["lat_max"].sub(emoc["lat_min"]).mean())

n_cols = int(round((lon_hi_fua - lon_lo_fua) / cell_w)) + 1
n_rows = int(round((lat_hi_fua - lat_lo_fua) / cell_h)) + 1

col_idx = ((emoc["longitude"] - lon_lo_fua) / cell_w).round().astype(int).clip(0, n_cols - 1)
row_idx = ((lat_hi_fua - emoc["latitude"]) / cell_h).round().astype(int).clip(0, n_rows - 1)

raster = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
rgba = np.array([COLORS[d] for d in emoc["deprivation"]], dtype=np.uint8)
raster[row_idx.to_numpy(), col_idx.to_numpy()] = rgba
deprivation_img = Image.fromarray(raster, mode="RGBA")

# Second raster: community-validation "focus" cells (transition zones)
focus_raster = np.zeros((n_rows, n_cols, 4), dtype=np.uint8)
foc = emoc[emoc["focused"]]
foc_rows = ((lat_hi_fua - foc["latitude"]) / cell_h).round().astype(int).clip(0, n_rows - 1)
foc_cols = ((foc["longitude"] - lon_lo_fua) / cell_w).round().astype(int).clip(0, n_cols - 1)
focus_raster[foc_rows.to_numpy(), foc_cols.to_numpy()] = (30, 30, 30, 150)
focus_img = Image.fromarray(focus_raster, mode="RGBA")

img_bounds = [[lat_lo_fua, lon_lo_fua], [lat_hi_fua, lon_hi_fua]]

# ---------------------------------------------------------------------------
# 3. Build the map
# ---------------------------------------------------------------------------

print("Building map...")
center = [(lat_lo_fua + lat_hi_fua) / 2, (lon_lo_fua + lon_hi_fua) / 2]
m = folium.Map(location=center, zoom_start=12, tiles=None, control_scale=True, prefer_canvas=True)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB positron", name="CartoDB Positron (light)").add_to(m)

# --- deprivation grid raster ---
folium.raster_layers.ImageOverlay(
    image=np.array(deprivation_img),
    bounds=img_bounds,
    name="EmOC access deprivation (grid)",
    opacity=0.75,
    interactive=False,
    cross_origin=False,
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=np.array(focus_img),
    bounds=img_bounds,
    name="Community-validation focus cells (transition zones)",
    opacity=0.6,
    interactive=False,
    cross_origin=False,
    show=False,
).add_to(m)

# --- GRID3 confirmed hospital-level facilities ---
hosp_fg = folium.FeatureGroup(name="GRID3 confirmed hospitals (General/Specialized/Teaching)", show=True)
for _, r in confirmed.iterrows():
    color = "#3498db" if r["ownership"] == "Public" else "#e67e22"
    folium.CircleMarker(
        location=[r["latitude"], r["longitude"]],
        radius=7,
        color="black",
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>{r['facility_name']}</b><br>"
            f"Type: {r['facility_level_option_clean']}<br>"
            f"Ownership: {r['ownership']}",
            max_width=260,
        ),
        tooltip=r["facility_name"],
    ).add_to(hosp_fg)
hosp_fg.add_to(m)

# --- all other GRID3 facilities (context), clustered ---
other_cluster = MarkerCluster(name="GRID3 other facilities (context)", show=False)
for _, r in other_facilities.iterrows():
    folium.CircleMarker(
        location=[r["latitude"], r["longitude"]],
        radius=3,
        color="#7f8c8d",
        weight=1,
        fill=True,
        fill_opacity=0.7,
        popup=f"{r['facility_name']} ({r['facility_level_option_clean']})",
    ).add_to(other_cluster)
other_cluster.add_to(m)

# --- Macharia verified comprehensive-EmOC facilities ---
mach_fg = folium.FeatureGroup(name="Macharia (2023) verified comprehensive EmOC", show=True)
for _, r in kano_emoc_fua.iterrows():
    color = "#3498db" if r["owner_label"] == "Public" else "#e67e22"
    level = r["level_label"] if pd.notna(r["level_label"]) else "Unknown"
    folium.CircleMarker(
        location=[r["latitude"], r["longitude"]],
        radius=5,
        color="black",
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{r['facility_name']}</b><br>"
            f"Level: {level}<br>"
            f"Ownership: {r['owner_label']}<br>"
            f"On-ground verified: {r.get('verif_onground', 'n/a')}<br>"
            f"Operating status: {r.get('operation_status', 'n/a')}",
            max_width=260,
        ),
        tooltip=r["facility_name"],
    ).add_to(mach_fg)
mach_fg.add_to(m)

# --- Murtala Mohammed / Standard Hospital highlight ---
anomaly_fg = folium.FeatureGroup(name="Murtala Mohammed vs Standard Hospital (report focus)", show=True)
for row, label, note in [
    (murtala, "Murtala Mohammed GH", "Not in Macharia verified list &mdash; 7.5% Low within 2 km"),
    (standard, "Standard Hospital", "In Macharia verified list &mdash; 92.7% Low within 2 km"),
]:
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        icon=folium.Icon(color="red" if "Murtala" in label else "green", icon="plus-sign"),
        popup=folium.Popup(f"<b>{label}</b><br>{note}", max_width=260),
        tooltip=label,
    ).add_to(anomaly_fg)
    folium.Circle(
        location=[row["latitude"], row["longitude"]],
        radius=2000,
        color="black",
        weight=1,
        dash_array="4 4",
        fill=False,
    ).add_to(anomaly_fg)
anomaly_fg.add_to(m)

# --- reference extents used in the report ---
extents_fg = folium.FeatureGroup(name="Report reference extents", show=False)
folium.Rectangle(
    bounds=[[lat_lo_fua, lon_lo_fua], [lat_hi_fua, lon_hi_fua]],
    color="#8e44ad",
    weight=2,
    dash_array="6 6",
    fill=False,
    popup=(
        "FUA bounding box used to filter facilities in the report "
        "(rectangle spanning the grid's lon/lat range, not the true FUA polygon)"
    ),
).add_to(extents_fg)
folium.Rectangle(
    bounds=[[11.94, 8.42], [12.12, 8.62]],
    color="#2c3e50",
    weight=2,
    fill=False,
    popup="City-centre zoom extent used in the report",
).add_to(extents_fg)
extents_fg.add_to(m)

# --- legend ---
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
            background: white; padding: 10px 14px; border: 1px solid #999;
            border-radius: 4px; font-size: 13px; line-height: 1.6;">
  <b>EmOC access deprivation</b><br>
  <span style="color:#2ecc71;">&#9632;</span> Low&nbsp;&nbsp;
  <span style="color:#f39c12;">&#9632;</span> Medium&nbsp;&nbsp;
  <span style="color:#e74c3c;">&#9632;</span> High<br>
  <b>Facility ownership</b><br>
  <span style="color:#3498db;">&#9679;</span> Public&nbsp;&nbsp;
  <span style="color:#e67e22;">&#9679;</span> Private
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)
Fullscreen().add_to(m)
MiniMap(toggle_display=True).add_to(m)
MeasureControl(primary_length_unit="kilometers").add_to(m)

out_path = "kano_emoc_interactive_map.html"
m.save(out_path)
print(f"Saved {out_path}")
