# Lunar Surface Renderer

Photorealistic Blender simulation of the lunar surface from orbital or low-altitude camera positions, using real NASA datasets.

## Datasets required

| Dataset | Description | Source |
|---|---|---|
| SLDEM2015 / LOLA | Elevation DEM (GeoTIFF) | [LOLA PDS](https://ode.rsl.wustl.edu/moon/) |
| WAC Hapke 3-band | Colour tiles (8 tiles, 90°×70° each) | [LROC WAC PDS](https://wms.lroc.asu.edu/lroc) |

Place the DEM at the path set in `config.json → paths.lola_dem` and the WAC tiles in `paths.wac_dir`.

> **Longitude convention:** both datasets use **0–360°E**. If you have a negative longitude, convert it: `lon_0360 = lon + 360`.

---

## Project structure

```
python_files/
├── config.json              ← all parameters live here
├── run.sh                   ← one-command pipeline
├── prepare_textures.py      ← step 1: crop DEM + stitch colour map
├── lunar_render.py          ← step 2: Blender scene + render
│
├── terrain/                 ← texture-preparation logic (system Python)
│   ├── patch.py             ← compute terrain patch size from camera params
│   ├── displacement.py      ← crop LOLA DEM, write disp_meta.json
│   └── color.py             ← stitch WAC Hapke tiles into color_patch.png
│
├── render/                  ← Blender scene logic (Blender Python)
│   ├── scene.py             ← clear_scene(), setup_renderer()
│   ├── terrain.py           ← build_terrain_mesh(), build_terrain_material()
│   ├── camera.py            ← place_camera()
│   ├── lighting.py          ← place_sun()
│   └── latlon.py            ← per-pixel lat/lon map
│
├── utils/
│   ├── geo.py               ← Moon geometry constants & helpers
│   └── config.py            ← load_config()
│
├── batch_render.py          ← iterate a SelenITA coordinates file and render all frames
│
└── output/                  ← generated files (created automatically)
    ├── disp_patch.tif
    ├── disp_meta.json
    ├── color_patch.png
    ├── lunar_render.png
    ├── lunar_render_latlon.json
    └── lunar_render_latlon.npz
```

---

## Quick start

### 1. Install dependencies (system Python)

```bash
pip install Pillow numpy
# GDAL (optional but recommended for better DEM cropping):
sudo apt install gdal-bin
```

### 2. Edit `config.json`

```json
{
  "camera": {
    "lat_deg":    10.0,
    "lon_deg":    45.0,
    "height_km":  5.0,
    "fov_deg":    75.0,
    "tilt_deg":   0.0,
    "azimuth_deg": 0.0
  },
  "sun": { "azimuth_deg": 135.0, "elevation_deg": 25.0, "strength": 5.0 },
  "render": { "width": 1920, "height": 1080, "samples": 256, "use_gpu": true },
  "paths": {
    "lola_dem":   "/path/to/lunar_dem.tif",
    "wac_dir":    "/path/to/color/",
    "wac_ext":    ".tif",
    "output_dir": "/path/to/python_files/output"
  }
}
```

### 3. Run

```bash
# Full pipeline (textures + render):
bash run.sh

# Custom config:
bash run.sh --config /path/to/my_config.json

# Steps individually:
python3 prepare_textures.py --config config.json
blender --background --python lunar_render.py -- --config config.json
```

---

## Batch rendering from a trajectory file

`batch_render.py` reads a **SelenITA coordinates file** (one row per second) and
renders one image per selected row, automatically converting negative longitudes
to the 0–360° convention.

### Usage

```bash
python3 batch_render.py \
    --input  ../real_data/SelenITA_CoordinatesMoon_Operational_70km.txt \
    [--config  config.json]           # base config — lat/lon/alt overridden per row
    [--output  output/batch_70km]     # defaults to output/batch
    [--blender blender]               # Blender executable
    [--interval N]                    # render every Nth second (default: 1)
    [--limit N]                       # stop after N rendered images
```

### Examples

```bash
# One image per minute, first 10 images only (quick test):
python3 batch_render.py \
    --input ./real_data/SelenITA_CoordinatesMoon_Commissioning.txt \
    --interval 60 --limit 300

# Full dataset, one image every 5 minutes:
python3 batch_render.py \
    --input ./real_data/SelenITA_CoordinatesMoon_Operational_30km.txt \
    --interval 300 \
    --output output/batch_30km
```

### Output structure

```
<output>/
    img/   HH_MM_SS-YYYYMMDD.png    16-bit PNG render
    json/  HH_MM_SS-YYYYMMDD.json   lat/lon corner summary
    npz/   HH_MM_SS-YYYYMMDD.npz    per-pixel lat/lon arrays
```

Intermediate files (`disp_patch.tif`, `color_patch.png`, `disp_meta.json`, etc.)
are **automatically deleted** after each frame to save disk space.

### File naming

The timestamp `1 Nov 2028 00:01:00.000` becomes `00_01_00-20281101`.

---

## Configuration reference

### `camera`

| Key | Type | Description |
|---|---|---|
| `lat_deg` | float | Latitude in degrees North (−90 to +90) |
| `lon_deg` | float | Longitude in degrees East (**0 to 360**) |
| `height_km` | float | Altitude above the terrain plane (km) |
| `fov_deg` | float | Horizontal field of view (degrees) |
| `tilt_deg` | float | Off-nadir tilt (0 = nadir look) |
| `azimuth_deg` | float | Tilt direction (0=N, 90=E, 180=S, 270=W) |

### `sun`

| Key | Type | Description |
|---|---|---|
| `azimuth_deg` | float | Sun direction (0=N, 90=E …) |
| `elevation_deg` | float | Sun angle above horizon |
| `strength` | float | Cycles lamp energy (watts) |

### `render`

| Key | Type | Description |
|---|---|---|
| `width` / `height` | int | Output image resolution in pixels |
| `samples` | int | Cycles path-tracing samples |
| `use_gpu` | bool | Enable GPU rendering (auto-detects CUDA/HIP) |

### `texture`

| Key | Type | Description |
|---|---|---|
| `color_patch_size` | int | Color patch resolution (default 1024) |
| `disp_patch_size` | int | Displacement patch size for PIL fallback (default 512) |

---

## Outputs

| File | Description |
|---|---|
| `output/lunar_render.png` | 16-bit PNG render |
| `output/lunar_render_latlon.json` | Corner lat/lon summary |
| `output/lunar_render_latlon.npz` | Per-pixel lat/lon arrays (H×W float32) |
| `output/disp_meta.json` | Height range metadata for the displacement shader |

### Reading the lat/lon map

```python
import numpy as np
data = np.load("output/lunar_render_latlon.npz")
lat = data["lat"]  # shape (H, W), NaN = sky
lon = data["lon"]
```

---

## Notes

- The terrain patch size is **computed automatically** from `height_km`, `fov_deg`, `tilt_deg`, and render aspect ratio — you never need to set it manually.
- Negative longitudes must be converted: `lon_0360 = lon_negative + 360`. `batch_render.py` does this automatically.
- Blender must be installed and the `blender` command must be on `$PATH` (install via `sudo snap install blender --classic`).
- **GPU device selection** gracefully skips unavailable backends (e.g. OPTIX on systems without the OptiX SDK) and falls back to CUDA → HIP → CPU automatically.
