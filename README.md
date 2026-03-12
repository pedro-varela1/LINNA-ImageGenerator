# Lunar Surface Renderer

Photorealistic Blender simulation of the lunar surface from orbital or low-altitude camera positions, using real NASA datasets.

## Datasets required

| Dataset | Description | Source |
|---|---|---|
| SLDEM2015 / LOLA | Elevation DEM (GeoTIFF) | [LOLA PDS](https://ode.rsl.wustl.edu/moon/) |
| WAC Hapke 3-band | Colour tiles (8 tiles, 90\u00b0\u00d770\u00b0 each) | [LROC WAC PDS](https://wms.lroc.asu.edu/lroc) |

Place the DEM at the path in `config.json \u2192 paths.lola_dem` and the WAC tiles in `paths.wac_dir`.

> **Longitude convention:** both datasets use **0\u2013360\u00b0E**. Convert negative longitudes: `lon_0360 = lon + 360`.

---

## Project structure

```
\u251c\u2500\u2500 config.json              \u2190 all parameters
\u251c\u2500\u2500 run.sh                   \u2190 one-command pipeline
\u251c\u2500\u2500 prepare_textures.py      \u2190 step 1: crop DEM + stitch colour map
\u251c\u2500\u2500 lunar_render.py          \u2190 step 2: Blender scene + render
\u251c\u2500\u2500 batch_render.py          \u2190 render all frames from a SelenITA trajectory file
\u251c\u2500\u2500 random_render.py         \u2190 render N images with randomised orbital/solar parameters
\u251c\u2500\u2500 annotate_craters.py      \u2190 overlay crater circles on a batch output folder
\u251c\u2500\u2500 plot_frames.py           \u2190 generate two-panel figures (sphere context + rendered image)
\u251c\u2500\u2500 plot_illustration.py     \u2190 generate orbital geometry illustrations per frame
\u251c\u2500\u2500 make_video.py            \u2190 assemble frames/ PNGs into an MP4 timelapse
\u2502
\u251c\u2500\u2500 terrain/                 \u2190 texture-preparation modules (system Python)
\u2502   \u251c\u2500\u2500 patch.py             \u2190 terrain patch extents from camera params
\u2502   \u251c\u2500\u2500 displacement.py      \u2190 crop DEM, write disp_meta.json
\u2502   \u2514\u2500\u2500 color.py             \u2190 stitch WAC Hapke tiles into color_patch.png
\u2502
\u251c\u2500\u2500 render/                  \u2190 Blender scene modules (Blender Python)
\u2502   \u251c\u2500\u2500 scene.py             \u2190 clear_scene(), setup_renderer()
\u2502   \u251c\u2500\u2500 terrain.py           \u2190 build_terrain_mesh(), build_terrain_material()
\u2502   \u251c\u2500\u2500 camera.py            \u2190 place_camera()
\u2502   \u251c\u2500\u2500 lighting.py          \u2190 place_sun()
\u2502   \u2514\u2500\u2500 latlon.py            \u2190 per-pixel lat/lon map
\u2502
\u2514\u2500\u2500 utils/
    \u251c\u2500\u2500 geo.py               \u2190 Moon geometry constants & shared helpers
    \u251c\u2500\u2500 sphere.py            \u2190 local scene frame (MCMF \u2194 local)
    \u2514\u2500\u2500 config.py            \u2190 load_config()
```

---

## Quick start

### 1. Install dependencies (system Python)

```bash
pip install Pillow numpy
sudo apt install gdal-bin   # recommended for better DEM cropping
```

### 2. Edit `config.json`

```json
{
  "camera": {
    "lat_deg": 10.0, "lon_deg": 45.0, "height_km": 5.0,
    "fov_deg": 75.0, "tilt_deg": 0.0, "azimuth_deg": 0.0
  },
  "sun": { "azimuth_deg": 135.0, "elevation_deg": 25.0, "strength": 5.0 },
  "render": { "width": 1920, "height": 1080, "samples": 256, "use_gpu": true },
  "paths": {
    "lola_dem": "/path/to/lunar_dem.tif",
    "wac_dir":  "/path/to/color/",
    "wac_ext":  ".tif",
    "output_dir": "/path/to/output"
  }
}
```

### 3. Run

```bash
bash run.sh                                          # full pipeline
bash run.sh --config /path/to/my_config.json         # custom config

# Steps individually:
python3 prepare_textures.py --config config.json
blender --background --python lunar_render.py -- --config config.json
```

---

## Scripts

### `batch_render.py` \u2014 render a SelenITA trajectory

Reads a SelenITA coordinates file (one row per second) and renders one image per selected row.

```bash
python3 batch_render.py \
    --input    real_data/SelenITA_CoordinatesMoon_Operational_70km.txt \
    --config   config.json \
    --output   output/batch_70km \
    --blender  blender \
    --interval 60 \
    --limit    300
```

| Argument | Default | Description |
|---|---|---|
| `--input` | required | SelenITA .txt trajectory file |
| `--config` | `config.json` | Base config (lat/lon/alt overridden per row) |
| `--output` | `output/batch/<auto>` | Output root directory |
| `--blender` | `blender` | Blender executable |
| `--interval N` | `1` | Render every Nth row |
| `--limit N` | \u2014 | Stop after N images |

Output naming: timestamp `5 Feb 2029 00:01:00` \u2192 `00_01_00-20290205`.

---

### `random_render.py` \u2014 synthetic dataset generation

Renders N images with randomised orbital and solar parameters.

```bash
python3 random_render.py \
    --n      100 \
    --config config.json \
    --output output/random \
    --seed   42 \
    --blender blender
```

| Argument | Default | Description |
|---|---|---|
| `--n` | required | Number of images to render |
| `--config` | `config.json` | Base config |
| `--output` | `output/random/<DEM>` | Output root directory |
| `--seed` | \u2014 | Random seed for reproducibility |

**Sampling ranges:** altitude 15\u2013150 km, latitude \u221260\u00b0\u2013+60\u00b0, longitude 0\u00b0\u2013360\u00b0, sun elevation 5\u00b0\u201315\u00b0, sun azimuth (relative) \u221245\u00b0\u2013+45\u00b0.

Each JSON includes a `render_params` block with all values used.

---

### `annotate_craters.py` \u2014 crater circle overlay

Draws crater circles from `craters_unified.parquet` on every image in a batch folder. Output goes to `<batch>/crater/`.

```bash
python3 annotate_craters.py \
    --batch     output/batch/70km_SunAz248.0_SunInc10.0_FOV120.0_GLD100 \
    --craters   craters_unified.parquet \
    --min-diam  1.0 \
    --color     0,0,255 \
    --thickness 2 \
    --offset-lat 0.0 \
    --offset-lon 0.0
```

| Argument | Default | Description |
|---|---|---|
| `--batch` | required | Batch folder (must contain `img/`, `json/`, `npz/`) |
| `--craters` | `craters_unified.parquet` | Crater catalogue |
| `--min-diam` | `0.0` | Skip craters smaller than N km |
| `--color` | `0,0,255` | BGR circle colour |
| `--thickness` | `2` | Line thickness in pixels |
| `--offset-lat/lon` | `0.0` | Shift circles N/S or E/W in degrees |

---

### `plot_frames.py` \u2014 two-panel frame figures

For every frame, generates a figure with a 3-D context sphere and the rendered image with km axes.

```bash
python3 plot_frames.py \
    --batch       output/batch/<name> \
    --coords-file real_data/SelenITA_CoordinatesMoon_Operational_70km.txt \
    --config      config.json
```

Output: `<batch>/frames/<stem>.png`

---

### `plot_illustration.py` \u2014 orbital geometry illustration

For every frame, generates a 2-D cross-section showing satellite altitude, FOV lines, and surface coverage arc.

```bash
python3 plot_illustration.py \
    --batch       output/batch/<name> \
    --coords-file real_data/SelenITA_CoordinatesMoon_Operational_70km.txt \
    --config      config.json
```

Output: `<batch>/illustration/<stem>.png`

---

### `make_video.py` \u2014 timelapse video

Assembles `frames/` PNGs into an MP4 using ffmpeg.

```bash
python3 make_video.py \
    --batch output/batch/<name> \
    --fps   5 \
    --crf   18 \
    --out   timelapse.mp4
```

| Argument | Default | Description |
|---|---|---|
| `--batch` | required | Batch folder (must contain `frames/`) |
| `--fps` | `5.0` | Frames per second |
| `--crf` | `18` | H.264 quality (0\u201351, lower = better) |
| `--out` | `timelapse.mp4` | Output filename (placed inside `frames/`) |

Requires ffmpeg: `sudo apt install ffmpeg`.

---

## Output structure

All batch scripts write to the same layout:

```
<output>/
    img/   <stem>.png     16-bit PNG render
    json/  <stem>.json    lat/lon corner summary  [+render_params for random_render]
    npz/   <stem>.npz     per-pixel lat/lon arrays (H\u00d7W float32)
```

Intermediate files are deleted automatically after each frame.

---

## Configuration reference

### `camera`

| Key | Type | Description |
|---|---|---|
| `lat_deg` | float | Latitude \u00b0N (\u221290 to +90) |
| `lon_deg` | float | Longitude \u00b0E (**0 to 360**) |
| `height_km` | float | Altitude above terrain (km) |
| `fov_deg` | float | Horizontal field of view (degrees) |
| `tilt_deg` | float | Off-nadir tilt (0 = nadir) |
| `azimuth_deg` | float | Tilt direction (0=N, 90=E, 180=S, 270=W) |

### `sun`

| Key | Type | Description |
|---|---|---|
| `azimuth_deg` | float | Sun direction (0=N, 90=E \u2026) |
| `elevation_deg` | float | Sun angle above horizon |
| `strength` | float | Cycles lamp energy (watts) |

### `render`

| Key | Type | Description |
|---|---|---|
| `width` / `height` | int | Output resolution in pixels |
| `samples` | int | Path-tracing samples |
| `use_gpu` | bool | GPU rendering (auto-detects CUDA/HIP) |

### `texture`

| Key | Type | Description |
|---|---|---|
| `color_patch_size` | int | Colour patch resolution (default 1024) |
| `disp_patch_size` | int | Displacement patch size \u2013 PIL fallback (default 512) |
| `use_legacy_dem` | bool | Use SLDEM2015 (\u00b160\u00b0 only) instead of GLD100 |
