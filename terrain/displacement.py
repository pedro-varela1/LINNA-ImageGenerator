import math
import os
import json
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# GLD100 tile catalogue
# ---------------------------------------------------------------------------

# Equirectangular tiles: suffix → (lat_min, lat_max, lon_min, lon_max)
GLD100_EQ_TILES = {
    "E300N0450": (  0, 60,   0,  90),
    "E300N1350": (  0, 60,  90, 180),
    "E300N2250": (  0, 60, 180, 270),
    "E300N3150": (  0, 60, 270, 360),
    "E300S0450": (-60,  0,   0,  90),
    "E300S1350": (-60,  0,  90, 180),
    "E300S2250": (-60,  0, 180, 270),
    "E300S3150": (-60,  0, 270, 360),
}

# Polar stereographic tiles
GLD100_POLAR_TILES = {
    "P900N0000": ( 60, 90, 0, 360),
    "P900S0000": (-90,-60, 0, 360),
}

_GLD100_BASENAME = "WAC_GLD100_{suffix}_100M"

# Moon equirectangular SRS matching the GLD100 equirectangular tile projection.
# gdalwarp uses this as the target CRS for all crops (reprojects polar tiles
# transparently).
MOON_EQC_SRS = (
    "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)

# Metres per degree on the Moon in the equirectangular projection
_DEG_TO_M = math.pi * 1737400.0 / 180.0


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _tile_path(gld100_dir, suffix, ext=".TIF"):
    """Resolve file path for a GLD100 tile, trying ext variants."""
    for e in (ext, ext.upper(), ext.lower()):
        p = os.path.join(gld100_dir, _GLD100_BASENAME.format(suffix=suffix) + e)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"GLD100 tile not found: {_GLD100_BASENAME.format(suffix=suffix)}{ext}  "
        f"(searched in {gld100_dir})"
    )


def find_gld100_tiles(lat_min, lat_max, lon_min, lon_max, gld100_dir, ext=".TIF"):
    """Return list of GLD100 tile paths whose coverage overlaps the patch."""
    all_tiles = dict(**GLD100_EQ_TILES, **GLD100_POLAR_TILES)
    paths = []
    for suffix, (tlat_min, tlat_max, tlon_min, tlon_max) in all_tiles.items():
        if tlat_max <= lat_min or tlat_min >= lat_max:
            continue
        if tlon_max <= lon_min or tlon_min >= lon_max:
            continue
        paths.append(_tile_path(gld100_dir, suffix, ext))
    if not paths:
        raise ValueError(
            f"No GLD100 tiles cover lat={lat_min}..{lat_max}, "
            f"lon={lon_min}..{lon_max}"
        )
    print(f"[DISP] Using {len(paths)} GLD100 tile(s)")
    return paths


# ---------------------------------------------------------------------------
# GDAL crop (uses geotransform — no manual pixel arithmetic)
# ---------------------------------------------------------------------------

def crop_gld100_with_gdal(tile_paths, lat_min, lat_max, lon_min, lon_max,
                          out_path, out_size):
    """
    Warp and mosaic GLD100 tile(s) to the patch using GDAL geotransform.

    gdalwarp -t_srs reprojects polar stereographic tiles to the Moon
    equirectangular CRS automatically.  The -te extent is expressed in
    equirectangular metres computed via gdal.ApplyGeoTransform-equivalent
    linear formula (lon/lat × π*R/180).
    """
    x_min = lon_min * _DEG_TO_M
    x_max = lon_max * _DEG_TO_M
    y_min = lat_min * _DEG_TO_M
    y_max = lat_max * _DEG_TO_M

    cmd = [
        "gdalwarp",
        "-t_srs",    MOON_EQC_SRS,
        "-te",       str(x_min), str(y_min), str(x_max), str(y_max),
        "-ts",       str(out_size), str(out_size),
        "-r",        "bilinear",
        "-dstnodata", "0",
        "-overwrite",
    ] + tile_paths + [out_path]

    print(f"[DISP] gdalwarp: {len(tile_paths)} tile(s) → {os.path.basename(out_path)}")
    subprocess.run(cmd, check=True)
    print(f"[DISP] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Pixel statistics (from cropped patch)
# ---------------------------------------------------------------------------

def get_disp_min_max(tif_path):
    """Read 8-bit pixel min/max from the cropped GeoTIFF (NoData=0 excluded)."""
    if shutil.which("gdalinfo"):
        result = subprocess.run(
            ["gdalinfo", "-mm", tif_path], capture_output=True, text=True
        )
        m = re.search(r"Computed Min/Max=([\d.]+),([\d.]+)", result.stdout)
        if m:
            v_min, v_max = float(m.group(1)), float(m.group(2))
            print(f"[DISP] Pixel range (gdalinfo): {v_min:.0f} .. {v_max:.0f}")
            return v_min, v_max

    # PIL fallback
    arr = np.array(Image.open(tif_path), dtype=np.float32)
    valid = arr[arr > 0]  # exclude NoData
    if valid.size == 0:
        return 1.0, 254.0
    v_min, v_max = float(valid.min()), float(valid.max())
    print(f"[DISP] Pixel range (PIL): {v_min:.0f} .. {v_max:.0f}")
    return v_min, v_max


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_displacement(gld100_dir, lat_min, lat_max, lon_min, lon_max,
                         out_dir, disp_patch_size=512, disp_scale_km=5.0,
                         dem_ext=".TIF"):
    """
    Crop the GLD100 DEM to the patch using GDAL geotransform and write
    disp_meta.json for the Blender material.

    The GLD100 GeoTIFF has the CRS embedded, so GDAL computes every pixel
    position directly via gdal.ApplyGeoTransform without any manual arithmetic.
    Polar stereographic tiles (±60°..90°) are reprojected to the equirectangular
    Moon CRS on the fly by gdalwarp.

    Pixel values are 8-bit (0–255, NoData=0).  Blender normalises them to
    [0, 1] for the Displacement node:
        displacement_km = (pixel/255 – midlevel) × scale

    Args:
        gld100_dir    : directory containing WAC_GLD100_*_100M.TIF tiles
        disp_scale_km : total height range (km) represented by pixel 0..255
                        (default 5.0 km; adjust in config texture.disp_scale_km)
        dem_ext       : tile file extension (default ".TIF")

    Returns the path to the saved GeoTIFF.
    """
    out_path = os.path.join(out_dir, "disp_patch.tif")

    tile_paths = find_gld100_tiles(
        lat_min, lat_max, lon_min, lon_max, gld100_dir, dem_ext
    )
    crop_gld100_with_gdal(
        tile_paths, lat_min, lat_max, lon_min, lon_max, out_path, disp_patch_size
    )

    v_min, v_max = get_disp_min_max(out_path)
    # Normalised midpoint that maps to zero displacement in Blender
    n_mid = (v_min + v_max) / 2.0 / 255.0

    meta = {
        "pixel_min": v_min,
        "pixel_max": v_max,
        "scale":     disp_scale_km,
        "midlevel":  n_mid,
    }
    meta_path = os.path.join(out_dir, "disp_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DISP] Metadata saved → {meta_path}")
    print(
        f"[DISP]   pixel=[{v_min:.0f}, {v_max:.0f}]  "
        f"scale={disp_scale_km} km  midlevel={n_mid:.3f}"
    )
    return out_path
