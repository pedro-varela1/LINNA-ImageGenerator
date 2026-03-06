import math
import os
import json
import re
import shutil
import struct
import subprocess

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Legacy SLDEM2015 / LOLA constants  (use_legacy_dem = true)
# ---------------------------------------------------------------------------

LOLA_LINES            = 15360
LOLA_LINE_SAMPLES     = 46080
LOLA_LAT_MIN          = -60.0
LOLA_LAT_MAX          =  60.0
LOLA_LON_MIN          =   0.0
LOLA_LON_MAX          = 360.0
LOLA_BYTES_PER_SAMPLE = 4          # 32-bit IEEE float (PC_REAL)


def _crop_lola_with_gdal(img_path, lat_min, lat_max, lon_min, lon_max, out_path):
    """Crop LOLA GeoTIFF with gdal_translate using a manually computed pixel window."""
    total_lon = LOLA_LON_MAX - LOLA_LON_MIN
    total_lat = LOLA_LAT_MAX - LOLA_LAT_MIN

    col_off  = int((lon_min - LOLA_LON_MIN) / total_lon * LOLA_LINE_SAMPLES)
    col_end  = int((lon_max - LOLA_LON_MIN) / total_lon * LOLA_LINE_SAMPLES)
    col_size = max(1, col_end - col_off)

    row_off  = int((LOLA_LAT_MAX - lat_max) / total_lat * LOLA_LINES)
    row_end  = int((LOLA_LAT_MAX - lat_min) / total_lat * LOLA_LINES)
    row_size = max(1, row_end - row_off)

    cmd = [
        "gdal_translate",
        "-srcwin", str(col_off), str(row_off), str(col_size), str(row_size),
        "-of", "GTiff",
        img_path, out_path,
    ]
    print(f"[DISP] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[DISP] Saved -> {out_path}")


def _crop_lola_raw(img_path, lat_min, lat_max, lon_min, lon_max,
                   out_path, out_size=512):
    """Pure-Python fallback for LOLA .IMG (row-major 32-bit floats)."""
    print("[DISP] Cropping LOLA IMG via raw binary read (no GDAL) ...")

    total_lat = LOLA_LAT_MAX - LOLA_LAT_MIN
    total_lon = LOLA_LON_MAX - LOLA_LON_MIN
    row_bytes = LOLA_LINE_SAMPLES * LOLA_BYTES_PER_SAMPLE

    row_top = int((LOLA_LAT_MAX - lat_max) / total_lat * LOLA_LINES)
    row_bot = int((LOLA_LAT_MAX - lat_min) / total_lat * LOLA_LINES)
    col_lft = int((lon_min - LOLA_LON_MIN) / total_lon * LOLA_LINE_SAMPLES)
    col_rgt = int((lon_max - LOLA_LON_MIN) / total_lon * LOLA_LINE_SAMPLES)

    row_top = max(0, min(row_top, LOLA_LINES - 1))
    row_bot = max(row_top + 1, min(row_bot, LOLA_LINES))
    col_lft = max(0, min(col_lft, LOLA_LINE_SAMPLES - 1))
    col_rgt = max(col_lft + 1, min(col_rgt, LOLA_LINE_SAMPLES))

    n_rows = row_bot - row_top
    n_cols = col_rgt - col_lft
    patch  = np.zeros((n_rows, n_cols), dtype=np.float32)

    with open(img_path, "rb") as f:
        for i, row_idx in enumerate(range(row_top, row_bot)):
            f.seek(row_idx * row_bytes + col_lft * LOLA_BYTES_PER_SAMPLE)
            raw = f.read(n_cols * LOLA_BYTES_PER_SAMPLE)
            patch[i, :] = struct.unpack(f"<{n_cols}f", raw)

    h_min, h_max = float(patch.min()), float(patch.max())
    if h_max > h_min:
        patch_norm = (patch - h_min) / (h_max - h_min)
    else:
        patch_norm = np.full_like(patch, 0.5)

    patch_u16 = (patch_norm * 65535).astype(np.uint16)
    img_out   = Image.fromarray(patch_u16, mode="I;16")
    img_out   = img_out.resize((out_size, out_size), resample=Image.LANCZOS)
    tif_path  = out_path.replace(".png", ".tif")
    img_out.save(tif_path)
    print(f"[DISP] Saved → {tif_path}")
    return tif_path, h_min, h_max


def _prepare_displacement_legacy(lola_img_path, lat_min, lat_max, lon_min, lon_max,
                                  out_dir, disp_patch_size):
    """
    Legacy pipeline: crop SLDEM2015/LOLA .IMG or GeoTIFF with manual pixel
    arithmetic (gdal_translate -srcwin).  Returns (tif_path, meta_dict).
    """
    out_path = os.path.join(out_dir, "disp_patch.tif")

    if shutil.which("gdal_translate"):
        _crop_lola_with_gdal(lola_img_path, lat_min, lat_max, lon_min, lon_max, out_path)
        h_min, h_max = _get_lola_min_max(out_path)
    else:
        print("[DISP] gdal_translate not found — using raw binary fallback.")
        out_path, h_min, h_max = _crop_lola_raw(
            lola_img_path, lat_min, lat_max, lon_min, lon_max,
            out_path, disp_patch_size,
        )

    h_mid = (h_max + h_min) / 2.0
    meta  = {
        "h_min_km": h_min,
        "h_max_km": h_max,
        "h_mid_km": h_mid,
        "scale":    1.0,
        "midlevel": h_mid,
    }
    return out_path, meta


def _get_lola_min_max(tif_path):
    if shutil.which("gdalinfo"):
        result = subprocess.run(
            ["gdalinfo", "-mm", tif_path], capture_output=True, text=True
        )
        m = re.search(r"Computed Min/Max=([-\d.]+),([-\d.]+)", result.stdout)
        if m:
            h_min, h_max = float(m.group(1)), float(m.group(2))
            print(f"[DISP] Height range (gdalinfo): {h_min:.3f} .. {h_max:.3f} km")
            return h_min, h_max
    arr = np.array(Image.open(tif_path), dtype=np.float32)
    h_min = float(arr[arr > -9000].min()) if (arr > -9000).any() else -2.0
    h_max = float(arr[arr > -9000].max()) if (arr > -9000).any() else 2.0
    print(f"[DISP] Height range (PIL): {h_min:.3f} .. {h_max:.3f} km")
    return h_min, h_max


# ---------------------------------------------------------------------------
# GLD100 tile catalogue  (use_legacy_dem = false, default)
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
                         dem_ext=".TIF", use_legacy_dem=False,
                         lola_img_path=None):
    """
    Crop the DEM patch and write disp_meta.json for the Blender material.

    When ``use_legacy_dem=True`` the old SLDEM2015/LOLA pipeline is used:
        • ``lola_img_path`` must point to the single .IMG or GeoTIFF file
        • pixel coordinates are computed manually with -srcwin
        • disp_meta stores real km values (scale=1, midlevel=h_mid_km)
    When ``use_legacy_dem=False`` (default) the new GLD100 pipeline is used:
        • ``gld100_dir`` must point to the directory of WAC_GLD100_*_100M.TIF
        • GDAL reads the embedded CRS geotransform; polar tiles are reprojected
        • disp_meta stores normalised values (scale=disp_scale_km, midlevel=0..1)

    Returns the path to the saved GeoTIFF.
    """
    out_path = os.path.join(out_dir, "disp_patch.tif")

    if use_legacy_dem:
        if not lola_img_path:
            raise ValueError(
                "use_legacy_dem=True requires paths.lola_dem to be set in config."
            )
        print("[DISP] Using legacy LOLA/SLDEM2015 pipeline")
        out_path, meta = _prepare_displacement_legacy(
            lola_img_path, lat_min, lat_max, lon_min, lon_max,
            out_dir, disp_patch_size,
        )
    else:
        print("[DISP] Using GLD100 pipeline")
        tile_paths = find_gld100_tiles(
            lat_min, lat_max, lon_min, lon_max, gld100_dir, dem_ext
        )
        crop_gld100_with_gdal(
            tile_paths, lat_min, lat_max, lon_min, lon_max, out_path, disp_patch_size
        )
        v_min, v_max = get_disp_min_max(out_path)
        n_mid = (v_min + v_max) / 2.0 / 255.0
        meta = {
            "pixel_min": v_min,
            "pixel_max": v_max,
            "scale":     disp_scale_km,
            "midlevel":  n_mid,
        }
        print(
            f"[DISP]   pixel=[{v_min:.0f}, {v_max:.0f}]  "
            f"scale={disp_scale_km} km  midlevel={n_mid:.3f}"
        )
    meta_path = os.path.join(out_dir, "disp_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DISP] Metadata saved → {meta_path}")
    return out_path
