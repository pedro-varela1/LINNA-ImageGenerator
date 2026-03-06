import os
import shutil
import struct
import json
import subprocess

import numpy as np
from PIL import Image

# SLDEM2015 / LOLA raster constants (do not change unless using a different dataset)
LOLA_LINES            = 15360
LOLA_LINE_SAMPLES     = 46080
LOLA_LAT_MIN          = -60.0
LOLA_LAT_MAX          =  60.0
LOLA_LON_MIN          =   0.0
LOLA_LON_MAX          = 360.0
LOLA_BYTES_PER_SAMPLE = 4          # 32-bit IEEE float (PC_REAL)


# ---------------------------------------------------------------------------
# DEM crop helpers
# ---------------------------------------------------------------------------

def crop_lola_with_gdal(img_path, lat_min, lat_max, lon_min, lon_max, out_path):
    """Crop LOLA GeoTIFF with gdal_translate using a pixel window."""
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


def crop_lola_raw(img_path, lat_min, lat_max, lon_min, lon_max,
                  out_path, out_size=512):
    """
    Pure-Python fallback: read the raw LOLA .IMG binary (row-major 32-bit
    floats, north-to-south), crop the patch, normalise to uint16 and save as
    a GeoTIFF that Blender can load.
    """
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
    print(f"[DISP] Height range in patch: {h_min:.3f} .. {h_max:.3f} km")

    if h_max > h_min:
        patch_norm = (patch - h_min) / (h_max - h_min)
    else:
        patch_norm = np.full_like(patch, 0.5)

    patch_u16 = (patch_norm * 65535).astype(np.uint16)
    img_out   = Image.fromarray(patch_u16, mode="I;16")
    img_out   = img_out.resize((out_size, out_size), resample=Image.LANCZOS)

    tif_path = out_path.replace(".png", ".tif")
    img_out.save(tif_path)
    print(f"[DISP] Saved → {tif_path}")
    return tif_path


# ---------------------------------------------------------------------------
# Min/max height reader
# ---------------------------------------------------------------------------

def get_disp_min_max(tif_path):
    """
    Read actual min/max height values (km) from the cropped GeoTIFF.
    Prefers gdalinfo; falls back to PIL pixel statistics.
    """
    import re as _re
    if shutil.which("gdalinfo"):
        result = subprocess.run(
            ["gdalinfo", "-mm", tif_path], capture_output=True, text=True
        )
        match = _re.search(r"Computed Min/Max=([-\d.]+),([-\d.]+)", result.stdout)
        if match:
            h_min, h_max = float(match.group(1)), float(match.group(2))
            print(f"[DISP] Height range (gdalinfo): {h_min:.3f} .. {h_max:.3f} km")
            return h_min, h_max

    print("[DISP] gdalinfo not available — estimating min/max via PIL pixel stats.")
    img = Image.open(tif_path)
    arr = np.array(img, dtype=np.float32)
    # Raw LOLA floats are in km; uint16 fallback values are >100 — use safe default
    if arr.max() > 100:
        h_min, h_max = -2.0, 2.0
    else:
        h_min, h_max = float(arr.min()), float(arr.max())
    print(f"[DISP] Height range (PIL): {h_min:.3f} .. {h_max:.3f} km")
    return h_min, h_max


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_displacement(lola_img_path, lat_min, lat_max, lon_min, lon_max,
                          out_dir, disp_patch_size=512):
    """
    Crop the LOLA DEM to the patch, compute the real height range, and write
    disp_meta.json so the Blender material uses the correct Scale/Midlevel.

    Returns the path to the saved GeoTIFF.
    """
    out_path = os.path.join(out_dir, "disp_patch.tif")

    if shutil.which("gdal_translate"):
        crop_lola_with_gdal(lola_img_path, lat_min, lat_max, lon_min, lon_max, out_path)
    else:
        print("[DISP] gdal_translate not found — using raw binary fallback.")
        out_path = crop_lola_raw(
            lola_img_path, lat_min, lat_max, lon_min, lon_max,
            out_path, disp_patch_size,
        )

    h_min, h_max = get_disp_min_max(out_path)
    h_mid = (h_max + h_min) / 2.0

    meta = {
        "h_min_km": h_min,
        "h_max_km": h_max,
        "h_mid_km": h_mid,
        "scale":    1.0,     # floats are already in km — scale is always 1
        "midlevel": h_mid,   # km value that maps to zero displacement
    }
    meta_path = os.path.join(out_dir, "disp_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DISP] Metadata saved -> {meta_path}")
    print(f"[DISP]   h_min={h_min:.3f} km  h_max={h_max:.3f} km  midlevel={h_mid:.3f} km")
    return out_path
