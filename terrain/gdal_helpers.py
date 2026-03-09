"""
terrain/gdal_helpers.py
=======================
Shared Moon projection constants and GDAL polar-tile helpers used by both
the displacement (GLD100) and colour (WAC EMP) pipelines.
"""

import os
import subprocess

import math

# ---------------------------------------------------------------------------
# Moon SRS strings
# ---------------------------------------------------------------------------

# Equirectangular (target CRS for all gdalwarp crops)
MOON_EQC_SRS = (
    "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)

# Explicit PROJ4 overrides for polar tiles whose embedded WKT has
# AXIS[south,south] / b=0 that confuses PROJ during inverse transforms.
MOON_POLAR_N_SRS = (
    "+proj=stere +lat_0=90 +k_0=1 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)
MOON_POLAR_S_SRS = (
    "+proj=stere +lat_0=-90 +k_0=1 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)

# Metres per degree on the Moon in the equirectangular projection
DEG_TO_M = math.pi * 1737400.0 / 180.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def to_proj_lon(lon_deg):
    """Normalize 0-360 longitude to the -180/180 range used by PROJ EQC."""
    return ((lon_deg + 180.0) % 360.0) - 180.0


def polar_src_srs(tile_path):
    """Return explicit source SRS string if tile is polar, else None."""
    name = os.path.basename(tile_path).upper()
    if "P900N" in name:
        return MOON_POLAR_N_SRS
    if "P900S" in name:
        return MOON_POLAR_S_SRS
    return None


# ---------------------------------------------------------------------------
# Polar-tile reprojection
# ---------------------------------------------------------------------------

def reproject_polar_to_eqc(tile_path, out_dir, label="GDAL"):
    """
    Reproject a polar stereographic tile to Moon equirectangular at FULL
    resolution (no clipping).  The result is stored in out_dir so it is
    reused if the same tile is needed again in the same render session.

    Returns tile_path unchanged for non-polar tiles.
    """
    src_srs = polar_src_srs(tile_path)
    if src_srs is None:
        return tile_path   # equirectangular — nothing to do

    cache_name = os.path.splitext(os.path.basename(tile_path))[0] + "_eqc.tif"
    cache_path = os.path.join(out_dir, cache_name)
    if os.path.isfile(cache_path):
        print(f"[{label}] Polar EQC cache hit: {cache_name}")
        return cache_path

    print(f"[{label}] Reprojecting polar tile → EQC: {os.path.basename(tile_path)}")
    # Force output extent to the same 0-360° x-range used by equatorial tiles
    # (x = lon_deg × DEG_TO_M).  Limit y to the valid hemisphere so that PROJ
    # is never asked to inverse-project points on the opposite pole (which
    # would cause "too many points failed to transform" errors and a
    # needlessly huge output file covering the full globe).
    x_min_p = str(0.0)
    x_max_p = str(360.0 * DEG_TO_M)
    if "P900N" in os.path.basename(tile_path).upper():
        y_min_p = str(55.0 * DEG_TO_M)   # 5° margin below the 60° boundary
        y_max_p = str(90.0 * DEG_TO_M)
    else:
        y_min_p = str(-90.0 * DEG_TO_M)
        y_max_p = str(-55.0 * DEG_TO_M)  # 5° margin above the -60° boundary
    cmd = [
        "gdalwarp",
        "-s_srs",    src_srs,
        "-t_srs",    MOON_EQC_SRS,
        "-te",       x_min_p, y_min_p, x_max_p, y_max_p,
        "-r",        "bilinear",
        "-of",       "GTiff",
        "-overwrite",
        tile_path, cache_path,
    ]
    subprocess.run(cmd, check=True)
    return cache_path


# ---------------------------------------------------------------------------
# Tile-seam correction
# ---------------------------------------------------------------------------

def smooth_tile_seam(tif_path, lat_min, lat_max, out_size, label="GDAL"):
    """
    Remove the DN calibration discontinuity at GLD100 / WAC tile boundaries
    (±60° latitude).  Equatorial tiles (E300*) and polar tiles (P900*) are
    calibrated independently; at the shared 60° boundary their DN values may
    differ by several counts, appearing as a visible ridge in the Blender
    displacement mesh or as a colour band in the albedo texture.

    Strategy: measure the per-column mean step at the seam, then apply a
    linear correction ramp over a blend zone [seam - blend_half,
    seam + blend_half] that gradually brings both sides into alignment.
    Fine terrain texture (local variation around the mean) is fully
    preserved because the correction is a smooth additive baseline shift.
    """
    import numpy as np
    from PIL import Image

    seam_lats = [s for s in (60.0, -60.0) if lat_min < s < lat_max]
    if not seam_lats:
        return

    img = Image.open(tif_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    lat_range = lat_max - lat_min
    blend_half = max(30, out_size // 100)
    modified = False

    for seam_lat in seam_lats:
        # Row 0 = lat_max (north edge), row h = lat_min (south edge)
        seam_row = int((lat_max - seam_lat) / lat_range * h)
        seam_row = max(1, min(h - 1, seam_row))

        r0 = max(0, seam_row - blend_half)
        r1 = min(h, seam_row + blend_half)
        if r0 >= r1:
            continue

        # Per-column mean DN on each side of the seam
        n_avg = max(3, blend_half // 6)
        above_rows = arr[max(0, seam_row - n_avg):seam_row, :]
        below_rows = arr[seam_row:min(h, seam_row + n_avg), :]
        if above_rows.shape[0] == 0 or below_rows.shape[0] == 0:
            continue

        step = below_rows.mean(axis=0) - above_rows.mean(axis=0)  # shape (w,)

        # Linear correction: ramps from 0 at zone edges to ±step/2 at seam.
        # Above seam: correction = +step * t  (lifts values toward seam level)
        # Below seam: correction = +step * (t - 1)  (lowers values toward seam level)
        # Both sides converge to a common midpoint — no discontinuity remains.
        rows = np.arange(r0, r1)
        t = (rows - r0).astype(float) / max(1, r1 - r0 - 1)  # 0→1 top→bottom
        correction = np.where(
            rows[:, np.newaxis] < seam_row,
            step[np.newaxis, :] * t[:, np.newaxis],           # above seam
            step[np.newaxis, :] * (t[:, np.newaxis] - 1.0),  # below seam
        )
        arr[r0:r1, :] += correction
        modified = True

        mean_step = float(np.abs(step).mean())
        print(f"[{label}] Corrected tile seam at lat={seam_lat}° "
              f"(rows {r0}–{r1}, mean Δ={mean_step:.1f} DN)")

    if modified:
        Image.fromarray(
            np.clip(arr, 0, 255).astype(np.uint8), mode="L"
        ).save(tif_path)
