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
    "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
)
MOON_POLAR_S_SRS = (
    "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
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
    cmd = [
        "gdalwarp",
        "-s_srs",    src_srs,
        "-t_srs",    MOON_EQC_SRS,
        "-r",        "bilinear",
        "-of",       "GTiff",
        "-overwrite",
        tile_path, cache_path,
    ]
    subprocess.run(cmd, check=True)
    return cache_path
