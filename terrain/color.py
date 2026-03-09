import os
import subprocess

from PIL import Image

from terrain.gdal_helpers import (
    MOON_EQC_SRS, DEG_TO_M,
    to_proj_lon as _to_proj_lon,
    reproject_polar_to_eqc as _reproject_polar_to_eqc_base,
    smooth_tile_seam as _smooth_tile_seam,
)


def _reproject_polar_to_eqc(tile_path, out_dir):
    return _reproject_polar_to_eqc_base(tile_path, out_dir, label="WAC")


# ---------------------------------------------------------------------------
# WAC EMP 643 nm tile catalogue
# ---------------------------------------------------------------------------

# Equirectangular tiles: suffix → (lat_min, lat_max, lon_min, lon_max)
WAC_EMP_EQ_TILES = {
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
WAC_EMP_POLAR_TILES = {
    "P900N0000": ( 60, 90, 0, 360),
    "P900S0000": (-90,-60, 0, 360),
}

_WAC_BASENAME = "WAC_EMP_643NM_{suffix}_304P"


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _tile_path(wac_dir, suffix, ext=".TIF"):
    """Resolve file path for a WAC EMP tile, trying ext variants."""
    for e in (ext, ext.upper(), ext.lower()):
        p = os.path.join(wac_dir, _WAC_BASENAME.format(suffix=suffix) + e)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"WAC EMP tile not found: {_WAC_BASENAME.format(suffix=suffix)}{ext}  "
        f"(searched in {wac_dir})"
    )


def find_wac_tiles(lat_min, lat_max, lon_min, lon_max, wac_dir, wac_ext=".TIF"):
    """Return list of WAC EMP 643 nm tile paths that overlap the patch."""
    all_tiles = dict(**WAC_EMP_EQ_TILES, **WAC_EMP_POLAR_TILES)
    paths = []
    for suffix, (tlat_min, tlat_max, tlon_min, tlon_max) in all_tiles.items():
        if tlat_max <= lat_min or tlat_min >= lat_max:
            continue
        if tlon_max <= lon_min or tlon_min >= lon_max:
            continue
        paths.append(_tile_path(wac_dir, suffix, wac_ext))
    if not paths:
        raise ValueError(
            f"No WAC EMP tiles cover lat={lat_min}..{lat_max}, "
            f"lon={lon_min}..{lon_max}"
        )
    print(f"[WAC] Using {len(paths)} WAC EMP 643 nm tile(s)")
    return paths


# ---------------------------------------------------------------------------
# GDAL crop and mosaic
# ---------------------------------------------------------------------------

def build_color_patch(lat_min, lat_max, lon_min, lon_max,
                      out_path, wac_dir, wac_ext=".TIF", size=1024):
    """
    Mosaic all required WAC EMP 643 nm tiles into a single grayscale PNG.

    Uses gdalwarp with the embedded GeoTIFF CRS (geotransform), which
    handles both equirectangular and polar stereographic source tiles
    transparently.  The target extent is expressed in Moon equirectangular
    metres computed as lon/lat × π*R/180.
    """
    tile_paths = find_wac_tiles(lat_min, lat_max, lon_min, lon_max,
                                wac_dir, wac_ext)

    # Use lon directly in 0-360 range (do NOT normalise to -180/180).
    # Equatorial tiles have x = lon_deg × DEG_TO_M in 0-360 range, and
    # polar tiles are reprojected to the same 0-360 EQC range by
    # reproject_polar_to_eqc.  Normalising to -180/180 would shift the
    # requested -te outside the tile's x-extent → all-zero output.
    lon_min_p = lon_min % 360.0
    lon_max_p = lon_max % 360.0
    if lon_max_p < lon_min_p:       # patch crosses 0°/360° line
        lon_max_p += 360.0

    x_min = lon_min_p * DEG_TO_M
    x_max = lon_max_p * DEG_TO_M
    y_min = lat_min   * DEG_TO_M
    y_max = lat_max   * DEG_TO_M

    out_dir  = os.path.dirname(out_path)
    tmp_tif  = out_path.replace(".png", "_tmp.tif")

    # Reproject polar tiles to full equirectangular before mosaicing.
    # This avoids the 'band in the middle' artefact that occurs when a
    # pre-clipped polar tile is mosaiced with a full equirectangular tile.
    ready_tiles = [_reproject_polar_to_eqc(tp, out_dir) for tp in tile_paths]

    cmd = [
        "gdalwarp",
        "-t_srs",     MOON_EQC_SRS,
        "-te",        str(x_min), str(y_min), str(x_max), str(y_max),
        "-ts",        str(size),  str(size),
        "-r",         "bilinear",
        "-dstnodata", "0",
        "-overwrite",
    ] + ready_tiles + [tmp_tif]

    print(f"[WAC] gdalwarp: {len(ready_tiles)} tile(s) → {os.path.basename(tmp_tif)}")
    subprocess.run(cmd, check=True)

    # Sanity-check: gdalwarp succeeds even when -te falls outside all source
    # tiles, producing an all-zero (NoData) image.  Catch that here so the
    # batch skips the frame instead of saving a black texture silently.
    import numpy as np
    _arr = np.array(Image.open(tmp_tif))
    if _arr.max() == 0:
        os.remove(tmp_tif)
        raise ValueError(
            f"[WAC] color patch is all zeros — patch extent "
            f"lon=[{lon_min_p:.2f},{lon_max_p:.2f}] lat=[{lat_min:.2f},{lat_max:.2f}] "
            f"does not overlap any tile."
        )

    _smooth_tile_seam(tmp_tif, lat_min, lat_max, size, label="WAC")

    # Convert GeoTIFF → plain grayscale PNG (strips geospatial metadata
    # so Blender loads it as a simple image)
    img = Image.open(tmp_tif).convert("L")
    img.save(out_path)
    os.remove(tmp_tif)

    print(f"[WAC] Color patch saved → {out_path}")
    return out_path
