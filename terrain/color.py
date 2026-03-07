import math
import os
import subprocess

from PIL import Image

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

# Moon equirectangular SRS — same target used by the DEM pipeline
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

_DEG_TO_M = math.pi * 1737400.0 / 180.0


def _polar_src_srs(tile_path):
    """Return explicit source SRS string if tile is polar, else None."""
    name = os.path.basename(tile_path).upper()
    if "P900N" in name:
        return MOON_POLAR_N_SRS
    if "P900S" in name:
        return MOON_POLAR_S_SRS
    return None


def _warp_tile(tile_path, src_srs, x_min, y_min, x_max, y_max, size, out_path):
    """Warp a single tile to equirectangular, overriding source SRS if given."""
    cmd = ["gdalwarp", "-t_srs", MOON_EQC_SRS]
    if src_srs:
        cmd += ["-s_srs", src_srs]
    cmd += [
        "-te",        str(x_min), str(y_min), str(x_max), str(y_max),
        "-ts",        str(size),  str(size),
        "-r",         "bilinear",
        "-dstnodata", "0",
        "-overwrite",
        tile_path, out_path,
    ]
    subprocess.run(cmd, check=True)


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

    x_min = lon_min * _DEG_TO_M
    x_max = lon_max * _DEG_TO_M
    y_min = lat_min * _DEG_TO_M
    y_max = lat_max * _DEG_TO_M

    tmp_tif = out_path.replace(".png", "_tmp.tif")
    out_dir = os.path.dirname(out_path)

    # Warp polar tiles individually with an explicit source SRS to work
    # around broken AXIS/b=0 WKT embedded in the LROC polar GeoTIFFs.
    warped_paths = []
    for i, tp in enumerate(tile_paths):
        src_srs = _polar_src_srs(tp)
        if src_srs is None:
            warped_paths.append(tp)           # equirectangular — pass through
        else:
            tmp_polar = os.path.join(out_dir, f"_polar_wac_{i}.tif")
            print(f"[WAC] Pre-warping polar tile: {os.path.basename(tp)}")
            _warp_tile(tp, src_srs, x_min, y_min, x_max, y_max, size, tmp_polar)
            warped_paths.append(tmp_polar)

    if len(warped_paths) == 1 and _polar_src_srs(tile_paths[0]):
        # Already warped to final size — just rename
        os.replace(warped_paths[0], tmp_tif)
    else:
        cmd = [
            "gdalwarp",
            "-t_srs",    MOON_EQC_SRS,
            "-te",       str(x_min), str(y_min), str(x_max), str(y_max),
            "-ts",       str(size),  str(size),
            "-r",        "bilinear",
            "-dstnodata", "0",
            "-overwrite",
        ] + warped_paths + [tmp_tif]
        print(f"[WAC] gdalwarp: {len(warped_paths)} tile(s) → {os.path.basename(tmp_tif)}")
        subprocess.run(cmd, check=True)
        # Clean up any pre-warped polar intermediates
        for i, tp in enumerate(tile_paths):
            if _polar_src_srs(tp):
                tmp_polar = os.path.join(out_dir, f"_polar_wac_{i}.tif")
                if os.path.isfile(tmp_polar):
                    os.remove(tmp_polar)

    # Convert GeoTIFF → plain grayscale PNG (strips geospatial metadata
    # so Blender loads it as a simple image)
    img = Image.open(tmp_tif).convert("L")
    img.save(out_path)
    os.remove(tmp_tif)

    print(f"[WAC] Color patch saved → {out_path}")
    return out_path
