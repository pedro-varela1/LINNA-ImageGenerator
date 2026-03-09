"""
annotate_craters.py
===================
For every rendered image in a batch output folder, draws the craters from
craters_unified.parquet as annotated circles and saves the results to a
crater/ sub-folder.

Usage
-----
    python3 annotate_craters.py \\
        --batch  output/batch/Commissioning_SunAz248.0_SunInc10.0_FOV120.0_GLD100 \\
        [--craters craters_unified.parquet]   # default: next to this script
        [--min-diam 0.0]                      # skip craters smaller than N km
        [--color   255,0,0]                   # BGR circle colour (default red)
        [--thickness 2]                       # circle line thickness in pixels
        [--offset-lat 0.0]                    # shift circles north (+) or south (-) in degrees
        [--offset-lon 0.0]                    # shift circles east (+) or west (-) in degrees

Output
------
    <batch>/crater/<stem>.png   – copy of the rendered PNG with crater circles
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Moon radius in km – matches utils/geo.py / utils/sphere.py
MOON_RADIUS_KM = 1737.4
KM_PER_DEG_LAT = np.pi * MOON_RADIUS_KM / 180.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_lon_360(lon):
    """Map any longitude to [0, 360)."""
    return lon % 360.0


def _shift_lon(lon_arr, center_lon):
    """Shift lon values to be relative to center_lon, result in (-180, 180]."""
    return ((lon_arr - center_lon + 180.0) % 360.0) - 180.0


def build_pixel_tree(lat_map, lon_map):
    """
    Build a cKDTree over valid pixels using centred, isometric coordinates.

    A fixed cos(lat_center) correction is applied to the longitude axis so
    that 1 unit ≈ 1° in both dimensions.  Coordinates are centred on the
    image's mean lat/lon to avoid the 0°/360° longitude wrap.

    Returns
    -------
    tree        : cKDTree
    rows_v, cols_v : 1-D index arrays mapping tree leaf → (row, col)
    center_lat, center_lon : float  – image centre (degrees)
    cos_lat_c   : float             – fixed scale factor applied to lon axis
    """
    valid = np.isfinite(lat_map) & np.isfinite(lon_map)
    rows_v, cols_v = np.where(valid)
    lats_px = lat_map[rows_v, cols_v]
    lons_px = lon_map[rows_v, cols_v]

    center_lat = float(np.nanmean(lat_map))
    center_lon = float(np.nanmean(lon_map))
    cos_lat_c  = float(np.cos(np.radians(center_lat)))

    # Relative, isometrically scaled coordinates
    rel_lat = lats_px - center_lat
    rel_lon = _shift_lon(lons_px, center_lon) * cos_lat_c
    pts = np.column_stack([rel_lat, rel_lon])

    return cKDTree(pts), rows_v, cols_v, center_lat, center_lon, cos_lat_c


def crater_radius_pixels(diam_km, lat_map):
    """
    Crater radius in pixels, derived from the image's latitude extent.
    """
    lat_valid = lat_map[np.isfinite(lat_map)]
    if lat_valid.size == 0:
        return 5
    lat_range_deg = float(lat_valid.max() - lat_valid.min())
    if lat_range_deg <= 0:
        return 5
    H = lat_map.shape[0]
    km_per_px = lat_range_deg * KM_PER_DEG_LAT / H
    return max(1, int(round((diam_km / 2.0) / km_per_px)))


# ---------------------------------------------------------------------------
# Per-image annotation
# ---------------------------------------------------------------------------

def annotate_image(img_path, npz_path, json_path,
                   craters_df, out_path,
                   min_diam_km, color_bgr, thickness,
                   offset_lat=0.0, offset_lon=0.0):
    """
    Load one rendered PNG + its NPZ lat/lon map, filter relevant craters,
    draw circles, and write the annotated image to out_path.

    Returns the number of craters drawn.
    """
    # Load render
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [WARN] Cannot read image: {img_path}")
        return 0

    # Convert to 8-bit BGR for drawing if it is 16-bit
    if img.dtype == np.uint16:
        img8 = (img / 256).astype(np.uint8)
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        elif img8.shape[2] == 4:
            img8 = cv2.cvtColor(img8, cv2.COLOR_BGRA2BGR)
    else:
        img8 = img.copy()
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        elif img8.shape[2] == 4:
            img8 = cv2.cvtColor(img8, cv2.COLOR_BGRA2BGR)

    # Load lat/lon maps
    data = np.load(npz_path)
    lat_map = data["lat"]    # (H, W) — NaN for sky pixels
    lon_map = data["lon"]    # (H, W) — [0, 360), NaN for sky

    # Load JSON summary for fast pre-filtering
    with open(json_path) as f:
        summary = json.load(f)
    lat_min, lat_max = summary["lat_range"]
    lon_min, lon_max = summary["lon_range"]  # both in [0, 360)

    # Add a margin equal to the largest possible crater radius in degrees
    max_diam_deg = craters_df["diam_km"].max() / KM_PER_DEG_LAT
    lat_lo = lat_min - max_diam_deg
    lat_hi = lat_max + max_diam_deg

    # Convert parquet lons (−180…180) → [0, 360)
    lon_360 = normalize_lon_360(craters_df["x_coord"].values)

    # Filter by latitude first (fast scalar comparison)
    lat_in = (craters_df["y_coord"].values >= lat_lo) & \
             (craters_df["y_coord"].values <= lat_hi)

    # Filter by longitude — handle the possible wrap across 0°/360°
    if lon_max >= lon_min:
        lon_lo = lon_min - max_diam_deg
        lon_hi = lon_max + max_diam_deg
        if lon_lo < 0:
            lon_in = (lon_360 >= lon_lo + 360) | (lon_360 <= lon_hi)
        elif lon_hi > 360:
            lon_in = (lon_360 >= lon_lo) | (lon_360 <= lon_hi - 360)
        else:
            lon_in = (lon_360 >= lon_lo) & (lon_360 <= lon_hi)
    else:
        # lon range wraps across 0° (e.g. 350 → 10)
        lon_in = (lon_360 >= lon_min - max_diam_deg) | \
                 (lon_360 <= lon_max + max_diam_deg)

    keep = lat_in & lon_in & (craters_df["diam_km"].values >= min_diam_km)
    subset = craters_df[keep].copy()
    subset["lon_360"] = lon_360[keep]

    if subset.empty:
        cv2.imwrite(out_path, img8)
        return 0

    # Build a pixel KD-tree with centred, isometric coordinates so that
    # the distance metric is consistent between tree points and queries.
    valid_mask = np.isfinite(lat_map) & np.isfinite(lon_map)
    if not valid_mask.any():
        cv2.imwrite(out_path, img8)
        return 0

    tree, rows_v, cols_v, center_lat, center_lon, cos_lat_c = \
        build_pixel_tree(lat_map, lon_map)

    drawn = 0
    for _, row in subset.iterrows():
        lat_c  = float(row["y_coord"])
        lon_c  = float(row["lon_360"])
        diam   = float(row["diam_km"])

        # Apply lat/lon offset before lookup
        query_lat = lat_c + offset_lat
        query_lon = normalize_lon_360(lon_c + offset_lon)

        # Query: same centred + cos-scaled coordinates as the tree
        rel_lat = query_lat - center_lat
        rel_lon = _shift_lon(query_lon, center_lon) * cos_lat_c
        _, idx = tree.query([[rel_lat, rel_lon]], k=1)
        px_row = int(rows_v[idx[0]])
        px_col = int(cols_v[idx[0]])

        # Radius in pixels  (correct: diam/2 / km_per_px)
        radius_px = crater_radius_pixels(diam, lat_map)

        cv2.circle(img8, (px_col, px_row), radius_px, color_bgr, thickness)
        drawn += 1

    cv2.imwrite(out_path, img8)
    return drawn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Annotate batch images with crater circles.")
    p.add_argument("--batch",    required=True,
                   help="Batch output directory (contains img/, json/, npz/)")
    p.add_argument("--craters",
                   default=os.path.join(SCRIPT_DIR, "craters_unified.parquet"),
                   help="Path to craters_unified.parquet")
    p.add_argument("--min-diam", type=float, default=0.0,
                   help="Skip craters smaller than this diameter in km")
    p.add_argument("--color",    default="0,0,255",
                   help="Circle colour as B,G,R (default: 0,0,255 = red)")
    p.add_argument("--thickness", type=int, default=2,
                   help="Circle line thickness in pixels (default: 2)")
    p.add_argument("--offset-lat", type=float, default=0.0,
                   help="Shift all circles north (+) or south (-) by N degrees (default: 0.0)")
    p.add_argument("--offset-lon", type=float, default=0.0,
                   help="Shift all circles east (+) or west (-) by N degrees (default: 0.0)")
    return p.parse_args()


def main():
    args = parse_args()

    batch_dir  = os.path.abspath(args.batch)
    img_dir    = os.path.join(batch_dir, "img")
    json_dir   = os.path.join(batch_dir, "json")
    npz_dir    = os.path.join(batch_dir, "npz")
    crater_dir = os.path.join(batch_dir, "crater")

    for d in (img_dir, json_dir, npz_dir):
        if not os.path.isdir(d):
            sys.exit(f"[ERROR] Directory not found: {d}")

    os.makedirs(crater_dir, exist_ok=True)

    # Parse colour
    try:
        b, g, r = [int(c) for c in args.color.split(",")]
        color_bgr = (b, g, r)
    except ValueError:
        sys.exit("[ERROR] --color must be three integers separated by commas, e.g. 0,0,255")

    # Load crater catalogue
    print(f"Loading crater catalogue: {args.craters}")
    craters_df = pd.read_parquet(args.craters)
    print(f"  {len(craters_df):,} craters total")

    # Collect image stems
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if not img_files:
        sys.exit("[ERROR] No PNG files found in img/")

    if args.offset_lat or args.offset_lon:
        print(f"Offset      : lat={args.offset_lat:+.4f}°  lon={args.offset_lon:+.4f}°")
    print(f"Annotating {len(img_files)} image(s) → {crater_dir}")
    print()

    total_drawn = 0
    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_path  = os.path.join(img_dir,    fname)
        json_path = os.path.join(json_dir,   f"{stem}.json")
        npz_path  = os.path.join(npz_dir,    f"{stem}.npz")
        out_path  = os.path.join(crater_dir, fname)

        missing = [p for p in (json_path, npz_path) if not os.path.isfile(p)]
        if missing:
            print(f"  [SKIP] {stem} — missing: {', '.join(missing)}")
            continue

        n = annotate_image(
            img_path, npz_path, json_path,
            craters_df, out_path,
            min_diam_km=args.min_diam,
            color_bgr=color_bgr,
            thickness=args.thickness,
            offset_lat=args.offset_lat,
            offset_lon=args.offset_lon,
        )
        print(f"  {stem}  →  {n} crater(s) drawn")
        total_drawn += n

    print()
    print(f"=== Done: {total_drawn} crater annotations across {len(img_files)} image(s) ===")
    print(f"  crater/ : {crater_dir}")


if __name__ == "__main__":
    main()
