"""
batch_render.py
===============
Iterates over a SelenITA coordinates file and renders one Blender image per row.

Usage
-----
    python3 batch_render.py \\
        --input  ../real_data/SelenITA_CoordinatesMoon_Operational_70km.txt \\
        [--config config.json]        \\  # defaults to config.json next to this script
        [--output output/batch_70km]  \\  # defaults to output/batch inside the repo
        [--blender blender]           \\  # Blender executable (default: blender)
        [--limit  N]                      # stop after N rows (useful for testing)

Output structure
----------------
    <output>/
        img/   <name>.png      16-bit PNG render
        json/  <name>.json     lat/lon corner summary
        npz/   <name>.npz      per-pixel lat/lon arrays

File naming convention (from the row timestamp)
------------------------------------------------
    "5 Feb 2029 00:00:00.000"  →  00_00_00-20290205
"""

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.geo import km_per_deg_lat, normalize_lon


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch lunar surface renderer.")
    p.add_argument("--input",   required=True,
                   help="Path to SelenITA coordinates .txt file")
    p.add_argument("--config",  default=os.path.join(SCRIPT_DIR, "config.json"),
                   help="Base config.json  (default: config.json next to this script)")
    p.add_argument("--output",  default=None,
                   help="Output root directory  (default: <config output_dir>/batch)")
    p.add_argument("--blender", default="blender",
                   help="Blender executable  (default: blender)")
    p.add_argument("--limit",    type=int, default=None,
                   help="Stop after rendering N images (omit to process all rows)")
    p.add_argument("--interval", type=int, default=1,
                   help="Render every Nth row, e.g. --interval 60 for one frame per minute")
    return p.parse_args()

# Ex.: python batch_render.py --input real_data/SelenITA_CoordinatesMoon_Operational_70km.txt --limit 300 --interval 60 --blender flathub run org.blender.Blender

# ---------------------------------------------------------------------------
# Legacy DEM coverage check
# ---------------------------------------------------------------------------

_LEGACY_LAT_LIMIT = 60.0   # LOLA/SLDEM2015 only covers ±60°


def _lat_half_deg(lat_deg, height_km, fov_deg, tilt_deg, width, height,
                  margin=1.5):
    """Compute lat patch half-extent in degrees (lon not needed for DEM check)."""
    fov_h  = math.radians(fov_deg)
    aspect = width / height
    fov_v  = 2.0 * math.atan(math.tan(fov_h / 2.0) / aspect)
    tilt_r = math.radians(tilt_deg)
    max_angle = tilt_r + fov_v / 2.0
    if max_angle >= math.radians(89.9):
        max_ground_dist_km = height_km * math.tan(math.radians(89.0))
    else:
        max_ground_dist_km = height_km * math.tan(max_angle)
    half_fov_h = height_km / math.cos(max_angle) * math.tan(fov_h / 2.0)
    max_ground_dist_km = math.sqrt(max_ground_dist_km**2 + half_fov_h**2) * margin
    kpd_lat = km_per_deg_lat()
    return max(0.1, min(max_ground_dist_km / kpd_lat, 15.0))


def should_use_legacy_dem(cfg):
    """
    Return True only when the full lat patch lies within ±60°.
    If any edge strays outside that belt, return False so GLD100 is used.
    """
    cam  = cfg["camera"]
    rend = cfg["render"]
    half = _lat_half_deg(
        cam["lat_deg"], cam["height_km"], cam["fov_deg"],
        cam.get("tilt_deg", 0.0), rend["width"], rend["height"],
    )
    lat_min = cam["lat_deg"] - half
    lat_max = cam["lat_deg"] + half
    inside  = (lat_min >= -_LEGACY_LAT_LIMIT) and (lat_max <= _LEGACY_LAT_LIMIT)
    print(f"[LegacyDEM] patch lat [{lat_min:.2f}, {lat_max:.2f}] → "
          f"use_legacy_dem={inside}")
    return inside


# ---------------------------------------------------------------------------
# Coordinate-file parser
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def timestamp_to_filename(time_str):
    """
    Convert "5 Feb 2029 00:00:00.000"  →  "00_00_00-20290205"
    """
    dt = datetime.strptime(time_str.strip(), "%d %b %Y %H:%M:%S.%f")
    return dt.strftime("%H_%M_%S-%Y%m%d")


def iter_rows(txt_path):
    """
    Yield (filename_stem, lat_deg, lon_deg_0360, alt_km) for every data row
    in a SelenITA coordinates file.  Header lines are skipped automatically.
    """
    in_data = False
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            # The dashes separator marks the end of the header block
            if not in_data:
                if re.match(r"^-{4,}", line):
                    in_data = True
                continue
            if not line:
                continue
            # Data lines look like:
            #  5 Feb 2029 00:00:00.000    22.604    -32.063    58.977456
            # Use a regex that matches the known format
            m = re.match(
                r"(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)"
                r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                line,
            )
            if not m:
                continue
            time_str, lat_s, lon_s, alt_s = m.groups()
            stem = timestamp_to_filename(time_str)
            lat  = float(lat_s)
            lon  = normalize_lon(float(lon_s))
            alt  = float(alt_s)
            yield stem, lat, lon, alt


# ---------------------------------------------------------------------------
# Per-row rendering
# ---------------------------------------------------------------------------

def render_row(stem, lat, lon, alt_km, base_cfg, out_root,
               blender_exe, tmp_dir):
    """
    Prepare textures and render one frame.
    Outputs are moved to <out_root>/img/, /json/, /npz/.
    Returns True on success, False on failure.
    """
    # Build a per-row config with updated camera lat/lon/height and a
    # temporary output_dir so concurrent runs don't overwrite each other.
    row_tmp = os.path.join(tmp_dir, stem)
    os.makedirs(row_tmp, exist_ok=True)

    cfg = json.loads(json.dumps(base_cfg))          # deep copy
    cfg["camera"]["lat_deg"]   = lat
    cfg["camera"]["lon_deg"]   = lon
    cfg["camera"]["height_km"] = alt_km
    cfg["paths"]["output_dir"] = row_tmp

    force_legacy = base_cfg["texture"].get("use_legacy_dem", False)
    if force_legacy:
        # Legacy DEM (LOLA/SLDEM2015) only covers ±60°.
        # If this frame's patch strays outside that belt, skip it entirely.
        if not should_use_legacy_dem(cfg):
            print(f"  [SKIP] lat={lat:.2f} outside ±60° legacy DEM coverage – polar frame skipped")
            shutil.rmtree(row_tmp, ignore_errors=True)
            return False
        cfg["texture"]["use_legacy_dem"] = True
    else:
        # Config says GLD100 → use it for every frame, no restriction.
        cfg["texture"]["use_legacy_dem"] = False

    tmp_cfg = os.path.join(row_tmp, "config.json")
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    # Step 1: prepare textures
    ret = subprocess.run(
        [sys.executable,
         os.path.join(SCRIPT_DIR, "prepare_textures.py"),
         "--config", tmp_cfg],
        capture_output=False,
    )
    if ret.returncode != 0:
        print(f"  [SKIP] prepare_textures failed for {stem}")
        shutil.rmtree(row_tmp, ignore_errors=True)
        return False

    # Step 2: Blender render
    blender_cmd = shlex.split(blender_exe)
    ret = subprocess.run(
        blender_cmd + ["--background",
         "--python", os.path.join(SCRIPT_DIR, "lunar_render.py"),
         "--", "--config", tmp_cfg],
        capture_output=False,
    )
    if ret.returncode != 0:
        print(f"  [SKIP] Blender render failed for {stem}")
        shutil.rmtree(row_tmp, ignore_errors=True)
        return False

    # Step 3: move the 3 final outputs; delete everything else in row_tmp
    moves = [
        ("lunar_render.png",         "img",  f"{stem}.png"),
        ("lunar_render_latlon.json", "json", f"{stem}.json"),
        ("lunar_render_latlon.npz",  "npz",  f"{stem}.npz"),
    ]
    try:
        for src_name, sub, dst_name in moves:
            src = os.path.join(row_tmp, src_name)
            dst = os.path.join(out_root, sub, dst_name)
            if os.path.isfile(src):
                shutil.move(src, dst)
    finally:
        # Always remove the per-row temp dir (disp_patch.tif, color_patch.png,
        # disp_meta.json, config.json, etc.) to avoid disk accumulation.
        shutil.rmtree(row_tmp, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load base config
    with open(args.config) as f:
        base_cfg = json.load(f)

    # Resolve output root
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    label      = input_stem.rsplit("_", 1)[-1]   # e.g. "70km" from "..._70km"
    sun        = base_cfg["sun"]
    fov        = base_cfg["camera"]["fov_deg"]
    dem_label  = "SLDEM" if base_cfg["texture"].get("use_legacy_dem", False) else "GLD100"
    auto_name  = (
        f"{label}"
        f"_SunAz{sun['azimuth_deg']}"
        f"_SunInc{sun['elevation_deg']}"
        f"_FOV{fov}"
        f"_{dem_label}"
    )
    out_root = args.output or os.path.join(
        base_cfg["paths"]["output_dir"], "batch", auto_name
    )
    for sub in ("img", "json", "npz"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    # Shared temp directory for intermediate Blender files
    tmp_dir = os.path.join(out_root, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Input   : {args.input}")
    print(f"Output  : {out_root}")
    print(f"Blender : {args.blender}")
    if args.limit:
        print(f"Limit   : {args.limit} images")
    if args.interval > 1:
        print(f"Interval: every {args.interval} rows")
    print()

    ok = 0
    fail = 0
    for i, (stem, lat, lon, alt) in enumerate(iter_rows(args.input)):
        if i % args.interval != 0:
            continue
        if args.limit and (ok + fail) >= args.limit:
            break

        print(f"[{i+1}] {stem}  lat={lat:.4f}  lon={lon:.4f}  alt={alt:.3f} km")

        success = render_row(stem, lat, lon, alt,
                             base_cfg, out_root, args.blender, tmp_dir)
        if success:
            ok += 1
        else:
            fail += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    print(f"=== Batch complete: {ok} rendered, {fail} failed ===")
    print(f"  img/  : {os.path.join(out_root, 'img')}")
    print(f"  json/ : {os.path.join(out_root, 'json')}")
    print(f"  npz/  : {os.path.join(out_root, 'npz')}")


if __name__ == "__main__":
    main()
