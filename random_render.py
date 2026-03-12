"""
random_render.py
================
Generate N synthetic lunar surface images with randomised orbital/solar
parameters sampled from the ranges below.

    python3 random_render.py --n 100 [--config config.json] [--output output/random]
                             [--blender blender] [--seed 42]

Sampling ranges
---------------
    altitude   : 15–150 km
    latitude   : -60° to 60°
    longitude  :  0° to 360°
    sun elevation (inclination): 5° to 15°
    sun azimuth (relative):    -45° to 45°  → converted to 0-360 for config

Other camera/render/texture settings are taken unchanged from config.json.

Output structure
----------------
    <output>/
        img/   rand_NNNN.png       16-bit PNG render
        json/  rand_NNNN.json      lat/lon corners + render parameters used
        npz/   rand_NNNN.npz       per-pixel lat/lon arrays

The JSON file merges the standard lat/lon summary with a `render_params` block
containing every parameter used to produce that image.
"""

import argparse
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.geo import km_per_deg_lat, lat_patch_half_deg

# ---------------------------------------------------------------------------
# Sampling ranges
# ---------------------------------------------------------------------------

ALT_MIN_KM    = 15.0
ALT_MAX_KM    = 150.0
LAT_MIN_DEG   = -60.0
LAT_MAX_DEG   =  60.0
LON_MIN_DEG   =   0.0
LON_MAX_DEG   = 360.0
SUN_INC_MIN   =   5.0   # elevation above horizon (degrees)
SUN_INC_MAX   =  15.0
SUN_AZ_REL_MIN = -45.0  # relative to North; converted to 0-360 before use
SUN_AZ_REL_MAX =  45.0

# SLDEM2015 only covers ±60° latitude
_LEGACY_LAT_LIMIT = 60.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render N random lunar surface images with varied parameters."
    )
    p.add_argument("--n",       type=int, required=True,
                   help="Number of images to render")
    p.add_argument("--config",  default=os.path.join(SCRIPT_DIR, "config.json"),
                   help="Base config.json  (default: config.json next to this script)")
    p.add_argument("--output",  default=None,
                   help="Output root directory  (default: <output_dir>/random)")
    p.add_argument("--blender", default="blender",
                   help="Blender executable  (default: blender)")
    p.add_argument("--seed",    type=int, default=None,
                   help="Random seed for reproducibility")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feasibility checks
# ---------------------------------------------------------------------------

def is_feasible(cfg):
    """
    Return (ok, reason_str).

    Checks:
    1. Sun elevation > 0 (guaranteed by sampling range, but verified here).
    2. The nadir surface normal faces the sun — dot(sun_dir_local, up) = sin(el) > 0.
    3. If use_legacy_dem is True, the full lat patch sits within ±60°.
    """
    cam    = cfg["camera"]
    sun    = cfg["sun"]
    rend   = cfg["render"]
    tex    = cfg.get("texture", {})

    sun_el = sun["elevation_deg"]
    if sun_el <= 0.0:
        return False, f"sun elevation {sun_el}° ≤ 0 — nadir in darkness"

    # SLDEM coverage
    if tex.get("use_legacy_dem", False):
        half = lat_patch_half_deg(
            cam["lat_deg"], cam["height_km"], cam["fov_deg"],
            cam.get("tilt_deg", 0.0), rend["width"], rend["height"],
        )
        lat_min = cam["lat_deg"] - half
        lat_max = cam["lat_deg"] + half
        if lat_min < -_LEGACY_LAT_LIMIT or lat_max > _LEGACY_LAT_LIMIT:
            return False, (
                f"patch lat [{lat_min:.2f}, {lat_max:.2f}] outside "
                f"SLDEM ±{_LEGACY_LAT_LIMIT}° coverage"
            )

    return True, "ok"


# ---------------------------------------------------------------------------
# Sample one random parameter set
# ---------------------------------------------------------------------------

def sample_params(rng):
    """Return a dict of sampled camera & sun parameters."""
    lat     = rng.uniform(LAT_MIN_DEG, LAT_MAX_DEG)
    lon     = rng.uniform(LON_MIN_DEG, LON_MAX_DEG)
    alt     = rng.uniform(ALT_MIN_KM,  ALT_MAX_KM)
    sun_inc = rng.uniform(SUN_INC_MIN, SUN_INC_MAX)
    # Sun azimuth sampled as relative (-45..45), converted to 0-360 for config
    sun_az_rel = rng.uniform(SUN_AZ_REL_MIN, SUN_AZ_REL_MAX)
    sun_az = sun_az_rel % 360.0
    return {
        "lat_deg":          lat,
        "lon_deg":          lon,
        "height_km":        alt,
        "sun_elevation_deg": sun_inc,
        "sun_azimuth_deg":   sun_az,
        "sun_azimuth_rel_deg": sun_az_rel,   # stored in JSON for reference
    }


# ---------------------------------------------------------------------------
# Per-sample rendering
# ---------------------------------------------------------------------------

def render_sample(idx, params, base_cfg, out_root, blender_exe, tmp_dir):
    """
    Build a config, run prepare_textures + Blender, collect outputs.
    Returns True on success, False on failure.
    """
    stem    = f"rand_{idx:04d}"
    row_tmp = os.path.join(tmp_dir, stem)
    os.makedirs(row_tmp, exist_ok=True)

    # Deep-copy base config and apply sampled values
    cfg = json.loads(json.dumps(base_cfg))
    cfg["camera"]["lat_deg"]    = params["lat_deg"]
    cfg["camera"]["lon_deg"]    = params["lon_deg"]
    cfg["camera"]["height_km"]  = params["height_km"]
    cfg["sun"]["azimuth_deg"]   = params["sun_azimuth_deg"]
    cfg["sun"]["elevation_deg"] = params["sun_elevation_deg"]
    cfg["paths"]["output_dir"]  = row_tmp

    ok, reason = is_feasible(cfg)
    if not ok:
        print(f"  [SKIP] {stem} — {reason}")
        shutil.rmtree(row_tmp, ignore_errors=True)
        return False

    tmp_cfg = os.path.join(row_tmp, "config.json")
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"  lat={params['lat_deg']:+.3f}°  lon={params['lon_deg']:.3f}°"
        f"  alt={params['height_km']:.1f} km"
        f"  sunAz={params['sun_azimuth_deg']:.1f}° (rel {params['sun_azimuth_rel_deg']:+.1f}°)"
        f"  sunInc={params['sun_elevation_deg']:.1f}°"
    )

    # Step 1 – prepare textures
    ret = subprocess.run(
        [sys.executable,
         os.path.join(SCRIPT_DIR, "prepare_textures.py"),
         "--config", tmp_cfg],
    )
    if ret.returncode != 0:
        print(f"  [FAIL] prepare_textures failed for {stem}")
        shutil.rmtree(row_tmp, ignore_errors=True)
        return False

    # Step 2 – Blender render
    blender_cmd = shlex.split(blender_exe)
    ret = subprocess.run(
        blender_cmd + [
            "--background",
            "--python", os.path.join(SCRIPT_DIR, "lunar_render.py"),
            "--", "--config", tmp_cfg,
        ],
    )
    if ret.returncode != 0:
        print(f"  [FAIL] Blender render failed for {stem}")
        shutil.rmtree(row_tmp, ignore_errors=True)
        return False

    # Step 3 – enrich the latlon JSON with render parameters
    latlon_json = os.path.join(row_tmp, "lunar_render_latlon.json")
    if os.path.isfile(latlon_json):
        with open(latlon_json) as f:
            summary = json.load(f)
        summary["render_params"] = {
            "camera": {
                "lat_deg":    params["lat_deg"],
                "lon_deg":    params["lon_deg"],
                "height_km":  params["height_km"],
                "fov_deg":    cfg["camera"]["fov_deg"],
                "tilt_deg":   cfg["camera"]["tilt_deg"],
                "azimuth_deg": cfg["camera"]["azimuth_deg"],
            },
            "sun": {
                "azimuth_deg":     params["sun_azimuth_deg"],
                "azimuth_rel_deg": params["sun_azimuth_rel_deg"],
                "elevation_deg":   params["sun_elevation_deg"],
                "strength":        cfg["sun"]["strength"],
            },
            "render": cfg["render"],
            "texture": cfg.get("texture", {}),
            "dem": (
                "SLDEM2015_256ppd"
                if cfg.get("texture", {}).get("use_legacy_dem", False)
                else "GLD100_100M"
            ),
        }
        with open(latlon_json, "w") as f:
            json.dump(summary, f, indent=2)

    # Step 4 – move outputs to final directories
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
        shutil.rmtree(row_tmp, ignore_errors=True)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[Random] seed={args.seed}")
    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    with open(args.config) as f:
        base_cfg = json.load(f)

    dem_label = (
        "SLDEM" if base_cfg.get("texture", {}).get("use_legacy_dem", False)
        else "GLD100"
    )
    out_root = args.output or os.path.join(
        base_cfg["paths"]["output_dir"], "random", dem_label
    )
    for sub in ("img", "json", "npz"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    tmp_dir = os.path.join(out_root, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Output  : {out_root}")
    print(f"Blender : {args.blender}")
    print(f"Target  : {args.n} images")
    print()

    ok   = 0
    skip = 0
    idx  = 0

    while ok < args.n:
        idx += 1
        params = sample_params(rng)
        print(f"[{ok+1}/{args.n}] Attempt #{idx}")

        success = render_sample(
            ok + 1, params, base_cfg, out_root, args.blender, tmp_dir
        )
        if success:
            ok += 1
        else:
            skip += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    print(f"=== Done: {ok} rendered, {skip} skipped (infeasible/failed) ===")
    print(f"  img/  : {os.path.join(out_root, 'img')}")
    print(f"  json/ : {os.path.join(out_root, 'json')}")
    print(f"  npz/  : {os.path.join(out_root, 'npz')}")


if __name__ == "__main__":
    main()
