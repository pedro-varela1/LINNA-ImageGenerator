"""
prepare_textures.py
====================
Run with system Python (not Blender):

    python prepare_textures.py [--config config.json]

Reads all parameters from config.json.
Outputs to paths.output_dir:
    disp_patch.tif    — cropped LOLA elevation map
    disp_meta.json    — height range metadata for the Blender material
    color_patch.png   — stitched WAC Hapke colour map
"""

import os
import sys
import argparse

# Ensure local packages (terrain/, utils/) are importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from utils.config import load_config
from terrain.patch import compute_patch_half_deg
from terrain.displacement import prepare_displacement
from terrain.color import build_color_patch


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LOLA displacement and WAC colour textures for Blender."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "config.json"),
        help="Path to config.json  (default: config.json next to this script)",
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    cam    = cfg["camera"]
    paths  = cfg["paths"]
    render = cfg["render"]
    tex    = cfg.get("texture", {})

    out_dir = paths["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    patch_half = compute_patch_half_deg(
        cam["height_km"], cam["fov_deg"], cam["tilt_deg"], cam["lat_deg"],
        render["width"], render["height"],
    )

    lat_min = cam["lat_deg"] - patch_half
    lat_max = cam["lat_deg"] + patch_half
    lon_min = cam["lon_deg"] - patch_half
    lon_max = cam["lon_deg"] + patch_half

    print(f"Patch: lat={lat_min:.4f}..{lat_max:.4f}  lon={lon_min:.4f}..{lon_max:.4f}")

    disp_out = prepare_displacement(
        paths["gld100_dem_dir"], lat_min, lat_max, lon_min, lon_max,
        out_dir, tex.get("disp_patch_size", 512),
        tex.get("disp_scale_km", 5.0),
        paths.get("dem_ext", ".TIF"),
        use_legacy_dem=tex.get("use_legacy_dem", False),
        lola_img_path=paths.get("lola_dem"),
    )

    color_out = os.path.join(out_dir, "color_patch.png")
    build_color_patch(
        lat_min, lat_max, lon_min, lon_max,
        color_out, paths["wac_dir"], paths.get("wac_ext", ".TIF"),
        tex.get("color_patch_size", 1024),
    )

    print("\n=== Textures ready ===")
    print(f"  Displacement : {disp_out}")
    print(f"  Color        : {color_out}")
    print(f"\nNext step — run Blender:")
    print(f"  blender --background --python lunar_render.py -- --config {args.config}")


if __name__ == "__main__":
    main()
