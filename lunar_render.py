"""
lunar_render.py
================
Run with Blender (after prepare_textures.py):

    blender --background --python lunar_render.py [-- --config config.json]

All parameters are read from config.json.
Expects disp_patch.tif and color_patch.png already in paths.output_dir.
"""

import bpy
import os
import sys
import argparse

# Add the repo root to sys.path so local packages (render/, terrain/, utils/)
# are importable from Blender's embedded Python interpreter.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils.config import load_config
from terrain.patch import compute_patch_half_deg
from render.scene import clear_scene, setup_renderer
from render.terrain import build_terrain_mesh, build_terrain_material
from render.camera import place_camera
from render.lighting import place_sun
from render.latlon import compute_pixel_latlon, save_latlon_map


def _parse_args():
    # Blender passes its own arguments before '--'; user arguments come after.
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "config.json"),
        help="Path to config.json",
    )
    return parser.parse_args(argv)


def main():
    print("=== Lunar Surface Renderer ===")

    args = _parse_args()
    cfg  = load_config(args.config)

    cam    = cfg["camera"]
    sun    = cfg["sun"]
    render = cfg["render"]
    paths  = cfg["paths"]

    out_dir     = paths["output_dir"]
    disp_path   = os.path.join(out_dir, "disp_patch.tif")
    color_path  = os.path.join(out_dir, "color_patch.png")
    disp_meta   = os.path.join(out_dir, "disp_meta.json")
    img_path    = os.path.join(out_dir, "lunar_render.png")
    latlon_path = os.path.join(out_dir, "lunar_render_latlon.json")

    for path, label in [(disp_path, "Displacement patch"), (color_path, "Color patch")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n  Run prepare_textures.py first!"
            )

    patch_half = compute_patch_half_deg(
        cam["height_km"], cam["fov_deg"], cam["tilt_deg"], cam["lat_deg"],
        render["width"], render["height"],
    )

    clear_scene()
    setup_renderer(
        render["width"], render["height"], render["samples"],
        img_path, render["use_gpu"],
    )

    print("[Terrain] Building mesh...")
    plane = build_terrain_mesh(cam["lat_deg"], cam["lon_deg"], patch_half)

    print("[Terrain] Applying material...")
    build_terrain_material(plane, disp_path, color_path, disp_meta)

    print("[Camera] Placing camera...")
    place_camera(cam["height_km"], cam["fov_deg"], cam["tilt_deg"], cam["azimuth_deg"])

    print("[Sun] Placing sun...")
    place_sun(sun["azimuth_deg"], sun["elevation_deg"], sun["strength"])

    print(f"[Render] {render['width']}x{render['height']} @ {render['samples']} samples...")
    bpy.ops.render.render(write_still=True)
    print(f"[Render] Saved → {img_path}")

    print("[LatLon] Computing per-pixel lat/lon...")
    lat_map, lon_map = compute_pixel_latlon(
        cam["lat_deg"], cam["lon_deg"], cam["height_km"],
        cam["fov_deg"], cam["tilt_deg"], cam["azimuth_deg"],
        render["width"], render["height"],
    )
    save_latlon_map(lat_map, lon_map, latlon_path)

    print("\n=== Done! ===")
    print(f"  Image   : {img_path}")
    print(f"  LatLon  : {latlon_path}  +  .npz")


if __name__ == "__main__":
    main()
