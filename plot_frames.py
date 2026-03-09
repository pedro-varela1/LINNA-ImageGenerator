"""
plot_frames.py
==============
For every frame in a batch output folder, generates a two-panel figure:

  Left   — 3-D context sphere with the current coverage patch highlighted
           + vertical altitude colorbar (global min–max from coords file).
  Right  — Rendered lunar image (from img/) with x/y axes in km;
           origin at image centre, +x East, +y North.

Usage
-----
    python3 plot_frames.py \\
        --batch output/batch/<name>                                     \\
        [--coords-file real_data/SelenITA_CoordinatesMoon_Operational_70km.txt] \\
        [--config config.json]

Output
------
    <batch>/frames/<stem>.png
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

# Re-use sphere renderer and helpers from plot_illustration
from plot_illustration import (
    _render_sphere_image,
    parse_coords_heights,
    stem_to_datetime,
    SPACE_BG, MOON_EDGE, DIM_COL, TEXT_COL,
    MOON_RADIUS_KM, KM_PER_DEG_LAT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALT_CMAP   = "plasma"


# ── Image loader ─────────────────────────────────────────────────────────────

def _load_image_rgb(img_path: str) -> np.ndarray:
    """Load a 16-bit RGBA PNG and return an 8-bit RGB array."""
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if raw.dtype == np.uint16:
        raw = (raw / 256).astype(np.uint8)
    if raw.ndim == 3 and raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    if raw.ndim == 3 and raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return raw  # grayscale fallback


# ── Frame figure ─────────────────────────────────────────────────────────────

def make_frame(stem, lat_min, lat_max, lon_min, lon_max,
               height_km, img_path, out_path,
               alt_min, alt_max) -> None:
    """
    Two-panel figure:
      Left  — 3-D context sphere  +  altitude colorbar
      Right — rendered lunar image with km axes
    """
    R = MOON_RADIUS_KM
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    ext_ns_km = (lat_max - lat_min) * KM_PER_DEG_LAT
    ext_ew_km = ((lon_max - lon_min)
                 * KM_PER_DEG_LAT
                 * abs(np.cos(np.radians(center_lat))))

    fig = plt.figure(figsize=(14, 7), facecolor=SPACE_BG)

    # ── Axes positions [left, bottom, width, height] ──────────────────────
    ax_sphere = fig.add_axes([0.030, 0.08, 0.400, 0.84])   # sphere
    ax_cbar   = fig.add_axes([0.450, 0.08, 0.022, 0.84])   # colorbar
    ax_img    = fig.add_axes([0.540, 0.08, 0.440, 0.84])   # image

    # ── Left: 3-D context sphere ──────────────────────────────────────────
    ax_sphere.set_facecolor(SPACE_BG)
    ax_sphere.set_aspect("equal")
    ax_sphere.axis("off")

    sphere_img = _render_sphere_image(
        500, center_lat, center_lon,
        lat_min, lat_max, lon_min, lon_max,
    )
    ax_sphere.imshow(sphere_img, extent=[-R, R, -R, R],
                     origin="upper", interpolation="bilinear", zorder=2)
    ax_sphere.add_patch(plt.Circle((0, 0), R,
                                   color=MOON_EDGE, fill=False,
                                   lw=1.0, zorder=3))
    ax_sphere.set_xlim(-R * 1.08, R * 1.08)
    ax_sphere.set_ylim(-R * 1.08, R * 1.08)
    ax_sphere.set_title("Context", color=TEXT_COL, fontsize=11, pad=6)

    # ── Middle: altitude colorbar ─────────────────────────────────────────
    norm = mcolors.Normalize(vmin=alt_min, vmax=alt_max)
    scm  = cm.ScalarMappable(cmap=ALT_CMAP, norm=norm)
    scm.set_array([])
    cbar = fig.colorbar(scm, cax=ax_cbar, orientation="vertical")
    cbar.set_label("Altitude (km)", color=TEXT_COL, fontsize=9, labelpad=6)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(colors=DIM_COL, labelsize=7, which="both",
                        length=3, width=0.7)
    plt.setp(cbar.ax.get_yticklabels(), color=DIM_COL)
    cbar.outline.set_edgecolor(DIM_COL)
    cbar.outline.set_linewidth(0.6)

    # Current-altitude marker: white line + label to the right
    ax_cbar.axhline(height_km, color="white", lw=2.0, zorder=5)
    ax_cbar.text(
        1.5, height_km, f" {height_km:.1f} km",
        color="white", fontsize=8, va="center", ha="left",
        transform=ax_cbar.get_yaxis_transform(), clip_on=False,
    )

    # ── Right: rendered image with km axes ────────────────────────────────
    img_rgb  = _load_image_rgb(img_path)
    half_ew  = ext_ew_km / 2.0
    half_ns  = ext_ns_km / 2.0

    # extent=[left, right, bottom, top]; origin='upper' → row-0 at top (North)
    ax_img.imshow(img_rgb,
                  extent=[-half_ew, half_ew, -half_ns, half_ns],
                  origin="upper", aspect="auto", zorder=2)
    ax_img.set_facecolor(SPACE_BG)
    ax_img.tick_params(colors=DIM_COL, labelsize=8)
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#22243a")

    ax_img.set_xlabel("Distance E–W (km)", color=DIM_COL, fontsize=9, labelpad=4)
    ax_img.set_ylabel("Distance N–S (km)", color=DIM_COL, fontsize=9, labelpad=4)

    # Cross-hair at image centre (nadir)
    ax_img.axhline(0, color="#ffffff28", lw=0.8, zorder=3)
    ax_img.axvline(0, color="#ffffff28", lw=0.8, zorder=3)

    # Cardinal labels on the border
    xlims = ax_img.get_xlim()
    ylims = ax_img.get_ylim()
    ax_img.text( half_ew * 0.95,  0, "E", color=DIM_COL, fontsize=7,
                va="center", ha="right")
    ax_img.text(-half_ew * 0.95,  0, "W", color=DIM_COL, fontsize=7,
                va="center", ha="left")
    ax_img.text(0,  half_ns * 0.95, "N", color=DIM_COL, fontsize=7,
                va="top", ha="center")
    ax_img.text(0, -half_ns * 0.95, "S", color=DIM_COL, fontsize=7,
                va="bottom", ha="center")

    dt = stem_to_datetime(stem)
    title_str = dt.strftime("%Y-%m-%d  %H:%M:%S UTC") if dt else stem
    ax_img.set_title(title_str, color=TEXT_COL, fontsize=10, pad=6)

    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=SPACE_BG, pad_inches=0.15)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate frame figures (sphere + image with km axes).")
    p.add_argument("--batch", required=True,
                   help="Batch output directory (contains img/ json/ sub-folders)")
    p.add_argument("--coords-file", default=None,
                   help="SelenITA coordinates .txt for per-frame height lookup")
    p.add_argument("--config",
                   default=os.path.join(SCRIPT_DIR, "config.json"),
                   help="Base config.json (fallback height, default: config.json)")
    return p.parse_args()


def main():
    args = parse_args()

    batch_dir  = os.path.abspath(args.batch)
    json_dir   = os.path.join(batch_dir, "json")
    img_dir    = os.path.join(batch_dir, "img")
    frames_dir = os.path.join(batch_dir, "frames")

    if not os.path.isdir(json_dir):
        sys.exit(f"[ERROR] Directory not found: {json_dir}")
    if not os.path.isdir(img_dir):
        sys.exit(f"[ERROR] Directory not found: {img_dir}")

    os.makedirs(frames_dir, exist_ok=True)

    # Per-frame heights
    heights_map: dict = {}
    if args.coords_file:
        heights_map = parse_coords_heights(args.coords_file)
        print(f"Loaded {len(heights_map)} height entries from {args.coords_file}")

    # Fallback height
    fallback_h = 50.0
    if os.path.isfile(args.config):
        with open(args.config) as fh:
            fallback_h = json.load(fh)["camera"]["height_km"]

    # Resolve heights for all frames first, so we can compute global min/max
    json_files = sorted(
        f for f in os.listdir(json_dir) if f.lower().endswith(".json")
    )
    if not json_files:
        sys.exit("[ERROR] No JSON files found in json/")

    frame_heights = {}
    for fname in json_files:
        stem = os.path.splitext(fname)[0]
        if stem in heights_map:
            frame_heights[stem] = heights_map[stem]
        else:
            tmp_cfg = os.path.join(batch_dir, "_tmp", stem, "config.json")
            if os.path.isfile(tmp_cfg):
                with open(tmp_cfg) as fh:
                    frame_heights[stem] = json.load(fh)["camera"]["height_km"]
            else:
                frame_heights[stem] = fallback_h

    all_heights = list(frame_heights.values())
    # If a coords-file was provided, use the full set of heights from it
    # (not just the frames present) for a stable colorbar range
    if heights_map:
        alt_min = min(heights_map.values())
        alt_max = max(heights_map.values())
    else:
        alt_min = min(all_heights)
        alt_max = max(all_heights)

    # Add a small margin so the min/max ticks are visible
    alt_range = alt_max - alt_min
    if alt_range < 1e-3:
        alt_min -= 5.0
        alt_max += 5.0
    else:
        alt_min -= alt_range * 0.02
        alt_max += alt_range * 0.02

    print(f"Altitude colorbar range: {alt_min:.1f} – {alt_max:.1f} km")
    print(f"Generating {len(json_files)} frame(s) → {frames_dir}")
    print()

    for fname in json_files:
        stem      = os.path.splitext(fname)[0]
        json_path = os.path.join(json_dir, fname)
        img_path  = os.path.join(img_dir, f"{stem}.png")
        out_path  = os.path.join(frames_dir, f"{stem}.png")

        if not os.path.isfile(img_path):
            print(f"  [SKIP] {stem}  (no image found)")
            continue

        with open(json_path) as fh:
            j = json.load(fh)

        lat_min, lat_max = j["lat_range"]
        lon_min, lon_max = j["lon_range"]
        height_km = frame_heights[stem]

        make_frame(
            stem, lat_min, lat_max, lon_min, lon_max,
            height_km, img_path, out_path,
            alt_min, alt_max,
        )
        print(f"  {stem}  h={height_km:.1f} km  → {os.path.basename(out_path)}")

    print()
    print(f"=== Done: {len(json_files)} frame(s) in {frames_dir} ===")


if __name__ == "__main__":
    main()
