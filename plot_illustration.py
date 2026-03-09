"""
plot_illustration.py
====================
For every frame in a batch output folder, generates an orbital geometry
illustration showing:
  - A 2D cross-section of the lunar sphere (N-S and E-W views)
  - The satellite at the correct altitude (in scale)
  - The camera FOV lines
  - The surface coverage arc highlighted
  - A text panel: lat/lon range, height, image extents in km

Usage
-----
    python3 plot_illustration.py \\
        --batch output/batch/<name>                                    \\
        [--coords-file real_data/SelenITA_CoordinatesMoon_Operational_70km.txt] \\
        [--config config.json]   # fallback for height if coords-file not given

Output
------
    <batch>/illustration/<stem>.png
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MOON_RADIUS_KM = 1737.4
KM_PER_DEG_LAT = np.pi * MOON_RADIUS_KM / 180.0

# ── Colour palette ──────────────────────────────────────────────────────────
SPACE_BG   = "#0b0d1c"
MOON_FILL  = "#4a4a5e"
MOON_EDGE  = "#8888aa"
ARC_COLOR  = "#ff7733"
FOV_COLOR  = "#33ffaa"
SAT_COLOR  = "#ffe033"
TEXT_COL   = "#ffffff"
DIM_COL    = "#778899"
PANEL_BG   = "#10121f"


# ── Geometry helpers ─────────────────────────────────────────────────────────

def km_per_deg_lon(lat_deg: float) -> float:
    return KM_PER_DEG_LAT * abs(np.cos(np.radians(lat_deg)))


# ── Coordinates-file parser (for per-frame heights) ─────────────────────────

def parse_coords_heights(txt_path: str) -> dict:
    """Return {stem: height_km} for every row in a SelenITA .txt file."""
    result = {}
    in_data = False
    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not in_data:
                if re.match(r"^-{4,}", line):
                    in_data = True
                continue
            if not line:
                continue
            m = re.match(
                r"(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)"
                r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                line,
            )
            if not m:
                continue
            time_str, _, _, alt_s = m.groups()
            dt = datetime.strptime(time_str.strip(), "%d %b %Y %H:%M:%S.%f")
            stem = dt.strftime("%H_%M_%S-%Y%m%d")
            result[stem] = float(alt_s)
    return result


def stem_to_datetime(stem: str):
    try:
        return datetime.strptime(stem, "%H_%M_%S-%Y%m%d")
    except ValueError:
        return None


# ── Cross-section drawing ────────────────────────────────────────────────────

def _draw_cross_section(ax, R, h, half_ang_deg, label_p, label_m, title, ext_km,
                         center_lat, center_lon, lat_min, lat_max, lon_min, lon_max):
    """
    Draw one 2-D cross-section panel.

    Parameters
    ----------
    half_ang_deg : angular half-extent of coverage measured on the sphere (deg)
    ext_km       : linear half-extent of coverage (km)  — used for axis limits
    """
    ax.set_facecolor(SPACE_BG)
    ax.set_aspect("equal")
    ax.tick_params(colors=DIM_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#22243a")
    ax.set_xlabel("km", color=DIM_COL, fontsize=7, labelpad=2)
    ax.set_ylabel("km", color=DIM_COL, fontsize=7, labelpad=2)
    ax.set_title(title, color=TEXT_COL, fontsize=10, pad=6)

    alpha = np.radians(half_ang_deg)

    # ── Moon circle ──────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((0, 0), R, color=MOON_FILL, zorder=1))
    ax.add_patch(plt.Circle((0, 0), R, color=MOON_EDGE, fill=False, lw=1.2, zorder=2))

    # ── Coverage arc (on sphere surface) ─────────────────────────────────────
    thetas = np.linspace(-alpha, alpha, 400)
    arc_x = R * np.sin(thetas)
    arc_y = R * np.cos(thetas)
    ax.plot(arc_x, arc_y, color=ARC_COLOR, lw=6,
            solid_capstyle="round", zorder=4)

    # Arc endpoints
    p_p = np.array([ R * np.sin(alpha), R * np.cos(alpha)])   # "plus"  side
    p_m = np.array([-R * np.sin(alpha), R * np.cos(alpha)])   # "minus" side

    # ── Satellite ────────────────────────────────────────────────────────────
    sat = np.array([0.0, R + h])
    ax.scatter(*sat, color=SAT_COLOR, s=160, marker="*", zorder=8, linewidths=0)
    ax.text(sat[0] + h * 0.55, sat[1],
            "Satellite", color=TEXT_COL, fontsize=8,
            va="center", ha="left")

    # ── FOV lines ─────────────────────────────────────────────────────────────
    ax.plot([sat[0], p_p[0]], [sat[1], p_p[1]], color=FOV_COLOR, lw=1.8, zorder=6)
    ax.plot([sat[0], p_m[0]], [sat[1], p_m[1]], color=FOV_COLOR, lw=1.8, zorder=6)

    # ── Nadir dashed line ─────────────────────────────────────────────────────
    ax.plot([0, 0], [R, R + h], color=TEXT_COL,
            lw=0.8, ls="--", alpha=0.35, zorder=3)

    # ── Height double-arrow ──────────────────────────────────────────────────
    arr_x = -ext_km * 0.55
    ax.annotate("", xy=(arr_x, R + h), xytext=(arr_x, R),
                arrowprops=dict(arrowstyle="<->", color=SAT_COLOR, lw=1.2),
                zorder=7)
    ax.text(arr_x - ext_km * 0.12, R + h / 2,
            f"h = {h:.1f} km", color=SAT_COLOR, fontsize=8,
            va="center", ha="right", rotation=90)

    # ── Coverage extent double-arrow (along arc, just outside sphere) ─────────
    arr_r = R * 1.065
    a_p2 = np.array([ arr_r * np.sin(alpha), arr_r * np.cos(alpha)])
    a_m2 = np.array([-arr_r * np.sin(alpha), arr_r * np.cos(alpha)])
    ax.annotate("", xy=a_p2, xytext=a_m2,
                arrowprops=dict(arrowstyle="<->", color=ARC_COLOR, lw=1.2),
                zorder=5)
    mid = (a_p2 + a_m2) / 2
    ax.text(mid[0], mid[1] + R * 0.025,
            f"{ext_km * 2:.0f} km", color=ARC_COLOR, fontsize=8,
            ha="center", va="bottom")

    # ── N / S (or E / W) endpoint labels ──────────────────────────────────────
    lbl_off = R * 0.04
    ax.text(p_p[0] + lbl_off, p_p[1], label_p,
            color=ARC_COLOR, fontsize=10, fontweight="bold",
            va="center", ha="left")
    ax.text(p_m[0] - lbl_off, p_m[1], label_m,
            color=ARC_COLOR, fontsize=10, fontweight="bold",
            va="center", ha="right")

    # ── Axis limits: zoomed to the relevant area ───────────────────────────────
    pad = max(ext_km * 1.6, h * 2.5)
    ax.set_xlim(-pad * 1.05, pad * 1.3)
    ax.set_ylim(R - pad * 0.5, R + h + pad * 1.1)

    # ── Context inset (3-D sphere with coverage patch) ─────────────────────────
    _draw_context_inset(ax, R, h, alpha, arc_x, arc_y, sat,
                        center_lat, center_lon, lat_min, lat_max, lon_min, lon_max)


def _render_sphere_image(size, center_lat_deg, center_lon_deg,
                          lat_min_deg, lat_max_deg, lon_min_deg, lon_max_deg):
    """
    Ray-cast a shaded Moon sphere with two-hemisphere compositing.

    * Front hemisphere: Lambertian shading + orange patch if in coverage.
    * Back hemisphere:  coverage patch glows through as a dim translucent
                        halo (Porter-Duff: back layer ∘ semi-transparent front).
    * The sphere is semi-transparent (alpha ≈ 0.82) so back-side patches
      are always visible regardless of their longitude.

    Camera fixed at lat=0, lon=0 — the patch appears at its true
    geographic position: horizontal ↔ longitude, vertical ↕ latitude.
    """
    y_i, x_i = np.mgrid[0:size, 0:size]
    half = (size - 1) / 2.0
    sx =  (x_i - half) / half       # East:  +1 = right
    sy =  (half - y_i) / half       # North: +1 = top
    r2 = sx ** 2 + sy ** 2
    on_sphere = r2 <= 1.0
    sz_f =  np.sqrt(np.where(on_sphere, 1.0 - r2, 0.0))  # front (+z)
    sz_b = -sz_f                                           # back  (-z)

    # Camera at lat=0, lon=0  →  ex=[0,1,0]  nx=[0,0,1]  zx=[1,0,0]
    # (simple identity-like basis, avoids re-deriving the full rotation)
    def _latlon(sz):
        Px = sz          # zx component
        Py = sx          # ex component
        Pz = sy          # nx component
        lat = np.degrees(np.arcsin(np.clip(Pz, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(Py, Px)) % 360.0
        return lat, lon

    def _in_cov(lat_px, lon_px):
        lmin = lon_min_deg % 360.0
        lmax = lon_max_deg % 360.0
        if lmax >= lmin:
            lon_ok = (lon_px >= lmin) & (lon_px <= lmax)
        else:
            lon_ok = (lon_px >= lmin) | (lon_px <= lmax)
        return (lat_px >= lat_min_deg) & (lat_px <= lat_max_deg) & lon_ok

    lat_f, lon_f = _latlon(sz_f)
    lat_b, lon_b = _latlon(sz_b)

    in_cov_f = on_sphere & _in_cov(lat_f, lon_f)
    in_cov_b = on_sphere & _in_cov(lat_b, lon_b)

    # ── Lambertian shading (front face only) ──────────────────────────────
    sun = np.array([-0.40, 0.55, 0.73])
    sun /= np.linalg.norm(sun)
    diffuse  = np.clip(sx * sun[0] + sy * sun[1] + sz_f * sun[2], 0.0, 1.0)
    intensity = 0.18 + 0.82 * diffuse

    # ── Back layer: dim orange glow for back-hemisphere coverage ──────────
    # This layer composites BEHIND the front sphere.
    SPHERE_ALPHA = 0.82   # front sphere transparency (1 = fully opaque)
    BACK_ALPHA   = 0.50   # opacity of the back-side patch glow
    back_rgb  = np.zeros((size, size, 3), dtype=np.float32)
    back_a    = np.zeros((size, size),    dtype=np.float32)
    back_rgb[in_cov_b] = [0.90, 0.38, 0.05]
    back_a[in_cov_b]   = BACK_ALPHA

    # ── Front layer: shaded sphere ────────────────────────────────────────
    mr, mg, mb = 0.28, 0.29, 0.36
    front_rgb = np.zeros((size, size, 3), dtype=np.float32)
    front_a   = np.zeros((size, size),    dtype=np.float32)
    front_rgb[on_sphere, 0] = mr * intensity[on_sphere]
    front_rgb[on_sphere, 1] = mg * intensity[on_sphere]
    front_rgb[on_sphere, 2] = mb * intensity[on_sphere]
    front_a[on_sphere]      = SPHERE_ALPHA
    # Front coverage: orange blended with shading
    blend = 0.65
    front_rgb[in_cov_f, 0] = blend * 1.00 + (1 - blend) * mr * intensity[in_cov_f]
    front_rgb[in_cov_f, 1] = blend * 0.47 + (1 - blend) * mg * intensity[in_cov_f]
    front_rgb[in_cov_f, 2] = blend * 0.10 + (1 - blend) * mb * intensity[in_cov_f]

    # ── Porter-Duff composite: front OVER back ────────────────────────────
    # alpha_out = a_f + a_b*(1 - a_f)
    # rgb_out   = (a_f*rgb_f + a_b*(1-a_f)*rgb_b) / alpha_out
    fa = front_a[:, :, np.newaxis]
    ba = back_a[:, :, np.newaxis]
    alpha_out = front_a + back_a * (1.0 - front_a)          # (H,W)
    safe_a    = np.where(alpha_out > 0, alpha_out, 1.0)[:, :, np.newaxis]
    rgb_out   = (fa * front_rgb + ba * (1.0 - fa) * back_rgb) / safe_a

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb_out, 0.0, 1.0)
    rgba[:, :,  3] = np.clip(alpha_out, 0.0, 1.0)
    return rgba


def _draw_context_inset(ax, R, h, alpha, arc_x, arc_y, sat,
                         center_lat, center_lon, lat_min, lat_max, lon_min, lon_max):
    # Upper-right corner
    ax_in = ax.inset_axes([0.62, 0.62, 0.37, 0.37])
    ax_in.set_facecolor(SPACE_BG)
    ax_in.set_aspect("equal")
    ax_in.axis("off")

    # 3-D shaded sphere image
    sphere_img = _render_sphere_image(
        220, center_lat, center_lon,
        lat_min, lat_max, lon_min, lon_max,
    )
    # imshow: origin='upper' so row-0 (North) is at the top; extent in km
    ax_in.imshow(sphere_img, extent=[-R, R, -R, R],
                 origin="upper", zorder=2, interpolation="bilinear")

    # Limb circle
    ax_in.add_patch(plt.Circle((0, 0), R,
                               color=MOON_EDGE, fill=False, lw=0.8, zorder=3))

    margin = R * 1.10
    ax_in.set_xlim(-margin, margin)
    ax_in.set_ylim(-margin, margin)

    ax_in.text(0, -margin * 0.88, "context", color=DIM_COL,
               fontsize=5, ha="center", va="top")


# ── Info panel ───────────────────────────────────────────────────────────────

def _draw_info_panel(ax, stem,
                     lat_min, lat_max, lon_min, lon_max,
                     height_km, ext_ns_km, ext_ew_km,
                     center_lat, center_lon):
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    def to_180(lon):
        return ((lon + 180.0) % 360.0) - 180.0

    dt = stem_to_datetime(stem)
    time_str = dt.strftime("%Y-%m-%d  %H:%M:%S UTC") if dt else stem

    # Decorative left bar
    ax.plot([0.06, 0.06], [0.01, 0.99], color="#2244aa",
            lw=3, transform=ax.transAxes)

    rows = [
        # (label, value, value_color, label_fs, val_fs)
        ("FRAME",      time_str,  "#ccddff", 7, 8),
        None,
        ("LAT  MIN",   f"{lat_min:.4f}°",  "#aaccff", 7, 9),
        ("LAT  MAX",   f"{lat_max:.4f}°",  "#aaccff", 7, 9),
        ("LON  MIN",   f"{to_180(lon_min):.4f}°", "#aaccff", 7, 9),
        ("LON  MAX",   f"{to_180(lon_max):.4f}°", "#aaccff", 7, 9),
        ("CENTER LAT", f"{center_lat:.3f}°",        "#aaccff", 7, 9),
        ("CENTER LON", f"{to_180(center_lon):.3f}°","#aaccff", 7, 9),
        None,
        ("HEIGHT",     f"{height_km:.2f} km",  "#ffe088", 7, 10),
        None,
        ("EXTENT  N–S", f"{ext_ns_km:.1f} km", "#88ff99", 7, 10),
        ("EXTENT  E–W", f"{ext_ew_km:.1f} km", "#88ff99", 7, 10),
        None,
        ("MOON  R",    f"{MOON_RADIUS_KM:.1f} km", DIM_COL, 7, 8),
    ]

    y = 0.97
    for item in rows:
        if item is None:
            y -= 0.025
            continue
        label, value, vcol, lfs, vfs = item
        ax.text(0.12, y, label,
                color=DIM_COL, fontsize=lfs, fontfamily="monospace",
                transform=ax.transAxes, va="top")
        y -= 0.042
        ax.text(0.12, y, value,
                color=vcol, fontsize=vfs, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.058


# ── Full figure ───────────────────────────────────────────────────────────────

def make_illustration(stem, lat_min, lat_max, lon_min, lon_max,
                      height_km, out_path) -> tuple:
    R = MOON_RADIUS_KM
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    half_lat_deg = (lat_max - lat_min) / 2.0
    # E-W angular half-extent on sphere (lon deg scaled by cos(lat))
    half_lon_ang = (lon_max - lon_min) / 2.0 * np.cos(np.radians(center_lat))

    ext_ns_km = (lat_max - lat_min) * KM_PER_DEG_LAT
    ext_ew_km = (lon_max - lon_min) * km_per_deg_lon(center_lat)

    fig = plt.figure(figsize=(17, 7.5), facecolor=SPACE_BG)
    fig.suptitle("Lunar Orbital Geometry", color=TEXT_COL,
                 fontsize=13, y=0.99, fontweight="bold")

    ax_ns  = fig.add_axes([0.02,  0.05, 0.37, 0.90])
    ax_ew  = fig.add_axes([0.415, 0.05, 0.37, 0.90])
    ax_txt = fig.add_axes([0.808, 0.03, 0.185, 0.93])

    _draw_cross_section(
        ax_ns, R, height_km, half_lat_deg,
        label_p="N", label_m="S",
        title="N – S  cross-section",
        ext_km=ext_ns_km / 2,
        center_lat=center_lat, center_lon=center_lon,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )
    _draw_cross_section(
        ax_ew, R, height_km, half_lon_ang,
        label_p="E", label_m="W",
        title="E – W  cross-section",
        ext_km=ext_ew_km / 2,
        center_lat=center_lat, center_lon=center_lon,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )
    _draw_info_panel(
        ax_txt, stem,
        lat_min, lat_max, lon_min, lon_max,
        height_km, ext_ns_km, ext_ew_km,
        center_lat, center_lon,
    )

    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=SPACE_BG, pad_inches=0.15)
    plt.close(fig)
    return ext_ns_km, ext_ew_km


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate orbital geometry illustrations for a batch.")
    p.add_argument("--batch", required=True,
                   help="Batch output directory (contains json/ sub-folder)")
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
    illust_dir = os.path.join(batch_dir, "illustration")

    if not os.path.isdir(json_dir):
        sys.exit(f"[ERROR] Directory not found: {json_dir}")

    os.makedirs(illust_dir, exist_ok=True)

    # Per-frame heights from coordinates file
    heights_map: dict = {}
    if args.coords_file:
        heights_map = parse_coords_heights(args.coords_file)
        print(f"Loaded {len(heights_map)} height entries from {args.coords_file}")

    # Fallback height from base config
    fallback_h = 50.0
    if os.path.isfile(args.config):
        with open(args.config) as fh:
            fallback_h = json.load(fh)["camera"]["height_km"]

    json_files = sorted(
        f for f in os.listdir(json_dir) if f.lower().endswith(".json")
    )
    if not json_files:
        sys.exit("[ERROR] No JSON files found in json/")

    print(f"Generating {len(json_files)} illustration(s) → {illust_dir}")
    print()

    for fname in json_files:
        stem      = os.path.splitext(fname)[0]
        json_path = os.path.join(json_dir, fname)
        out_path  = os.path.join(illust_dir, f"{stem}.png")

        with open(json_path) as fh:
            j = json.load(fh)

        lat_min, lat_max = j["lat_range"]
        lon_min, lon_max = j["lon_range"]

        # Height lookup priority: coords-file > _tmp config > base config
        if stem in heights_map:
            height_km = heights_map[stem]
        else:
            tmp_cfg = os.path.join(batch_dir, "_tmp", stem, "config.json")
            if os.path.isfile(tmp_cfg):
                with open(tmp_cfg) as fh:
                    height_km = json.load(fh)["camera"]["height_km"]
            else:
                height_km = fallback_h

        ext_ns, ext_ew = make_illustration(
            stem, lat_min, lat_max, lon_min, lon_max, height_km, out_path
        )
        print(f"  {stem}  h={height_km:.1f} km  "
              f"N-S={ext_ns:.0f} km  E-W={ext_ew:.0f} km  "
              f"→ {os.path.basename(out_path)}")

    print()
    print(f"=== Done: {len(json_files)} illustration(s) in {illust_dir} ===")


if __name__ == "__main__":
    main()
