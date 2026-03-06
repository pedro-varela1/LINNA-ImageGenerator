import math
import json

import numpy as np

MOON_RADIUS_KM = 1737.4


def compute_pixel_latlon(cam_lat, cam_lon, cam_height_km, fov_deg,
                          tilt_deg, azimuth_deg, render_w, render_h):
    """
    Compute per-pixel geographic coordinates via pinhole-camera ray casting
    onto the flat terrain plane (Z = 0).  Fully vectorised with NumPy.

    Returns
    -------
    lat_map, lon_map : ndarray of shape (H, W), dtype float64
        Degrees.  NaN where the ray misses the terrain (sky pixels).
    """
    fov_h  = math.radians(fov_deg)
    aspect = render_w / render_h
    fov_v  = 2 * math.atan(math.tan(fov_h / 2) / aspect)

    t_r = math.radians(tilt_deg)
    a_r = math.radians(azimuth_deg)

    km_per_deg_lat = math.pi * MOON_RADIUS_KM / 180.0
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(cam_lat))

    u_frac = (np.arange(render_w) + 0.5) / render_w - 0.5   # (W,)
    v_frac = (np.arange(render_h) + 0.5) / render_h - 0.5   # (H,)

    # Camera-space ray directions (camera looks along −Z)
    cx = np.tan(u_frac * fov_h)[np.newaxis, :]   # (1, W)
    cy = -np.tan(v_frac * fov_v)[:, np.newaxis]  # (H, 1)
    cz = -1.0

    # Apply tilt (rotation around X)
    ct, st = math.cos(t_r), math.sin(t_r)
    ry = cy * ct - cz * st
    rz = cy * st + cz * ct
    rx = cx

    # Apply azimuth (rotation around Z), broadcast to (H, W)
    ca, sa = math.cos(a_r), math.sin(a_r)
    rx = np.broadcast_to(rx, (render_h, render_w)).copy()
    ry = np.broadcast_to(ry, (render_h, render_w)).copy()
    rz = np.broadcast_to(rz, (render_h, render_w)).copy()

    wx = rx * ca - ry * sa   # East  component
    wy = rx * sa + ry * ca   # North component
    wz = rz                  # Up component (negative = pointing down)

    valid   = wz < -1e-9
    safe_wz = np.where(valid, wz, -1e-9)
    t_hit   = np.where(valid, -cam_height_km / safe_wz, np.nan)

    lat_map = cam_lat + (t_hit * wy) / km_per_deg_lat
    lon_map = cam_lon + (t_hit * wx) / km_per_deg_lon
    lat_map[~valid] = np.nan
    lon_map[~valid] = np.nan
    return lat_map, lon_map


def save_latlon_map(lat_map, lon_map, json_path):
    """
    Save lat/lon arrays to a compressed .npz file and write a human-readable
    corner-coordinate summary to a .json file.
    """
    npz_path = json_path.replace(".json", ".npz")
    np.savez_compressed(npz_path, lat=lat_map, lon=lon_map)
    print(f"[LatLon] Arrays  → {npz_path}")

    summary = {
        "description": "Per-pixel lunar lat/lon. Full arrays in .npz (np.load).",
        "array_shape_HxW": list(lat_map.shape),
        "corners_deg": {
            "top_left":     [float(lat_map[0,  0]),  float(lon_map[0,  0])],
            "top_right":    [float(lat_map[0, -1]),  float(lon_map[0, -1])],
            "bottom_left":  [float(lat_map[-1, 0]),  float(lon_map[-1, 0])],
            "bottom_right": [float(lat_map[-1,-1]),  float(lon_map[-1,-1])],
        },
        "lat_range": [float(np.nanmin(lat_map)), float(np.nanmax(lat_map))],
        "lon_range": [float(np.nanmin(lon_map)), float(np.nanmax(lon_map))],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[LatLon] Summary → {json_path}")
