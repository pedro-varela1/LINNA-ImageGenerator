import math
import json

import numpy as np

from utils.sphere import MOON_RADIUS_KM, local_frame


def compute_pixel_latlon(cam_lat, cam_lon, cam_height_km, fov_deg,
                          tilt_deg, azimuth_deg, render_w, render_h):
    """
    Compute per-pixel geographic coordinates via pinhole-camera ray casting
    onto the spherical Moon surface.  Fully vectorised with NumPy.

    The camera is placed at (0, 0, cam_height_km) in the local East-North-Up
    frame centred at the nadir (cam_lat, cam_lon).  Each pixel ray is
    intersected with the sphere of radius MOON_RADIUS_KM.  The near
    intersection (facing the camera) is taken, giving accurate coordinates
    even for large FOVs at polar latitudes where the flat-plane approximation
    introduces significant distortion.

    Returns
    -------
    lat_map, lon_map : ndarray of shape (H, W), dtype float64
        Degrees.  NaN where the ray misses the lunar surface (sky pixels).
    """
    fov_h  = math.radians(fov_deg)
    aspect = render_w / render_h
    fov_v  = 2 * math.atan(math.tan(fov_h / 2) / aspect)

    t_r = math.radians(tilt_deg)
    a_r = math.radians(azimuth_deg)

    # Camera position in local frame (km)
    cam_pos = np.array([0.0, 0.0, cam_height_km])    # (3,)

    # Local-to-MCMF rotation matrix
    R_frame, nadir_mcmf = local_frame(cam_lat, cam_lon)

    # Camera position in MCMF (km)
    cam_mcmf = R_frame @ cam_pos + nadir_mcmf         # (3,)

    # Build pixel ray directions in local frame ---------------------------------
    u_frac = (np.arange(render_w) + 0.5) / render_w - 0.5   # (W,)
    v_frac = (np.arange(render_h) + 0.5) / render_h - 0.5   # (H,)

    # Camera-space: looks along −Z (local Up), X = East, Y = North
    cx = np.tan(u_frac * fov_h)[np.newaxis, :]    # (1, W)
    cy = -np.tan(v_frac * fov_v)[:, np.newaxis]   # (H, 1)
    cz = -1.0

    # Apply tilt (rotation around camera X)
    ct, st = math.cos(t_r), math.sin(t_r)
    ry = cy * ct - cz * st
    rz = cy * st + cz * ct
    rx = cx

    # Apply azimuth (rotation around local Z/Up)
    ca, sa = math.cos(a_r), math.sin(a_r)
    rx = np.broadcast_to(rx, (render_h, render_w)).copy()
    ry = np.broadcast_to(ry, (render_h, render_w)).copy()
    rz = np.broadcast_to(rz, (render_h, render_w)).copy()

    # Local-frame ray direction (East, North, Up) — not yet normalised
    d_local = np.stack([
        rx * ca - ry * sa,    # East
        rx * sa + ry * ca,    # North
        rz,                   # Up
    ], axis=-1)               # (H, W, 3)

    # Rotate to MCMF
    d_mcmf = d_local @ R_frame.T              # (H, W, 3)

    # Ray-sphere intersection --------------------------------------------------
    # Ray: P(t) = cam_mcmf + t * d
    # Sphere: |P|² = R²
    # => (d·d)t² + 2(A·d)t + (A·A - R²) = 0  where A = cam_mcmf
    R2 = MOON_RADIUS_KM ** 2
    A  = cam_mcmf                                      # (3,)

    a_coef = np.sum(d_mcmf * d_mcmf, axis=-1)         # (H, W)
    b_coef = 2.0 * np.einsum("ij,j->ij", a_coef[:, :1] * 0 + 1,  # broadcast trick
                              np.zeros(1))              # placeholder — computed below
    # Proper dot A·d
    b_coef = 2.0 * np.sum(d_mcmf * A, axis=-1)        # (H, W)
    c_coef = float(np.dot(A, A)) - R2                 # scalar

    discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef  # (H, W)

    valid = discriminant >= 0.0
    safe_disc = np.where(valid, discriminant, 0.0)

    # Two solutions; take the smaller positive t (near surface)
    sqrt_d = np.sqrt(safe_disc)
    t1 = (-b_coef - sqrt_d) / (2.0 * a_coef)
    t2 = (-b_coef + sqrt_d) / (2.0 * a_coef)

    # We want the smallest t > 0  (camera is above the surface)
    t_hit = np.where((t1 > 0) & valid, t1,
            np.where((t2 > 0) & valid, t2, np.nan))
    valid  = np.isfinite(t_hit) & (t_hit > 0)

    # Hit point in MCMF
    # P_hit (H, W, 3)  =  cam_mcmf  +  t_hit[:,:,None] * d_mcmf
    P_hit = cam_mcmf + t_hit[:, :, np.newaxis] * d_mcmf  # (H, W, 3)

    # Convert MCMF → lat/lon
    norm = np.linalg.norm(P_hit, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    P_unit = P_hit / norm

    lat_map = np.degrees(np.arcsin(np.clip(P_unit[:, :, 2], -1.0, 1.0)))
    lon_map = np.degrees(np.arctan2(P_unit[:, :, 1], P_unit[:, :, 0]))
    # Normalise lon to [0, 360) to match the rest of the pipeline
    lon_map = lon_map % 360.0

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
