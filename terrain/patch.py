import math
from utils.geo import km_per_deg_lat, km_per_deg_lon


def compute_patch_half_deg(height_km, fov_deg, tilt_deg, cam_lat_deg,
                            render_width=1920, render_height=1080, margin=1.5):
    """
    Return the minimum terrain-patch half-extent in degrees so that the mesh
    borders stay outside the camera frustum, with a safety margin.

    Args:
        height_km     : camera altitude above the terrain plane (km)
        fov_deg       : horizontal FOV in degrees
        tilt_deg      : off-nadir tilt angle in degrees (0 = nadir)
        cam_lat_deg   : camera latitude – needed for lon km/deg conversion
        render_width  : image width  in pixels (used for aspect ratio)
        render_height : image height in pixels
        margin        : safety multiplier (1.5 → 50 % extra beyond visible area)

    Returns:
        patch_half_deg (float)
    """
    fov_h  = math.radians(fov_deg)
    aspect = render_width / render_height
    fov_v  = 2 * math.atan(math.tan(fov_h / 2) / aspect)
    tilt_r = math.radians(tilt_deg)

    max_angle = tilt_r + fov_v / 2.0

    if max_angle >= math.radians(89.9):
        max_ground_dist_km = height_km * math.tan(math.radians(89.0))
    else:
        max_ground_dist_km = height_km * math.tan(max_angle)

    half_fov_h_at_far  = height_km / math.cos(max_angle) * math.tan(fov_h / 2)
    max_ground_dist_km = math.sqrt(max_ground_dist_km**2 + half_fov_h_at_far**2)
    max_ground_dist_km *= margin

    kpd_lat = km_per_deg_lat()
    kpd_lon = km_per_deg_lon(cam_lat_deg)
    patch_half_deg = max_ground_dist_km / min(kpd_lat, kpd_lon)
    patch_half_deg = max(0.1, min(patch_half_deg, 15.0))

    print(f"[Patch] Max visible ground distance: {max_ground_dist_km / margin:.2f} km")
    print(f"[Patch] Patch half-extent (with {margin}x margin): {patch_half_deg:.3f} deg")
    return patch_half_deg
