import math
from utils.geo import km_per_deg_lat, km_per_deg_lon


def compute_patch_extents(height_km, fov_deg, tilt_deg, cam_lat_deg,
                           render_width=1920, render_height=1080, margin=1.5):
    """
    Return (lat_half_deg, lon_half_deg) — separate half-extents for lat and lon.

    Near the poles, lon degrees are physically much shorter than lat degrees,
    so the two halves diverge significantly.  Using a single value (as done
    previously) caused a square-in-km patch to appear as a distorted rectangle
    in the EQC texture, stretching the colour map horizontally.

    Caps:
        lat_half : max 15°  (keeps equatorial patches reasonable)
        lon_half : max 89°  (allows polar patches to cover the full FOV in km)
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

    lat_half = max(0.1, min(max_ground_dist_km / kpd_lat, 15.0))
    lon_half = max(0.1, min(max_ground_dist_km / kpd_lon, 89.0))

    print(f"[Patch] Max visible ground distance: {max_ground_dist_km / margin:.2f} km")
    print(f"[Patch] lat_half={lat_half:.3f} deg  lon_half={lon_half:.3f} deg  (margin={margin}x)")
    return lat_half, lon_half
