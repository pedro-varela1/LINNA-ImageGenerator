import math

MOON_RADIUS_KM = 1737.4
KM_PER_DEG_LAT = math.pi * MOON_RADIUS_KM / 180.0


def km_per_deg_lat():
    """kilometres per degree of latitude on the Moon."""
    return KM_PER_DEG_LAT


def km_per_deg_lon(lat_deg):
    """kilometres per degree of longitude at a given latitude."""
    return KM_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def normalize_lon(lon_deg):
    """Normalize any longitude value to the [0, 360) range."""
    return lon_deg % 360.0


def lat_patch_half_deg(lat_deg, height_km, fov_deg, tilt_deg, width, height,
                       margin=1.5):
    """Estimate lat patch half-extent in degrees (used for legacy DEM coverage checks)."""
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
    return max(0.1, min(max_ground_dist_km / KM_PER_DEG_LAT, 15.0))
