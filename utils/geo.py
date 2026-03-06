import math

MOON_RADIUS_KM = 1737.4


def km_per_deg_lat():
    """kilometres per degree of latitude on the Moon."""
    return math.pi * MOON_RADIUS_KM / 180.0


def km_per_deg_lon(lat_deg):
    """kilometres per degree of longitude at a given latitude."""
    return km_per_deg_lat() * math.cos(math.radians(lat_deg))


def normalize_lon(lon_deg):
    """Normalize any longitude value to the [0, 360) range."""
    return lon_deg % 360.0
