"""
utils/sphere.py
===============
Shared helpers for the spherical Moon model.

The local Blender scene is centred at the nadir point on the lunar surface.
We define a right-handed frame:
    E  = East  unit vector  (X axis in Blender)
    N  = North unit vector  (Y axis in Blender)
    U  = Up (radially outward from Moon centre)  (Z axis in Blender)

The camera sits at (0, 0, cam_height_km) in this local frame.
"""

import math
import numpy as np

MOON_RADIUS_KM = 1737.4


def local_frame(lat_deg, lon_deg):
    """
    Return the 3×3 rotation matrix R whose columns are [E, N, U] expressed
    in Moon-centred Cartesian coordinates (MCMF), i.e.:

        P_mcmf = R @ P_local  +  nadir_mcmf

    Parameters
    ----------
    lat_deg, lon_deg : float  (degrees)

    Returns
    -------
    R      : (3,3) ndarray, columns = [E, N, U] in MCMF
    nadir  : (3,) ndarray  — nadir point in MCMF (km)
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    cl, sl = math.cos(lat), math.sin(lat)
    co, so = math.cos(lon), math.sin(lon)

    # Unit vectors in MCMF
    U = np.array([ cl * co,  cl * so,  sl])          # radially outward
    E = np.array([-so,        co,        0.0])        # East
    N = np.array([-sl * co,  -sl * so,  cl])          # North = U × E (normalised)

    R = np.column_stack([E, N, U])                    # columns = axes
    nadir = MOON_RADIUS_KM * U
    return R, nadir


def latlon_to_local(lat_deg, lon_deg, cam_lat_deg, cam_lon_deg):
    """
    Convert a surface point (lat_deg, lon_deg) to local scene coordinates
    relative to the nadir at (cam_lat_deg, cam_lon_deg).

    Returns (x_local, y_local, z_local) in km.
    """
    R, nadir = local_frame(cam_lat_deg, cam_lon_deg)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    P_mcmf = MOON_RADIUS_KM * np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    ])
    P_local = R.T @ (P_mcmf - nadir)
    return P_local  # (x=E, y=N, z=U_rel)
