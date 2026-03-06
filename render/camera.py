import bpy
import math
from mathutils import Vector


def place_camera(cam_height_km, fov_deg, tilt_deg, azimuth_deg):
    """
    Place the camera at the centre of the terrain patch (0, 0, cam_height_km).
    tilt_deg=0  → nadir look-down.
    tilt_deg=30 → 30° off-nadir in the direction of azimuth_deg.
    """
    cam_data            = bpy.data.cameras.new(name="LunarCamera")
    cam_data.lens_unit  = "FOV"
    cam_data.angle      = math.radians(fov_deg)
    cam_data.clip_start = 0.00001
    cam_data.clip_end   = 10000.0

    cam_obj = bpy.data.objects.new("LunarCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location       = Vector((0.0, 0.0, cam_height_km))
    cam_obj.rotation_mode  = "ZXY"
    cam_obj.rotation_euler = (math.radians(tilt_deg), 0, math.radians(azimuth_deg))
    bpy.context.scene.camera = cam_obj
    return cam_obj
