import bpy
import math
from mathutils import Vector


def place_sun(azimuth_deg, elevation_deg, strength):
    """
    Add a directional Sun lamp and set the world background to black (no atmosphere).
    azimuth_deg  : 0=North, 90=East, 180=South, 270=West
    elevation_deg: angle above the horizon
    strength     : Cycles lamp energy in watts
    """
    light_data        = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = strength
    light_data.angle  = math.radians(0.53)   # realistic solar angular diameter

    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    sx =  math.sin(az) * math.cos(el)
    sy =  math.cos(az) * math.cos(el)
    sz =  math.sin(el)

    sun_dir = Vector((-sx, -sy, -sz))
    light_obj.rotation_euler = sun_dir.to_track_quat("-Z", "Y").to_euler()
    light_obj.location       = Vector((sx * 1000, sy * 1000, sz * 1000))

    # Black vacuum sky (Moon has no atmosphere)
    world = bpy.data.worlds.new("Space")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value    = (0, 0, 0, 1)
    bg.inputs["Strength"].default_value = 0.0
    return light_obj
