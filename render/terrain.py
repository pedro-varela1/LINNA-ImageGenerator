import bpy
import math
import os
import json


def build_terrain_mesh(cam_lat, cam_lon, lat_half, lon_half):
    """
    Create a flat plane whose physical size in km matches the terrain patch.
    UV [0,1] covers the full patch extent.
    Patch bounds are stored as custom properties for reference.

    lat_half and lon_half are kept separate so the mesh is correctly sized
    near the poles, where 1° of longitude covers far less ground than 1° of
    latitude.
    """
    lat_min = cam_lat - lat_half
    lat_max = cam_lat + lat_half
    lon_min = cam_lon - lon_half
    lon_max = cam_lon + lon_half

    km_per_deg_lat = math.pi * 1737.4 / 180.0
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(cam_lat))
    size_x = (lon_max - lon_min) * km_per_deg_lon
    size_y = (lat_max - lat_min) * km_per_deg_lat

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name    = "LunarTerrain"
    plane.scale   = (size_x, size_y, 1)
    bpy.ops.object.transform_apply(scale=True)

    plane["patch_lat_min"] = lat_min
    plane["patch_lat_max"] = lat_max
    plane["patch_lon_min"] = lon_min
    plane["patch_lon_max"] = lon_max
    return plane


def build_terrain_material(plane, disp_path, color_path, disp_meta_path):
    """
    Cycles node material using displacement_method='BOTH':
      - color_path  → Base Color
      - disp_path   → geometry Displacement node  (large-scale terrain shape)
      - disp_path   → Bump node → BSDF Normal     (micro-detail shading)

    Scale and Midlevel are read from disp_meta.json written by prepare_displacement().
    """
    mat = bpy.data.materials.new(name="LunarSurface")
    mat.use_nodes            = True
    mat.displacement_method  = "BOTH"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new("ShaderNodeOutputMaterial"); out.location  = (1100,   0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = ( 600,   0)
    bsdf.inputs["Roughness"].default_value           = 0.95   # lunar regolith
    bsdf.inputs["Specular IOR Level"].default_value  = 0.02
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    uv = nodes.new("ShaderNodeTexCoord"); uv.location = (-800, 0)

    # --- Colour map ---
    color_img               = bpy.data.images.load(color_path)
    color_tex               = nodes.new("ShaderNodeTexImage")
    color_tex.location      = (-300,  300)
    color_tex.image         = color_img
    color_tex.interpolation = "Linear"
    links.new(uv.outputs["UV"],           color_tex.inputs["Vector"])
    links.new(color_tex.outputs["Color"], bsdf.inputs["Base Color"])

    # --- Displacement map (shared by both paths; must be Non-Color) ---
    disp_img                          = bpy.data.images.load(disp_path)
    disp_img.colorspace_settings.name = "Non-Color"
    disp_tex                          = nodes.new("ShaderNodeTexImage")
    disp_tex.location                 = (-300, -50)
    disp_tex.image                    = disp_img
    disp_tex.interpolation            = "Linear"
    links.new(uv.outputs["UV"], disp_tex.inputs["Vector"])

    # Read scale + midlevel from disp_meta.json
    disp_scale, disp_midlevel = 1.0, 0.0
    if os.path.isfile(disp_meta_path):
        with open(disp_meta_path) as f:
            meta = json.load(f)
        disp_scale    = meta["scale"]
        disp_midlevel = meta["midlevel"]
        print(f"[Material] Displacement scale={disp_scale:.3f} km  midlevel={disp_midlevel:.3f}")
    else:
        print("[Material] disp_meta.json not found — using fallback scale=1.0, midlevel=0.0")

    # Path 1: geometry displacement
    disp_node          = nodes.new("ShaderNodeDisplacement")
    disp_node.location = (600, -300)
    disp_node.space    = "WORLD"    # absolute km units
    disp_node.inputs["Scale"].default_value    = disp_scale
    disp_node.inputs["Midlevel"].default_value = disp_midlevel
    links.new(disp_tex.outputs["Color"],         disp_node.inputs["Height"])
    links.new(disp_node.outputs["Displacement"], out.inputs["Displacement"])

    # Path 2: bump for micro-detail normals
    bump_node          = nodes.new("ShaderNodeBump")
    bump_node.location = (200, 150)
    bump_node.inputs["Strength"].default_value = 1.5
    bump_node.inputs["Distance"].default_value = 0.01   # km ≈ 10 m
    links.new(disp_tex.outputs["Color"],   bump_node.inputs["Height"])
    links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])

    plane.data.materials.append(mat)

    # Adaptive subdivision modifier
    subd                          = plane.modifiers.new(name="Subdivision", type="SUBSURF")
    subd.subdivision_type         = "SIMPLE"
    subd.levels                   = 0
    subd.render_levels            = 8
    subd.use_adaptive_subdivision = True

    bpy.ops.object.select_all(action="DESELECT")
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.shade_smooth()
    return mat
