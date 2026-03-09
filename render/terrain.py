import bpy
import bmesh
import math
import os
import json

import numpy as np

from utils.sphere import MOON_RADIUS_KM, local_frame


# Number of grid divisions along each axis of the spherical cap mesh.
# 64×64 = 4096 quads — fine enough for subdivision + displacement.
_GRID_DIV = 64


def build_terrain_mesh(cam_lat, cam_lon, lat_half, lon_half):
    """
    Create a spherical-cap mesh whose vertices lie on the Moon's surface,
    centred at the nadir point (cam_lat, cam_lon).

    The local coordinate frame is:
        X = East,  Y = North,  Z = Up (radially outward)
    The camera sits at (0, 0, cam_height_km) in this frame.

    UV [0,1]×[0,1] maps the (lat_min..lat_max) × (lon_min..lon_max) patch
    exactly as in the old flat-plane pipeline, so textures are unchanged.

    Patch bounds are stored as custom properties for reference.
    """
    lat_min = cam_lat - lat_half
    lat_max = cam_lat + lat_half
    lon_min = cam_lon - lon_half
    lon_max = cam_lon + lon_half

    R_frame, nadir_mcmf = local_frame(cam_lat, cam_lon)

    n = _GRID_DIV + 1          # vertices per side
    lats = np.linspace(lat_min, lat_max, n)
    lons = np.linspace(lon_min, lon_max, n)

    # Build vertex positions on the sphere
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")  # (n, n)
    lat_r = np.radians(lat_g)
    lon_r = np.radians(lon_g)

    # MCMF Cartesian
    P_mcmf = MOON_RADIUS_KM * np.stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ], axis=-1)  # (n, n, 3)

    # Transform to local scene frame
    delta = P_mcmf - nadir_mcmf  # (n, n, 3)
    P_local = delta @ R_frame    # (n, n, 3)  — R_frame columns are [E,N,U]

    # UV coordinates:
    #   U = longitude fraction  (follows j-axis)
    #   V = latitude  fraction  (follows i-axis)
    # With indexing="ij": meshgrid(lat_fracs, lon_fracs) → first output varies
    # with i (lat), second varies with j (lon).
    lat_fracs = np.linspace(0.0, 1.0, n)
    lon_fracs = np.linspace(0.0, 1.0, n)
    v_g, u_g = np.meshgrid(lat_fracs, lon_fracs, indexing="ij")
    # v_g[i,j] = lat_fracs[i]  → V increases with latitude  ✓
    # u_g[i,j] = lon_fracs[j]  → U increases with longitude ✓

    # Build mesh with BMesh
    me = bpy.data.meshes.new("LunarTerrainMesh")
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    # Create all vertices
    verts = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            x, y, z = P_local[i, j]
            verts[i, j] = bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Create quads and assign UVs
    for i in range(_GRID_DIV):
        for j in range(_GRID_DIV):
            v00 = verts[i,   j  ]
            v10 = verts[i+1, j  ]
            v11 = verts[i+1, j+1]
            v01 = verts[i,   j+1]
            face = bm.faces.new([v00, v10, v11, v01])
            uvs = [
                (u_g[i,   j  ], v_g[i,   j  ]),
                (u_g[i+1, j  ], v_g[i+1, j  ]),
                (u_g[i+1, j+1], v_g[i+1, j+1]),
                (u_g[i,   j+1], v_g[i,   j+1]),
            ]
            for loop, uv in zip(face.loops, uvs):
                loop[uv_layer].uv = uv

    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new("LunarTerrain", me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    obj["patch_lat_min"] = lat_min
    obj["patch_lat_max"] = lat_max
    obj["patch_lon_min"] = lon_min
    obj["patch_lon_max"] = lon_max
    return obj


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

    # Path 1: geometry displacement  (along vertex normals — radially outward)
    disp_node          = nodes.new("ShaderNodeDisplacement")
    disp_node.location = (600, -300)
    disp_node.space    = "OBJECT"   # normals are radial, not all parallel to Z
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
