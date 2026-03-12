import bpy
import os


def clear_scene():
    """Remove all objects and data-blocks from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:    bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:   bpy.data.cameras.remove(block)
    for block in bpy.data.lights:    bpy.data.lights.remove(block)
    for block in bpy.data.images:    bpy.data.images.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)


def setup_renderer(render_width, render_height, render_samples,
                   output_image_path, use_gpu=True):
    """Configure the Cycles renderer.  Tries OPTIX → CUDA → HIP → METAL for GPU."""
    scene = bpy.context.scene
    scene.render.engine                      = "CYCLES"
    scene.cycles.use_adaptive_sampling       = True
    scene.cycles.adaptive_threshold          = 0.02
    scene.cycles.samples                     = render_samples
    scene.render.resolution_x                = render_width
    scene.render.resolution_y                = render_height
    scene.render.filepath                    = output_image_path
    scene.render.image_settings.file_format  = "PNG"
    scene.render.image_settings.color_depth  = "16"

    scene.view_settings.view_transform = "Filmic"
    # scene.view_settings.exposure       = 10.0

    # Better shadow ray sampling (Blender 3.5+)
    scene.cycles.use_light_tree = True

    # Finer adaptive tessellation at render time — reduces staircase shadow edges.
    # Default is 1.0 px/polygon; 0.5 doubles precision along each axis.
    scene.cycles.dicing_rate         = 0.5
    scene.cycles.offscreen_dicing_scale = 4.0

    # Denoise the final render — removes residual shadow noise
    scene.cycles.use_denoising          = True
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    _nvoptix = "/usr/share/nvidia/nvoptix.bin"
    if os.path.isfile(_nvoptix):
        scene.cycles.denoiser = "OPTIX"
        print("[Renderer] Denoiser: OPTIX")
    else:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        print("[Renderer] Denoiser: OPENIMAGEDENOISE")

    if use_gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.refresh_devices()
        for device_type in ("OPTIX", "CUDA", "HIP", "METAL"):
            try:
                prefs.compute_device_type = device_type
            except TypeError:
                continue          # this device type is not available on this system
            prefs.refresh_devices()
            devices = prefs.get_devices_for_type(device_type)
            if devices:
                for d in devices:
                    d.use = True
                scene.cycles.device = "GPU"
                print(f"[Renderer] GPU: {device_type}")
                break
        else:
            scene.cycles.device = "CPU"
            print("[Renderer] No GPU found — using CPU.")
    else:
        scene.cycles.device = "CPU"
