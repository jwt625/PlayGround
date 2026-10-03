"""Render presets, output helpers and camera helpers for the film (Blender 4.2, EEVEE Next). 1080x1350 (4:5), 30 fps.

Usage in a Blender session:
    import sys; sys.path.insert(0, "<repo>/scripts/assets/materials_vfx")
    import render_presets as RP
    RP.apply_render_preset(scene, "standard")        # draft | standard | hero
    RP.set_output_h264(scene, "/path/out.mp4")       # or RP.set_output_png(scene, "/path/frames/f_")
    cam, target = RP.camera_rig(scene, "CAM_s1")
    RP.add_shake(cam, amp_loc=0.01, amp_rot=0.004, frame_range=(1, 300))
    RP.whip_pan(target, f0=120, f1=126, to_loc=(5, 0, 1.2))
    RP.hard_cut(scene, [(1, cam_a), (91, cam_b)])
    RP.speed_ramp(obj, '["p_t"]', [(1, 0.0), (60, 1.0)], speeds=[0.3, 2.5, 0.3])
Measured per-frame costs are in the JSON metadata (render_presets.json) and DevLog-003-materials_vfx.md.
"""
import math

import bpy

W, H, FPS = 1080, 1350, 30

PRESETS = {
    # taa samples, ray tracing, fast GI, shadow rays/steps, motion blur, soft shadows, volumetric samples
    "draft": dict(samples=8, raytracing=False, shadow_rays=1, shadow_steps=6, motion_blur=False, gtao=False, res_pct=50),
    "standard": dict(samples=24, raytracing=False, shadow_rays=2, shadow_steps=8, motion_blur=False, gtao=False, res_pct=100),
    "hero": dict(samples=48, raytracing=True, shadow_rays=3, shadow_steps=12, motion_blur=True, gtao=True, res_pct=100),
}


def apply_render_preset(scn, name="standard", w=W, h=H, fps=FPS):
    p = PRESETS[name]
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.render.resolution_x, scn.render.resolution_y = w, h
    scn.render.resolution_percentage = p["res_pct"]
    scn.render.fps = fps
    scn.render.fps_base = 1.0
    ev = scn.eevee
    ev.taa_render_samples = p["samples"]
    ev.taa_samples = max(4, p["samples"] // 2)
    ev.use_shadows = True
    ev.shadow_ray_count = p["shadow_rays"]
    ev.shadow_step_count = p["shadow_steps"]
    ev.use_raytracing = p["raytracing"]
    if p["raytracing"]:
        ev.ray_tracing_method = "SCREEN"
        try:
            ev.fast_gi_method = "GLOBAL_ILLUMINATION"
        except Exception:
            pass
    scn.render.use_motion_blur = p["motion_blur"]
    if p["motion_blur"]:
        scn.render.motion_blur_shutter = 0.5
    ev.use_volumetric_shadows = False
    # colour management: Standard, no look
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0
    scn.display_settings.display_device = "sRGB"
    scn.render.film_transparent = False
    return p


def set_output_png(scn, prefix, color_depth="8", compression=15):
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = color_depth
    scn.render.image_settings.compression = compression
    scn.render.filepath = prefix
    return scn.render.filepath


def set_output_h264(scn, path, crf="HIGH"):
    """H.264 in MP4 (yuv420p, +faststart behaviour of Blender's ffmpeg muxer is not exposed; re-mux with ffmpeg -movflags +faststart if needed)."""
    r = scn.render
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = crf
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.ffmpeg.gopsize = 30
    r.ffmpeg.audio_codec = "NONE"
    r.filepath = path
    return path


# ----------------------------------------------------------------------------- camera helpers
def camera_rig(scn, name="CAM", lens=28.0, loc=(0, -5, 1.5), target_loc=(0, 0, 1.0)):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.clip_start, cd.clip_end = 0.05, 2000
    cam = bpy.data.objects.new(name, cd)
    scn.collection.objects.link(cam)
    cam.location = loc
    tg = bpy.data.objects.new(name + "_TARGET", None)
    tg.location = target_loc
    scn.collection.objects.link(tg)
    c = cam.constraints.new("TRACK_TO")
    c.target = tg
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    if scn.camera is None:
        scn.camera = cam
    return cam, tg


def attach_dof(cam, focus_obj=None, fstop=2.8, distance=None):
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = fstop
    if focus_obj is not None:
        cam.data.dof.focus_object = focus_obj
    elif distance is not None:
        cam.data.dof.focus_distance = distance


def add_shake(obj, amp_loc=0.01, amp_rot=0.004, freq=0.12, frame_range=None, seed=1):
    """Handheld shake: Noise modifiers on location and rotation F-curves (additive). Frequency = 1/scale in frames. Keyframes the base
    transform at the range start so the F-curves exist. Rotation shakes are in radians (0.004 rad = 0.23 deg)."""
    f0 = frame_range[0] if frame_range else bpy.context.scene.frame_start
    for path, amp in (("location", amp_loc), ("rotation_euler", amp_rot)):
        for i in range(3):
            obj.keyframe_insert(path, index=i, frame=f0)
    ad = obj.animation_data
    for fc in ad.action.fcurves:
        if fc.data_path not in ("location", "rotation_euler"):
            continue
        amp = amp_loc if fc.data_path == "location" else amp_rot
        m = fc.modifiers.new("NOISE")
        m.scale = 1.0 / max(freq, 1e-3)
        m.strength = amp
        m.phase = seed * 7.3 + fc.array_index * 3.1
        m.depth = 1
        if frame_range:
            m.use_restricted_range = True
            m.frame_start, m.frame_end = frame_range
            m.blend_in = 3
            m.blend_out = 3


def whip_pan(target, f0, f1, to_loc, back_to=None):
    """Whip pan: the camera TARGET empty moves fast (bezier ease) from its current location to to_loc between f0 and f1 (usually 4-8 frames);
    use with motion blur on (hero preset) or the motion_streaks effect for the blur feel."""
    target.keyframe_insert("location", frame=f0)
    target.location = to_loc
    target.keyframe_insert("location", frame=f1)
    for fc in target.animation_data.action.fcurves:
        if fc.data_path == "location":
            for kp in fc.keyframe_points:
                if kp.co[0] in (f0, f1):
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"


def hard_cut(scn, cuts):
    """Hard cuts via timeline markers bound to cameras: cuts = [(frame, camera_object), ...]."""
    for f, cam in cuts:
        m = scn.timeline_markers.new("cut_%d" % f, frame=f)
        m.camera = cam
    return scn.timeline_markers


def speed_ramp(obj, data_path, keys, index=-1):
    """Keyframe a property with smooth (bezier, auto-clamped) speed changes. keys = [(frame, value), ...]: the slope between keys is the
    speed, so slow -> fast -> slow is expressed by key spacing, e.g. speed_ramp(root, '["p_t"]', [(1, 0.0), (30, 0.3), (36, 2.0), (66, 2.4)])
    for a time-remap of an effect, or speed_ramp(obj, 'location', [(1, 0), (30, 1), (40, 9)], index=0)."""
    for f, v in keys:
        if data_path.startswith("["):
            obj[data_path[2:-2]] = v
            obj.keyframe_insert(data_path, frame=f)
        else:
            if index >= 0:
                getattr(obj, data_path)[index] = v
            else:
                setattr(obj, data_path, v)
            obj.keyframe_insert(data_path, frame=f, index=index)
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"


def setup_comp(scn, preset="cartoon"):
    """Put the NG_comp_post node group (append it from lighting_and_world.blend first) between Render Layers and Composite."""
    presets = {"off": (0.0, 0.0, 0.0, 0.0), "light": (0.25, 0.2, 0.0015, 0.04), "cartoon": (0.6, 0.3, 0.003, 0.08), "hero_glow": (1.2, 0.4, 0.005, 0.12)}
    scn.use_nodes = True
    nt = scn.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    g = nt.nodes.new("CompositorNodeGroup")
    g.node_tree = bpy.data.node_groups["NG_comp_post"]
    cp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], g.inputs["Image"])
    nt.links.new(g.outputs["Image"], cp.inputs["Image"])
    b, v, c, gr = presets[preset]
    g.inputs["Bloom Amount"].default_value = b
    g.inputs["Vignette Amount"].default_value = v
    g.inputs["CA Amount"].default_value = c
    g.inputs["Grain Amount"].default_value = gr
    return g
