"""Verification renders for motion v2: contact strips at 540x675 EEVEE 16 samples, camera follows the root (or fixed), checker floor.

Run inside Blender (rig blend open):  render_strip(rig_root, action, frames, out_png_prefix, ...)
Tiles are written as PNG files; compose with ffmpeg (see motion_v2.compose_sheet).
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

W, H = 540, 675


def setup_scene(fps=30):
    scn = bpy.context.scene
    scn.render.fps = fps
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = 16
    scn.render.resolution_x, scn.render.resolution_y = W, H
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.compression = 90
    scn.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("PV_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.62, 0.70, 0.80, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    scn.world = w
    sun = bpy.data.lights.new("PV_sun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("PV_sun", sun)
    so.rotation_euler = (0.9, 0.15, 0.55)
    scn.collection.objects.link(so)
    fill = bpy.data.lights.new("PV_fill", "AREA")
    fill.energy = 250
    fill.size = 3
    fo = bpy.data.objects.new("PV_fill", fill)
    fo.location = (-2.5, -3, 2)
    fo.rotation_euler = (1.2, 0, -0.6)
    scn.collection.objects.link(fo)
    # checker floor with 0.25 m squares (procedural, rendered lit)
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "PV_floor"
    m = bpy.data.materials.new("PV_floor")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    chk = nt.nodes.new("ShaderNodeTexChecker")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    chk.inputs["Scale"].default_value = 4.0
    chk.inputs["Color1"].default_value = (0.50, 0.52, 0.50, 1)
    chk.inputs["Color2"].default_value = (0.36, 0.38, 0.36, 1)
    nt.links.new(tc.outputs["Object"], chk.inputs["Vector"])
    nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    fl.data.materials.append(m)
    cam = bpy.data.objects.new("PV_cam", bpy.data.cameras.new("PV_cam"))
    scn.collection.objects.link(cam)
    scn.camera = cam
    return cam


def aim(cam, loc, tgt, lens=40):
    cam.location = loc
    cam.data.lens = lens
    d = Vector(tgt) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


VIEWS = {
    # name: (camera offset from the focus point, lens)
    "front34": ((2.3, -3.3, 0.45), 42),
    "side": ((4.2, 0.0, 0.25), 42),
    "back34": ((-2.3, 3.3, 0.45), 42),
    "wide34": ((3.2, -4.6, 0.7), 40),
    "low34": ((2.2, -3.0, -0.2), 38),
}


def render_frames(root, arm, act, frames, out_prefix, view="front34", focus_z=0.9, speed=0.0, yaw=0.0, follow=True, cam=None,
                  lens=None, extra=None, travel_axis=(0, -1, 0)):
    """Render each action frame in `frames` to <out_prefix>_<i>.png. speed (m/s, world) moves the root along travel_axis."""
    scn = bpy.context.scene
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = act
    off, ln = VIEWS[view]
    paths = []
    ax = np.array(travel_axis, float)
    for i, f in enumerate(frames):
        t = f / 30.0
        p = ax * speed * t
        root.location = (p[0], p[1], 0.0)
        root.rotation_euler = (0, 0, yaw)
        if extra:
            extra(f)
        scn.frame_set(int(round(f)))
        bpy.context.view_layer.update()
        fx = p if follow else np.zeros(3)
        ca = cam if cam is not None else bpy.data.objects["PV_cam"]
        aim(ca, (fx[0] + off[0], fx[1] + off[1], off[2] + focus_z), (fx[0], fx[1], focus_z), lens or ln)
        path = "%s_%02d.png" % (out_prefix, i)
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    return paths
