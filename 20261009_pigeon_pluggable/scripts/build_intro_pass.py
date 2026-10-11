"""Intro pass for the pigeon film (owner item F8, DevLog-015 / DevLog-017): the Tree agent's card set rendered as its own
pass (Standard view transform; AgX flattens its colours), (v4: captions are burned in post by A). The street is
cross-dissolved in at encode time (ffmpeg xfade, offset 4.5 s, 0.8 s), so the intro is never overlaid inside the street.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/build_intro_pass.py -- scenes/intro_v2.blend
Frames 1-160 (0-5.3 s). Env: FILM_TIER / FILM_W / FILM_H as in build_film.py.
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "intro"))
import build_film as BF  # noqa: E402  (caption layout, narration.json)
import build_intro as BI  # noqa: E402  (Tree agent, DevLog-017)

L = BF.L
F_END = 160


def build(out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    tier = os.environ.get("FILM_TIER", "draft")
    tw, th = {"preview": (360, 450), "draft": (540, 674), "review": (1080, 1350)}.get(tier, (540, 674))
    scn.render.resolution_x = int(os.environ.get("FILM_W", tw))
    scn.render.resolution_y = int(os.environ.get("FILM_H", th))
    scn.eevee.taa_render_samples = int(os.environ.get("FILM_SAMPLES", "1"))
    for attr, val in (("use_shadows", False), ("use_raytracing", False), ("use_gtao", False)):
        if hasattr(scn.eevee, attr):
            setattr(scn.eevee, attr, val)
    scn.render.use_persistent_data = True
    scn.render.use_compositing = False
    scn.render.fps = 30
    scn.frame_start, scn.frame_end = 1, F_END
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("INTRO_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*BI.C_BG, 1)
    scn.world = w
    root = BI.append_intro(location=(0.0, 0.0, 0.0))
    L.HOLD_Z = 0.06
    L.HOLD_S0 = L.HOLD_Z / 3.0
    L.setup_camera()
    L.CAM.data.clip_start = 0.004
    tgt = BI.key_camera(L.CAM, root, f_start=1)
    L.HOLD.scale = (L.HOLD_S0 * L.LENS0 / L.CAM.data.lens,) * 3     # captions at the same screen size as the main film
    # v4: A burns captions into the bottom 18 %: lower camera + target together so the cards sit higher in frame
    # (content moves up ~9 % of the frame height), and darken the band with a soft gradient
    import mathutils
    cams = {fc.array_index: fc for fc in L.CAM.animation_data.action.fcurves if fc.data_path == "location"}
    tgts = {fc.array_index: fc for fc in tgt.animation_data.action.fcurves if fc.data_path == "location"}
    for kc, kt in zip(cams[2].keyframe_points, tgts[2].keyframe_points):
        f = kc.co.x
        c = mathutils.Vector([cams[k].evaluate(f) for k in range(3)])
        t = mathutils.Vector([tgts[k].evaluate(f) for k in range(3)])
        dz = 0.085 * (c - t).length
        for kp in (kc, kt):
            kp.co.y -= dz
            kp.handle_left.y -= dz
            kp.handle_right.y -= dz
    for fc in list(cams.values()) + list(tgts.values()):
        fc.update()
    # v4f: keep every visible card face above 0.80 * H at the opening keys (the climb then exits through the opaque band)
    from bpy_extras.object_utils import world_to_camera_view
    faces = [o for o in bpy.data.objects if o.type == "MESH" and o.name.split(".")[0].endswith("_face")]
    scn_ = bpy.context.scene
    vh = 2 * 18.0 * 674.0 / 540.0 / L.CAM.data.lens      # frame height / distance
    for kc, kt in zip(cams[2].keyframe_points, tgts[2].keyframe_points):
        f = int(round(kc.co.x))
        if f > 5:
            continue
        scn_.frame_set(f)
        lo = 1.0
        for o in faces:
            pts = [world_to_camera_view(scn_, L.CAM, o.matrix_world @ mathutils.Vector(v)) for v in o.bound_box]
            if any(0.0 <= q.x <= 1.0 and q.y <= 1.0 and q.z > 0 for q in pts) and max(q.y for q in pts) > 0.0:
                lo = min(lo, min(q.y for q in pts))
        if lo < 0.23:
            c = mathutils.Vector([cams[k].evaluate(f) for k in range(3)])
            t = mathutils.Vector([tgts[k].evaluate(f) for k in range(3)])
            dz = (0.23 - lo) * vh * (c - t).length
            for kp in (kc, kt):
                kp.co.y -= dz
                kp.handle_left.y -= dz
                kp.handle_right.y -= dz
            print("INTRO key", f, "lowest card %.3f -> lowered %.3f m" % (lo, dz))
    for fc in list(cams.values()) + list(tgts.values()):
        fc.update()
    m = bpy.data.materials.new("intro_caption_band")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0, 0, 0, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.86
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = 1.0
    ramp.inputs["To Max"].default_value = 0.0
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    band = L.plane("intro_caption_band", (0.0, -2.41 + 0.55, -0.02), (3.9, 1.1), m, rot=(0, 0, 0), parent=L.HOLD)   # opaque to 0.80 H
    print("INTRO lens", L.CAM.data.lens, "(v4: no rendered captions; A burns them in post)")
    L.finalize_visibility(F_END)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("SAVED", out_path)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build(argv[0] if argv else "intro_v2.blend")
