"""Preview stills of the environment blends (workstream E, DevLog-013). EEVEE draft, 540 x 674 (film 4:5).

Usage: scripts/bslot.sh -b <blend> --python scripts/env/render_previews.py -- <set> [view ...]
  set = opening | ending | hatchery ; views default to all of the set. Outputs to outputs/assets/environment/.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_lib as E  # noqa: E402

import vfx  # noqa: E402

VIEWS = {
    "opening": {
        "wide": ((0.6, 30.0, 2.4), (0.0, 13.0, 3.6), 24),
        "s1_start": ((0.55, 19.9, 0.75), (0.1, 17.0, 1.0), 24),
        "flap": ((0.05, 15.5, 3.45), (0.0, 13.0, 3.25), 26),
        "facade": ((-2.5, 21.0, 1.7), (1.0, 13.0, 3.4), 24),
        "tree_alt": ((0.4, 24.0, 1.6), (0.0, 17.0, 1.6), 24),
        "plan": ((0.0, 23.0, 45.0), (0.0, 23.0, 0.0), "ORTHO:26"),
    },
    "ending": {
        "door_out": ((0.0, 13.7, 1.6), (0.4, 24.0, 2.6), 24),
        "crane": ((-1.2, 16.5, 7.0), (0.6, 24.5, 3.2), 24),
        "perches": ((1.4, 17.2, 3.4), (-1.2, 21.0, 3.6), 24),
        "wires": ((2.4, 17.0, 4.0), (5.0, 21.5, 5.4), 26),
        "sky": ((0.5, 19.0, 1.5), (4.0, 38.0, 26.0), 30),
        "sat_roofcam": ((-4.3, 18.3, 8.3), (-2.0, 26.0, 16.0), 22),
        "puddles_low": ((0.6, 15.2, 1.1), (0.4, 24.0, 0.0), 26),
        "plan": ((0.0, 23.0, 45.0), (0.0, 23.0, 0.0), "ORTHO:26"),
        # S's S6 camera keys (scripts/build_film.py, film v2.1): door 2.0 s, sidewalk 4.75 s, up-shot 5.05 s, CCTV 7.15 s
        "sf_overview": ((0.0, 20.0, 45.0), (60.0, 560.0, 120.0), 30),
        "s6_door": ((0.1, 12.0, 2.05), (-0.5, 16.0, 1.8), 26),
        "s6_sidewalk": ((-2.9, 14.2, 1.6), (-1.2, 19.6, 1.45), 24),
        "s6_up": ((-3.2, 14.4, 1.4), (-2.941, 15.366, 1.977), 18),
        "s6_cc": ((-3.1, 14.5, 1.45), (-1.732, 18.259, 1.61), 24),
        "bb_right": ((1.5, 19.0, 4.5), (7.4, 28.5, 10.5), 30),        # v4 billboard close checks
        "bb_front_artisand": ((2.63, 19.5, 10.9), (7.6, 27.0, 10.9), 40),
        "bb_front_nonlinear": ((3.31, 23.18, 15.4), (7.8, 31.0, 15.4), 40),
        "bb_left": ((1.5, 20.0, 3.0), (-6.5, 28.0, 6.5), 34),
        "s6_wide": ((-2.5, 14.8, 3.85), (-1.732, 18.259, 2.51), 16),   # v4 'Internet of Birds' wide (8.7 s)
    },
    "hatchery": {
        "front": ((0.2, -3.6, 1.75), (0.2, 0.3, 1.1), 26),
        "station": ((-0.75, -1.35, 1.45), (-1.05, 0.1, 0.98), 35),
        "eyes": ((2.1, -0.95, 1.15), (1.67, 0.0, 1.03), 45),
        "hop": ((1.45, -2.7, 1.35), (2.55, -0.7, 0.65), 30),
    },
}


def markers(kind="perch"):
    """Preview-only spheres at hooks (not saved)."""
    m = E.principled("PREVIEW_marker", (1.0, 0.1, 0.9), 0, 0.4, (1.0, 0.1, 0.9), 6.0)
    for h in [o for o in bpy.data.objects if o.name.startswith("HOOK_perch_")]:
        me = bpy.data.meshes.new("mk")
        mb = E.MB()
        mb.ellipsoid((0, 0, 0.09), (0.09, 0.09, 0.09), 10, 6)
        mb.rod((0, 0, 0.09), (0.25, 0, 0.09), 0.02, 6)
        me = mb.mesh("PREVIEW_marker")
        me.materials.append(m)
        o = bpy.data.objects.new("PREVIEW_marker", me)
        bpy.context.scene.collection.objects.link(o)
        o.matrix_world = h.matrix_world


def render(set_name, views, tag=""):
    scn = bpy.context.scene
    fast = os.environ.get("FAST", "0") == "1"      # S's fast film tier: 1 sample, no ray tracing / volumetrics / shadows
    vfx.setup_eevee(scn, "draft", raytracing=not fast, volumetrics=not fast, view_transform="AgX", look="AgX - Punchy")
    if fast:
        scn.eevee.taa_render_samples = int(os.environ.get("FAST_SAMPLES", "1"))
        scn.eevee.use_shadows = False
        scn.eevee.use_gtao = False
        scn.render.use_compositing = False
    else:
        vfx.set_fast_reflections(False)
        vfx.setup_compositor(scn, bloom_threshold=1.0, bloom_size=7, vignette=0.25, grain=0.03)
    scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = 540, 674, 100
    if os.environ.get("RES") == "1080":
        scn.render.resolution_x, scn.render.resolution_y = 1080, 1350
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    cd = bpy.data.cameras.new("PREVIEW_cam")
    cam = bpy.data.objects.new("PREVIEW_cam", cd)
    scn.collection.objects.link(cam)
    cd.clip_start, cd.clip_end = 0.02, 3000.0
    scn.camera = cam
    scn.frame_set(int(os.environ.get("PREVIEW_FRAME", "15")))
    for v in views:
        loc, tgt, lens = VIEWS[set_name][v]
        if isinstance(lens, str):
            cd.type = "ORTHO"
            cd.ortho_scale = float(lens.split(":")[1])
        else:
            cd.type = "PERSP"
            cd.lens = lens
        cam.location = loc
        d = Vector(tgt) - Vector(loc)
        up = "Y" if abs(d.normalized().z) < 0.99 else "Y"
        cam.rotation_euler = d.to_track_quat("-Z", up).to_euler()
        if v == "plan":
            cam.rotation_euler = (0, 0, 0)
        out = os.path.join(E.OUT_PREVIEWS, "%s_%s%s.png" % (set_name if set_name == "hatchery" else "street_" + set_name, v, tag))
        scn.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print("WROTE", out)


if __name__ == "__main__":
    a = E.argv_after_dashes()
    s = a[0]
    views = a[1:] or list(VIEWS[s])
    if s in ("opening", "ending") and os.environ.get("MARKERS", "1") == "1":
        markers()
    render(s, views, os.environ.get("TAG", ""))
