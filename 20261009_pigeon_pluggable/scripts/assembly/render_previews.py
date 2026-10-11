"""Fast-tier previews of the assembly hatchery (workstream M, DevLog-019). 540 x 674, EEVEE 1 sample, no shadows / AO /
ray tracing (S's FILM_FAST tier), AgX Punchy. Usage:
  scripts/bslot.sh -b assets/components/assembly/assembly_hatchery.blend --python scripts/assembly/render_previews.py -- stills <out_dir> [view ...]
  scripts/bslot.sh -b assets/components/assembly/assembly_hatchery.blend --python scripts/assembly/render_previews.py -- anim <frames_dir> [f_a f_b]
"""
import json
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
import vfx  # noqa: E402
import assembly_api as AA  # noqa: E402

side = json.load(open(AA.SIDECAR))
XS, CZ, BZ = side["stations_x"], side["chuck_z"], side["bench_z"]
HK = {h["name"]: Vector(h["location"]) for h in side["hooks"]}


def setup():
    scn = bpy.context.scene
    vfx.setup_eevee(scn, "draft", raytracing=False, volumetrics=False, view_transform="AgX", look="AgX - Punchy")
    scn.eevee.taa_render_samples = int(os.environ.get("FAST_SAMPLES", "1"))
    for attr, val in (("use_shadows", False), ("use_gtao", False)):
        if hasattr(scn.eevee, attr):
            setattr(scn.eevee, attr, val)
    scn.render.use_compositing = False
    scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = 540, 674, 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    w = bpy.data.worlds.new("ASM_preview_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.004, 0.004, 0.006, 1)
    scn.world = w
    cd = bpy.data.cameras.new("PREVIEW_cam")
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36           # same as the film camera (blender_lib.setup_camera)
    cd.clip_start, cd.clip_end = 0.004, 200.0
    cam = bpy.data.objects.new("PREVIEW_cam", cd)
    scn.collection.objects.link(cam)
    scn.camera = cam
    return scn, cam


def look(cam, loc, tgt, lens):
    cam.location = loc
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens


def plan(shot, k, frame):
    """A view taken from the suggested camera plan (sidecar) key k of `shot`, rendered at `frame`."""
    sh = next(x for x in side["camera_plan"] if x["name"] == shot)
    t, loc, tgt, lens = sh["keys"][k]
    return (frame, Vector(loc), Vector(tgt), lens)


def views():
    s2 = HK["HOOK_S2_work"]
    s3 = HK["HOOK_S3_work"]
    s1 = HK["HOOK_S1_work"]
    b4 = Vector((XS["S4"], -0.05, BZ + 0.18))
    b5 = Vector((XS["S5"], -0.05, BZ + 0.16))
    return {
        "overview": (16, (0.9, -3.6, 1.75), (-0.1, 0.2, 1.0), 26),
        "s1_station": (20, s1 + Vector((0.35, -1.0, 0.45)), s1 + Vector((0, 0, 0.12)), 35),
        "s1_macro": (20, s1 + Vector((0.06, -0.22, 0.14)), s1 + Vector((0.02, 0.0, 0.004)), 85),
        "s2_station": (16, s2 + Vector((0.3, -1.0, 0.4)), s2 + Vector((0, 0, 0.1)), 35),
        "s2_iso_place": plan("A_align_macro", 0, 15),
        "s2_dfb_off_tool_on_lens": plan("A_align_macro", 1, 38),
        "s2_dfb_on": plan("A_align_macro", 1, 43),
        "s2_power_peak": plan("A_align_macro", 2, 55),
        "s2_uv_cure": plan("A_align_macro", 2, 60),
        "s3_station": (40, s3 + Vector((0.3, -1.0, 0.4)), s3 + Vector((0, 0, 0.1)), 35),
        "s3_bonding": plan("B_bond_macro", 0, 96),
        "s4_dispense": plan("C_press_to_test", 0, 136),
        "s4_lid_closing": plan("C_press_to_test", 2, 160),
        "s4_lid_closed": plan("C_press_to_test", 3, 166),
        "s5_hop_in": plan("D_test_stand", 0, 174),
        "s5_power_on": plan("D_test_stand", 1, 186),
        "s5_cmis_ddm": plan("D_test_stand", 3, 201),
        "s5_heads_docked": plan("D_test_stand", 4, 208),
        "s5_tx_lane3": plan("D_test_stand", 5, 221),
        "s5_tx_lane8": plan("D_test_stand", 6, 247),
        "s5_rx_link": plan("D_test_stand", 7, 264),
        "s5_pass": plan("D_test_stand", 8, 272),
        "s5_retract": plan("D_test_stand", 9, 279),
        "s5_hop_off": plan("D_test_stand", 10, 286),
        "s5_board_overview": (190, Vector((XS["S5"] + 0.12, -0.42, BZ + 0.36)), Vector((XS["S5"], -0.04, BZ + 0.02)), 40),
        "s5_stand_wide": (200, Vector((XS["S5"] - 0.05, -1.45, 1.45)), Vector((XS["S5"], 0.1, 0.98)), 30),
        "s5_socket_macro": (190, Vector((XS["S5"] + 0.02, -0.25, BZ + 0.1)), Vector((XS["S5"] - 0.01, -0.06, BZ + 0.025)), 50),
    }


def stills(out_dir, names):
    scn, cam = setup()
    os.makedirs(out_dir, exist_ok=True)
    V = views()
    for n in (names or list(V)):
        fr, loc, tgt, lens = V[n]
        scn.frame_set(fr)
        look(cam, loc, tgt, lens)
        scn.render.filepath = os.path.join(out_dir, "assembly_%s_f%03d.png" % (n, fr))
        bpy.ops.render.render(write_still=True)
        print("ASM_PREVIEW", n, fr)


def anim(frames_dir, fa=None, fb=None):
    scn, cam = setup()
    root = bpy.data.objects["assembly_root"]
    AA.apply_camera(cam, AA.camera_keys(root, f0=side["frame_start"]))
    os.makedirs(frames_dir, exist_ok=True)
    f0 = side["frame_start"]
    scn.frame_start = fa if fa is not None else f0
    scn.frame_end = fb if fb is not None else f0 + side["duration_frames"] - 1
    scn.render.filepath = os.path.join(frames_dir, "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]
    if a[0] == "stills":
        stills(a[1], a[2:])
    else:
        anim(a[1], *(int(v) for v in a[2:4]))
