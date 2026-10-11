"""Build optics_tx_fx.blend (optics_tx.blend + FX layer) with a 150-frame test animation and render test frames.

Run from the project folder:
  scripts/bslot.sh -b assets/components/optics_tx/optics_tx.blend --python scripts/assets/optics_tx/make_tx_fx_test.py \
      -- assets/components/optics_tx outputs/assets/optics_tx/fx_test
Writes <asset_dir>/optics_tx_fx.blend, <asset_dir>/optics_tx_fx.json (API, distances, timings, front track) and
raw frames <out>/fxA_####.png (light-up, camera follows the front) and <out>/fxB_####.png (RF signal glow close-up),
540x674. compose_tx_fx_strip.py tiles them into strips.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tx_fx  # noqa: E402
import tx_geom as g  # noqa: E402

ARGS = sys.argv[sys.argv.index("--") + 1:]
ASSET_DIR, OUT = os.path.abspath(ARGS[0]), ARGS[1]
os.makedirs(OUT, exist_ok=True)
MM = 0.001
SCN = bpy.context.scene
bpy.context.preferences.filepaths.save_version = 0
SCN.render.fps = 30
SCN.frame_start, SCN.frame_end = 1, 150

root = bpy.data.objects["OPTICS_TX_ROOT"]
fx = tx_fx.build_tx_fx(root)
LU = tx_fx.tx_light_up(fx, f_on=10, f_front_end=80)
SG = tx_fx.tx_signal_glow(fx, f0=40, f1=150, rate=40.0, seed=7)
track = tx_fx.tx_front_track(fx, 10, 80, world=False)
print("FX distances: optical %.2f mm (laser->SSC %.2f, MZM out %.2f, output SSC %.2f, eye %.2f), electrical %.2f mm"
      % (fx["d_opt"], fx["d_laser_ssc"], fx["d_mzm_out"], fx["d_ssc_out"], fx["d_eye"], fx["d_el"]))
print("TIMING", LU, SG)
bpy.data.collections["tx_beams_viz"].hide_render = True

# ---------------- render setup (fast tier)
SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 24
SCN.render.resolution_x, SCN.render.resolution_y = 540, 674
SCN.render.resolution_percentage = 100
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Punchy"
SCN.render.image_settings.file_format = "PNG"
world = bpy.data.worlds.new("dark")
SCN.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.024, 0.032, 1)
SCN.use_nodes = True
nt = SCN.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type = "FOG_GLOW"
gl.quality = "MEDIUM"
gl.threshold = 1.0
gl.size = 6
gl.mix = -0.4
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])


def area(name, loc, power, size, target):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = power, size
    ob = bpy.data.objects.new(name, ld)
    SCN.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).normalized().to_track_quat("-Z", "Y").to_euler()


C = Vector((g.X_W, -4.5, 0)) * MM
area("key", C + Vector((0.03, 0.01, 0.02)), 0.02, 0.02, C)
area("fill", C + Vector((0.02, -0.02, -0.02)), 0.008, 0.02, C)

cd = bpy.data.cameras.new("fxcam")
cam = bpy.data.objects.new("fxcam", cd)
SCN.collection.objects.link(cam)
SCN.camera = cam
cd.lens = 50
cd.clip_start = 0.0001
cd.clip_end = 1.0


def aim(loc, target, up):
    d = (Vector(target) - Vector(loc)).normalized()
    q = d.to_track_quat("-Z", "Y")
    cam.rotation_mode = "QUATERNION"
    cam.location = loc
    cam.rotation_quaternion = q
    bpy.context.view_layer.update()
    cu = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))
    u = Vector(up) - d * Vector(up).dot(d)
    u.normalize()
    cam.rotation_quaternion = Quaternion(d, math.atan2(d.dot(cu.cross(u)), cu.dot(u))) @ q


def render(name, frame):
    SCN.frame_set(frame)
    SCN.render.filepath = os.path.join(OUT, "%s_%04d.png" % (name, frame))
    bpy.ops.render.render(write_still=True)


# ---------------- camera A: looks down on the PIC face (+X), optical edge up; follows the optical front (keyed)
offs = Vector((14.0, 2.5, 0.0))
for fr, p in track:
    t = Vector((g.X_W, max(min(p[1], -0.5), -7.6), 0.0))
    aim((t + offs) * MM, t * MM, (0, 1, 0))
    cam.keyframe_insert("location", frame=fr)
    cam.keyframe_insert("rotation_quaternion", frame=fr)
for fc in cam.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
SCN.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ASSET_DIR, "optics_tx_fx_tmp.blend"))
FRAMES_A = [9, 14, 20, 28, 38, 48, 60, 72, 82, 110]
for f in FRAMES_A:
    render("fxA", f)

# ---------------- camera B: RF entry close-up (carrier traces, GSSG bonds, TW electrodes of lanes 4-6)
if cam.animation_data:
    cam.animation_data_clear()
tB = Vector((g.X_W - 0.05, g.Y_BACK + 0.9, 1.0))
aim((tB + Vector((4.0, -2.6, 2.0))) * MM, tB * MM, (1, 0, 0))
cam.keyframe_insert("location", frame=1)
FRAMES_B = [41, 44, 47, 52, 70, 100]
for f in FRAMES_B:
    render("fxB", f)

# ---------------- save the FX blend (without the test cameras/lights) + JSON
for o in [o for o in bpy.data.objects if o.type in ("CAMERA", "LIGHT")]:
    bpy.data.objects.remove(o, do_unlink=True)
SCN.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ASSET_DIR, "optics_tx_fx.blend"))
os.remove(os.path.join(ASSET_DIR, "optics_tx_fx_tmp.blend"))
doc = dict(
    date="2026-10-10", api="scripts/assets/optics_tx/tx_fx.py",
    functions={
        "build_tx_fx(parent, unit=0.001)": "creates fx_tx_optical_pic, fx_tx_optical_free_space, fx_tx_electrical_rf, fx_tx_dfb_on under parent (collection tx_fx)",
        "tx_light_up(fx, f_on, f_front_end, f_mod_on=None, dfb_ramp=4, f_off=None)": "DFB on, optical front 0 -> d_opt mm linearly f_on..f_front_end, PAM4 blink on MZM outputs from f_mod_on",
        "tx_signal_glow(fx, f0, f1, rate=40, seed=7, sym_per_mm=3, arrive_frames=12, fade=6)": "amber RF pulses flex -> carrier -> GSSG bonds -> TW electrodes -> terminations",
        "tx_front_track(fx, f_on, f_front_end, world=True)": "[(frame, position)] of the optical front for camera aiming",
    },
    colours=dict(optical=tx_fx.OPT_COL, electrical=tx_fx.ELEC_COL),
    distances_mm=dict(optical_total=fx["d_opt"], laser_facet_to_pic_ssc=fx["d_laser_ssc"], to_mzm_outputs=fx["d_mzm_out"],
                      to_output_ssc=fx["d_ssc_out"], to_eye_plate=fx["d_eye"], electrical_flex_to_termination=fx["d_el"]),
    test_timing=dict(fps=30, light_up=LU, signal_glow=SG),
    front_track_mm=[(round(d, 3), [round(c, 3) for c in p]) for d, p in fx["track_mm"]],
    order="optical front: DFB facets (Y~0, Z +/-2.5) -> Si lenses / isolator -> PIC front edge (Y -2.25) -> side feeds to the back "
          "(Y ~ -9.3) -> MMI trees -> 8 MZM arms travelling toward +Y -> output fan-in -> 1x8 SSCs (Y -2.25) -> staircase / "
          "rhomboids -> 2x4 eye (X 0) -> out",
)
with open(os.path.join(ASSET_DIR, "optics_tx_fx.json"), "w") as fh:
    json.dump(doc, fh, indent=1)
print("TX_FX_TEST_DONE frames A=%d B=%d" % (len(FRAMES_A), len(FRAMES_B)))
