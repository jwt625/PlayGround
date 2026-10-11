"""Build optics_rx_fx.blend (optics_rx.blend + fixture cleanup + FX layer + 150-frame test keys) and render test frames.

  scripts/bslot.sh -b assets/components/optics_rx/optics_rx.blend --python scripts/assets/optics_rx/make_rx_fx_test.py \
      -- outputs/assets/optics_rx/fx_test
Writes assets/components/optics_rx/optics_rx_fx.blend and optics_rx_fx.json (API, colours, distances, timing, tracks);
raw frames <out>/fxA_####.png (light arrival) and <out>/fxB_####.png (signal glow close-up), 540x674, EEVEE Next
(the film's engine). optics_rx.blend / .glb / _parts.json are not modified.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rx_fx  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1]
os.makedirs(OUT, exist_ok=True)
SCN = bpy.context.scene
bpy.context.preferences.filepaths.save_version = 0
SCN.render.fps = 30
SCN.frame_start, SCN.frame_end = 1, 150

root = bpy.data.objects["RX_OPTICS_ROOT"]
touched = rx_fx.rx_clean_fixtures(root, si_plates=True, lenslets=True)
fx = rx_fx.build_rx_fx(root)
LA = rx_fx.rx_light_arrive(fx, f_on=10, f_front_end=60)
LAND = rx_fx.rx_land_frames(fx, 10, 60)
SG = rx_fx.rx_signal_glow(fx, f0=LAND[0], f1=150, rate=40.0, seed=7, arrive_frames=14, intensity=1.0)
otrack = rx_fx.rx_front_track(fx, 10, 60, f_sig0=50, f_sig_end=64, world=False)
bpy.data.collections["viz_beams"].hide_render = True
bpy.data.collections["viz_beams"].hide_viewport = True
print("FX: optical %.3f mm (lens at %.3f), electrical %.3f mm; touched %s" % (fx["d_opt"], fx["d_lens"], fx["d_el"],
                                                                             [o.name for o in touched]))
print("TIMING", LA, SG)
blend = os.path.join(rx_fx.ADIR, "optics_rx_fx.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
info = dict(
    asset="optics_rx_fx", date="2026-10-10", base="optics_rx.blend (unchanged) + rx_fx layer",
    api=["rx_fx.rx_clean_fixtures(parent, si_plates=True, lenslets=False, facets=True) -> touched objects",
         "rx_fx.build_rx_fx(parent, unit=0.001) -> fx",
         "rx_fx.rx_light_arrive(fx, f_on, f_front_end, f_off=None, rate=30.0, seed=7)",
         "rx_fx.rx_land_frames(fx, f_on, f_front_end) -> (first, last) landing frame",
         "rx_fx.rx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6, intensity=1.0)",
         "rx_fx.rx_front_track(fx, f_on, f_front_end, f_sig0=None, f_sig_end=None, world=True) -> [(frame, xyz)]"],
    colours=dict(optical_rx_cyan=rx_fx.OPT_COL, electrical_amber=rx_fx.ELEC_COL),
    objects={k: o.name for k, o in fx["objs"].items()}, fixtures_ghosted=[o.name for o in touched],
    distances_mm=dict(optical_total=round(fx["d_opt"], 4), start_to_lens=round(fx["d_lens"], 4),
                      lane_land=fx["d_land"], electrical_pd_to_flex=round(fx["d_el"], 4)),
    test_timing=dict(fps=30, light=LA, land_frames=[round(x, 2) for x in LAND], signal=SG),
    track_mm=[(round(f, 2), [round(c, 4) for c in p]) for f, p in otrack])
json.dump(info, open(os.path.join(rx_fx.ADIR, "optics_rx_fx.json"), "w"), indent=1)

# ---------------- test render (not saved): x1000 like the film macro set, EEVEE Next
root.scale = (1000.0, 1000.0, 1000.0)
for m in bpy.data.materials:
    if m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "BUMP":
                n.inputs["Distance"].default_value *= 1000.0
SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 24
SCN.render.resolution_x, SCN.render.resolution_y = 540, 674
SCN.render.resolution_percentage = 100
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Punchy"
world = bpy.data.worlds.new("dark")
SCN.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
SCN.use_nodes = True
nt = SCN.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type, gl.quality, gl.threshold, gl.size, gl.mix = "FOG_GLOW", "MEDIUM", 1.0, 6, -0.4
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])
for nm, rot, e in (("key", (40, 0, 135), 2.5), ("fill", (65, 0, -40), 0.9)):
    ld = bpy.data.lights.new(nm, "SUN")
    ld.energy, ld.angle = e, 0.4
    ob = bpy.data.objects.new(nm, ld)
    ob.rotation_euler = [math.radians(a) for a in rot]
    SCN.collection.objects.link(ob)
cd = bpy.data.cameras.new("cam")
cd.clip_start, cd.clip_end = 0.01, 500
cam = bpy.data.objects.new("cam", cd)
SCN.collection.objects.link(cam)
SCN.camera = cam


def aim(loc, tgt, lens):
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    cd.lens = lens


SHOTS = {"A": (((1.6, 6.2, 4.6), (-2.3, 0.0, -0.6), 30), [9, 24, 40, 56, 64]),
         "B": (((-6.6, 1.9, 0.9), (-4.4, 0.1, -1.3), 32), [55, 58, 61, 64, 68, 72, 90, 120]),
         "C": (((-4.6, 1.6, 0.4), (-4.7, -0.6, -1.25), 30), [57, 60, 63, 66, 69, 72, 100, 130])}
for tag, ((loc, tgt, lens), frames) in SHOTS.items():
    aim(loc, tgt, lens)
    for f in frames:
        SCN.frame_set(f)
        SCN.render.filepath = os.path.join(OUT, "fx%s_%04d.png" % (tag, f))
        bpy.ops.render.render(write_still=True)
        print("RENDERED", tag, f)
