"""Test renders for the owner interim items (workstream E, DevLog-013 section 8): F4 HUD, F6 constellation + beam
hand-off, F7 data emitters / data beams. Fast tier (1 sample, no ray tracing, no shadows), 540 x 674.

Usage:
  scripts/bslot.sh -b --factory-startup --python scripts/vfx/build_fx_tests.py -- hud <frame> [...]
  scripts/bslot.sh -b assets/components/environment/street_ending.blend --python scripts/vfx/build_fx_tests.py -- sky <frame> [...]
  scripts/bslot.sh -b --factory-startup --python scripts/vfx/build_fx_tests.py -- data <frame> [...]
Writes outputs/assets/environment/fx_<mode>_f<frame>.png
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "env"))  # env_lib, constellation, render_previews
import env_lib as E  # noqa: E402
import vfx  # noqa: E402

PI = math.pi


def fast(scn):
    vfx.setup_eevee(scn, "draft", raytracing=False, volumetrics=False, view_transform="AgX", look="AgX - Punchy")
    scn.eevee.taa_render_samples = 1
    scn.eevee.use_shadows = False
    scn.eevee.use_gtao = False
    scn.render.use_compositing = False
    scn.render.resolution_x, scn.render.resolution_y, scn.render.resolution_percentage = 540, 674, 100
    scn.render.fps = 30
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"


def camera(scn, loc, tgt, lens, name="TEST_cam"):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.lens = lens
    cd.clip_start, cd.clip_end = 0.01, 800
    cam = bpy.data.objects.new(name, cd)
    scn.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam
    return cam


def gary(c, name="GARY_standin"):
    mb = E.MB()
    mb.cyl((0, 0, 0.45), 0.08, 0.9, seg=12)
    mb.ellipsoid((0, 0, 1.2), (0.2, 0.13, 0.35), 16, 10)
    mb.ellipsoid((0, 0, 1.68), (0.11, 0.11, 0.13), 14, 8)
    return E.obj(name, mb, [E.principled("GARY_mat", (0.16, 0.14, 0.2), 0, 0.6)], c, smooth=True)


def build_hud():
    E.reset()
    scn = bpy.context.scene
    c = E.coll("HUD_TEST")
    mb = E.MB()
    mb.box((0, 5, -0.05), (6, 16, 0.1))
    E.obj("floor", mb, ["tile_dark"], c)
    mb = E.MB()
    for k in range(7):
        for sx in (-1, 1):
            mb.box((sx * 1.6, 1.0 + 1.6 * k, 1.1), (1.0, 0.7, 2.2))
            for r in range(6):
                mb.box((sx * 1.09, 1.0 + 1.6 * k, 0.3 + 0.32 * r), (0.02, 0.6, 0.02), m=1)
    E.obj("racks", mb, ["metal_dark", "led_cyan"], c)
    g = gary(c)
    g.location = (0.1, 7.5, 0)
    g.keyframe_insert("location", frame=1)
    g.location = (-0.2, 4.6, 0)
    g.keyframe_insert("location", frame=120)
    birds = []
    for i, (x, y, z) in enumerate(((-1.1, 2.0, 1.5), (1.1, 3.4, 1.2), (-1.1, 5.2, 0.9), (1.1, 6.6, 1.8), (-1.1, 8.2, 1.4),
                                   (1.1, 9.4, 0.6), (-0.4, 0.4, 2.3), (1.1, 1.6, 2.0))):
        mb = E.MB()
        mb.ellipsoid((0, 0, 0), (0.15, 0.08, 0.08), 10, 6)
        birds.append(E.obj("bird_%d" % i, mb, ["metal_paint"], c, loc=(x, y, z)))
    E.light("key", "AREA", (0, 5, 3.8), 300, (0.8, 0.85, 1.0), c, size=4, rot=(0, 0, 0))
    E.light("rim", "SPOT", (0, 10, 2.5), 400, (1.0, 0.4, 0.8), c, spot_deg=60, target=(0, 5, 1.2))
    vfx.night_sky_world(scn, strength=0.0)
    cam = camera(scn, (-0.85, 0.2, 1.55), (0.0, 6.0, 1.2), 30)
    cam.rotation_mode = "XYZ"
    e0 = cam.rotation_euler.copy()
    cam.keyframe_insert("rotation_euler", frame=1)
    cam.rotation_euler = (e0.x, e0.y, e0.z + 0.12)
    cam.keyframe_insert("rotation_euler", frame=120)
    fast(scn)
    vfx.hud_overlay(scn, cam, g, 1, 120, style="intel", seed=4, blips=birds, step=2)
    return scn


def build_sky():
    scn = bpy.context.scene
    import constellation as CN
    c = E.coll("FX_SKY")
    sats = CN.constellation_train(c, n=10, spacing=4.5, scale=1.0)
    CN.constellation_pass(sats[0].parent, CN.S6_PASS[0], CN.S6_PASS[1], 1, 120)
    for o in bpy.data.objects:            # drop the single v2 satellite from this test
        if o.name.startswith("satellite"):
            o.hide_render = True
    eye = Vector((-6.04, 19.6, 9.1 + 0.22))
    eye2 = Vector((-5.85, 23.0, 6.06 + 0.22))
    eye3 = Vector((4.9, 23.0, 5.6))
    trk = vfx.make_beam("fx_track", eye, eye + Vector((0, 0, 1)), radius=0.06, color=vfx.TX_RED, collection=c)
    vfx.beam_track(trk, eye, sats[1:5], 6, 116, step=1, jump=3)
    sw = vfx.make_beam("fx_sweep", eye2, eye2 + Vector((0, 1, 0)), radius=0.05, color=vfx.RX_CYAN, collection=c)
    vfx.beam_sweep(sw, eye2, 1, 120, yaw0=0.2, yaw1=1.4, pitch0=0.25, pitch1=0.6, length=16.0, wobble=0.04)
    hp = vfx.make_beam("fx_hops", eye3, eye3 + Vector((0, 0, 1)), radius=0.03, color=vfx.TX_RED, collection=c)
    nodes = [eye3, Vector((-3.7, 16.2, 4.82)), Vector((-3.35, 20.1, 1.25)), Vector((5.72, 19.0, 3.5)), Vector((-5.4, 17.5, 3.92))]
    vfx.beam_hops(hp, nodes, 4, hop=6, hold=10)
    if os.environ.get("SKY_CAM") == "side":      # hand-off strip: source bird and the train both in frame
        camera(scn, (3.0, 11.5, 12.5), (-3.5, 27.0, 19.0), 20)
    else:                                         # near S's S6 roof camera
        camera(scn, (-3.0, 15.5, 6.5), (-1.0, 26.0, 16.0), 22)
    fast(scn)
    return scn


def build_data():
    E.reset()
    scn = bpy.context.scene
    c = E.coll("FX_DATA")
    pitch = 0.23
    em = vfx.data_emitter_material("FX_emit_tx", vfx.TX_RED, strength=6.0, rate=20.0, pitch=pitch, offset=(pitch * 2, pitch, 0))
    mb = E.MB()
    mb.box((0, 0, -0.06), (1.1, 0.6, 0.05))
    E.obj("plate", mb, ["metal_dark"], c)
    mb = E.MB()
    for r in range(2):
        for k in range(4):
            mb.ellipsoid((-0.345 + pitch * k, -0.115 + pitch * r, 0.0), (0.09, 0.09, 0.05), 16, 8)
    lens = E.obj("lenslets", mb, [em], c, smooth=True)
    for r in range(2):
        for k in range(4):
            p0 = Vector((-0.345 + pitch * k, -0.115 + pitch * r, 0.05))
            b = vfx.make_data_beam("fx_db_%d%d" % (r, k), p0, p0 + Vector((0.15 * (k - 1.5), 0.1 * (r - 0.5), 6.0)), radius=0.02,
                                   color=vfx.TX_RED, collection=c, symbol_length=0.12, speed=4.0)
    E.light("fill", "AREA", (0, -2, 2), 40, (0.7, 0.75, 0.9), c, size=3, target=(0, 0, 0))
    vfx.night_sky_world(scn, strength=0.0)
    camera(scn, (1.6, -1.6, 0.9), (0.0, 0.3, 1.2), 24)
    fast(scn)
    return scn


def _hooks():
    import json
    d = json.load(open(os.path.join(E.OUT_ASSETS, "street_ending.json")))
    return {h["name"]: Vector(h["location"]) for h in d["hooks"]}


def _s6cam(scn, key):
    """S's S6 camera keys (scripts/build_film.py, film v2.1)."""
    import render_previews as RP
    loc, tgt, lens = RP.VIEWS["ending"][key]
    return camera(scn, loc, tgt, lens, "S6_" + key)


def build_v3sky():
    """v3: train at 0.5x scale on the 2x-speed pass (S's S6 3.6-8.2 s window = 139 frames) + MIMO uplinks, from S's up-shot."""
    scn = bpy.context.scene
    import constellation as CN
    c = E.coll("FX_V3SKY")
    for o in bpy.data.objects:
        if o.name.startswith("satellite"):
            o.hide_render = True
    sats = CN.constellation_train(c, n=10)
    CN.constellation_pass(sats[0].parent, CN.S6_PASS_V3[0], CN.S6_PASS_V3[1], 1, 139)
    H = _hooks()
    eye = Vector((0, 0, 0.22))
    srcs = [H[n] + eye for n in ("HOOK_perch_x_R3_parapet", "HOOK_perch_roof_L", "HOOK_perch_roof_R", "HOOK_perch_ledge_a")]
    cam = _s6cam(scn, "s6_up")
    up = vfx.data_beam_material("FX_uplink", vfx.TX_RED, strength=10.0, symbol_length=0.5, speed=12.0)
    res = vfx.mimo_links(srcs, sats, 30, 130, k=2, step=2, min_elev_deg=15.0, cam=cam, seed=3, radius=0.05, material=up,
                         collection=c, isl=True, isl_radius=0.025)
    print("MIMO sessions", len(res["sessions"]), [(s[0], s[1], s[2], s[3]) for s in res["sessions"]])
    fast(scn)
    return scn


def build_choreo():
    scn = bpy.context.scene
    c = E.coll("FX_CHOREO")
    H = _hooks()
    eye = Vector((0, 0, 0.22))
    names = ["lamp1_top", "lamp1_arm", "lamp2_arm", "wire_18p2", "wire_20p7", "rail_a", "rail_b", "ac_1", "ac_2", "ledge_a",
             "ledge_b", "sign", "bin", "cctv_pole", "comp0", "comp1", "x_lamp_1_top", "x_wire_E_21p8", "x_ac_3"]
    birds = [H["HOOK_perch_" + n] + eye for n in names if "HOOK_perch_" + n in H]
    m = vfx.data_beam_material("FX_street", vfx.TX_RED, strength=9.0, symbol_length=0.05, speed=2.0)
    res = vfx.street_beam_choreo(birds, 1, 120, n_sweep=4, n_p2p=3, seed=5, mat_sweep=None, mat_p2p=m, collection=c,
                                 radius_sweep=0.02, radius_p2p=0.015)
    print("CHOREO sweeps", len(res["sweeps"]), "p2p sessions", [(a, b, fs, fe) for a, b, fs, fe, _ in res["p2p"]])
    _s6cam(scn, os.environ.get("S6CAM", "s6_sidewalk"))
    fast(scn)
    return scn


if __name__ == "__main__":
    a = E.argv_after_dashes()
    mode, frames = a[0], [int(x) for x in a[1:]] or [30]
    scn = {"hud": build_hud, "sky": build_sky, "data": build_data, "v3sky": build_v3sky, "choreo": build_choreo}[mode]()
    for f in frames:
        scn.frame_set(f)
        tag = "_" + os.environ["SKY_CAM"] if mode == "sky" and os.environ.get("SKY_CAM") else ""
        tag = "_" + os.environ["S6CAM"] if mode == "choreo" and os.environ.get("S6CAM") else tag
        scn.render.filepath = os.path.join(E.OUT_PREVIEWS, "fx_%s%s_f%03d.png" % (mode, tag, f))
        bpy.ops.render.render(write_still=True)
        print("WROTE", scn.render.filepath)
