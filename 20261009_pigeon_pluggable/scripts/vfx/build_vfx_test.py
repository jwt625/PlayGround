"""VFX test sheet (workstream E, DevLog-013): one small stage with every VFX library effect, rendered with the compositor.

Usage: scripts/bslot.sh -b --python scripts/vfx/build_vfx_test.py -- [out.png] [frame] [--blend out.blend]
"""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
import env_lib as E  # noqa: E402
import vfx  # noqa: E402

PI = math.pi


def label(txt, x, c):
    t = E.text("lbl_" + txt, txt, 0.17, c, (x, -0.9, 0.06), (PI / 2 - 0.5, 0, 0), mat=E.principled("lbl", (1, 1, 1), 0, 0.5, (1, 1, 1), 1.2),
               fnt="Arial Bold")
    return t


def build():
    E.reset()
    scn = bpy.context.scene
    c = E.coll("VFX_TEST")
    # stage: wet floor, back wall
    mb = E.MB()
    mb.box((0, -1.0, -0.05), (20, 14, 0.1))
    E.obj("floor", mb, ["asphalt"], c)
    mb = E.MB()
    mb.box((0, 3.2, 2.5), (16, 0.2, 5))
    E.obj("wall", mb, ["concrete"], c)
    # 1 hologram mini tree
    holo = vfx.hologram_material("VFX_test_holo", strength=2.0)
    holo_txt = vfx.hologram_material("VFX_test_holo_text", base=1.0, strength=1.5, scan_mix=0.3)
    mb = E.MB()
    mb.cyl((0, 0, 1.2), 0.02, 2.2, seg=8)
    for k in range(6):
        z = 0.5 + 0.33 * k
        x = 0.35 if k % 2 else -0.35
        mb.rod((0, 0, z - 0.1), (x, 0, z), 0.008, 6)
        mb.ellipsoid((x, 0, z), (0.09, 0.07, 0.06), 12, 8)
    mb.ellipsoid((0, 0, 2.45), (0.16, 0.12, 0.1), 16, 10)
    o = E.obj("holo_tree", mb, [holo], c, loc=(-5.5, 0.5, 0), shadow=False)
    E.text("holo_txt", "CYBER PIGEON", 0.14, c, (-5.5, 0.5, 2.72), mat=holo_txt, fnt="Arial Bold")
    vfx.light_cone("holo_emitter_cone", (-5.5, 0.5, 0.02), (0, 0, 1), 0.9, 40, (0.2, 0.9, 1.0), 0.35, c)
    mb = E.MB()
    mb.cyl((0, 0, 0.02), 0.35, 0.04, seg=32)
    E.obj("holo_puck", mb, ["metal_dark"], c, loc=(-5.5, 0.5, 0))
    mb = E.MB()
    mb.ring((0, 0, 0.045), 0.3, 0.26, 0.01, seg=32)
    E.obj("holo_puck_ring", mb, ["led_cyan"], c, loc=(-5.5, 0.5, 0))
    label("HOLOGRAM", -5.5, c)
    # 2 beams between two posts
    mb = E.MB()
    for x in (-3.4, -1.6):
        mb.cyl((x, 0.5, 1.1), 0.05, 2.2, seg=12)
    E.obj("beam_posts", mb, ["metal_paint"], c)
    for z, col, a in ((0.9, vfx.TX_RED, 1.0), (1.3, vfx.TX_RED, 0.35), (1.7, vfx.RX_CYAN, 1.0)):
        b = vfx.make_beam("beam_%.1f" % z, (-3.35, 0.5, z), (-1.65, 0.5, z), radius=0.006, color=col, collection=c)
        b.color = (1, 1, 1, a)
    label("BEAM / LINK", -2.5, c)
    # 3 neon signs
    mb = E.MB()
    mb.box((0, 3.05, 2.1), (2.6, 0.06, 1.4))
    E.obj("neon_backer", mb, ["sign_back"], c)
    E.text("neon_ramen", "RAMEN 24H", 0.42, c, (0, 3.0, 2.45), mat="neon_orange", tube=True, bevel=0.012)
    E.text("neon_link", "LINK UP", 0.42, c, (0, 3.0, 1.8), mat="neon_mag_fl", tube=True, bevel=0.012)
    E.light("neon_spill", "AREA", (0, 2.4, 2.1), 120, (1.0, 0.3, 0.5), c, size=2.0, rot=(-PI / 2, 0, 0))
    label("NEON", 0.0, c)
    # 4 eye glow lenslet grids (TX on, RX on, RX off)
    tx = vfx.eye_glow_material("VFX_test_eye_tx", vfx.TX_RED, strength=6)
    rx = vfx.eye_glow_material("VFX_test_eye_rx", vfx.RX_CYAN, strength=6, pulse=0.5)
    for k, (m, pw, z) in enumerate(((tx, 1.0, 1.9), (rx, 1.0, 1.25), (rx, 0.0, 0.6))):
        mb = E.MB()
        mb.box((0, 0.03, 0), (0.62, 0.04, 0.34))
        E.obj("eye_plate_%d" % k, mb, ["metal_dark"], c, loc=(2.5, 0.5, z))
        mb = E.MB()
        for r in range(2):
            for cc in range(4):
                mb.ellipsoid((-0.21 + 0.14 * cc, 0, -0.07 + 0.14 * r), (0.055, 0.03, 0.055), 14, 8)
        o = E.obj("eye_lens_%d" % k, mb, [m], c, loc=(2.5, 0.5, z), smooth=True)
        o.color = (1, 1, 1, pw)
    label("EYE GLOW", 2.5, c)
    # 5 haze cards + light cone under a lamp
    mb = E.MB()
    mb.cyl((5.0, 0.8, 1.6), 0.05, 3.2, seg=12)
    mb.box((5.0, 0.35, 3.15), (0.12, 0.9, 0.08))
    E.obj("lamp", mb, ["metal_paint"], c)
    mb = E.MB()
    mb.box((5.0, 0.0, 3.1), (0.25, 0.4, 0.03))
    E.obj("lamp_lens", mb, ["lamp_lens"], c)
    E.light("lamp_spot", "SPOT", (5.0, 0.0, 3.05), 900, (1.0, 0.78, 0.55), c, spot_deg=80, blend=0.6, rot=(0, 0, 0), radius=0.1)
    vfx.light_cone("lamp_cone", (5.0, 0.0, 3.08), (0, 0, -1), 3.1, 70, (1.0, 0.78, 0.55), 0.22, c)
    vfx.haze_cards("haze", (5.0, 0.3, 1.5), (3.0, 2.0, 3.0), n=5, density=0.08, normal_axis="Y", collection=c)
    label("HAZE + CONE", 5.0, c)
    label("WET GROUND + PUDDLES", -1.0, c).location = (-1.0, -2.4, 0.06)
    # lights, sky, haze
    E.light("moon", "SUN", (0, 0, 20), 0.15, (0.6, 0.7, 1.0), c, rot=(0.9, 0.2, 0.4))
    E.light("fill", "AREA", (0, -6, 5), 80, (0.6, 0.65, 0.8), c, size=8, target=(0, 0.5, 1))
    vfx.night_sky_world(scn, stars=1.0)
    vfx.world_haze(scn, density=0.006)
    cam_d = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_d)
    c.objects.link(cam)
    cam.location = (0, -7.6, 1.9)
    cam.rotation_euler = (math.radians(84), 0, 0)
    cam_d.lens = 20
    scn.camera = cam
    vfx.setup_eevee(scn, "draft", view_transform=os.environ.get("VT", "AgX"), look=os.environ.get("LOOK", "AgX - Punchy"))
    vfx.setup_compositor(scn, bloom_threshold=1.0, vignette=0.3, grain=0.04)
    scn.render.resolution_x, scn.render.resolution_y = 960, 600
    scn.render.image_settings.file_format = "PNG"
    return scn


if __name__ == "__main__":
    args = E.argv_after_dashes()
    out = args[0] if args and not args[0].startswith("--") else os.path.join(E.OUT_PREVIEWS, "vfx_test_sheet.png")
    frame = int(args[1]) if len(args) > 1 and not args[1].startswith("--") else 20
    scn = build()
    if "--blend" in args:
        E.save_blend(args[args.index("--blend") + 1])
    scn.frame_set(frame)
    scn.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("WROTE", out)
