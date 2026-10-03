"""els_laser_source: OIF ELSFP external-laser-source pluggable (dimensions from OIF-ELSFP-01.0 Fig. 7/9) in closed, open (top removed,
8 laser packages + PM fibres to two MT-12 ferrules) and front-fibre-pigtail variants, plus a 14-pin butterfly laser with fibre pigtail and FC/APC.
Layout: ELSFP closed at origin (rear/blind-mate end at +Y, pull tab toward -Y); open at x=+40 mm; pigtail variant at x=+80 mm; butterfly at x=-45 mm.
Run: Blender -b --python build_els_laser_source.py -- <out_dir>
"""
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
ASSET = "els_laser_source"
MM, UM, PI = P.MM, P.UM, math.pi

W, L, H = 21.18 * MM, 55.2 * MM, 9.3 * MM
WALL = 1.0 * MM
TAB_L = 44.8 * MM  # 100.0 total minus 55.2 body (Fig. 7)


def elsfp(coll, root, name, cx, open_top=False, pigtail=False):
    y0, y1 = -L / 2, L / 2  # front (pull tab side) .. rear (blind-mate side)
    body = P.mat("oe_body")
    g = P.Geo()
    g.box(cx - W / 2, y0, 0, cx + W / 2, y1, WALL)  # bottom
    g.box(cx - W / 2, y0, WALL, cx - W / 2 + WALL, y1, H - (0 if open_top else WALL))
    g.box(cx + W / 2 - WALL, y0, WALL, cx + W / 2, y1, H - (0 if open_top else WALL))
    g.box(cx - W / 2 + WALL, y0, WALL, cx + W / 2 - WALL, y0 + WALL, H - WALL)  # front wall
    # rear wall with window for the optical connector shroud / paddle card
    g.box(cx - W / 2 + WALL, y1 - WALL, WALL, cx - 10.0 * MM, y1, H - WALL)
    g.box(cx + 10.0 * MM, y1 - WALL, WALL, cx + W / 2 - WALL, y1, H - WALL)
    g.box(cx - 10.0 * MM, y1 - WALL, H - WALL - 2.0 * MM, cx + 10.0 * MM, y1, H - WALL)
    g.build(name + "_shell", body, coll, root, bevel=(0.4 * MM, 2))
    if not open_top:
        P.Geo().box(cx - W / 2, y0, H - WALL, cx + W / 2, y1, H).build(name + "_shell_top", body, coll, root, bevel=(0.4 * MM, 2))
        # heat-sink contact area (17.03 mm wide) as a raised thin plate
        P.Geo().box(cx - 17.03 * MM / 2, y0 + 3 * MM, H, cx + 17.03 * MM / 2, y0 + 3 * MM + 50 * MM, H + 0.02 * MM).build(name + "_heatsink_contact_area", P.mat("lid_nickel"), coll, root)
    # latch block and pull tab (flat loop with an oval window), orange plastic
    tab_m = P.mat("pull_tab", (1.0, 0.45, 0.05), 0.0, 0.5)
    P.Geo().box(cx - W / 2 + 1.5 * MM, y0 - 8.0 * MM, 2.5 * MM, cx + W / 2 - 1.5 * MM, y0, 4.0 * MM).build(name + "_latch", tab_m, coll, root, bevel=(0.3 * MM, 2))
    ty0 = y0 - 8.0 * MM
    gt = P.Geo()
    rail = 3.0 * MM
    tw = 21.0 * MM
    tz0, tz1 = 2.5 * MM, 4.0 * MM
    gt.box(cx - tw / 2, ty0 - TAB_L + 8.0 * MM, tz0, cx - tw / 2 + rail, ty0, tz1)
    gt.box(cx + tw / 2 - rail, ty0 - TAB_L + 8.0 * MM, tz0, cx + tw / 2, ty0, tz1)
    gt.box(cx - tw / 2, ty0 - TAB_L + 8.0 * MM, tz0, cx + tw / 2, ty0 - TAB_L + 8.0 * MM + rail, tz1)
    gt.box(cx - tw / 2, ty0 - 4.0 * MM, tz0, cx + tw / 2, ty0, tz1)
    gt.build(name + "_pull_tab", tab_m, coll, root, bevel=(0.5 * MM, 2))
    # paddle PCB + two MT-12 ferrules (module side: unpinned, guide holes) behind the rear window
    P.Geo().box(cx - 18.94 * MM / 2, y1 - 12.0 * MM, 2.0 * MM, cx + 18.94 * MM / 2, y1 + 2.0 * MM, 3.0 * MM).build(name + "_paddle_card", P.mat("pcb_green"), coll, root)
    gf = P.Geo()
    for i in range(10):
        gf.box(cx - 8.5 * MM + i * 1.8 * MM, y1 - 1.0 * MM, 1.99 * MM, cx - 8.5 * MM + i * 1.8 * MM + 1.0 * MM, y1 + 2.0 * MM, 2.0 * MM)
    gf.build(name + "_card_edge_fingers", P.mat("gold_pad"), coll, root)
    P.Geo().box(cx - 11.2 * MM, y1 - 10.0 * MM, 3.0 * MM, cx + 11.2 * MM, y1 + 0.5 * MM, 8.0 * MM).build(name + "_optical_connector_housing", P.mat("anodized"), coll, root, bevel=(0.3 * MM, 2))
    mts = []
    for k, sx in enumerate((-3.6 * MM, 3.6 * MM)):
        info = P.build_mt_ferrule(coll, root, name + "_mt%d" % k, 12, 250 * UM, loc=(cx + sx, y1 + 0.5 * MM, 6.5 * MM), with_pins=False)
        info["root"].rotation_euler = (0, 0, PI / 2)
        mts.append(info)
    if open_top:
        # 8 laser packages with gold leads and 8 PM fibres to the MT ferrules
        gl, gld = P.Geo(), P.Geo()
        fibs = []
        for r in range(2):
            for c in range(4):
                lx = cx + (-6.0 + r * 12.0) * MM
                ly = y0 + (6.0 + c * 9.0) * MM
                gl.box_c(lx, ly, 3.0 * MM + 2.0 * MM, 5.0 * MM, 8.0 * MM, 4.0 * MM)
                for q in range(5):
                    gld.box_c(lx - 3.0 * MM, ly - 3.0 * MM + q * 1.5 * MM, 3.4 * MM, 1.0 * MM, 0.4 * MM, 0.15 * MM)
                    gld.box_c(lx + 3.0 * MM, ly - 3.0 * MM + q * 1.5 * MM, 3.4 * MM, 1.0 * MM, 0.4 * MM, 0.15 * MM)
        gl.build(name + "_laser_packages", P.mat("kovar"), coll, root, bevel=(0.2 * MM, 2))
        gld.build(name + "_laser_leads", P.mat("gold_pad"), coll, root)
        i = 0
        for r in range(2):
            for c in range(4):
                lx = cx + (-6.0 + r * 12.0) * MM
                ly = y0 + (6.0 + c * 9.0) * MM
                k = r
                tx = cx + (-3.6 if k == 0 else 3.6) * MM + (c - 1.5) * 0.25 * MM
                path = P.catmull([(lx, ly + 4.0 * MM, 5.5 * MM), (lx + (tx - lx) * 0.2, ly + 8.0 * MM, 6.5 * MM), (tx, y1 - 14.0 * MM, 6.5 * MM), (tx, y1 - 9.0 * MM, 6.5 * MM)], 8)
                g = P.Geo().tube([(p.x, p.y, p.z) for p in path], 0.45 * MM, n=8)
                g.build("%s_pm_fiber_%d" % (name, i), P.mat("plastic_blue"), coll, root, smooth=True)
                i += 1
    if pigtail:
        paths = []
        for i in range(8):
            xo = (i % 4 - 1.5) * 0.9 * MM
            zo = (i // 4 - 0.5) * 0.9 * MM
            xe = (i - 3.5) * 0.25 * MM
            path = P.catmull([(cx + xo, y0 + 1.0 * MM, 6.0 * MM + zo), (cx + xo, y0 - 20.0 * MM, 6.0 * MM + zo), (cx + xe * 3, y0 - 75.0 * MM, 5.0 * MM), (cx + xe, y0 - 100.0 * MM, 5.0 * MM)], 10)
            g = P.Geo().tube([(p.x, p.y, p.z) for p in path], 0.45 * MM if False else 0.45 * MM, n=8)
            g.build("%s_pigtail_pm_fiber_%d" % (name, i), P.mat("plastic_blue"), coll, root, smooth=True)
        # MT-12 at the pigtail end, mating face toward -Y
        info = P.build_mt_ferrule(coll, root, name + "_pigtail_mt", 12, 250 * UM, loc=(cx, y0 - 108.0 * MM, 5.0 * MM), with_pins=True)
        info["root"].rotation_euler = (0, 0, -PI / 2)
        P.Geo().box_c(cx, y0 - 100.0 * MM, 5.0 * MM, 7.0 * MM, 10.0 * MM, 3.5 * MM).build(name + "_pigtail_boot", P.mat("boot_black"), coll, root, bevel=(0.5 * MM, 3))
    return mts


def butterfly(coll, root, name, cx):
    Lb, Wb, Hb = 30.0 * MM, 12.7 * MM, 9.0 * MM
    g = P.Geo().box(cx - Wb / 2, -Lb / 2, 0, cx + Wb / 2, Lb / 2, Hb - 1.5 * MM)
    g.build(name + "_body", P.mat("kovar"), coll, root, bevel=(0.5 * MM, 2))
    P.Geo().box(cx - Wb / 2 + 0.6 * MM, -Lb / 2 + 0.6 * MM, Hb - 1.5 * MM, cx + Wb / 2 - 0.6 * MM, Lb / 2 - 0.6 * MM, Hb).build(name + "_lid", P.mat("steel"), coll, root, bevel=(0.3 * MM, 2))
    gp = P.Geo()
    for side in (-1, 1):
        for i in range(7):
            y = (i - 3) * 2.54 * MM
            x0 = cx + side * Wb / 2
            gp.box(x0 if side > 0 else x0 - 5.0 * MM, y - 0.25 * MM, 2.8 * MM, x0 + 5.0 * MM if side > 0 else x0, y + 0.25 * MM, 3.1 * MM)
    gp.build(name + "_pins_14", P.mat("gold_pad"), coll, root)
    # snout, boot and fibre pigtail toward -Y
    ys = -Lb / 2
    g = P.Geo().tube([(cx, ys, 4.0 * MM), (cx, ys - 7.0 * MM, 4.0 * MM)], 2.0 * MM, n=20)
    g.build(name + "_snout", P.mat("steel"), coll, root, smooth=True)
    gb = P.Geo()
    gb.cyl_axis((cx, ys - 7.0 * MM, 4.0 * MM), (cx, ys - 27.0 * MM, 4.0 * MM), 1.4 * MM, n=16)
    gb.build(name + "_strain_relief_boot", P.mat("boot_black"), coll, root, smooth=True)
    path = P.catmull([(cx, ys - 27.0 * MM, 4.0 * MM), (cx, ys - 60.0 * MM, 3.0 * MM), (cx + 25.0 * MM, ys - 90.0 * MM, 2.5 * MM), (cx + 25.0 * MM, ys - 140.0 * MM, 2.5 * MM)], 10)
    P.Geo().tube([(p.x, p.y, p.z) for p in path], 0.45 * MM, n=8).build(name + "_fiber_900um", P.mat("fiber_coat_08", (1.0, 0.85, 0.05), 0.0, 0.5), coll, root, smooth=True)
    # FC/APC connector at the end
    ye = ys - 140.0 * MM
    xe = cx + 25.0 * MM
    P.Geo().tube([(xe, ye, 2.5 * MM), (xe, ye - 10.0 * MM, 2.5 * MM)], 1.2 * MM, n=12).build(name + "_fc_boot", P.mat("plastic_green"), coll, root, smooth=True)
    P.Geo().tube([(xe, ye - 10.0 * MM, 2.5 * MM), (xe, ye - 22.0 * MM, 2.5 * MM)], 4.2 * MM, n=24).build(name + "_fc_body", P.mat("steel"), coll, root, smooth=True)
    P.Geo().tube([(xe, ye - 22.0 * MM, 2.5 * MM), (xe, ye - 28.0 * MM, 2.5 * MM)], 1.25 * MM, n=16).build(name + "_fc_ferrule", P.mat("ferrule_white"), coll, root, smooth=True)


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="B")
    v1 = C.sub_collection(coll, "VARIANT_elsfp_closed")
    v2 = C.sub_collection(coll, "VARIANT_elsfp_open")
    v3 = C.sub_collection(coll, "VARIANT_elsfp_front_pigtail")
    v4 = C.sub_collection(coll, "VARIANT_butterfly_laser")
    elsfp(v1, root, ASSET + "_elsfp", 0.0)
    elsfp(v2, root, ASSET + "_elsfp_open", 40.0 * MM, open_top=True)
    elsfp(v3, root, ASSET + "_elsfp_pt", 80.0 * MM, pigtail=True)
    butterfly(v4, root, ASSET + "_butterfly", -45.0 * MM)
    C.hook("blindmate_face", v1, root, loc=(0, L / 2 + 2.0 * MM, 6.5 * MM))
    C.hook("pull_tab_end", v1, root, loc=(0, -L / 2 - TAB_L - 8.0 * MM + 8 * MM, 3.0 * MM))
    C.hook("pigtail_mt_face", v3, root, loc=(80.0 * MM, -L / 2 - 108.0 * MM - 8.0 * MM, 5.0 * MM))
    C.hook("butterfly_fc_face", v4, root, loc=(-45.0 * MM + 25.0 * MM, -15.0 * MM - 140.0 * MM - 28.0 * MM, 2.5 * MM))
    root["p_laser_on"] = 0.0
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("module outer width (shell body)", 21.18, "mm", "OIF-ELSFP-01.0 Fig. 7/9 (21.18 max); end face 22.58 incl. shroud", "A"),
        P.dim("module height", 9.3, "mm", "OIF-ELSFP-01.0 Fig. 9 (9.3 +-0.1)", "A"),
        P.dim("module length", "55.2 body to latch datum; 100.0 overall with pull tab", "mm", "OIF-ELSFP-01.0 Fig. 7 (55.2 +-0.08; 100.00)", "A"),
        P.dim("heat-sink contact area", "17.03 wide (length 59.29 incl. beyond shell; modelled 50)", "mm", "OIF-ELSFP-01.0 Fig. 7", "A"),
        P.dim("paddle card width / 2 ferrules", "18.94 / primary + optional secondary MT-12 ferrule", "mm", "OIF-ELSFP-01.0 Fig. 9", "A"),
        P.dim("shell wall thickness, latch, tab shape", "1.0 mm walls; OSFP-style latch, orange flat tab", "mm", "estimate (spec shows outline only)", "C"),
        P.dim("optical power class (reference)", "up to 26 dBm per lambda per core (SHP); PM fibres (blue)", "-", "OIF-ELSFP Table 6", "A"),
        P.dim("internal layout (8 laser packages, PM fibre loops)", "schematic", "-", "spec: 4 or 8 PM fibres via 1-2 MT-12; internal layout not specified", "C"),
        P.dim("front fibre pigtail (pass-through style)", "8 PM fibres to MT-12, 100 mm", "mm", "spec allows front-side optical connectors for pass-through ELSFP (Sec. 2.1/6.0); geometry estimated", "C"),
        P.dim("14-pin butterfly laser", "30 x 12.7 x 9 mm, 7 pins per side at 2.54 mm", "mm", "typical 14-pin butterfly package class (datasheet range 30-33 x 12.7-13 x 8.9-9); not from one datasheet", "C"),
        P.dim("fibre pigtail / FC-APC", "900 um buffered fibre 140 mm to FC connector 22 mm long", "mm", "generic", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "ELSFP envelope per OIF-ELSFP-01.0. Variants side by side along x: closed (x=0), open top (x=+40 mm), front fibre pigtail (x=+80 mm), butterfly laser (x=-45 mm). Rear/blind-mate end at +Y, pull tab toward -Y (faceplate side). Shell MAT_photonics_oe_body, tab MAT_photonics_pull_tab.",
        [dict(src="https://www.oiforum.com/wp-content/uploads/OIF-ELSFP-01.0.pdf", accessed="2026-10-02", used_for="all ELSFP dimensions (Fig. 7, 9; Table 6)"),
         dict(src="20260320_OFC M4B.5 (ELSFP with 4 x 21 dBm DFB at 1310 nm)", accessed="2026-10-02", used_for="typical use: 4 PM fibres, 1:4 split per fibre"),
         dict(src="blog 2025/20251219_CPO/ranovus-external-internal-laser.webp", accessed="2026-10-02", used_for="external laser source with fibre tray")],
        dims, [h.name for h in coll.all_objects if h.name.startswith("HOOK_")], P.used_material_names(), ["p_laser_on (unused placeholder 0..1)"],
        "S3 (external laser box vs in-package laser), S5/S6 context.",
        ["Latch mechanics simplified", "Internal electronics omitted", "Butterfly dims typical not datasheet-specific", "No logos (label area blank)"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    prev = os.path.join(out_dir, "previews")
    shots = [
        dict(name="overview_top", target=(0.01, -0.03, 0.0), dist=0.2, az=0, el=89.5),
        dict(name="front", target=(0.01, -0.02, 0.005), dist=0.3, az=0, el=12),
        dict(name="three_quarter", target=(0.01, -0.03, 0.004), dist=0.17, az=-35, el=28),
        dict(name="elsfp_closed_three_quarter", target=(0.0, -0.01, 0.005), dist=0.09, az=35, el=28),
        dict(name="elsfp_open_interior", target=(0.04, 0.0, 0.004), dist=0.06, az=25, el=55),
        dict(name="elsfp_rear_blindmate_face", target=(0.0, 0.028, 0.006), dist=0.03, az=180 - 25, el=15),
        dict(name="pigtail_variant", target=(0.08, -0.07, 0.004), dist=0.17, az=-30, el=35),
        dict(name="butterfly_laser", target=(-0.04, -0.07, 0.003), dist=0.17, az=-35, el=35),
        dict(name="butterfly_closeup", target=(-0.045, -0.01, 0.004), dist=0.045, az=25, el=35),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots, floor=-0.0001)
    print(outs)


if __name__ == "__main__":
    main()
