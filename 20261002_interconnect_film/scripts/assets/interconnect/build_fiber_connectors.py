"""Build fiber_connectors.blend: MPO-12 / MPO-16 plugs, MPO and LC adapters, LC duplex, SC, APC faces, dust caps, boots, end-face macro.

Every connector local frame: mating face at y = 0 facing -Y, body toward +Y, key/latch up (+Z). Items are laid out along X.
Run: Blender -b --python scripts/assets/interconnect/build_fiber_connectors.py -- assets/components/interconnect
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import ic_common as I
import ic_optics as O
from ic_common import MB, _v

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "fiber_connectors"

# MPO plug housing (estimates, level B/C: public product photos, US Conec MT footprint 6.4 x 2.5 mm)
MPO = dict(housing_w=12.4, housing_h=8.3, housing_len=13.0, sleeve_w=13.2, sleeve_h=9.0, sleeve_len=15.0, boot_len=32.0)


def hollow(mb, xr, yr, zr, wall, bev=0.2, front_open=True, rear_open=True):
    x0, x1 = xr
    y0, y1 = yr
    z0, z1 = zr
    mb.box((x0, x0 + wall), (y0, y1), (z0, z1), bev=bev, seg=1)
    mb.box((x1 - wall, x1), (y0, y1), (z0, z1), bev=bev, seg=1)
    mb.box((x0 + wall, x1 - wall), (y0, y1), (z1 - wall, z1), bev=bev, seg=1)
    mb.box((x0 + wall, x1 - wall), (y0, y1), (z0, z0 + wall), bev=bev, seg=1)
    if not front_open:
        mb.box((x0, x1), (y0, y0 + wall), (z0, z1), bev=bev, seg=1)
    if not rear_open:
        mb.box((x0, x1), (y1 - wall, y1), (z0, z1), bev=bev, seg=1)


def cone_boot(M, name, y0, length, r0, r1, mat, segs=16, sx=1.0, sz=1.0):
    mb = MB()
    # cone along +Y: radius1 at -Y end (y0) = r0, radius2 at +Y end = r1
    mb.cyl((0, y0 + length / 2, 0), r0, length, axis="Y", segs=segs, r2=r1)
    ob = mb.build(name, mat, smooth_deg=40)
    ob.scale = (sx, 1.0, sz)
    return ob


def mpo_plug(M, key, nfib, apc, male, body_color, coll, parent, loc_x, with_cap=False):
    """key: item name. Returns the group empty."""
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, 0, 0))
    pfx = ASSET + "_" + key
    for o in O.mt_ferrule(M, nfib, 1, apc=apc, pins=male, name=pfx + "_mt"):
        I.place(o, coll, g)
    # front housing (coloured) and sliding sleeve
    mb = MB()
    hollow(mb, (-MPO["housing_w"] / 2, MPO["housing_w"] / 2), (1.0, 1.0 + MPO["housing_len"]), (-MPO["housing_h"] / 2, MPO["housing_h"] / 2), 1.3)
    mb.box((-1.6 if nfib == 12 else 0.8, 1.6 if nfib == 12 else 3.8), (1.0, 1.0 + MPO["housing_len"] * 0.8), (MPO["housing_h"] / 2, MPO["housing_h"] / 2 + 0.5), bev=0.15, seg=1)   # key ridge (offset for MPO-16)
    I.place(mb.build(pfx + "_housing", M[body_color], smooth_deg=35), coll, g)
    mb = MB()
    hollow(mb, (-MPO["sleeve_w"] / 2, MPO["sleeve_w"] / 2), (1.0 + 9.0, 1.0 + 9.0 + MPO["sleeve_len"]), (-MPO["sleeve_h"] / 2, MPO["sleeve_h"] / 2), 0.9, bev=0.3)
    I.place(mb.build(pfx + "_sleeve", M["plastic_black"] if apc is False else M["plastic_black"], smooth_deg=35), coll, g)
    # ferrule carrier inside + crimp
    mb = MB()
    mb.box((-4.8, 4.8), (8.0, 20.0), (-2.4, 2.4), bev=0.2, seg=1)
    I.place(mb.build(pfx + "_carrier", M["plastic_beige"], smooth_deg=35), coll, g)
    y_b = 1.0 + 9.0 + MPO["sleeve_len"]
    mb = MB()
    mb.box((-4.2, 4.2), (y_b - 1.0, y_b + 6.0), (-3.0, 3.0), bev=0.8, seg=2)
    I.place(mb.build(pfx + "_crimp", M["steel"], smooth_deg=35), coll, g)
    I.place(cone_boot(M, pfx + "_boot", y_b + 6.0, MPO["boot_len"], 3.6, 1.7, M[body_color if body_color != "plastic_aqua" else "plastic_aqua"], segs=18, sx=1.25, sz=0.85), coll, g)
    # cable (3.0 mm jacket for 12/16F ribbon-in-tube)
    pts = np.array([[0, y_b + 6.0 + MPO["boot_len"] - 1, 0], [0, y_b + 6 + MPO["boot_len"] + 25, 0], [0, y_b + 6 + MPO["boot_len"] + 50, 0]]) * 0.001
    I.place(I.tube_mesh(pfx + "_cable", I.catmull(pts, 6), 0.0015, sides=10, mat=M["jacket_aqua"] if body_color == "plastic_aqua" else M["jacket_yellow"]), coll, g)
    C.hook(key + "_mate", coll, g, loc=_v(0, 0, 0))
    return g


def hook_note(h):
    h.empty_display_size = 0.006


def lc_simplex(M, pfx, apc, body_color, coll, g, x0, sm=True):
    # ferrule 1.25 mm, 6.6 long: 3.4 mm protrudes from the housing front (y 3.2)
    for o in O.round_ferrule(M, 1.25, 6.6, apc=apc, name=pfx + "_ferrule"):
        I.place(o, coll, g, loc_mm=(x0, 0, 0))
    mb = MB()
    mb.box((x0 - 2.8, x0 + 2.8), (3.4, 20.0), (-3.0, 3.0), bev=0.5, seg=2)       # housing body 5.6 wide x 6.0 high
    I.place(mb.build(pfx + "_housing", M[body_color], smooth_deg=35), coll, g)
    # latch arm: thin plate rising at ~25 deg from the housing top front to the rear
    mb = MB()
    mb.prism_yz([(5.0, 3.0), (17.0, 3.0), (20.5, 5.4), (9.5, 7.2), (8.0, 6.4)], x0 - 1.7, x0 + 1.7)
    I.place(mb.build(pfx + "_latch", M[body_color], smooth_deg=35), coll, g)
    mb = MB()
    mb.box((x0 - 1.5, x0 + 1.5), (9.0, 12.5), (3.0, 4.6), bev=0.2, seg=1)
    I.place(mb.build(pfx + "_latch_stop", M["plastic_black"], smooth_deg=35), coll, g)
    # boot (2.0 mm cord)
    I.place(cone_boot(M, pfx + "_boot", 20.0, 14.0, 2.4, 1.2, M[body_color], segs=14), coll, g)
    ob = bpy.data.objects[pfx + "_boot"]
    ob.location.x += x0 * 0.001
    pts = np.array([[x0, 33.5, 0], [x0, 55, 0], [x0 + 3, 90, -2]]) * 0.001
    I.place(I.tube_mesh(pfx + "_cord", I.catmull(pts, 8), 0.001, sides=10, mat=M["jacket_yellow"]), coll, g)


def lc_duplex(M, key, apc, body_color, coll, parent, loc_x):
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, 0, 0))
    pfx = ASSET + "_" + key
    for s in (-1, 1):
        lc_simplex(M, pfx + ("_T" if s < 0 else "_R"), apc, body_color, coll, g, s * 3.125)
    # duplex clip: bridging bar at the rear of the housings
    mb = MB()
    mb.box((-5.9, 5.9), (13.0, 19.5), (-3.8, -2.6), bev=0.2, seg=1)
    mb.box((-6.2, -4.6), (13.0, 19.5), (-3.8, 3.0), bev=0.2, seg=1) if False else None
    I.place(mb.build(pfx + "_clip", M["plastic_black"], smooth_deg=35), coll, g)
    C.hook(key + "_mate", coll, g, loc=_v(0, 0, 0))
    return g


def sc_simplex(M, key, apc, body_color, coll, parent, loc_x):
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, 0, 0))
    pfx = ASSET + "_" + key
    for o in O.round_ferrule(M, 2.5, 10.5, apc=apc, name=pfx + "_ferrule"):
        I.place(o, coll, g)
    mb = MB()
    hollow(mb, (-4.2, 4.2), (6.0, 24.0), (-4.5, 4.5), 1.0, bev=0.4, front_open=True)
    mb.box((-4.2, 4.2), (6.0, 7.5), (-4.5, 4.5), bev=0.4, seg=1)
    mb.box((-1.5, 1.5), (6.0, 18.0), (4.5, 5.2), bev=0.2, seg=1)      # key
    I.place(mb.build(pfx + "_housing", M[body_color], smooth_deg=35), coll, g)
    mb = MB()
    mb.box((-3.2, 3.2), (24.0, 32.0), (-3.2, 3.2), bev=0.6, seg=2)
    I.place(mb.build(pfx + "_grip", M["plastic_black"], smooth_deg=35), coll, g)
    I.place(cone_boot(M, pfx + "_boot", 32.0, 20.0, 2.8, 1.6, M["plastic_black"], segs=14), coll, g)
    pts = np.array([[0, 51, 0], [0, 75, 0], [0, 100, 0]]) * 0.001
    I.place(I.tube_mesh(pfx + "_cord", I.catmull(pts, 6), 0.001, sides=10, mat=M["jacket_yellow"]), coll, g)
    C.hook(key + "_mate", coll, g, loc=_v(0, 0, 0))
    return g


def mpo_adapter(M, key, coll, parent, loc_x, y0=0.0):
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, y0, 0))
    pfx = ASSET + "_" + key
    mb = MB()
    hollow(mb, (-7.9, 7.9), (-15.0, 15.0), (-5.1, 5.1), 1.1, bev=0.3)
    mb.box((-1.6, 1.6), (-15.0, 15.0), (5.1, 5.7), bev=0.15, seg=1)     # key rail
    I.place(mb.build(pfx + "_body", M["plastic_aqua"], smooth_deg=35), coll, g)
    mb = MB()
    mb.box((-11.5, 11.5), (-1.0, 1.5), (-6.8, 6.8), bev=0.4, seg=1)      # mounting flange (mm: estimate)
    mb.box((-11.5, 11.5), (-1.0, 1.5), (-6.8, 6.8), bev=0.4, seg=1)
    I.place(mb.build(pfx + "_flange", M["plastic_black"], smooth_deg=35), coll, g)
    mb = MB()
    mb.cyl((-2.3, 0, 0), 0.7, 14.0, axis="Y", segs=12)
    mb.cyl((2.3, 0, 0), 0.7, 14.0, axis="Y", segs=12)
    I.place(mb.build(pfx + "_guide_pins_slot", M["steel"], smooth_deg=30), coll, g)
    return g


def dust_cap_mpo(M, key, coll, parent, loc_x, color):
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, 0, 0))
    mb = MB()
    hollow(mb, (-6.9, 6.9), (-14.0, 4.0), (-4.7, 4.7), 0.8, bev=0.4, front_open=False, rear_open=True)
    mb.box((-6.9, 6.9), (-14.0, -13.2), (-4.7, 4.7), bev=0.4, seg=1)
    I.place(mb.build(ASSET + "_" + key + "_cap", M[color], smooth_deg=35), coll, g)
    return g


def dust_cap_lc(M, key, coll, parent, loc_x, color):
    g = I.empty(key + "_group", coll, parent, loc_mm=(loc_x, 0, 0))
    mb = MB()
    mb.cyl((0, -4.0, 0), 1.3, 12.0, axis="Y", segs=20)
    mb.cyl((0, -9.0, 0), 1.9, 6.0, axis="Y", segs=20)
    I.place(mb.build(ASSET + "_" + key + "_cap", M[color], smooth_deg=35), coll, g)
    return g


def endface_macro(M, coll, parent, loc_mm, scale):
    """LC ferrule end face at true size under a scaled empty: 1.25 mm ferrule, 125 um cladding, 9 um core (x scale)."""
    g = I.empty("endface_macro_group", coll, parent, loc_mm=loc_mm)
    g.scale = (scale, scale, scale)
    R = 0.625
    mb = MB()
    # annular ferrule face (ring between r=0.0635 and R-0.12 with chamfer to R), seen head-on (face toward -Y)
    prof = [(0.0635, 0.8), (0.0635, 0.0), (R - 0.12, 0.0), (R, 0.12), (R, 0.8)]
    O.lathe_y(mb, prof, segs=72)
    I.place(mb.build(ASSET + "_endface_ferrule", M["ferrule_ceramic"], smooth_deg=40), coll, g)
    mb = MB()
    mb.cyl((0, 0.4, 0), 0.0625, 0.8, axis="Y", segs=48)
    I.place(mb.build(ASSET + "_endface_cladding", M["fiber_glass"], smooth_deg=0), coll, g)
    mb = MB()
    mb.cyl((0, 0.0004, 0), 0.0045, 0.0009, axis="Y", segs=24)
    I.place(mb.build(ASSET + "_endface_core", M["fiber_core"], smooth_deg=0), coll, g)
    return g


def main():
    C.reset()
    M = I.Mats()
    coll, root = C.new_asset(ASSET, accuracy="B")
    items = []
    x = 0.0
    sub = {}

    def sc(name):
        sub[name] = C.sub_collection(coll, "VARIANT_" + name)
        return sub[name]

    mpo_plug(M, "mpo12_pc_male", 12, False, True, "plastic_aqua", sc("mpo12_pc_male"), root, x); x += 28
    mpo_plug(M, "mpo12_apc_female", 12, True, False, "plastic_green", sc("mpo12_apc_female"), root, x); x += 28
    mpo_plug(M, "mpo16_apc_male", 16, True, True, "plastic_green", sc("mpo16_apc_male"), root, x); x += 28
    mpo_adapter(M, "mpo_adapter", sc("mpo_adapter"), root, x); x += 34
    # mated pair demonstration: adapter + two plugs facing each other (linked duplicates, plug B rotated 180 deg about Z)
    cm = sc("mpo12_mated_pair")
    ga = mpo_adapter(M, "mpo_adapter_mated", cm, root, x, y0=0.0)
    pa = mpo_plug(M, "mpo12_mated_a", 12, False, True, "plastic_aqua", cm, ga, 0.0)
    pb = mpo_plug(M, "mpo12_mated_b", 12, False, False, "plastic_aqua", cm, ga, 0.0)
    pa.location = _v(0, -0.2, 0)
    pb.location = _v(0, 0.2, 0)
    pa.rotation_euler = (0, 0, math.pi)
    x += 70
    lc_duplex(M, "lc_duplex_sm_upc", False, "plastic_blue", sc("lc_duplex_sm_upc"), root, x); x += 30
    lc_duplex(M, "lc_duplex_sm_apc", True, "plastic_green", sc("lc_duplex_sm_apc"), root, x); x += 30
    lc_duplex(M, "lc_duplex_mm_om4", False, "plastic_aqua", sc("lc_duplex_mm_om4"), root, x); x += 30
    sc_simplex(M, "sc_sm_upc", False, "plastic_blue", sc("sc_sm_upc"), root, x); x += 26
    sc_simplex(M, "sc_sm_apc", True, "plastic_green", sc("sc_sm_apc"), root, x); x += 28
    dust_cap_mpo(M, "dust_cap_mpo", sc("dust_caps"), root, x, "plastic_blue")
    dust_cap_lc(M, "dust_cap_lc", sub["dust_caps"], root, x + 25, "plastic_blue")
    # LC duplex adapter (simple hollow body with two split sleeves)
    ca = sc("lc_duplex_adapter")
    ga2 = I.empty("lc_adapter_group", ca, root, loc_mm=(x + 55, 0, 0))
    mb = MB()
    hollow(mb, (-6.9, 6.9), (-12.0, 12.0), (-4.6, 4.6), 1.0, bev=0.3)
    I.place(mb.build(ASSET + "_lc_adapter_body", M["plastic_blue"], smooth_deg=35), ca, ga2)
    mb = MB()
    for s in (-1, 1):
        mb.cyl((s * 3.125, 0, 0), 1.2, 10.0, axis="Y", segs=20)
    I.place(mb.build(ASSET + "_lc_adapter_sleeves", M["ferrule_ceramic"], smooth_deg=35), ca, ga2)
    x_end = x + 90
    # end-face macro (true size built, x40 scale on the group), placed on a second row behind the connectors
    cm2 = sc("endface_macro")
    endface_macro(M, cm2, root, (-70, 0, 0), 40.0)
    I.add_prop(root, "p_note", 0, doc="no animated properties; connectors are static")
    del root["p_note"]
    meta = dict(
        description="Fibre connectors: MPO-12 PC/APC male/female, MPO-16 APC (offset key, 16-fibre 1x16 MT), MPO adapter and mated pair, LC duplex (SM UPC blue, SM APC green, MM aqua), SC simplex UPC/APC, LC and MPO dust caps, LC duplex adapter, LC end-face macro.",
        sources=[
            {"what": "US Conec product page (MT-12 ferrule: 0.7 mm guide pins, 4.6 mm pin pitch, 0.25 mm fibre pitch)", "url": "https://www.usconec.com/products/mt-ferrule-with-boot-12f-multimode-mt-elite-dimple", "accessed": "2026-10-02"},
            {"what": "US Conec white paper (MT ferrule width 6.4 mm; 250 um pitch)", "url": "https://www.usconec.com/media/vbhbl0da/a-very-small-form-factor-multi-row-multi-fiber-connector-with-multi-vendor-interoperability-white-paper.pdf", "accessed": "2026-10-02"},
            {"what": "US Conec MTP-16 handout (same footprint as MT-12, 0.25/0.50 mm pitch, offset key)", "url": "https://www.usconec.com/media/1i4pg2b5/mtp-16_connector_handout.pdf", "accessed": "2026-10-02"},
            {"what": "QSFP-DD HW Rev 5.1 Fig 15/16/18 (MPO-12 / MPO-16 / dual LC receptacle pictures; TIA-604-5, 604-18, 604-10 referenced)", "url": "http://www.qsfp-dd.com/wp-content/uploads/2020/08/QSFP-DD-Hardware-rev5.1.pdf", "accessed": "2026-10-02"},
            {"what": "US Conec catalog URL from the task returned HTTP 404 on 2026-10-02: MT ferrule 2.5 mm height and 8.0 mm length are level B (task statement + public knowledge), not read from the catalog", "url": "https://www.usconec.com/media/2bsp1emu/us-conec-product-catalog.pdf", "accessed": "2026-10-02"},
        ],
        dimension_table=[
            dict(item="MT ferrule width", value=6.4, unit="mm", source="US Conec white paper", accuracy="A"),
            dict(item="MT ferrule height", value=2.5, unit="mm", source="task statement; catalog not reachable", accuracy="B"),
            dict(item="MT ferrule length", value=8.0, unit="mm", source="public knowledge, not verified from a primary doc", accuracy="B"),
            dict(item="guide pin pitch / diameter", value="4.6 / 0.7", unit="mm", source="US Conec product page", accuracy="A"),
            dict(item="fibre pitch (row), row pitch (2-row)", value="0.25 / 0.50", unit="mm", source="US Conec product page, MTP-16 handout", accuracy="A"),
            dict(item="fibre hole 126 um, cladding 125 um, core 9 um (SM)", value="0.126 / 0.125 / 0.009", unit="mm", source="standard SMF (G.652) geometry; hole 126 um is an estimate", accuracy="B"),
            dict(item="APC polish angle", value=8, unit="deg", source="IEC 61755 / industry standard", accuracy="A"),
            dict(item="MPO plug housing 12.4 x 8.3 x 13 (front) + sleeve 13.2 x 9.0 x 15", value="12.4 x 8.3", unit="mm", source="estimate from public product photos; TIA-604-5 not read", accuracy="C"),
            dict(item="LC ferrule diameter / length", value="1.25 / 6.6", unit="mm", source="IEC 61754-20 (ferrule dia 1.25 mm); length estimate", accuracy="B"),
            dict(item="LC duplex pitch", value=6.25, unit="mm", source="TIA-604-10 as referenced by QSFP-DD HW Fig 18", accuracy="B"),
            dict(item="LC body 5.6 x 6.0 x 17 mm", value=5.6, unit="mm", source="estimate", accuracy="C"),
            dict(item="SC ferrule diameter", value=2.5, unit="mm", source="IEC 61754-4", accuracy="B"),
            dict(item="SC housing 8.4 x 9.0 x 18", value=8.4, unit="mm", source="estimate", accuracy="C"),
            dict(item="boot lengths / cord diameters", value="MPO 32, LC 14, SC 20 / 3.0 and 2.0 mm", unit="mm", source="estimate; cord diameters from task (2.0 and 3.0 mm)", accuracy="C"),
        ],
        hooks={"HOOK_<item>_mate": "mating face centre of each connector (local -Y is the mating direction: the partner approaches from -Y); items: mpo12_pc_male, mpo12_apc_female, mpo16_apc_male, mpo12_mated_a/b, lc_duplex_sm_upc, lc_duplex_sm_apc, lc_duplex_mm_om4, sc_sm_upc, sc_sm_apc"},
        custom_properties={},
        layout="Items in a row along +X (28-34 mm spacing) with faces toward -Y; the LC end-face macro is at x = -70 mm, built at true size (1.25 mm ferrule) under an empty scaled x40 (so ferrule 50 mm, cladding 5 mm, core 0.36 mm across). Each item is a VARIANT_<item> sub-collection under one parent empty named <item>_group.",
        origin="ROOT at the origin; each item's group empty origin is at its mating face centre.",
        colour_code="SM UPC blue, SM APC green, MM OM3/OM4 aqua (TIA-598 convention).",
        simplifications=["MPO housing is hollow boxes (no latch springs, no push-pull internals)", "no alignment-pin clips", "LC latch is a plain wedge plate", "SC adapter not modeled", "dust caps are plain moulded shapes", "fibre cores glow via an emission material (shader-only)"],
        intended_usage="S6 connector-jam macro (MT face, LC end-face macro), S1/S2 background; module receptacles use the same MT ferrule builder (ic_optics.mt_ferrule).",
    )
    meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
    meta["triangles_unique_meshes"] = I.unique_tris(coll)
    blend = os.path.join(OUT, ASSET + ".blend")
    C.finish(ASSET, blend, coll, meta, preview_dir=None)
    return coll, root


if __name__ == "__main__":
    coll, root = main()
    prev = os.path.join(OUT, "previews")
    I.preview_views(coll, prev, ASSET, [("overview_front", (60, 0, 0), (0.0, -1, 0.25), 560, 40), ("overview_three_quarter", (60, 20, 0), (0.7, -1, 0.5), 560, 40),
                                       ("mpo_face_closeup", (0, 0, 0), (0.25, -1, 0.15), 16, 85), ("mpo16_apc_face_closeup", (56, 0, 0), (0.5, -1, 0.1), 16, 85),
                                       ("lc_duplex_closeup", (188, 0, 0), (0.35, -1, 0.3), 45, 60), ("endface_macro", (-70, 0, 0), (0.0, -1, 0.0), 120, 50)])
