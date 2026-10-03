"""Build fiber_cables_and_trays.blend: yellow SM patch cords (2.0 / 3.0 mm), trunk cable cross-sections (8F..6912F), cable tie, velcro,
splice tray and a 1U patch panel with LC duplex and MPO adapters.
Run: Blender -b --python scripts/assets/interconnect/build_fiber_cables_and_trays.py -- assets/components/interconnect
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import ic_common as I
from ic_common import MB, _v
import build_fiber_connectors as FC

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "fiber_cables_and_trays"
FC.ASSET = ASSET

# (label, fibres, cable OD mm): ODs from 20260612_fiber_bundle/ctx-summary.md (own project note, level B)
TRUNKS = [("8F", 8, 4.0), ("144F", 144, 9.6), ("288F", 288, 12.1), ("864F", 864, 11.4), ("3456F", 3456, 23.5), ("6912F", 6912, 29.0)]
COLORS12 = ["plastic_blue", "plastic_orange", "plastic_green", "jacket_blue", "plastic_beige", "plastic_white", "plastic_red", "plastic_black",
            "plastic_yellow", "jacket_violet", "jacket_red", "plastic_aqua"]


def hexpts(n, rmax, pitch):
    pts = []
    k = int(rmax / pitch) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = pitch * (i + 0.5 * (j % 2))
            y = pitch * math.sqrt(3) / 2 * j
            r = math.hypot(x, y)
            if r <= rmax:
                pts.append((r, x, y))
    pts.sort()
    return [(x, y) for _, x, y in pts[:n]], len(pts)


def trunk_section(M, coll, parent, label, n, od, x_mm):
    g = I.empty("trunk_%s_group" % label, coll, parent, loc_mm=(x_mm, 0, 0))
    wall = 0.8 if od < 6 else 1.3 if od < 15 else 2.0
    r_in = od / 2 - wall
    rf = 0.125
    pts, cap = hexpts(n, r_in - 0.45, 2 * rf * 1.0)
    if len(pts) < n:
        rf = 0.1
        pts, cap = hexpts(n, r_in - 0.45, 2 * rf)
    assert len(pts) >= n, (label, len(pts), n)
    ang = np.linspace(0, 2 * np.pi, 56, endpoint=False)
    mb = MB()
    mb.ring_xz([(od / 2 * math.cos(a), od / 2 * math.sin(a)) for a in ang], [(r_in * math.cos(a), r_in * math.sin(a)) for a in ang], 0.0, 60.0)
    I.place(mb.build("trunk_%s_jacket" % label, M["jacket_black"], smooth_deg=40), coll, g)
    mb = MB()
    mb.ring_xz([((r_in) * math.cos(a), (r_in) * math.sin(a)) for a in ang], [((r_in - 0.25) * math.cos(a), (r_in - 0.25) * math.sin(a)) for a in ang], 0.0, 60.0)
    I.place(mb.build("trunk_%s_core_wrap" % label, M["plastic_white"], smooth_deg=40), coll, g)
    # strength members (FRP rods) in the jacket wall for the larger cables
    if od > 6:
        mb = MB()
        for k in range(6):
            a = 2 * math.pi * k / 6
            rr = od / 2 - wall / 2
            mb.cyl((rr * math.cos(a), 30.0, rr * math.sin(a)), min(0.5, wall * 0.3), 60.0, axis="Y", segs=8)
        I.place(mb.build("trunk_%s_strength_rods" % label, M["steel"], smooth_deg=30), coll, g)
    P = np.array(pts)
    idx = np.arange(len(P)) % 12
    for c in range(12):
        sel = P[idx == c]
        if len(sel) == 0:
            continue
        paths = np.stack([np.column_stack([sel[:, 0], np.full(len(sel), 0.0), sel[:, 1]]), np.column_stack([sel[:, 0], np.full(len(sel), 3.0), sel[:, 1]])], axis=1) * 0.001
        I.place(I.tube_mesh("trunk_%s_fibres_c%02d" % (label, c), paths, rf * 0.001, sides=6, mat=M[COLORS12[c]], caps=True), coll, g)
    return g, dict(label=label, fibres=n, od=od, fibre_radius_mm=rf, packing_capacity=cap)


def main():
    C.reset()
    M = I.Mats()
    coll, root = C.new_asset(ASSET, accuracy="B")
    info = {}
    # ---- trunk cross-sections (row at y = 0)
    sub = C.sub_collection(coll, "VARIANT_trunk_cross_sections")
    x = 0.0
    trunk_info = []
    for label, n, od in TRUNKS:
        x += od / 2 + 3
        g, ti = trunk_section(M, sub, root, label, n, od, x)
        trunk_info.append(ti)
        x += od / 2 + 3
    # ---- patch cords (row at y = 300)
    subc = C.sub_collection(coll, "VARIANT_patch_cords")
    gp = I.empty("patch_cords_root", subc, root, loc_mm=(0, 300, 0))
    L = 400.0
    for label, rad, xo in (("2p0mm", 1.0, 0.0), ("3p0mm", 1.5, 40.0)):
        g = I.empty("cord_%s_group" % label, subc, gp, loc_mm=(xo, 0, 0))
        FC.lc_duplex(M, "cord_%s_plug_a" % label, False, "plastic_blue", subc, g, 0.0)
        gb = FC.lc_duplex(M, "cord_%s_plug_b" % label, False, "plastic_blue", subc, g, 0.0)
        gb.location = _v(0, L, 0)
        gb.rotation_euler = (0, 0, math.pi)
        paths = []
        for s in (-1, 1):
            x0 = s * 3.125
            pts = np.array([[x0 + 3, 90, -2], [x0 + 3, 90 + 40, -2], [x0 * 0.4, L * 0.5, 15], [x0 + 3 if False else x0 * 0.4 * 0, L - 130, -2] if False else [-(x0 + 3), L - 130, -2], [-(x0 + 3), L - 90, -2]]) * 0.001
            paths.append(I.catmull(pts, 8))
        cord = I.tube_mesh("cord_%s_jackets" % label, np.array(paths), rad * 0.001, sides=12, mat=M["jacket_yellow"], caps=True)
        I.place(cord, subc, g)
        C.hook("cord_%s_end_a" % label, subc, g, loc=_v(0, 0, 0))
        hb = C.hook("cord_%s_end_b" % label, subc, g, loc=_v(0, L, 0))
    # ---- cable tie + velcro around a bundle of 8 cords (row at y = -150)
    subt = C.sub_collection(coll, "VARIANT_ties_and_velcro")
    gt = I.empty("ties_root", subt, root, loc_mm=(0, -150, 0))
    rng = np.random.default_rng(5)
    paths = []
    for k in range(8):
        a = 2 * math.pi * k / 8
        paths.append(np.array([[3.2 * math.cos(a), -60 + 0.0, 3.2 * math.sin(a) + 6], [3.2 * math.cos(a), 0.0, 3.2 * math.sin(a) + 6], [3.2 * math.cos(a), 60.0, 3.2 * math.sin(a) + 6]]) * 0.001)
    bund = I.tube_mesh("bundle_cords", np.array([I.catmull(p, 6) for p in paths]), 0.001, sides=10, mat=M["jacket_yellow"], caps=True)
    I.place(bund, subt, gt)
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    mb = MB()
    mb.ring_xz([(5.2 * math.cos(a), 6 + 5.2 * math.sin(a)) for a in ang], [(4.6 * math.cos(a), 6 + 4.6 * math.sin(a)) for a in ang], -22.0, -19.0)
    mb.box((-2.0, 2.0), (-23.0, -18.0), (10.4, 13.6), bev=0.3, seg=1)
    mb.box((-0.8, 0.8), (-22.5, -18.5), (13.6, 40.0 / 1.0 * 0.0 + 30.0), bev=0.2, seg=1)
    I.place(mb.build("cable_tie", M["cable_tie"], smooth_deg=35), subt, gt)
    mb = MB()
    mb.ring_xz([(5.5 * math.cos(a), 6 + 5.5 * math.sin(a)) for a in ang], [(4.65 * math.cos(a), 6 + 4.65 * math.sin(a)) for a in ang], 15.0, 40.0)
    mb.box((5.0, 5.8), (15.0, 40.0), (3.0, 14.0), bev=0.2, seg=1)
    mb.box((5.0, 5.8), (15.0, 40.0), (3.0, 14.0), bev=0.2, seg=1)
    I.place(mb.build("velcro_strap", M["velcro_black"], smooth_deg=35), subt, gt)
    mb = MB()
    mb.box((5.8, 6.2), (30.0, 40.0), (3.0, 12.0), bev=0.1, seg=1)
    I.place(mb.build("velcro_hook_pad", M["plastic_grey"], smooth_deg=35), subt, gt)
    # ---- splice tray (x = 300, y = 0)
    subs = C.sub_collection(coll, "VARIANT_splice_tray")
    gs = I.empty("splice_tray_root", subs, root, loc_mm=(320, 0, 0))
    mb = MB()
    mb.box((-80, 80), (-55, 55), (0, 2.0), bev=0.5, seg=1)
    for (xr, yr) in (((-80, -78.5), (-55, 55)), ((78.5, 80), (-55, 55)), ((-80, 80), (-55, -53.5)), ((-80, 80), (53.5, 55))):
        mb.box(xr, yr, (2.0, 10.0), bev=0.4, seg=1)
    for (cx, cy) in ((-35, 0), (35, 0)):
        mb.cyl((cx, cy, 5.0), 14.0, 6.0, axis="Z", segs=32)          # fibre spool posts (bend radius > 15 mm control)
    I.place(mb.build("splice_tray_base", M["plastic_grey"], smooth_deg=35), subs, gs)
    mb = MB()
    for k in range(12):
        mb.box((-72 + (k % 6) * 5.0 + 0, -72 + (k % 6) * 5.0 + 3.5), (-48 + (k // 6) * 8.0, -48 + (k // 6) * 8.0 + 4.0), (2.0, 5.0), bev=0.2, seg=1)
    I.place(mb.build("splice_protector_holders", M["plastic_black"], smooth_deg=35), subs, gs)
    mb = MB()
    for k in range(12):
        mb.cyl((-70.3 + (k % 6) * 5.0, -46 + (k // 6) * 8.0, 3.8), 1.4, 3.0 if False else 42.0, axis="Y", segs=10)
    # splice protectors: heat-shrink sleeves 40 mm long, 2.6 mm dia (estimate) lying in the holders
    sp = MB()
    for k in range(12):
        sp.cyl((-70.2 + (k % 6) * 5.0 + 1.7, -48 + (k // 6) * 8.0 + 2.0, 4.0), 1.4, 3.0, axis="Z", segs=10)
    I.place(sp.build("splice_protectors", M["plastic_red"], smooth_deg=30), subs, gs)
    loops = []
    for k in range(10):
        R = 17 + 1.2 * k
        t = np.linspace(0, 2 * np.pi, 48)
        cx = -35 if k % 2 == 0 else 35
        loops.append(np.column_stack([cx + R * np.cos(t), R * np.sin(t) * 0.9, np.full(48, 6.5 + 0.35 * (k % 3))]) * 0.001)
    I.place(I.tube_mesh("splice_tray_fibre_loops", np.array(loops), 0.00045, sides=6, mat=M["jacket_yellow"]), subs, gs)
    mb = MB()
    mb.box((-80, 80), (-55, 55), (11.5, 12.0), bev=0.2, seg=1)
    lid = mb.build("splice_tray_lid", M["glass"], smooth_deg=30)
    I.place(lid, subs, gs)
    lid.hide_render = True
    lid.location = _v(0, 0, 0)
    C.hook("splice_tray_lid", subs, gs, loc=_v(0, 0, 12.0))
    # ---- 1U patch panel (y = 700)
    subp = C.sub_collection(coll, "VARIANT_patch_panel")
    gpp = I.empty("patch_panel_root", subp, root, loc_mm=(0, 700, 0))
    mb = MB()
    W, H = 482.6, 43.66            # EIA-310 19 in panel width, 1U panel height (1.75 in - 1/32 in)
    n_lc, pitch_lc, x_lc0 = 12, 14.4, -190.0
    n_mpo, pitch_mpo, x_mpo0 = 6, 34.0, 60.0
    # plate built from frame pieces around the adapter openings (opening z 12..32)
    zo0, zo1 = 13.0, 31.0
    mb.box((-W / 2, W / 2), (0, 2.0), (0, zo0), bev=0.3, seg=1)
    mb.box((-W / 2, W / 2), (0, 2.0), (zo1, H), bev=0.3, seg=1)
    xs_edges = [-W / 2]
    openings = [(x_lc0 + i * pitch_lc - 6.6, x_lc0 + i * pitch_lc + 6.6) for i in range(n_lc)] + [(x_mpo0 + i * pitch_mpo - 8.0, x_mpo0 + i * pitch_mpo + 8.0) for i in range(n_mpo)]
    cur = -W / 2
    for (a, b) in openings:
        if a > cur:
            mb.box((cur, a), (0, 2.0), (zo0, zo1), bev=0.2, seg=1)
        cur = b
    mb.box((cur, W / 2), (0, 2.0), (zo0, zo1), bev=0.2, seg=1)
    mb.box((-W / 2 - 17.5, -W / 2), (0, 2.0), (0, H), bev=0.3, seg=1)     # rack ears
    mb.box((W / 2, W / 2 + 17.5), (0, 2.0), (0, H), bev=0.3, seg=1)
    I.place(mb.build("patch_panel_plate", M["painted_black"], smooth_deg=35), subp, gpp)
    mb = MB()
    for i in range(n_lc):
        cx = x_lc0 + i * pitch_lc
        mb.box((cx - 6.4, cx + 6.4), (-0.5, 24.0), (zo0 + 1.0, zo1 - 1.0), bev=0.3, seg=1)
    I.place(mb.build("patch_panel_lc_adapters_body", M["plastic_blue"], smooth_deg=35), subp, gpp)
    mb = MB()
    for i in range(n_mpo):
        cx = x_mpo0 + i * pitch_mpo
        mb.box((cx - 7.8, cx + 7.8), (-0.5, 28.0), (zo0 + 0.5, zo1 - 0.5), bev=0.3, seg=1)
    I.place(mb.build("patch_panel_mpo_adapters_body", M["plastic_aqua"], smooth_deg=35), subp, gpp)
    # dark port recesses (front openings)
    mb = MB()
    for i in range(n_lc):
        cx = x_lc0 + i * pitch_lc
        mb.box((cx - 5.2, cx + 5.2), (-0.7, -0.4), (zo0 + 3.0, zo1 - 3.0))
    for i in range(n_mpo):
        cx = x_mpo0 + i * pitch_mpo
        mb.box((cx - 6.4, cx + 6.4), (-0.7, -0.4), (zo0 + 2.2, zo1 - 2.2))
    I.place(mb.build("patch_panel_port_openings", M["dark_void"], smooth_deg=0), subp, gpp)
    # label strip + plugged LC duplex cord on port 3 and MPO on port 2 (linked duplicates of the connector plugs)
    mb = MB()
    mb.box((-W / 2 + 10, -W / 2 + 120), (-0.1, 0.0), (3.0, 9.0))
    I.place(mb.build("patch_panel_label_strip", M["label_white"], smooth_deg=0), subp, gpp)
    gl = FC.lc_duplex(M, "panel_plug_lc3", False, "plastic_blue", subp, gpp, x_lc0 + 2 * pitch_lc)
    gl.location = _v(x_lc0 * 0.001 + 2 * pitch_lc * 0.001, -0.040, 0.0225)
    gl.scale = (0.0, 0.0, 0.0) if False else (1, 1, 1)
    gm = FC.mpo_plug(M, "panel_plug_mpo2", 12, False, False, "plastic_aqua", subp, gpp, x_mpo0 + pitch_mpo)
    gm.location = _v((x_mpo0 + pitch_mpo) * 0.001, -0.040, 0.0225)
    C.hook("panel_port_lc_1", subp, gpp, loc=_v(x_lc0, 0, 22.0))
    C.hook("panel_port_mpo_1", subp, gpp, loc=_v(x_mpo0, 0, 22.0))
    meta = dict(
        description="Fibre cables and trays: yellow SM duplex-LC patch cords (2.0 and 3.0 mm jackets), trunk cable end sections with correct fibre counts and ODs (8F, 144F, 288F, 864F, 3456F, 6912F), cable tie, velcro strap, splice tray with loops/protectors, 1U patch panel with 12 LC duplex and 6 MPO adapters.",
        sources=[
            {"what": "Cable ODs 4.0 / 9.6 / 12.1 / 11.4 / 23.5 / 29.0 mm for 8F / 144F / 288F / 864F / 3456F / 6912F; 9,072 fibres per rack (hypothetical, own estimate)", "path": "20260612_fiber_bundle/ctx-summary.md", "accessed": "2026-10-02"},
            {"what": "EIA-310 19-inch panel 482.6 mm, 1U = 44.45 mm (panel 43.66 mm)", "url": "n/a (standard)", "accessed": "2026-10-02"},
        ],
        dimension_table=[dict(item="trunk %s OD / fibres in the model" % t["label"], value="%s / %s" % (t["od"], t["fibres"]), unit="mm / count", source="fiber_bundle ctx-summary.md", accuracy="B") for t in trunk_info]
        + [dict(item="fibre radius used (coated 250 um or 200 um where needed to fit)", value=[t["fibre_radius_mm"] for t in trunk_info], unit="mm", source="packing check, hex lattice inside the core wrap", accuracy="C"),
           dict(item="jacket wall thickness / core wrap", value="0.8-2.0 / 0.25", unit="mm", source="estimate", accuracy="C"),
           dict(item="patch cord jacket", value="2.0 and 3.0", unit="mm", source="task spec", accuracy="B"), dict(item="cord length", value=L, unit="mm", source="model parameter", accuracy="C"),
           dict(item="patch panel 19 in x 1U", value="482.6 x 43.66", unit="mm", source="EIA-310", accuracy="A"), dict(item="LC adapter pitch 14.4, MPO pitch 34", value=14.4, unit="mm", source="estimate", accuracy="C"),
           dict(item="splice tray 160 x 110 x 12 mm", value=160, unit="mm", source="estimate", accuracy="C")],
        trunk_packing=trunk_info,
        rollable_ribbon_note="Cross-sections show the correct fibre count packed hex inside a core wrap; rollable ribbon folding is not represented (a flat fibre array is shown), colour code 12 colours repeating.",
        hooks={"HOOK_cord_<2p0mm|3p0mm>_end_a/_end_b": "cord ends (plug mating faces)", "HOOK_panel_port_lc_1 / HOOK_panel_port_mpo_1": "first LC and MPO port faces", "HOOK_splice_tray_lid": "lid centre (lid object splice_tray_lid is hidden for render by default)"},
        custom_properties={},
        layout="trunk sections along X at y=0 (60 mm slices, fibres exposed 3 mm at the end face); patch cords at y=300; ties/velcro at y=-150; splice tray at x=320; patch panel at y=700 (front toward -Y).",
        origin="ROOT at the origin; groups carry offsets.",
        simplifications=["trunk interiors show one end face and a 60 mm jacket stub only", "fibres are straight 250 um discs, no rollable-ribbon folds, no loose-tube separation", "panel cassettes not modeled: adapters are blocks with openings", "cord is a fixed S-curve, no slack control (cord length parameter in the build script)"],
        intended_usage="S6 fibre mess context (trunk size gag from 9,072 fibres), tray and panel props.",
    )
    meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
    meta["triangles_unique_meshes"] = I.unique_tris(coll)
    blend = os.path.join(OUT, ASSET + ".blend")
    C.finish(ASSET, blend, coll, meta, preview_dir=None)
    return coll


if __name__ == "__main__":
    coll = main()
    prev = os.path.join(OUT, "previews")
    I.preview_views(coll, prev, ASSET, [("trunk_sections", (60, 0, 0), (0, -1, 0.05), 330, 50), ("patch_cords_three_quarter", (20, 500, 0), (0.7, -1.0, 0.7), 750, 40),
                                       ("ties_velcro", (0, -150, 6), (0.7, -1, 0.5), 110, 50), ("splice_tray", (320, 0, 5), (0.5, -0.8, 0.9), 260, 40),
                                       ("patch_panel_front", (0, 700, 22), (0, -1, 0.15), 600, 40), ("patch_panel_closeup", (-130, 700, 22), (0.5, -1, 0.4), 120, 50)])
