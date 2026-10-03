"""Build osfp_cage_and_host_connector.blend: OSFP 1x1 cage (perforated top, EMI spring fingers, compliant pins, light pipes), 60-contact host
connector, belly-to-belly pair, and a 1U faceplate section with 32 OSFP ports (2 stacked rows x 16, populated with closed modules).
Frame: origin = host PCB top surface under the cage forward-stop plane (module datum B), +Y toward the connector, -Y front (module enters from -Y).
A module plugged in has its ROOT at (x, 0, 1.0 mm) (module bottom rests 1.0 mm above the PCB: 'effective floor height' in the spec).
Run: Blender -b --python scripts/assets/interconnect/build_osfp_cage_and_host_connector.py -- assets/components/interconnect
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
import ic_common as I
import ic_modules as MOD
from ic_common import MB, _v

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "osfp_cage_and_host_connector"

CW, CH = 23.48, 14.65           # external width (w/o fingers), height above PCB
Y_F, Y_R = -69.8, 21.2          # cage front / rear (length 91.0)
HOLD = 0.25                     # sheet thickness (estimate)


def finger(mb, c, size, axis, ang_deg):
    verts = mb.box((c[0] - size[0] / 2, c[0] + size[0] / 2), (c[1] - size[1] / 2, c[1] + size[1] / 2), (c[2] - size[2] / 2, c[2] + size[2] / 2))
    bmesh.ops.rotate(mb.bm, verts=verts, cent=Vector(_v(*c)), matrix=Matrix.Rotation(math.radians(ang_deg), 3, axis))


def build_cage(M, coll, parent, name):
    hw = CW / 2
    mb = MB()
    # side walls
    for s in (-1, 1):
        mb.box(((hw - HOLD, hw) if s > 0 else (-hw, -hw + HOLD)), (Y_F, Y_R), (0.2, CH), bev=0.08, seg=1)
    # top plate: frame + lattice of vents in y -60..10
    zt = (CH - HOLD, CH)
    mb.box((-hw, hw), (Y_F, -60.0), zt, bev=0.05, seg=1)
    mb.box((-hw, hw), (10.0, Y_R), zt, bev=0.05, seg=1)
    for k in range(15):
        yc = -60.0 + k * (70.0 / 14)
        mb.box((-hw + HOLD, hw - HOLD), (yc - 0.8, yc + 0.8), zt, bev=0.0)
    for j in range(9):
        xc = -10.4 + j * 2.6
        mb.box((xc - 0.45, xc + 0.45), (-60.0, 10.0), zt, bev=0.0)
    mb.box((-hw, hw), (Y_R - HOLD, Y_R), (0.2, CH), bev=0.05, seg=1)           # rear wall
    sh = mb.build(name + "_shell", M["nickel"], smooth_deg=30)
    I.place(sh, coll, parent)
    # EMI spring fingers around the front opening (pitch about 1.85 mm, 9 mm long, bent inward 12 deg / outward tips)
    fm = MB()
    for k in range(12):
        xc = -hw + 1.3 + k * (CW - 2.6) / 11
        finger(fm, (xc, Y_F + 4.0, CH + 0.3), (1.2, 9.0, 0.12), "X", -10)
        finger(fm, (xc, Y_F + 4.0, 0.3), (1.2, 9.0, 0.12), "X", 10)
    for k in range(6):
        zc = 1.5 + k * (CH - 3.0) / 5
        for s in (-1, 1):
            finger(fm, (s * (hw + 0.3), Y_F + 4.0, zc), (0.12, 9.0, 1.2), "Z", 10 * s)
    fing = fm.build(name + "_emi_fingers", M["steel"], smooth_deg=0)
    I.place(fing, coll, parent)
    # compliant pins (eye of needle) along the lower side edges
    pm = MB()
    ys = [Y_F + 15.5 + 4.0 * i for i in range(16)] + [Y_R - 2.0 * i for i in range(7)]
    for s in (-1, 1):
        for y in ys:
            pm.box((s * (hw - HOLD / 2) - 0.4, s * (hw - HOLD / 2) + 0.4), (y - 0.3, y + 0.3), (-1.4, 0.2), bev=0.0)
    I.place(pm.build(name + "_compliant_pins", M["tinned_copper"], smooth_deg=0), coll, parent)
    # light pipes (two) on the front top
    lp = MB()
    for s in (-1, 1):
        lp.cyl((s * 6.0, Y_F + 8.0, CH + 0.9), 0.8, 14.0, axis="Y", segs=12)
    I.place(lp.build(name + "_light_pipes", M["lightpipe"], smooth_deg=30), coll, parent)
    tip = MB()
    for s in (-1, 1):
        tip.cyl((s * 6.0, Y_F + 0.9, CH + 0.9), 0.82, 0.3, axis="Y", segs=12)
    I.place(tip.build(name + "_light_pipe_tips", M["led_green"], smooth_deg=0), coll, parent)
    # host connector (60 contacts): body 22.48 x 8.96 x 8.45, slot, gold contacts, SMT tails
    cy0, cy1 = 1.6, 10.56
    cb = MB()
    cb.box((-11.24, 11.24), (cy0, cy1), (1.68, 10.13), bev=0.25, seg=1)
    conn = cb.build(name + "_host_connector", M["plastic_black"], smooth_deg=35)
    I.place(conn, coll, parent)
    sl = MB()
    sl.box((-9.6, 9.6), (cy0 - 0.05, cy0 + 0.3), (5.15, 6.65), bev=0.0)
    I.place(sl.build(name + "_connector_slot", M["dark_void"], smooth_deg=0), coll, parent)
    ct = MB()
    for i in range(30):
        x = (i - 14.5) * 0.6
        ct.box((x - 0.11, x + 0.11), (cy0 - 0.06, cy0 + 1.2), (6.0, 6.08))
        ct.box((x - 0.11, x + 0.11), (cy0 - 0.06, cy0 + 1.2), (5.72, 5.8))
    I.place(ct.build(name + "_contacts", M["gold"], smooth_deg=0), coll, parent)
    tl = MB()
    for s in (-1, 1):
        tl.box((s * 9.0 - 0.6, s * 9.0 + 0.6), (cy0 + 1.0, cy1 - 1.0), (0.0, 1.68))
    I.place(tl.build(name + "_smt_tails", M["tinned_copper"], smooth_deg=0), coll, parent)
    return [sh, fing, conn]


def main():
    C.reset()
    M = I.Mats()
    coll, root = C.new_asset(ASSET, accuracy="A")
    # ---- variant 1: single cage on a host PCB
    v1 = C.sub_collection(coll, "VARIANT_cage_1x1")
    g1 = I.empty("cage_1x1_group", v1, root, loc_mm=(0, 0, 0))
    build_cage(M, v1, g1, "osfp_cage")
    mb = MB()
    mb.box((-20, 20), (-30, 40), (-1.6, 0.0), bev=0.1, seg=1)
    I.place(mb.build("osfp_cage_host_pcb", M["fr4_green"], smooth_deg=30), v1, g1)
    C.hook("module_insert_point", v1, g1, loc=_v(0, 0, 1.0))
    C.hook("cage_front_center", v1, g1, loc=_v(0, Y_F, 8.0))
    # ---- variant 2: belly-to-belly (two cages on opposite PCB sides)
    v2 = C.sub_collection(coll, "VARIANT_belly_to_belly")
    g2 = I.empty("belly_group", v2, root, loc_mm=(60, 0, 0))
    mb = MB()
    mb.box((-20, 20), (-30, 40), (-1.6, 0.0), bev=0.1, seg=1)
    I.place(mb.build("osfp_belly_host_pcb", M["fr4_green"], smooth_deg=30), v2, g2)
    gt = I.empty("belly_top_cage", v2, g2)
    build_cage(M, v2, gt, "osfp_belly_top")
    gb = I.empty("belly_bottom_cage", v2, g2, loc_mm=(0, 0, -1.6), rot=(math.pi, 0, 0))
    build_cage(M, v2, gb, "osfp_belly_bottom")
    C.hook("belly_module_top", v2, g2, loc=_v(0, 0, 1.0))
    C.hook("belly_module_bottom", v2, g2, loc=_v(0, 0, -2.6), rot=(math.pi, 0, 0))
    # ---- variant 3: 1U faceplate, 2 rows x 16 ports (stacked cages, 14.9 mm vertical pitch, 23.23 mm horizontal pitch)
    v3 = C.sub_collection(coll, "VARIANT_faceplate_1u_2x16")
    g3 = I.empty("faceplate_group", v3, root, loc_mm=(0, 250, 0))
    src = C.sub_collection(v3, "LOD_source")
    gs = I.empty("faceplate_cage_source", src, g3)
    cage_objs = build_cage(M, src, gs, "osfp_fp_cage")
    for o in src.objects:
        o.hide_render = True
        o.hide_viewport = True
    # module source (closed shells only, from the OSFP module builder, hidden)
    msrc = C.sub_collection(v3, "LOD_module_source")
    mroot = I.empty("faceplate_module_source", msrc, g3)
    MOD.build_osfp(msrc, mroot, M)
    keep = ["shell_bottom", "shell_top", "pull_tab", "latch_rails", "receptacle", "tongue", "pcb", "gold_fingers"]
    mod_objs = [o for o in msrc.objects if o.type == "MESH" and any(o.name.endswith(k) for k in keep)]
    for o in msrc.objects:
        o.hide_render = True
        o.hide_viewport = True
    pitch_x, pitch_z, z0 = 23.23, 14.9, 0.0
    ncol = 16
    pcb_z = 0.0
    mb = MB()
    mb.box((-ncol * pitch_x / 2 - 8, ncol * pitch_x / 2 + 8), (-30, 40), (-1.6, 0.0), bev=0.1, seg=1)
    I.place(mb.build("faceplate_host_pcb", M["fr4_green"], smooth_deg=30), v3, g3)
    # panel: 482.6 x 43.66 x 2 with one ganged cutout 24.28 + 15 x 23.23 wide, height 14.7 + 14.9 (derived)
    W, H = 482.6, 43.66
    cw = 24.28 + (ncol - 1) * pitch_x
    ch = 14.70 + pitch_z
    zc0 = (CH + pitch_z) / 2 + 0.0
    zlo, zhi = zc0 - ch / 2, zc0 + ch / 2
    zb0 = zc0 - H / 2
    pm = MB()
    pm.box((-W / 2, W / 2), (Y_F - 2.0, Y_F), (zb0, zlo), bev=0.2, seg=1)
    pm.box((-W / 2, W / 2), (Y_F - 2.0, Y_F), (zhi, zb0 + H), bev=0.2, seg=1)
    pm.box((-W / 2, -cw / 2), (Y_F - 2.0, Y_F), (zlo, zhi), bev=0.2, seg=1)
    pm.box((cw / 2, W / 2), (Y_F - 2.0, Y_F), (zlo, zhi), bev=0.2, seg=1)
    pm.box((-W / 2 - 17.5, -W / 2), (Y_F - 2.0, Y_F), (zb0, zb0 + H), bev=0.2, seg=1)
    pm.box((W / 2, W / 2 + 17.5), (Y_F - 2.0, Y_F), (zb0, zb0 + H), bev=0.2, seg=1)
    I.place(pm.build("faceplate_panel", M["painted_grey"], smooth_deg=35), v3, g3)
    n = 0
    for row in range(2):
        for col in range(ncol):
            x = (col - (ncol - 1) / 2) * pitch_x
            z = row * pitch_z
            e = I.empty("port_r%d_c%02d" % (row, col), v3, g3, loc_mm=(x, 0, z))
            for o in cage_objs:
                I.link_dup(o, v3, e, (0, 0, 0), name="fp_r%d_c%02d_%s" % (row, col, o.name))
            em = I.empty("module_r%d_c%02d" % (row, col), v3, e, loc_mm=(0, 0, 1.0))
            for o in mod_objs:
                I.link_dup(o, v3, em, (0, 0, 0), name="fp_mod_r%d_c%02d_%s" % (row, col, o.name))
            n += 1
    C.hook("port_r0_c00", v3, g3, loc=_v(-(ncol - 1) / 2 * pitch_x, Y_F, 7.0))
    C.hook("port_r1_c15", v3, g3, loc=_v((ncol - 1) / 2 * pitch_x, Y_F, pitch_z + 7.0))
    C.hook("faceplate_center", v3, g3, loc=_v(0, Y_F - 1.0, zc0))
    meta = dict(
        description="OSFP 1x1 cage with perforated top, EMI spring fingers, compliant pins, light pipes; 60-contact SMT host connector; belly-to-belly pair; 1U faceplate section with 32 OSFP ports (2 stacked rows x 16) populated with closed modules.",
        sources=[
            {"what": "OSFP MSA Rev 5.22: cage dimensions (Fig 5-3/5-4/5-5), bezel cut-out (Fig 5-23/5-24), SMT connector (Fig 5-25), stacked cage 14.9 mm pitch (Sec 7.2)", "url": "https://www.osfpmsa.org/assets/pdf/OSFP_Module_Specification_Rev5_22.pdf", "accessed": "2026-10-02"},
        ],
        dimension_table=[
            dict(item="cage external width (w/o EMI fingers) / internal width", value="23.48 / 22.88", unit="mm", source="OSFP MSA Rev 5.22 Fig 5-4", accuracy="A"),
            dict(item="cage height above PCB / internal height", value="14.65 / 13.30", unit="mm", source="Fig 5-5 / 5-4", accuracy="A"),
            dict(item="cage length", value=91.0, unit="mm", source="Fig 5-5 (91.0 in parentheses = reference)", accuracy="B"),
            dict(item="effective cage floor height above PCB", value=1.0, unit="mm", source="Fig 5-5", accuracy="A"),
            dict(item="compliant pin pitch", value="4.00 (15 x) and 2.00 (7 x)", unit="mm", source="Fig 5-3", accuracy="A"),
            dict(item="host connector body", value="22.48 x 8.96 x 8.45", unit="mm", source="Fig 5-25 (max / +-0.20)", accuracy="A"),
            dict(item="connector contacts / pitch", value="60 / 0.60", unit="count / mm", source="Fig 5-25", accuracy="A"),
            dict(item="cage horizontal pitch / stacked vertical pitch", value="23.23 / 14.9", unit="mm", source="Fig 5-23 text / Sec 7.2", accuracy="A"),
            dict(item="bezel cut-out 1x1 24.28 x 14.70; 1x4 93.97 (=24.28 + 3 x 23.23)", value="24.28 x 14.70", unit="mm", source="Fig 5-23/5-24", accuracy="A"),
            dict(item="ganged 16-port cut-out 372.7 mm and 2-row height 29.6 mm", value="372.7 x 29.6", unit="mm", source="derived from 24.28 + 15 x 23.23; 14.70 + 14.9 (own arithmetic)", accuracy="B"),
            dict(item="sheet thickness 0.25, vent lattice, finger length 9, light pipe size", value=0.25, unit="mm", source="estimates", accuracy="C"),
            dict(item="1U panel 482.6 x 43.66", value=482.6, unit="mm", source="EIA-310", accuracy="A"),
        ],
        hooks={"HOOK_module_insert_point": "module ROOT location when fully plugged in the 1x1 cage (0,0,1.0 mm); insert along +Y from the front (-Y)", "HOOK_cage_front_center": "cage front opening centre",
               "HOOK_belly_module_top / _bottom": "plug points for the belly-to-belly pair (bottom is inverted: rotate module 180 deg about Y)", "HOOK_port_r0_c00 / HOOK_port_r1_c15": "faceplate port corner positions (front opening centre)", "HOOK_faceplate_center": "panel centre"},
        custom_properties={},
        layout="cage_1x1 at x=0; belly_to_belly at x=+60 mm; faceplate section at y=+250 mm (32 populated ports, module closed shells are linked duplicates of osfp_module geometry built by ic_modules.build_osfp). Source collections LOD_source / LOD_module_source are hidden.",
        stagger_interpretation="'2 x 16 stagger' is built as two stacked rows of 16 (stacked 2x1 cages at the spec's 14.9 mm pitch), not laterally offset rows; each cage is an individual shell (ganged walls overlap by 0.25 mm since pitch 23.23 < width 23.48).",
        origin="ROOT at the PCB top surface under the cage forward-stop plane; faceplate at y = +250 mm.",
        simplifications=["no top latch flap or ground tabs; no spring-finger contact on the bezel", "stacked cages share no walls and the upper connector is a copy (real stacked connectors are one body)", "host connector has no internal contact geometry beyond 2 x 30 gold tips", "modules in the faceplate are the module's closed shells (no internals)", "vent lattice is bars, not punched slots"],
        intended_usage="S2/S3 faceplate context; assembler can plug modules at HOOK_module_insert_point.",
    )
    meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
    meta["triangles_unique_meshes"] = I.unique_tris(coll)
    blend = os.path.join(OUT, ASSET + ".blend")
    C.finish(ASSET, blend, coll, meta, preview_dir=None)
    return coll


if __name__ == "__main__":
    coll = main()
    prev = os.path.join(OUT, "previews")
    I.preview_views(coll, prev, ASSET, [("cage_three_quarter", (0, -20, 8), (0.8, -0.9, 0.7), 190, 50), ("cage_front_closeup", (0, -69.8, 8), (0.3, -1, 0.2), 60, 50),
                                       ("connector_closeup", (0, 5, 6), (0.3, 1.0, 0.5), 45, 60), ("belly_three_quarter", (60, -10, 0), (0.8, -0.9, 0.5), 200, 50),
                                       ("faceplate_front", (0, 180, 22), (0, -1, 0.1), 650, 40), ("faceplate_three_quarter", (-100, 180, 22), (0.6, -1, 0.5), 330, 40)])
