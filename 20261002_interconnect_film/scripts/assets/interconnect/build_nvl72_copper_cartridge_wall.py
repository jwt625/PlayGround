"""Build nvl72_copper_cartridge_wall.blend: NVL72-style rear copper NVLink spine.

Contains (one ASSET collection, sub-collections):
  VARIANT_cartridge_detail : one cartridge, 18 connector bays, 1296 individual cables (generator: cartridge_cable_paths)
  VARIANT_real_4           : 4 cartridges as linked duplicates (2 columns x 2 stacked) = 5184 cables (hidden by default)
  VARIANT_wall_12x5        : crude-film layout, 12 x 5 low-LOD cartridges (hidden by default)
  VARIANT_bundle_14        : 14 loose twinax-style cables between HOOK_bundle_a and HOOK_bundle_b (curve based, parametric)
Run: Blender -b --python scripts/assets/interconnect/build_nvl72_copper_cartridge_wall.py -- assets/components/interconnect
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import ic_common as I
from ic_common import MB, _v, mm

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "nvl72_copper_cartridge_wall"

# ---- cartridge dimensions (mm). NONE of these is verified from a drawing: estimates (level C)
CW, CD, CH = 160.0, 90.0, 800.0        # width, depth, height (9 x 2RU = 9 x 88.9 = 800 mm from the '9x2RU' listing title)
BAYS = 9
BAY_PITCH = 88.9                        # 2RU
PINS_COLS, PINS_ROWS = 12, 12           # 144 contacts per bay block (estimate; 1296 cables per cartridge = 5184 / 4)
BLOCK_W, BLOCK_H = 62.0, 62.0
COL_X = (-40.0, 40.0)                   # left (compute-side) and right (switch-side) block columns
CABLE_R = 1.4                           # mm, individual cable radius (estimate)


def bay_z(k):
    return 44.45 + k * BAY_PITCH


def cartridge_cable_paths(rng, n_pts=18):
    """(1296, n_pts, 3) cable paths in metres. Left bay a -> right bay b: a group of 16 cables for every (a, b) pair (all-to-all 9 x 9).
    Each group travels as a 4 x 4 bundle through its own lane (x lane by b, depth lane by a)."""
    paths = []
    pc = np.arange(PINS_COLS) - (PINS_COLS - 1) / 2
    pr = np.arange(PINS_ROWS) - (PINS_ROWS - 1) / 2
    pin_xy = [(c * 4.8, r * 4.8) for r in pr for c in pc]    # 144 pin offsets in the block plane (x, z)
    for a in range(BAYS):
        order = list(range(144))
        rng.shuffle(order)
        order_r = list(range(144))
        rng.shuffle(order_r)
        for b in range(BAYS):
            for t in range(16):
                lx, lz = pin_xy[order[(b * 16 + t) % 144]]
                rx, rz = pin_xy[order_r[(a * 16 + t) % 144]]
                xl, zl = COL_X[0] + lx, bay_z(a) + lz
                xr, zr = COL_X[1] + rx, bay_z(b) + rz
                ox, oy = ((t % 4) - 1.5) * 3.1, ((t // 4) - 1.5) * 3.1
                xm = (b - 4) * 8.0 + ox
                ym = -6.0 - 4.4 * a + oy
                zl_c, zr_c = bay_z(a), bay_z(b)
                zm = (zl_c + zr_c) / 2
                pts = [(xl, 52.0, zl), (xl, 24.0, zl), (COL_X[0] * 0.6 + ox * 0.6, 4.0 + oy, zl_c + 0.25 * (zr_c - zl_c)),
                       (xm, ym, zm), (COL_X[1] * 0.6 + ox * 0.6, 4.0 + oy, zl_c + 0.75 * (zr_c - zl_c)), (xr, 24.0, zr), (xr, 52.0, zr)]
                paths.append(I.catmull(np.array(pts) * 0.001, per_seg=3))
    out = []
    for p in paths:
        s_ = np.linspace(0, len(p) - 1, n_pts)
        i0 = np.floor(s_).astype(int)
        i1 = np.minimum(i0 + 1, len(p) - 1)
        f = (s_ - i0)[:, None]
        out.append(p[i0] * (1 - f) + p[i1] * f)
    return np.array(out)


def build_cartridge_detail(M, coll, parent, name="cartridge"):
    rng = np.random.default_rng(72)
    import random
    r = random.Random(72)
    objs = []
    # frame (sheet metal)
    mb = MB()
    for s in (-1, 1):
        mb.box((s * (CW / 2 - 2.0) - 2.0, s * (CW / 2 - 2.0) + 2.0), (-CD / 2, CD / 2), (0, CH), bev=0.6, seg=1)
    mb.box((-CW / 2, CW / 2), (-CD / 2, CD / 2), (0, 6.0), bev=0.8, seg=1)
    mb.box((-CW / 2, CW / 2), (-CD / 2, CD / 2), (CH - 6.0, CH), bev=0.8, seg=1)
    for k in range(BAYS + 1):
        zc = k * BAY_PITCH
        mb.box((-CW / 2 + 2, CW / 2 - 2), (CD / 2 - 6.0, CD / 2), (min(max(zc - 2.0, 6.0), CH - 6.0), min(max(zc + 2.0, 8.0), CH - 4.0)), bev=0.4, seg=1)
    for zc in (CH * 0.25, CH * 0.5, CH * 0.75):
        mb.box((-CW / 2 + 2, CW / 2 - 2), (-CD / 2, -CD / 2 + 3.0), (zc - 4, zc + 4), bev=0.5, seg=1)    # rear cross straps
    objs.append(mb.build(name + "_frame", M["nvl_cartridge"], smooth_deg=35))
    # connector blocks on the +Y face, 18 (9 bays x 2 columns)
    mb = MB()
    for k in range(BAYS):
        for cx in COL_X:
            mb.box((cx - BLOCK_W / 2, cx + BLOCK_W / 2), (CD / 2 - 6.0, CD / 2 + 14.0), (bay_z(k) - BLOCK_H / 2, bay_z(k) + BLOCK_H / 2), bev=1.2, seg=1)
    objs.append(mb.build(name + "_connector_blocks", M["nvl_connector"], smooth_deg=35))
    # contact cavities (dark recesses on the block faces): 144 per block
    items = []
    pc = np.arange(PINS_COLS) - (PINS_COLS - 1) / 2
    for k in range(BAYS):
        for cx in COL_X:
            for rr in range(PINS_ROWS):
                for cc in range(PINS_COLS):
                    items.append((cx + pc[cc] * 4.8, CD / 2 + 14.0, bay_z(k) + (rr - (PINS_ROWS - 1) / 2) * 4.8, 2.4, 0.3, 2.4))
    mb = MB()
    mb.boxes(items)
    objs.append(mb.build(name + "_contacts", M["gold"], smooth_deg=0))
    # fabric sleeves where the cables leave each block
    mb = MB()
    for k in range(BAYS):
        for cx in COL_X:
            mb.cyl((cx, CD / 2 - 14.0, bay_z(k)), 32.0, 16.0, axis="Y", segs=16)
    objs.append(mb.build(name + "_sleeves", M["nvl_sleeve"], smooth_deg=30))
    # cables (3 sides, 1296 paths)
    P = cartridge_cable_paths(r)
    objs.append(I.tube_mesh(name + "_cables", P, CABLE_R * 0.001, sides=3, mat=M.named("nvl_cable", (0.06, 0.06, 0.07), 0.0, 0.7)))
    for o in objs:
        I.place(o, coll, parent)
    return objs


def build_cartridge_low(M, coll, parent, name="cartridge_low"):
    objs = []
    mb = MB()
    mb.box((-CW / 2, CW / 2), (-CD / 2, CD / 2), (0, CH), bev=1.5, seg=1)
    objs.append(mb.build(name + "_body", M["painted_black"], smooth_deg=35))
    mb = MB()
    for k in range(BAYS):
        for cx in COL_X:
            mb.box((cx - BLOCK_W / 2, cx + BLOCK_W / 2), (CD / 2, CD / 2 + 14.0), (bay_z(k) - BLOCK_H / 2, bay_z(k) + BLOCK_H / 2), bev=1.0, seg=1)
    objs.append(mb.build(name + "_blocks", M["nvl_connector"], smooth_deg=35))
    # rear: three sleeved cable bundles per column running vertically (visible from the viewer side, -Y)
    mb = MB()
    for cx in COL_X:
        for dx in (-22.0, 0.0, 22.0):
            mb.cyl((cx * 0.55 + dx, -CD / 2 + 14.0, CH / 2), 15.0, CH - 20.0, axis="Z", segs=10)
    objs.append(mb.build(name + "_bundles", M["nvl_sleeve"], smooth_deg=30))
    for o in objs:
        I.place(o, coll, parent)
    return objs


def make_bundle_14(M, coll, root, n=14, od_mm=7.5, length_m=2.0, sag_m=0.12, stub_mm=80.0):
    """14 loose cables between two hook empties; each cable is a Bezier curve (NURBS order 4, 9 control points) whose points are driven by the hooks."""
    cu_coll = C.sub_collection(coll, "VARIANT_bundle_14")
    ha = bpy.data.objects.new("HOOK_bundle_a", None)
    hb = bpy.data.objects.new("HOOK_bundle_b", None)
    hm = bpy.data.objects.new("HOOK_bundle_mid", None)
    for h, loc in ((ha, (0.0, 0.0, 1.0)), (hb, (length_m, 0.0, 1.0)), (hm, (length_m / 2, 0.0, 1.0 - sag_m))):
        h.empty_display_type = "ARROWS"
        h.empty_display_size = 0.04
        cu_coll.objects.link(h)
        h.parent = root
        h.location = loc
    I.add_prop(root, "p_sag_m", sag_m, 0.0, 2.0, "mid-span sag of the 14-cable bundle (m); taut = 0")
    I.add_prop(root, "p_cable_od_mm", od_mm, 1.0, 40.0, "documentation only: cable OD used by the build (bevel depth baked)")
    # mid hook driven: midpoint of a and b minus sag
    for k, comp in enumerate(("X", "Y", "Z")):
        fc = hm.driver_add("location", k)
        d = fc.driver
        d.type = "SCRIPTED"
        for nm, ob in (("a", ha), ("b", hb)):
            v = d.variables.new()
            v.name = nm
            v.type = "TRANSFORMS"
            v.targets[0].id = ob
            v.targets[0].transform_type = "LOC_" + comp
            v.targets[0].transform_space = "LOCAL_SPACE"
        if comp == "Z":
            v = d.variables.new()
            v.name = "s"
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["p_sag_m"]'
            d.expression = "(a+b)/2-s"
        else:
            d.expression = "(a+b)/2"
    # hex-like cross-section layout of 14 cables (centres in mm, in the YZ plane)
    pitch = od_mm * 1.02
    offs = []
    for row, cnt in enumerate((4, 5, 5)):
        for i in range(cnt):
            offs.append(((i - (cnt - 1) / 2) * pitch, (row - 1) * pitch * 0.87))
    offs = offs[:n]
    mats = [M["jacket_black"], M["jacket_grey"]]
    for ci, (oy, oz) in enumerate(offs):
        cu = bpy.data.curves.new("bundle14_cable_%02d" % ci, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = od_mm * 0.0005
        cu.bevel_resolution = 2
        cu.resolution_u = 6
        sp = cu.splines.new("NURBS")
        npts = 9
        sp.points.add(npts - 1)
        sp.order_u = 4
        sp.use_endpoint_u = True
        ts = [0.0, None, 0.2, 0.35, 0.5, 0.65, 0.8, None, 1.0]
        for i, bp in enumerate(sp.points):
            tt = ts[i]
            xx = stub_mm * 0.001 if i == 1 else (length_m - stub_mm * 0.001 if i == 7 else length_m * tt)
            zz = 0.0 if tt is None else -4 * sag_m * tt * (1 - tt)
            bp.co = (xx, oy * 0.001, 1.0 + oz * 0.001 + zz, 1.0)
        ob = bpy.data.objects.new("bundle14_cable_%02d" % ci, cu)
        cu_coll.objects.link(ob)
        ob.parent = root
        ob.data.materials.append(mats[ci % 2])
        # drivers: point at fraction t along a->b: a + t (b - a) + offset - 4 s t (1-t) (parabolic sag in Z); stubs are straight
        pts_t = [(t, (stub_mm * 0.001 if i == 1 else -stub_mm * 0.001 if i == 7 else 0.0)) for i, t in enumerate([0.0, None, 0.2, 0.35, 0.5, 0.65, 0.8, None, 1.0])]
        for pi, (t, dx) in enumerate(pts_t):
            for k, comp in enumerate(("X", "Y", "Z")):
                fc = cu.driver_add("splines[0].points[%d].co" % pi, k)
                d = fc.driver
                d.type = "SCRIPTED"
                va = d.variables.new(); va.name = "a"; va.type = "TRANSFORMS"
                va.targets[0].id = ha; va.targets[0].transform_type = "LOC_" + comp; va.targets[0].transform_space = "LOCAL_SPACE"
                vb = d.variables.new(); vb.name = "b"; vb.type = "TRANSFORMS"
                vb.targets[0].id = hb; vb.targets[0].transform_type = "LOC_" + comp; vb.targets[0].transform_space = "LOCAL_SPACE"
                vs = d.variables.new(); vs.name = "s"; vs.type = "SINGLE_PROP"
                vs.targets[0].id = root; vs.targets[0].data_path = '["p_sag_m"]'
                off = (0.0, oy * 0.001, oz * 0.001)[k]
                if t is None:
                    base = "a+%r" % dx if dx > 0 else "b+%r" % dx
                    d.expression = "%s+%r" % (base, off)
                else:
                    sag = "-4*s*%r" % (t * (1 - t)) if comp == "Z" else ""
                    d.expression = "a+%r*(b-a)+%r%s" % (t, off, sag)
    # end blocks (plain connector housings) following the hooks
    for hob, sgn in ((ha, 1), (hb, -1)):
        mb = MB()
        mb.box((-30 if sgn > 0 else -5, 5 if sgn > 0 else 30), (-30, 30), (-22, 22), bev=2.0, seg=2)
        blk = mb.build("bundle14_end_block_" + ("a" if sgn > 0 else "b"), M["nvl_connector"], smooth_deg=35)
        C.add(blk, cu_coll, hob)
    return cu_coll, ha, hb, hm


def main():
    C.reset()
    M = I.Mats()
    coll, root = C.new_asset(ASSET, accuracy="C")
    I.add_prop(root, "p_wall_cols", 12, 1, 12, "documentation: crude wall columns")
    # ---- detail cartridge
    sub = C.sub_collection(coll, "VARIANT_cartridge_detail")
    cart = I.empty("cartridge_detail_root", sub, root)
    detail_objs = build_cartridge_detail(M, sub, cart)
    # ---- real_4 layout (2 columns x 2 stacked) of linked duplicates
    sub4 = C.sub_collection(coll, "VARIANT_real_4")
    r4 = I.empty("real4_root", sub4, root)
    cart_w, cart_h = CW + 20.0, CH + 20.0
    gap_mid = 400.0   # switch-tray band (9 x 1RU = 400 mm) between the two stacked groups
    k = 0
    for ci in range(2):
        for ri in range(2):
            e = I.empty("real4_cartridge_%d" % k, sub4, r4, loc_mm=((ci - 0.5) * cart_w, 0, 100.0 + ri * (CH + gap_mid)))
            for o in detail_objs:
                I.link_dup(o, sub4, e, (0, 0, 0), name="real4_%d_%s" % (k, o.name))
            k += 1
    I.hide_collection(sub4, True)
    # ---- crude wall 12 x 5 (low LOD)
    sub_w = C.sub_collection(coll, "VARIANT_wall_12x5")
    wall_root = I.empty("wall_root", sub_w, root, loc_mm=(0, 0, 0))
    low_src = build_cartridge_low(M, sub_w, wall_root, name="cartridge_low_src")
    for o in low_src:
        o.hide_render = True
        o.hide_viewport = True
    for col in range(12):
        for row in range(5):
            e = I.empty("wall_cartridge_c%02d_r%d" % (col, row), sub_w, wall_root, loc_mm=((col - 5.5) * (CW + 6.0), 0, row * (CH + 6.0)))
            for o in low_src:
                I.link_dup(o, sub_w, e, (0, 0, 0), name="wall_c%02d_r%d_%s" % (col, row, o.name))
    I.hide_collection(sub_w, True)
    # ---- 14 loose cables
    cu_coll, ha, hb, hm = make_bundle_14(M, coll, root)
    I.hide_collection(cu_coll, True)
    # hooks on the wall
    wh = C.hook("wall_left", coll, wall_root, loc=_v(-6.5 * (CW + 6.0), -CD, 2000.0))
    wh2 = C.hook("wall_right", coll, wall_root, loc=_v(6.5 * (CW + 6.0), -CD, 2000.0))
    C.hook("cartridge_front_center", coll, cart, loc=_v(0, -CD / 2, CH / 2))
    meta = {
        "description": "NVL72-style rear copper NVLink spine: detail cartridge (1296 individual cables, 18 contact blocks), 4-cartridge real layout, 12x5 crude wall, and a 14-cable loose bundle with end hooks.",
        "sources": [
            {"what": "4 rear NVLink cable cartridges, >5,000 copper cables, ~2 miles (SemiAnalysis 'Nvidia's Optical Boogeyman'; Lenovo Press GB300 NVL72 guide lp2357; ServeTheHome DGX GB200 NVL72 article)", "url": "https://newsletter.semianalysis.com/p/nvidias-optical-boogeyman-nvl72-infiniband", "accessed": "2026-10-02"},
            {"what": "18 compute trays, 9 switch trays, 18 NVLink5 links per GPU (one per NVSwitch ASIC)", "url": "https://lenovopress.lenovo.com/lp2357-lenovo-nvidia-gb300-nvl72-rack-scale-ai", "accessed": "2026-10-02"},
            {"what": "eBay listing title 'NVIDIA GB200 NVL72 NVLink Spine Cartridge Amphenol 9x2RU CBL' (page returned HTTP 403; only the search-result title was seen)", "url": "https://www.ebay.com/itm/277334793017", "accessed": "2026-10-02"},
        ],
        "verified": ["4 cartridges at the rack rear", "more than 5,000 copper cables, about 2 miles (vendor/CEO-level statements)", "18 compute trays and 9 switch trays", "18 NVLink5 links per GPU"],
        "not_verified_estimates": [
            "cartridge size 160 x 90 x 800 mm (800 = 9 x 2RU from a listing title); no drawing seen",
            "arrangement of the 4 cartridges (modeled 2 columns x 2 stacked with a 400 mm switch band between) is a guess",
            "1296 cables per cartridge = 5184 / 4, where 5184 = 72 GPUs x 18 links x 4 differential pairs per link (own arithmetic, assumes one twinax/coax per pair)",
            "18 connector blocks per cartridge, 144 contacts each, 62 x 62 mm; cable radius 1.4 mm; routing is a synthetic all-to-all (16 cables per bay pair)",
            "connector style (blind-mate, Amphenol-like) is generic: no real product geometry",
        ],
        "dimension_table": [
            dict(item="cartridge width", value=CW, unit="mm", source="estimate", accuracy="C"),
            dict(item="cartridge depth", value=CD, unit="mm", source="estimate", accuracy="C"),
            dict(item="cartridge height (9 x 2RU)", value=CH, unit="mm", source="eBay listing title '9x2RU' x 88.9 mm per 2RU (EIA-310 1RU = 44.45 mm)", accuracy="C"),
            dict(item="cables per cartridge / total (4)", value="1296 / 5184", unit="count", source="derived from 72 x 18 x 4 pairs; matches '>5,000 cables'", accuracy="C"),
            dict(item="individual cable radius", value=CABLE_R, unit="mm", source="estimate", accuracy="C"),
            dict(item="loose bundle cable OD", value=7.5, unit="mm", source="estimate for 14 twinax cables (200G/lane DAC class)", accuracy="C"),
        ],
        "origin": "ROOT at the bottom centre of the detail cartridge footprint; detail cartridge spans z 0..0.8 m; -Y is the cable-viewing side (rack rear), +Y the connector (tray) face. Wall layouts are centred on x = 0 with z from 0.",
        "hooks": {
            "HOOK_bundle_a / HOOK_bundle_b": "ends of the 14-cable bundle (children of root, TRANSLATION drives the curves: move them to stretch/slacken; rotation is ignored by the drivers); defaults at x=0 and x=2.0 m, z=1.0 m",
            "HOOK_bundle_mid": "driven (midpoint of a and b minus p_sag_m); do not move manually",
            "HOOK_wall_left / HOOK_wall_right": "camera/pulse start and end points in front of the 12x5 wall (parent wall_root)",
            "HOOK_cartridge_front_center": "front centre of the detail cartridge",
        },
        "custom_properties": {"p_sag_m": "0..2 mid-span sag of the 14-cable bundle in metres (taut = 0)", "p_cable_od_mm": "documentation only", "p_wall_cols": "documentation only"},
        "variants": {"VARIANT_cartridge_detail": "visible by default", "VARIANT_real_4": "hidden: unhide collection (hide_render + hide_viewport)", "VARIANT_wall_12x5": "hidden: low-LOD wall, 60 cartridges at real size = 1.9 x 4.1 m (assembler scales as needed)", "VARIANT_bundle_14": "hidden: loose 14-cable bundle between racks"},
        "cable_bundle_generator": "scripts/assets/interconnect/build_nvl72_copper_cartridge_wall.py: cartridge_cable_paths() builds the 1296 numpy paths in about 1 s; I.tube_mesh() turns any (m, n, 3) path array into one 3-sided-tube mesh (about 100 triangles per cable); change n_pts/sides for cost; make_bundle_14() builds the curve-based hooked bundle (parametric via hooks and p_sag_m).",
        "simplifications": ["no cable sleeves along the runs, only at the block exits", "no alignment pins, keying or latches on the blocks", "frame is a simple lattice"],
        "intended_usage": "S1 hook fly-along (wall, 'about 5,000 copper cables, 2 miles' card) and the 14-cable bundle between two racks at 5 m / 2 m / 1 m.",
    }
    blend = os.path.join(OUT, ASSET + ".blend")
    meta["triangles_unique_meshes"] = I.unique_tris(coll)
    meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
    C.finish(ASSET, blend, coll, meta, preview_dir=None)
    return coll, root, sub, sub4, sub_w, cu_coll, ha, hb


if __name__ == "__main__":
    coll, root, sub, sub4, sub_w, cu_coll, ha, hb = main()
    prev = os.path.join(OUT, "previews")
    # preview 1: detail cartridge (front = viewer side -Y), plus close-up of connector face
    I.preview_views(coll, prev, ASSET, [("cartridge_three_quarter", (0, 0, 400), (0.7, -1.0, 0.35), 1600, 40),
                                       ("cartridge_closeup", (-40, -20, 400), (0.2, -1.0, 0.25), 330, 50)])
    sub4.hide_render = False
    sub.hide_render = True
    I.preview_views(coll, prev, ASSET, [("real4_three_quarter", (0, 0, 1050), (0.9, -1.0, 0.3), 3800, 40)])
    sub4.hide_render = True
    sub_w.hide_render = False
    I.preview_views(coll, prev, ASSET, [("wall_12x5_front", (0, 0, 2000), (0.25, -1.0, 0.1), 7000, 40)])
    sub_w.hide_render = True
    cu_coll.hide_render = False
    bpy.context.view_layer.update()
    I.preview_views(coll, prev, ASSET, [("bundle14_three_quarter", (1000, 0, 950), (0.1, -1.0, 0.15), 2200, 40), ("bundle14_end_closeup", (0, 0, 1000), (0.5, -1.0, 0.4), 160, 50)])
