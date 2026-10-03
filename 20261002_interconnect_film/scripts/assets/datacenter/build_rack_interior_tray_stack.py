"""rack_interior_tray_stack: S2 set. Inside of a rack seen from tray level: backplane with blind-mate connector rows, side walls
with slide rails, N tray boards with 4 connectors each and free space for retimer rows (hooks only; chips come from packaging).

Builds two files in one run: rack_interior_tray_stack.blend (stylized: s=4, pitch 0.7 m, depth 1.6 m, N=8) and
rack_interior_tray_stack_real.blend (s=1, pitch 44.45 mm, depth 0.82 m, N=8).
Overrides: n=12 pitch=0.5 s=4 depth=1.6 variant=stylized|real   (key=value after the output dir; then only that variant is built)

Run: Blender -b --python scripts/assets/datacenter/build_rack_interior_tray_stack.py -- assets/components/datacenter
"""
import math
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_parts as P
import common as C
import dc_common as D
from dc_common import MB
import bpy

OUT, KV = D.parse_args()

VARIANTS = {
    "stylized": dict(aid="rack_interior_tray_stack", s=4.0, pitch=0.7, depth=1.6, n=8),
    "real": dict(aid="rack_interior_tray_stack_real", s=1.0, pitch=0.04445, depth=0.82, n=8),
}


def build(var, par):
    s, pitch, depth, N = par["s"], par["pitch"], par["depth"], par["n"]
    AID = par["aid"]
    C.reset()
    D.reset_mats()
    L = D.lib()
    pcb_trace = D.M("pcb_trace", (0.05, 0.22, 0.12), 0.0, 0.3)
    silk = D.M("pcb_silk", (0.9, 0.9, 0.85), 0.0, 0.6)
    gold = L["gold"]
    coll, root = C.new_asset(AID, accuracy="C")
    root["p_tray_count"] = N
    root["p_pitch"] = pitch
    root["p_detail_scale"] = s
    root["p_tray_depth"] = depth
    cs = {k: C.sub_collection(coll, k) for k in ("STATIC_shell", "TRAYS", "LABELS", "HOOKS")}

    k_ = lambda v: v * 0.001 * s            # feature mm -> m with detail scale
    W = 537.0                                # ORV3 21-inch opening (mm, feature units)
    Wm = k_(W)
    Yb = depth / 2                           # backplane front face y
    z_floor = 2.0                            # tray floor thickness (mm)
    board_top = z_floor + 4.0 + 1.6          # mm (standoff 4, PCB 1.6)
    wall_h = 38.0
    H = (N - 1) * pitch + k_(wall_h) + k_(60)
    xk = [(k - 1.5) * 118.0 for k in range(4)]

    # ---------------- static shell: backplane plate, ribs, side walls, base plate, conduits
    sh = MB()
    bpt = k_(25)
    sh.box((0, Yb + bpt / 2, H / 2 - k_(10) / 2), (Wm + k_(30), bpt, H + k_(10)), 0)
    for dx in (-1, 1):
        sh.box((dx * (Wm / 2 + k_(12)), Yb - k_(6), H / 2), (k_(14), k_(24), H), 1)     # vertical posts at the corners
    # horizontal stiffener ledge under each tray's connector row (on the backplane face)
    for j in range(N):
        zl = j * pitch + k_(board_top + 14 - 20)
        sh.box((0, Yb - k_(2), zl), (Wm - k_(30), k_(4), k_(3)), 1)
    parts_shell = P.Part("backplane", sh, [L["panel"], L["dark_steel"]], bevel=(0.0015 * s, 1))
    wl = MB()
    for dx in (-1, 1):
        wl.box((dx * (Wm / 2 + k_(3)), 0, H / 2), (k_(6), depth + k_(25), H), 0)
    wl.box((0, 0, -k_(3)), (Wm + k_(12), depth + k_(25), k_(6)), 0)
    # slide rails: one per tray per side, at tray-floor height (alu inner members ride on steel outer members)
    ra = MB()
    for j in range(N):
        zc = j * pitch + k_(22)
        for dx in (-1, 1):
            wl.box((dx * (Wm / 2 - k_(2)), -k_(10), zc), (k_(5), depth - k_(100), k_(10)), 1)
            ra.box((dx * (Wm / 2 - k_(7.5)), -k_(10), zc), (k_(5), depth - k_(130), k_(24)), 0)
    parts = [parts_shell,
             P.Part("side_walls", wl, [L["dark_steel"], L["steel"]], bevel=(0.0008 * s, 1)),
             P.Part("slide_rails", ra, [L["alu"]], bevel=(0.0006 * s, 1))]
    # vertical cable conduits at the backplane edges with cable bundles inside (cable routing)
    cd = MB()
    cb = MB()
    rnd = random.Random(7)
    for dx in (-1, 1):
        x0 = dx * (Wm / 2 - k_(22))
        cd.box((x0 - dx * k_(15), Yb - k_(14), H / 2), (k_(2), k_(26), H - k_(40)), 0)
        cd.box((x0 + dx * k_(15), Yb - k_(14), H / 2), (k_(2), k_(26), H - k_(40)), 0)
        for q in range(6):
            cb.tube([(x0 + (q - 2.5) * k_(4.2), Yb - k_(10), k_(30)), (x0 + (q - 2.5) * k_(4.2), Yb - k_(10), H - k_(30))],
                    k_(2.0), 8, q % 2)
        for j in range(N):
            zc = j * pitch + k_(board_top + 14)
            cd.box((x0, Yb - k_(22), zc + k_(16)), (k_(34), k_(3), k_(3)), 0)    # cable-tie bars across the conduit
    parts.append(P.Part("cable_conduits", cd, [L["steel"]], bevel=(0.0005 * s, 1)))
    parts.append(P.Part("cable_bundles", cb, [L["black"], L["grey"]], smooth=True))
    P.instantiate(parts, coll, root, AID, subcolls=None)
    for o in [o for o in bpy.data.objects if o.name.startswith(AID + "_") and o.parent == root]:
        C.add(o, cs["STATIC_shell"], root)

    # ---------------- tray unit (identical for all trays): pan, PCB, traces, silk, connectors, receptacles, components
    tp = []
    pan = MB()
    pan.box((0, 0, k_(z_floor) / 2), (k_(526) - k_(4), depth - k_(60), k_(z_floor)), 0)
    for dx in (-1, 1):
        pan.box((dx * (k_(526) / 2 - k_(3)), 0, k_(wall_h) / 2), (k_(1.5), depth - k_(60), k_(wall_h)), 0)
    tp.append(P.Part("pan", pan, [L["steel"]], bevel=(0.0004 * s, 1)))
    pcb = MB()
    zb = k_(z_floor + 4.0 + 0.8)
    pcb.box((0, 0, zb), (k_(500), depth - k_(100), k_(1.6)), 0)
    for dx in (-200, 0, 200):
        for dy in (-0.35, 0.0, 0.35):
            pcb.box((dx * 0.001 * s, dy * depth, k_(z_floor + 2.0)), (k_(6), k_(6), k_(4.0)), 1)
    tp.append(P.Part("board", pcb, [L["pcb"], L["dark_steel"]]))
    zt = k_(board_top)                                     # board top surface z (rel. tray floor bottom)
    # traces: 8 differential-pair-ish lines per connector heading toward -y (front) to row 2 and beyond
    tr = MB()
    y_conn_front = Yb - k_(46)
    for kk, x in enumerate(xk):
        for q in range(8):
            xx = k_(x + (q - 3.5) * 5.5)
            tr.box((xx, (y_conn_front - (Yb - k_(300))) / 2 + Yb - k_(300) - k_(0) + 0.0, zt + k_(0.06)),
                   (k_(0.9), (y_conn_front - (Yb - k_(300))), k_(0.12)), 0)
    tp.append(P.Part("traces", tr, [pcb_trace]))
    # silkscreen outlines at the retimer positions + tray-edge marks
    sk = MB()
    row_y = lambda r: Yb - k_(30 + 14 + 22 + r * 44)       # centres: housing front at Yb-46 mm, row0 22 mm clear
    for r in range(3):
        for kk, x in enumerate(xk):
            cx, cy = k_(x), row_y(r)
            w_, h_ = k_(28), k_(22)
            for ddx, ddy, ww, hh in ((0, h_ / 2, w_, k_(0.5)), (0, -h_ / 2, w_, k_(0.5)), (w_ / 2, 0, k_(0.5), h_),
                                     (-w_ / 2, 0, k_(0.5), h_)):
                sk.box((cx + ddx, cy + ddy, zt + k_(0.1)), (ww, hh, k_(0.2)), 0)
    tp.append(P.Part("silk", sk, [silk]))
    # board components (seeded): MLCC strips, small ICs, inductors, near the rear corners and the far front region
    cp = MB()
    rr = random.Random(11)
    for _ in range(70):
        xx = rr.choice([-1, 1]) * rr.uniform(150, 240)
        yy = Yb - rr.uniform(70, 330) * 0.001 * s
        kind = rr.random()
        if kind < 0.55:
            cp.box((k_(xx), yy, zt + k_(0.4)), (k_(1.6), k_(0.8), k_(0.8)), 0)       # MLCC 0603-ish
        elif kind < 0.85:
            cp.box((k_(xx), yy, zt + k_(1.0)), (k_(6), k_(5), k_(2.0)), 1)           # SOIC/QFN
        else:
            cp.box((k_(xx), yy, zt + k_(2.0)), (k_(8), k_(8), k_(4.0)), 2)           # inductor
    # front region: power input connector and pull-latch housing marks (front end of the board, open front)
    cp.box((0, -depth / 2 + k_(90), zt + k_(3)), (k_(60), k_(14), k_(6)), 1)
    tp.append(P.Part("components", cp, [L["gold"], L["mold"], L["dark_steel"]]))
    # connector plug housings (4), mated to backplane receptacles (4) + power blade pair
    pl = MB()    # plug body mat0 black, latch mat1 alu, contacts mat2 gold
    rc = MB()    # receptacle mat0 grey, lip mat1 dark
    zc = zt + k_(14)
    for x in xk:
        pl.box((k_(x), Yb - k_(18 + 14), zc), (k_(86), k_(28), k_(26)), 0)
        pl.box((k_(x), Yb - k_(18 + 14), zc + k_(13.1)), (k_(70), k_(20), k_(0.4)), 2)      # contact comb window (top)
        for q in range(20):
            pl.box((k_(x - 33 + q * 3.5), Yb - k_(18 + 18), zc + k_(13.4)), (k_(1.1), k_(9), k_(0.5)), 2)
            pl.box((k_(x - 33 + q * 3.5), Yb - k_(18 + 8), zc + k_(13.4)), (k_(1.1), k_(9), k_(0.5)), 2)
        for dx in (-1, 1):
            pl.box((k_(x + dx * 45), Yb - k_(18 + 22), zc), (k_(4), k_(18), k_(14)), 1)     # latch levers
        rc.box((k_(x), Yb - k_(9), zc), (k_(96), k_(18), k_(34)), 0)
        rc.box((k_(x), Yb - k_(19), zc), (k_(98), k_(2), k_(36)), 1)                         # lip
    pw = MB()
    pw.box((0, Yb - k_(14), zc), (k_(40), k_(28), k_(28)), 0)
    pw.box((k_(-8), Yb - k_(30), zc), (k_(10), k_(6), k_(22)), 1)
    pw.box((k_(8), Yb - k_(30), zc), (k_(10), k_(6), k_(22)), 1)
    tp.append(P.Part("plugs", pl, [L["black"], L["alu"], gold], bevel=(0.0005 * s, 1)))
    tp.append(P.Part("receptacles", rc, [L["grey"], L["dark_steel"]], bevel=(0.0006 * s, 1)))
    tp.append(P.Part("power_blade", pw, [L["black"], L["copper"]], bevel=(0.0005 * s, 1)))
    for dz, dsl in (("", None),):
        pass
    # instantiate trays
    for j in range(N):
        e = bpy.data.objects.new("%s_tray%d" % (AID, j), None)
        e.empty_display_size = 0.05 * s
        cs["TRAYS"].objects.link(e)
        e.parent = root
        e.location = (0, 0, j * pitch)
        P.instantiate(tp, cs["TRAYS"], e, "%s_tray%d" % (AID, j), key=var + "_tray")
        # hooks (rows r=0..2, connectors k=0..3), at the board top surface
        for r in range(3):
            for kk in range(4):
                h = C.hook("retimer_row%d_tray%d_%d" % (r, j, kk), cs["HOOKS"], e, (k_(xk[kk]), row_y(r), zt))
                h.empty_display_size = 0.02 * s
        C.hook("tray%d_surface" % j, cs["HOOKS"], e, (0, 0, zt))
    # labels: tray numbers on white plates + connector ids, text meshes
    lm = L["label"]
    tm = L["text"]
    for j in range(N):
        e = bpy.data.objects[AID + "_tray%d" % j]
        for dx in (-1, 1):
            pl_ = MB()
            pl_.box((dx * (Wm / 2 - k_(60)), Yb - k_(23.5), j * pitch + zt + k_(10)), (k_(70), k_(0.8), k_(12)), 0)
            o = pl_.obj("%s_label_plate_t%d_%s" % (AID, j, "L" if dx < 0 else "R"), [lm])
            D.put(o, cs["LABELS"], root)
            t = D.text("%s_label_t%d_%s" % (AID, j, "L" if dx < 0 else "R"), "T%02d" % (j + 1), k_(7.0),
                       (dx * (Wm / 2 - k_(60)), Yb - k_(24.1), j * pitch + zt + k_(10)), (math.pi / 2, 0, 0), tm,
                       cs["LABELS"], root, extrude=k_(0.1))
        for kk in range(4):
            t = D.text("%s_label_c_t%d_%d" % (AID, j, kk), "C%d" % kk, k_(5.0),
                       (k_(xk[kk]), Yb - k_(19.7), j * pitch + zt + k_(14) + k_(18.5)), (math.pi / 2, 0, 0), D.M("text_light", (0.85, 0.85, 0.8), 0, 0.6),
                       cs["LABELS"], root, extrude=k_(0.1))
    # parody sign hook, camera aids
    C.hook("nvl_sign", cs["HOOKS"], root, (0, Yb - 0.02 * s, (N - 1) * pitch * 0.25 + 0.2 * s))
    C.hook("cam_start", cs["HOOKS"], root, (0, -depth / 2 + 0.2 * s, k_(board_top) + min(pitch * 0.5, 0.35)))
    C.hook("cam_end_top", cs["HOOKS"], root, (0, -depth / 2 + 0.2 * s, H))
    bpy.context.view_layer.update()
    blend = os.path.join(OUT, AID + ".blend")
    meta = dict(
        sources=[
            dict(what="EIA-310 rack unit 44.45 mm (real pitch variant)", url="https://en.wikipedia.org/wiki/19-inch_rack", accessed="2026-10-02"),
            dict(what="ORV3 21 in (537 mm) opening", url="https://www.opencompute.org/", accessed="2026-10-02"),
            dict(what="NVL72: trays blind-mate onto a copper backplane/spine; tray 1U; NVLink copper", url="https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component", accessed="2026-10-02"),
            dict(what="Layout of connector rows: storyboard S2 (4 per tray) ", url="DevLog/DevLog-001-story-scenes-assets-proposal.md Section 4", accessed="2026-10-02"),
        ],
        dimensions=[
            D.dim("opening width", W * s, "mm", "ORV3 537 mm x detail scale s=%g" % s, "B"),
            D.dim("tray pitch", pitch * 1000, "mm", "param: real=44.45 (EIA-310), stylized=700 (crude storyboard)", "A" if var == "real" else "C"),
            D.dim("connector plug", "86 x 28 x 26 (x s)", "mm", "plausible blind-mate footprint, estimate (range 60-110 wide)", "C"),
            D.dim("connector spacing", 118 * s, "mm", "estimate", "C"),
            D.dim("retimer row spacing", 44 * s, "mm", "estimate: leaves ~22 x 28 mm silkscreen footprint per chip", "C"),
            D.dim("tray depth", depth * 1000, "mm", "param", "C"),
            D.dim("detail scale s", s, "-", "uniform scale of all hardware features (not pitch or depth)", "-"),
        ],
        simplifications=["trays are pans with boards (no cold plates or chassis lids); tray fronts removed",
                         "connector contact fields are decorative", "backplane rear side not modelled (plate only)"],
        hooks_doc={"HOOK_retimer_row<r>_tray<j>_<k>": "r=0..2 row index (0 nearest the connectors), j=0..N-1 tray index from bottom, k=0..3 connector; at board top surface, +x right, row axis along x",
                   "HOOK_tray<j>_surface": "centre of tray board top", "HOOK_nvl_sign": "backplane sign anchor (valley sign from packaging/props)",
                   "HOOK_cam_start / cam_end_top": "camera rise start/end aids"},
        custom_properties_doc={"p_*": "build-time parameters, documentation only (geometry is static; rebuild with key=value overrides)"},
        origin="x=0, y=0 mid-depth, z=0 bottom of the lowest tray floor; front at -y (open), backplane at +y",
        intended_usage="S2 camera rise inside the rack; real-pitch variant for macro or documentation",
        build_params=par,
    )
    D.use_visible_bbox()
    C.finish(AID, blend, coll, meta)
    D.extend_meta(blend, dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll)[:12] + ["... %d hooks total" % len(D.hook_list(coll))],
                              custom_properties=D.custom_props(root)))
    pv = os.path.join(OUT, "previews")
    zr = 2 * pitch + k_(board_top)  # tray 2 board top
    ceil_l = [((0, -depth * 0.1, zr + pitch * 0.9), 20 * s * s, 0.8 * s), ((0, -depth * 0.4, zr + pitch * 0.9), 20 * s * s, 0.8 * s)] if var == "stylized" else []
    views = []
    if var == "stylized":
        views = [dict(name="front", loc=(0, -depth * 1.1, H * 0.5), target=(0, 0, H * 0.45), lens=40),
                 dict(name="three_quarter", loc=(depth * 0.9, -depth * 0.9, H * 0.9), target=(0, 0, H * 0.4), lens=35),
                 dict(name="top", loc=(0, -0.01, H + 7), target=(0, 0, 0), lens=35),
                 dict(name="interior_tray2", loc=(0, -depth * 0.2, 2 * pitch + k_(board_top) + 0.3 * s),
                      target=(0, Yb, 2 * pitch + k_(board_top) + 0.3 * s), lens=22),
                 dict(name="interior_rise", loc=(0.2, -depth * 0.2, 4 * pitch + k_(board_top) + 0.3 * s),
                      target=(0, Yb, 4.6 * pitch), lens=22),
                 dict(name="interior_closeup", loc=(-0.3, -0.5, 2 * pitch + k_(board_top) + 0.12 * s),
                      target=(-0.2, Yb, 2 * pitch + k_(board_top) + 0.1 * s), lens=30)]
    else:
        views = [dict(name="front", loc=(0, -1.4, 0.2), target=(0, 0, 0.14), lens=40),
                 dict(name="three_quarter", loc=(0.9, -1.1, 0.5), target=(0, 0, 0.14), lens=35),
                 dict(name="top", loc=(0, -0.01, 1.6), target=(0, 0, 0), lens=35),
                 dict(name="interior_closeup", loc=(0.12, -0.15, 0.05), target=(0.0, Yb, 0.02), lens=30)]
    D.render_views(coll, pv, AID, views, floor=False, lights=[((0.3 * s, -depth * 0.6, H + 1.5 * s), 150 * s * s, 2.0 * s),
                                                           ((0.0, -depth * 0.1, pitch * 2.8), 15 * s * s, 1.0 * s)] ,
                   sun=1.0, world=0.6, clip=(0.01 * s * 0.1, 200))


todo = [KV["variant"]] if "variant" in KV else list(VARIANTS)
for v in todo:
    par = dict(VARIANTS[v])
    for k in ("s", "pitch", "depth", "n"):
        if k in KV:
            par[k] = KV[k]
    build(v, par)
