"""Office gag props family: envelope_bonus, wall_calendar, price_balloon, pager, coffee_mug, coffee_machine, clipboard,
hard_hat, cardboard_box, wafer_map_printout, paper_stack. Each is an ASSET_<id> collection with ROOT_<id>; roots are laid out along +X.

Run: Blender -b --python scripts/assets/props/build_office_gags.py -- assets/components/props
All dimensions in mm in the build code (generic office objects, estimates flagged in the JSON).
"""
import calendar
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()
PI = math.pi
V3 = lambda x, y, z: Vector((mm(x), mm(y), mm(z)))  # noqa: E731


def prism(name, pts, z0, z1, mat, bevel=0):
    """Extrude a polygon (list of (x, y) in mm) between z0 and z1 (mm)."""
    bm = bmesh.new()
    lo = [bm.verts.new(V3(x, y, z0)) for x, y in pts]
    hi = [bm.verts.new(V3(x, y, z1)) for x, y in pts]
    n = len(pts)
    bm.faces.new(lo[::-1])
    bm.faces.new(hi)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    return L.obj_from_bm(name, bm, mat, True, True, 40)


class Ctx:
    def __init__(self):
        self.items = []
        self.xcur = 0.0

    def item(self, aid, width_mm, accuracy="C"):
        coll, root = C.new_asset(aid, accuracy=accuracy)
        root.location = (mm(self.xcur), 0, 0)
        self.xcur += mm(width_mm) + 0.12
        self.cur = (aid, coll, root)
        return coll, root

    def done(self, meta):
        aid, coll, root = self.cur
        self.items.append(dict(asset_id=aid, coll=coll, root=root, meta=meta))


def A(o, coll, root):
    C.add(o, coll, root)
    return o


# ------------------------------------------------------------------ materials
def make_mats():
    M = {}
    M["paper"] = C.principled("MAT_props_paper", (0.93, 0.92, 0.88), rough=0.85)
    M["paper_cream"] = C.principled("MAT_props_paper_cream", (0.92, 0.86, 0.68), rough=0.85)
    M["ink"] = C.principled("MAT_props_ink", (0.04, 0.04, 0.05), rough=0.7)
    M["ink_green"] = C.principled("MAT_props_ink_green", (0.05, 0.4, 0.12), rough=0.6)
    M["red_clay"] = L.clay("MAT_props_clay_red", (0.8, 0.07, 0.05), bump=0.3, scale=60, ring=14)
    M["white_clay"] = L.clay("MAT_props_clay_white", (0.95, 0.95, 0.93), bump=0.3, scale=60, ring=14)
    M["steel"] = C.principled("MAT_props_steel", (0.7, 0.7, 0.72), metallic=1.0, rough=0.3)
    M["dark_steel"] = C.principled("MAT_props_dark_steel", (0.2, 0.2, 0.22), metallic=1.0, rough=0.4)
    M["plastic_black"] = C.principled("MAT_props_plastic_black", (0.03, 0.03, 0.035), rough=0.4)
    M["plastic_gray"] = C.principled("MAT_props_plastic_gray", (0.55, 0.56, 0.58), rough=0.45)
    M["plastic_blue"] = C.principled("MAT_props_plastic_blue", (0.06, 0.1, 0.25), rough=0.35)
    M["ceramic"] = C.principled("MAT_props_ceramic", (0.9, 0.92, 0.95), rough=0.2, coat=0.5)
    M["coffee"] = C.principled("MAT_props_coffee", (0.07, 0.035, 0.015), rough=0.08)
    M["glass"] = C.principled("MAT_props_glass", (0.85, 0.92, 0.95), rough=0.03, alpha=0.25)
    M["cardboard"] = C.principled("MAT_props_cardboard", (0.55, 0.4, 0.24), rough=0.9)
    M["tape"] = C.principled("MAT_props_packing_tape", (0.62, 0.5, 0.3), rough=0.25)
    M["hardboard"] = C.principled("MAT_props_hardboard", (0.4, 0.26, 0.14), rough=0.8)
    M["yellow"] = C.principled("MAT_props_hardhat_yellow", (0.95, 0.72, 0.02), rough=0.3, coat=0.3)
    M["web"] = C.principled("MAT_props_webbing", (0.08, 0.08, 0.09), rough=0.9)
    M["led"] = C.principled("MAT_props_led_green", (0.1, 0.8, 0.2), emit=(0.1, 1.0, 0.25), emit_strength=3.0)
    M["green"] = C.principled("MAT_props_wafer_green", (0.05, 0.7, 0.12), rough=0.5)
    M["red"] = C.principled("MAT_props_wafer_red", (0.8, 0.06, 0.05), rough=0.5)
    M["string"] = C.principled("MAT_props_string", (0.9, 0.88, 0.8), rough=0.8)
    M["balloon_tag"] = C.principled("MAT_props_price_tag", (0.95, 0.9, 0.5), rough=0.7)
    # screen: emission shader plus an EMPTY image texture node for the assembler
    sc = C.principled("MAT_props_screen", (0.45, 0.62, 0.4), rough=0.3, emit=(0.45, 0.62, 0.4), emit_strength=0.8)
    nt = sc.node_tree
    img = nt.nodes.new("ShaderNodeTexImage")
    img.name = "SCREEN_IMAGE"
    img.label = "plug image sequence here -> Emission Color"
    img.location = (-400, 0)
    M["screen"] = sc
    return M


# ------------------------------------------------------------------ envelope
def envelope(cx, M):
    coll, root = cx.item("envelope_bonus", 240)
    body = L.rbox("envelope_bonus_body", (mm(229), mm(162), mm(9)), (0, 0, mm(4.5)), mm(2.5), 2, M["paper_cream"])
    A(body, coll, root)
    flap = prism("envelope_bonus_flap", [(-114.5, 81), (114.5, 81), (0, 6)], 9.0, 9.9, M["paper_cream"])
    A(flap, coll, root)
    lab = L.rbox("envelope_bonus_label", (mm(150), mm(46), mm(0.6)), (0, -mm(36), mm(9.0 + 0.3)), 0, 1, M["white_clay"])
    A(lab, coll, root)
    t1 = L.text_obj("envelope_bonus_text", "YEAR-END BONUS", mm(13), loc=(0, -mm(32), mm(9.6)), extrude=mm(0.4), mat=M["ink_green"])
    A(t1, coll, root)
    t2 = L.text_obj("envelope_bonus_text_sub", "OPEN IMMEDIATELY", mm(6), loc=(0, -mm(48), mm(9.6)), extrude=mm(0.3), mat=M["ink"])
    A(t2, coll, root)
    L.hook_at("handoff", coll, root, (0, 0, mm(10)), size=0.04)
    L.hook_at("label", coll, root, (0, -mm(36), mm(10)), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre, lying flat, flap side up, flap point towards -Y; size 229 x 162 mm (C5) x 9 mm stuffed",
                 hooks={"HOOK_handoff": "centre of the envelope (hand it over)", "HOOK_label": "label centre; the text objects envelope_bonus_text(_sub) are removable"},
                 dims=[{"item": "envelope", "value": "229 x 162 x 9", "provenance": "C5 envelope size (ISO 269) plus estimated stuffing", "level": "B"}]))


# ------------------------------------------------------------------ wall calendar
def wall_calendar(cx, M):
    coll, root = cx.item("wall_calendar", 320, "B")
    YEAR = 2027
    names = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
    pcols = [(0.55, 0.7, 0.85), (0.85, 0.6, 0.65), (0.6, 0.8, 0.55), (0.9, 0.8, 0.45), (0.5, 0.78, 0.7), (0.95, 0.65, 0.4),
             (0.4, 0.6, 0.9), (0.85, 0.5, 0.3), (0.7, 0.55, 0.8), (0.9, 0.55, 0.2), (0.6, 0.55, 0.45), (0.75, 0.2, 0.25)]
    H, W, PH, PW = 412.0, 300.0, 400.0, 280.0
    back = L.rbox("wall_calendar_backboard", (mm(W), mm(4), mm(430)), (0, mm(2.2), mm(215)), mm(1.5), 1, M["white_clay"])
    A(back, coll, root)
    yr = L.text_obj("wall_calendar_year_back", str(YEAR), mm(40), loc=(0, mm(-0.02), mm(10)), rot=(PI / 2, 0, 0), extrude=mm(0.4), mat=M["ink"])
    # binding: bar + coil rings
    bm = bmesh.new()
    L.lathe_bm(bm, [(mm(1.8), -mm(135)), (mm(1.8), mm(135))], 12, xf=Matrix.Translation((0, -mm(3), mm(H + 4))) @ Matrix.Rotation(PI / 2, 4, "Y"))
    for k in range(21):
        x = -120 + 12 * k
        L.lathe_bm(bm, [(mm(5.5), -mm(1.1)), (mm(6.5), -mm(1.1)), (mm(6.5), mm(1.1)), (mm(5.5), mm(1.1))], 12, xf=Matrix.Translation((mm(x), -mm(3), mm(H + 4))) @ Matrix.Rotation(PI / 2, 4, "Y"), closed_profile=True)
    A(L.obj_from_bm("wall_calendar_binding", bm, M["steel"], True, True, 50), coll, root)
    A(L.rbox("wall_calendar_hanger", (mm(30), mm(3), mm(10)), (0, mm(5), mm(428)), mm(1.5), 1, M["steel"]), coll, root)
    # rig
    bones = [dict(name="page_%02d" % (k + 1), head=V3(0, -(12 - k) * 0.6 - 1.0, H), tail=V3(0, -(12 - k) * 0.6 - 1.0, H + 10), roll_axis=(0, -1, 0)) for k in range(12)]
    arm = L.make_armature("wall_calendar_rig", bones, coll, root, display="STICK")
    L.add_prop(root, "p_flip", 0.0, 0.0, 12.0, "pages flipped: 0 = January showing, 1 = January flipped, ... 12 = all flipped (rigid pages hinged at the binding, each up to 172 deg)")
    pitch = L.line_pitch(mm(11))
    for k in range(12):
        py = -(12 - k) * 0.6 - 1.0
        parts = []
        parts.append(L.rbox("p_base", (mm(PW), mm(0.4), mm(PH)), (0, mm(py), mm(H - PH / 2)), 0, 1, M["paper"]))
        panel = L.rbox("p_panel", (mm(250), mm(0.2), mm(185)), (0, mm(py - 0.3), mm(H - 12 - 92.5)), 0, 1, C.principled("MAT_props_cal_panel_%02d" % (k + 1), pcols[k], rough=0.7))
        parts.append(panel)
        parts.append(L.text_obj("p_month", names[k], mm(22), loc=(0, mm(py - 0.35), mm(H - 215)), rot=(PI / 2, 0, 0), extrude=mm(0.3), mat=M["ink"]))
        first, ndays = calendar.monthrange(YEAR, k + 1)  # Monday = 0
        first = (first + 1) % 7  # Sunday = 0
        cols = [[] for _ in range(7)]
        for c in range(7):
            cols[c].append("SMTWTFS"[c])
        for d in range(1, ndays + 1):
            cols[(first + d - 1) % 7].append(str(d))
        for c in range(7):
            body = "\n".join(cols[c])
            sp = mm(21) / pitch
            parts.append(L.text_obj("p_col", body, mm(11), loc=(mm((c - 3) * 35 + 12), mm(py - 0.35), mm(H - 232)), rot=(PI / 2, 0, 0), extrude=mm(0.25), mat=M["ink"], align="RIGHT", align_y="TOP", space_line=sp))
        pg = C.join(parts, "wall_calendar_page_%02d" % (k + 1))
        A(pg, coll, root)
        L.bone_parent(pg, arm, "page_%02d" % (k + 1))
        h = L.hook_at("page_%02d" % (k + 1), coll, root, (0, mm(py - 1.0), mm(H - 200)), size=0.02)
        L.bone_parent(h, arm, "page_%02d" % (k + 1))
        L.driver(arm.pose.bones["page_%02d" % (k + 1)], "rotation_euler", 0, root, ["p_flip"], "min(max(v0-%d,0),1)*3.0" % k)
    # removable +3 MONTHS gag label, stuck on page 1
    lab = L.rbox("wall_calendar_label_plus3", (mm(130), mm(0.6), mm(44)), (0, -mm(13.2 + 0.6), mm(H - 110)), mm(0.2), 1, M["red_clay"], rot=Matrix.Rotation(0.1, 3, "Y"))
    txt = L.text_obj("wall_calendar_label_plus3_text", "+3 MONTHS", mm(15), loc=(0, -mm(14.4), mm(H - 110)), rot=(PI / 2, -0.1, 0), extrude=mm(0.4), mat=M["white_clay"])
    for o in (lab, txt):
        A(o, coll, root)
        L.bone_parent(o, arm, "page_01")
    L.hook_at("hang", coll, root, (0, 0, mm(428)), size=0.03)
    L.hook_at("label", coll, root, (0, -mm(15), mm(H - 110)), size=0.02)
    cx.done(dict(accuracy="B", origin="bottom centre of the backboard, hanging on the wall plane y = 0 (front towards -Y); pages 280 x 400 mm, backboard 300 x 430 mm, year 2027",
                 hooks={"HOOK_hang": "hanger hole at the top", "HOOK_label": "centre of the +3 MONTHS sticker (objects wall_calendar_label_plus3 and ..._text are removable, parented to bone page_01)", "HOOK_page_01..12": "centre of each page, follows its page bone"},
                 custom_properties={"p_flip": "0..12 pages flipped (driver on bones page_01..page_12, 172 deg max, rigid pages; they stand above the binding when flipped)"},
                 rig="Armature wall_calendar_rig: page_01 (front, January) .. page_12, hinge axis world X at the binding; pages are 0.6 mm apart in Y.",
                 dims=[{"item": "page / backboard", "value": "280 x 400 / 300 x 430", "provenance": "generic wall calendar, estimate", "level": "C"}],
                 notes="Day numbers are extruded text meshes (2027, Sunday first); weekday letters in the first line of each column."))


# ------------------------------------------------------------------ price balloon
def price_balloon(cx, M):
    coll, root = cx.item("price_balloon", 520)
    R = 250.0
    prof = []
    n = 40
    for i in range(n + 1):
        t = PI * i / n
        r = R * math.sin(t) * (1.0 + 0.08 * (-math.cos(t)))
        z = -R * 1.12 * math.cos(t) * (1.0 if math.cos(t) < 0 else 0.92) + 0.0
        prof.append((mm(r), mm(z)))
    prof[0] = (0, prof[0][1])
    prof[-1] = (0, prof[-1][1])
    parts = []
    bal = L.lathe("pb_balloon", prof, 56, mat=M["red_clay"], sharp_deg=None)
    bal.data.transform(Matrix.Translation((0, 0, mm(R * 0.1))))
    parts.append(bal)
    zb = -R * 1.12 + R * 0.1 + R * 0.0
    knot = L.lathe("pb_knot", [(0, mm(zb + 6)), (mm(9), mm(zb)), (mm(14), mm(zb - 12)), (mm(5), mm(zb - 22)), (0, mm(zb - 24))], 16, mat=M["red_clay"])
    parts.append(knot)
    sp = [Vector((mm(18 * math.sin(i * 0.9) * (1 + i * 0.03)), mm(10 * math.sin(i * 0.6 + 1)), mm(zb - 22 - i * 40))) for i in range(14)]
    bm = bmesh.new()
    L.tube_path_bm(bm, L.catmull(sp, 6), [(mm(a), mm(b)) for a, b in L.circle2d(1.6, 8)], binormal=Vector((1, 0, 0)), cap=True)
    parts.append(L.obj_from_bm("pb_string", bm, M["string"], True, True, None))
    ztag = zb - 22 - 13 * 40 - 20
    parts.append(L.rbox("pb_tag", (mm(80), mm(3), mm(50)), (0, 0, mm(ztag)), mm(3), 2, M["balloon_tag"]))
    parts.append(L.text_obj("pb_tag_text", "$", mm(34), loc=(0, -mm(2.2), mm(ztag)), rot=(PI / 2, 0, 0), extrude=mm(2.0), mat=M["ink"]))
    parts.append(L.text_obj("pb_text", "$$$", mm(150), loc=(0, -mm(R * 0.98 - 6), mm(R * 0.1 + 10)), rot=(PI / 2, 0, 0), extrude=mm(14), mat=M["white_clay"]))
    o = C.join(parts, "price_balloon")
    A(o, coll, root)
    # shape keys: Basis = nominal x 0.2, 'inflate' = nominal x 1.6 (scale 0.2 + 1.4 v)
    nominal = [v.co.copy() for v in o.data.vertices]
    o.shape_key_add(name="Basis", from_mix=False)
    kb = o.shape_key_add(name="inflate", from_mix=False)
    kb.slider_min, kb.slider_max = 0.0, 1.0
    o.data.shape_keys.key_blocks["Basis"].data.foreach_set("co", [c * 0.2 for v in nominal for c in v])
    kb.data.foreach_set("co", [c * 1.6 for v in nominal for c in v])
    o.data.vertices.foreach_set("co", [c * 0.2 for v in nominal for c in v])
    o.data.update()
    L.add_prop(root, "p_inflate_scale", 1.0, 0.2, 1.6, "uniform scale of the nominal balloon (0.2 deflated .. 1.6 fully inflated); drives shape key 'inflate' = (scale - 0.2) / 1.4")
    L.driver(o.data.shape_keys, 'key_blocks["inflate"].value', None, root, ["p_inflate_scale"], "(v0-0.2)/1.4")
    L.hook_at("center", coll, root, (0, 0, 0), size=0.05)
    L.hook_at("front_text", coll, root, (0, -mm(R), mm(R * 0.1)), rot=(PI / 2, 0, 0), size=0.04)
    cx.done(dict(accuracy="C", origin="balloon centre (mounting origin); nominal radius 250 mm at p_inflate_scale = 1 (matches the v0.2 crude prop: radius 0.25 m scaled 0.2 to 1.6)",
                 hooks={"HOOK_center": "balloon centre (does not scale: the shape key scales the mesh about this point)", "HOOK_front_text": "centre of the $$$ text, facing -Y"},
                 custom_properties={"p_inflate_scale": "0.2..1.6 -> shape key 'inflate' value (scale - 0.2) / 1.4; Basis = 0.2 x nominal, inflate = 1.6 x nominal"},
                 shape_keys={"Basis": "deflated, 0.2 x nominal (radius 50 mm)", "bbox_note": "bbox_mm in this JSON is measured on the Basis (0.2 x nominal) mesh; at p_inflate_scale = 1 expect about 500 x 500 x 1150 mm including string and tag", "inflate": "fully inflated, 1.6 x nominal (radius 400 mm); balloon, knot, string, hang tag and $$$ text scale together"},
                 dims=[{"item": "balloon radius (nominal)", "value": 250, "provenance": "design: matches crude v0.2", "level": "C"}]))


# ------------------------------------------------------------------ pager
def pager(cx, M):
    coll, root = cx.item("pager", 90)
    piv = bpy.data.objects.new("pager_buzz_pivot", None)
    piv.empty_display_type = "PLAIN_AXES"
    piv.empty_display_size = 0.01
    bpy.context.scene.collection.objects.link(piv)
    A(piv, coll, root)
    parts = []
    parts.append(L.rbox("pager_body", (mm(62), mm(44), mm(15)), (0, 0, mm(7.5)), mm(4), 3, M["plastic_blue"]))
    parts.append(L.rbox("pager_bezel", (mm(48), mm(22), mm(0.8)), (0, mm(6), mm(15.2)), mm(1.5), 2, M["plastic_black"]))
    scr = L.rbox("pager_screen", (mm(40), mm(15), mm(0.2)), (0, mm(7), mm(15.7)), 0, 1, M["screen"])
    # UV 0..1 over the screen top face
    me = scr.data
    me.uv_layers.new(name="UV_screen")
    top = max(p.center.z for p in me.polygons)
    for p in me.polygons:
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            me.uv_layers["UV_screen"].data[li].uv = ((co.x + mm(20)) / mm(40), (co.y - mm(7) + mm(7.5)) / mm(15))
    parts.append(scr)
    parts.append(L.text_obj("pager_text", "BUZZ", mm(8), loc=(0, mm(7), mm(15.9)), extrude=mm(0.2), mat=M["ink"]))
    for k in range(3):
        parts.append(L.rbox("pager_button_%d" % k, (mm(10), mm(5), mm(1.5)), (mm((k - 1) * 14), -mm(12), mm(15.6)), mm(1), 1, M["plastic_gray"]))
    for k in range(5):
        parts.append(L.rbox("pager_slot_%d" % k, (mm(0.8), mm(8), mm(0.3)), (mm(-14 + 7 * k), -mm(3), mm(15.1)), 0, 1, M["plastic_black"]))
    parts.append(L.rbox("pager_clip", (mm(24), mm(30), mm(2)), (0, mm(2), -mm(1.2)), mm(0.8), 1, M["steel"]))
    parts.append(L.rbox("pager_clip_tab", (mm(24), mm(4), mm(6)), (0, -mm(14), -mm(3.5)), mm(0.8), 1, M["steel"]))
    for o in parts:
        A(o, coll, root)
        o.parent = piv
    L.hook_at("buzz", coll, piv, (0, 0, mm(8)), size=0.02)
    L.add_prop(root, "p_buzz", 0.0, 0.0, 1.0, "vibration amount 0..1 (drivers shake pager_buzz_pivot: about 0.7 mm and 1.7 deg at 1, uses the scene frame)")
    L.driver(piv, "location", 0, root, ["p_buzz"], "v0*0.0007*sin(frame*4.3)")
    L.driver(piv, "location", 1, root, ["p_buzz"], "v0*0.0005*sin(frame*5.1+1)")
    L.driver(piv, "rotation_euler", 2, root, ["p_buzz"], "v0*0.03*sin(frame*3.7)")
    cx.done(dict(accuracy="C", origin="bottom centre, lying on its back (screen up), 62 x 44 x 15 mm; generic design, no brand",
                 hooks={"HOOK_buzz": "centre of the pager body (parented to pager_buzz_pivot); the pivot is driven by p_buzz"},
                 custom_properties={"p_buzz": "0..1 vibration (location 0.7 mm, rotation 1.7 deg, frame-driven)"},
                 screen="plane pager_screen, material MAT_props_screen: emission shader, UV map UV_screen 0..1; the image texture node SCREEN_IMAGE is left empty (connect its Color output to Emission Color); default emission is a flat LCD green; text 'BUZZ' is geometry above it",
                 dims=[{"item": "body", "value": "62 x 44 x 15", "provenance": "generic pager, estimate +-5", "level": "C"}]))


# ------------------------------------------------------------------ mug
def coffee_mug(cx, M):
    coll, root = cx.item("coffee_mug", 120)
    prof = [(0, 0), (37, 0), (40, 2), (41, 8), (41, 94.5), (40, 95.2), (37.8, 95.2), (37.0, 94), (37, 9), (35, 7), (0, 7)]
    A(L.lathe("coffee_mug_body", [(mm(r), mm(h)) for r, h in prof], 64, closed_profile=True, mat=M["ceramic"], sharp_deg=None), coll, root)
    pts = [Vector((mm(x), 0, mm(z))) for x, z in ((40, 78), (56, 82), (68, 66), (66, 42), (54, 26), (40, 22))]
    bm = bmesh.new()
    L.tube_path_bm(bm, L.catmull(pts, 6), [(mm(a), mm(b)) for a, b in L.superellipse2d(9, 12, 14, 2.4)], binormal=Vector((0, 1, 0)), cap=True)
    A(L.obj_from_bm("coffee_mug_handle", bm, M["ceramic"], True, True, None), coll, root)
    cf = C.cylinder("coffee_mug_coffee", mm(37.3), mm(0.8), loc=(0, 0, mm(84)), verts=48, mat=M["coffee"])
    A(cf, coll, root)
    L.hook_at("steam", coll, root, (0, 0, mm(88)), size=0.02)
    L.hook_at("grip", coll, root, (mm(62), 0, mm(52)), size=0.02)
    L.hook_at("sip", coll, root, (0, -mm(38), mm(95)), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre; handle towards +X; 82 mm diameter x 95 mm high, 3.5 mm wall (typical mug, estimate)",
                 hooks={"HOOK_steam": "above the coffee surface", "HOOK_grip": "handle centre", "HOOK_sip": "front rim"},
                 dims=[{"item": "mug diameter x height", "value": "82 x 95", "provenance": "typical 11 oz mug, estimate +-5", "level": "C"}]))


# ------------------------------------------------------------------ coffee machine
def coffee_machine(cx, M):
    coll, root = cx.item("coffee_machine", 260)
    A(L.rbox("cm_base", (mm(230), mm(300), mm(28)), (0, 0, mm(14)), mm(6), 3, M["plastic_black"]), coll, root)
    A(C.cylinder("cm_warm_plate", mm(88), mm(2), loc=(0, -mm(10), mm(29)), verts=48, mat=M["dark_steel"]), coll, root)
    A(L.rbox("cm_column", (mm(120), mm(80), mm(330)), (0, mm(108), mm(165)), mm(8), 3, M["plastic_gray"]), coll, root)
    A(L.rbox("cm_head", (mm(230), mm(230), mm(55)), (0, mm(33), mm(327.5)), mm(10), 3, M["plastic_gray"]), coll, root)
    A(L.rbox("cm_basket_holder", (mm(130), mm(130), mm(30)), (0, -mm(10), mm(290)), mm(8), 3, M["plastic_black"]), coll, root)
    A(L.rbox("cm_window", (mm(50), mm(2), mm(220)), (0, mm(66.5), mm(165)), mm(1), 1, M["glass"]), coll, root)
    # carafe
    cz0, cz1 = 30.0, 235.0
    prof = [(0, cz0), (58, cz0), (66, cz0 + 8), (72, cz0 + 40), (70, cz0 + 120), (62, cz0 + 170), (56, cz1 - 5), (54, cz1), (51, cz1), (53, cz1 - 6), (60, cz0 + 170), (68, cz0 + 120), (70, cz0 + 40), (64, cz0 + 10), (0, cz0 + 4)]
    A(L.lathe("cm_carafe", [(mm(r), mm(h)) for r, h in prof], 64, xf=Matrix.Translation((0, -mm(10), 0)), closed_profile=True, mat=M["glass"], sharp_deg=None), coll, root)
    A(C.cylinder("cm_carafe_lid", mm(56), mm(8), loc=(0, -mm(10), mm(cz1 + 3)), verts=40, mat=M["plastic_black"]), coll, root)
    hp = [Vector((mm(x), -mm(10), mm(z))) for x, z in ((66, 190), (92, 205), (96, 150), (80, 105), (70, 100))]
    bm = bmesh.new()
    L.tube_path_bm(bm, L.catmull(hp, 6), [(mm(a), mm(b)) for a, b in L.superellipse2d(10, 14, 12, 2.4)], binormal=Vector((0, 1, 0)), cap=True)
    A(L.obj_from_bm("cm_carafe_handle", bm, M["plastic_black"], True, True, None), coll, root)
    # coffee level (origin at its bottom; driven scale)
    cof = C.cylinder("cm_coffee", mm(62), mm(100), loc=(0, -mm(10), mm(cz0 + 6 + 50)), verts=40, mat=M["coffee"])
    cof.data.transform(Matrix.Translation((0, 0, -mm(cz0 + 6))) )
    cof.data.transform(Matrix.Translation((0, mm(10), 0)))
    cof.location = (0, -mm(10), mm(cz0 + 6))
    A(cof, coll, root)
    L.add_prop(root, "p_fill", 0.6, 0.0, 1.0, "coffee level in the carafe 0..1 (drives the z scale of cm_coffee; 1 = 160 mm column)")
    L.driver(cof, "scale", 2, root, ["p_fill"], "max(v0*1.6,0.001)")
    # controls
    for k in range(3):
        A(L.rbox("cm_button_%d" % k, (mm(16), mm(4), mm(6)), (mm(-30 + 30 * k), -mm(146), mm(16)), mm(1.5), 1, M["plastic_gray"]), coll, root)
    A(C.cylinder("cm_led", mm(3), mm(1.5), loc=(mm(60), -mm(147), mm(16)), axis="Y", verts=14, mat=M["led"]), coll, root)
    pts = [Vector((mm(0), mm(150 + 40 * i), mm(2 + 0 * i))) for i in range(8)]
    bm = bmesh.new()
    L.tube_path_bm(bm, L.catmull(pts, 4), [(mm(a), mm(b)) for a, b in L.circle2d(3.5, 10)], binormal=Vector((1, 0, 0)), cap=True)
    A(L.obj_from_bm("cm_cord", bm, M["plastic_black"], True, True, None), coll, root)
    L.hook_at("pour", coll, root, (mm(60), -mm(10), mm(cz1)), size=0.03)
    L.hook_at("steam", coll, root, (0, -mm(10), mm(300)), size=0.03)
    L.hook_at("button_power", coll, root, (mm(-30), -mm(148), mm(16)), size=0.02)
    L.hook_at("basket", coll, root, (0, -mm(10), mm(272)), size=0.03)
    cx.done(dict(accuracy="C", origin="bottom centre of the base (230 x 300 mm), front towards -Y; 355 mm high; generic 10-cup drip machine (estimate)",
                 hooks={"HOOK_pour": "carafe rim", "HOOK_steam": "under the head, steam origin", "HOOK_button_power": "left front button", "HOOK_basket": "filter basket"},
                 custom_properties={"p_fill": "0..1 coffee level (driver scale z of cm_coffee)"},
                 dims=[{"item": "machine W x D x H", "value": "230 x 300 x 355", "provenance": "generic drip coffee maker, estimate +-30", "level": "C"}],
                 notes="MAT_props_led_green is the power LED (emissive)."))


# ------------------------------------------------------------------ clipboard
def clipboard(cx, M):
    coll, root = cx.item("clipboard", 250)
    A(L.rbox("clipboard_board", (mm(230), mm(4.5), mm(320)), (0, 0, mm(160)), mm(2), 2, M["hardboard"]), coll, root)
    A(L.rbox("clipboard_paper", (mm(210), mm(0.5), mm(297)), (0, -mm(2.6), mm(160 - 4)), 0, 1, M["paper"]), coll, root)
    A(L.rbox("clipboard_clip_plate", (mm(90), mm(2), mm(34)), (0, -mm(4.0), mm(311)), mm(0.8), 1, M["steel"]), coll, root)
    A(L.rbox("clipboard_clip_lever", (mm(80), mm(7), mm(14)), (0, -mm(7.5), mm(321)), mm(1.5), 2, M["steel"]), coll, root)
    A(L.rbox("clipboard_clip_arm", (mm(76), mm(1.4), mm(24)), (0, -mm(5.2), mm(298)), mm(0.4), 1, M["steel"]), coll, root)
    parts = []
    rnd = random.Random(11)
    for k in range(14):
        ln = rnd.uniform(110, 175)
        parts.append(L.rbox("l", (mm(ln), mm(0.15), mm(2.0)), (-mm(90 - ln / 2 - 10 * 0) + mm(0), -mm(2.95), mm(262 - k * 15.5)), 0, 1, M["ink"]))
    for k in range(4):
        parts.append(L.rbox("cb", (mm(7), mm(0.15), mm(7)), (-mm(92), -mm(2.95), mm(262 - k * 31)), 0, 1, M["ink"]))
    ln = C.join(parts, "clipboard_text_lines")
    A(ln, coll, root)
    A(L.text_obj("clipboard_title", "CHECKLIST", mm(16), loc=(0, -mm(2.95), mm(282)), rot=(PI / 2, 0, 0), extrude=mm(0.2), mat=M["ink"]), coll, root)
    L.hook_at("hold", coll, root, (-mm(115), 0, mm(160)), size=0.03)
    L.hook_at("clip", coll, root, (0, -mm(8), mm(311)), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre, standing upright, front towards -Y; board 230 x 320 x 4.5 mm, A4 sheet 210 x 297 mm",
                 hooks={"HOOK_hold": "left edge centre (hand grip)", "HOOK_clip": "clip lever"},
                 dims=[{"item": "board", "value": "230 x 320 x 4.5", "provenance": "A4 clipboard, typical", "level": "B"}, {"item": "paper", "value": "210 x 297", "provenance": "ISO 216 A4", "level": "A"}]))


# ------------------------------------------------------------------ hard hat
def hard_hat(cx, M):
    coll, root = cx.item("hard_hat", 330)
    # shell: lathe (circular), then scaled to an oval 285 x 235 mm, front peak towards -Y
    prof = [(0, 150), (40, 148), (80, 140), (100, 124), (110, 100), (114, 60), (116, 12), (118, 0), (115, 0), (113, 12), (110, 60), (106, 100), (96, 122), (76, 137), (38, 145), (0, 147)]
    sh = L.lathe("hard_hat_shell", [(mm(r), mm(h)) for r, h in prof], 72, closed_profile=True, mat=M["yellow"], sharp_deg=None)
    sh.data.transform(Matrix.Diagonal((1.0, 1.215, 1.0, 1.0)))
    A(sh, coll, root)
    # brim all around + front peak
    brim = L.lathe("hard_hat_brim", [(mm(113), 0), (mm(133), mm(-2)), (mm(135), mm(1.5)), (mm(114), mm(5))], 72, closed_profile=True, mat=M["yellow"])
    brim.data.transform(Matrix.Diagonal((1.0, 1.215, 1.0, 1.0)))
    A(brim, coll, root)
    peak = L.rbox("hard_hat_peak", (mm(150), mm(55), mm(4)), (0, -mm(155), mm(-1)), mm(2), 2, M["yellow"], rot=Matrix.Rotation(-0.12, 3, "X"))
    A(peak, coll, root)
    # ridges along the crown (centre + two side ridges)
    for nm, dx in (("c", 0), ("l", -44), ("r", 44)):
        pts = []
        for i in range(0, 15):
            y = -130 + 260 * i / 14
            u = abs(y) / 138.0
            h = (147 * (1 - u ** 2.6) ** 0.5) * (1.0 if dx == 0 else 0.9) if u < 1 else 0
            xx = dx * (1 - 0.35 * u * u)
            pts.append(Vector((mm(xx), mm(y), mm(max(h, 60 if dx else 40) + 2.5))))
        bm = bmesh.new()
        L.tube_path_bm(bm, pts, [(mm(a), mm(b)) for a, b in L.circle2d(5.0, 10)], binormal=Vector((1, 0, 0)), cap=True)
        A(L.obj_from_bm("hard_hat_ridge_" + nm, bm, M["yellow"], True, True, None), coll, root)
    # suspension: headband ring + 4 straps
    A(L.lathe("hard_hat_headband", [(mm(100), mm(-4)), (mm(104), mm(-4)), (mm(104), mm(26)), (mm(100), mm(26))], 48, closed_profile=True, mat=M["web"], sharp_deg=None), coll, root)
    coll.objects["hard_hat_headband"].data.transform(Matrix.Diagonal((1.0, 1.1, 1.0, 1.0)))
    for k in range(4):
        a = k * PI / 2 + PI / 4
        pts = [Vector((mm(100 * math.cos(a)), mm(110 * math.sin(a)), mm(22))), Vector((mm(60 * math.cos(a)), mm(66 * math.sin(a)), mm(95))), Vector((0, 0, mm(122)))]
        bm = bmesh.new()
        L.tube_path_bm(bm, L.catmull(pts, 8), [(mm(a_), mm(b_)) for a_, b_ in L.superellipse2d(14, 1.6, 8, 3.0)], binormal=Vector((-math.sin(a), math.cos(a), 0)), cap=True)
        A(L.obj_from_bm("hard_hat_strap_%d" % k, bm, M["web"], True, True, None), coll, root)
    L.hook_at("head_top", coll, root, (0, 0, mm(2)), size=0.04)
    L.hook_at("top", coll, root, (0, 0, mm(152)), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre of the shell rim (z = 0), opening down, peak towards -Y; about 285 x 270 (with peak) x 152 mm",
                 hooks={"HOOK_head_top": "rim plane centre: align the top of the wearer's head about 40 mm above this point", "HOOK_top": "crown apex"},
                 dims=[{"item": "shell length x width x height", "value": "285 x 235 x 150", "provenance": "generic Type-I hard hat proportions, estimate +-15 (accepts head circumference 53-63 cm per EN 397, not verified)", "level": "C"}]))


# ------------------------------------------------------------------ cardboard box
def cardboard_box(cx, M):
    coll, root = cx.item("cardboard_box", 420)
    Lx, Ly, Lz, t = 400.0, 300.0, 250.0, 3.0
    A(L.rbox("cardboard_box_floor", (mm(Lx), mm(Ly), mm(t)), (0, 0, mm(t / 2)), mm(0.6), 1, M["cardboard"]), coll, root)
    for nm, (sx, sy, cx_, cy_) in {"front": (Lx, t, 0, -Ly / 2 + t / 2), "back": (Lx, t, 0, Ly / 2 - t / 2), "left": (t, Ly - 2 * t, -Lx / 2 + t / 2, 0), "right": (t, Ly - 2 * t, Lx / 2 - t / 2, 0)}.items():
        A(L.rbox("cardboard_box_wall_" + nm, (mm(sx), mm(sy), mm(Lz)), (mm(cx_), mm(cy_), mm(Lz / 2)), mm(0.6), 1, M["cardboard"]), coll, root)
    L.add_prop(root, "p_open", 0.0, 0.0, 1.0, "flaps: 0 closed (taped), 1 fully open (outward 125 deg); drives the four flap objects")
    flaps = {}
    # (name, size x, size y, hinge location, direction, axis)
    specs = [("front", Lx, Ly / 2, (0, -Ly / 2, Lz), -1, "x"), ("back", Lx, Ly / 2, (0, Ly / 2, Lz), +1, "x"),
             ("left", Lx / 2 - 0.0, Ly - 2 * t, (-Lx / 2, 0, Lz), -1, "y"), ("right", Lx / 2, Ly - 2 * t, (Lx / 2, 0, Lz), +1, "y")]
    for nm, sx, sy, hinge, d, ax in specs:
        # build local geometry about the hinge, flap pointing inward
        if ax == "x":
            ycen = (sy / 2) * (1 if d < 0 else -1)
            o = L.rbox("cardboard_box_flap_" + nm, (mm(sx), mm(sy), mm(2.6)), (0, mm(ycen), mm(1.3 + (2.6 if nm == "back" else 0))), mm(0.5), 1, M["cardboard"])
        else:
            xcen = (sx / 2) * (1 if d < 0 else -1)
            o = L.rbox("cardboard_box_flap_" + nm, (mm(sx), mm(sy), mm(2.6)), (mm(xcen), 0, mm(1.3 - 0.0)), mm(0.5), 1, M["cardboard"])
        o.location = V3(*hinge)
        A(o, coll, root)
        flaps[nm] = o
    # drivers: rotation about the hinge
    L.driver(flaps["front"], "rotation_euler", 0, root, ["p_open"], "v0*2.18")
    L.driver(flaps["back"], "rotation_euler", 0, root, ["p_open"], "-v0*2.18")
    L.driver(flaps["left"], "rotation_euler", 1, root, ["p_open"], "-v0*2.18")
    L.driver(flaps["right"], "rotation_euler", 1, root, ["p_open"], "v0*2.18")
    A(L.rbox("cardboard_box_tape", (mm(48), mm(Ly + 4), mm(0.6)), (0, 0, mm(Lz + 5.6)), mm(0.2), 1, M["tape"]), coll, root)
    tp = coll.objects["cardboard_box_tape"]
    # tape disappears when open (scale driver on z)
    L.driver(tp, "scale", 2, root, ["p_open"], "1.0 if v0 < 0.01 else 0.0001")
    L.hook_at("lid_front", coll, root, (0, -mm(Ly / 2), mm(Lz)), size=0.03)
    L.hook_at("inside", coll, root, (0, 0, mm(10)), size=0.03)
    cx.done(dict(accuracy="C", origin="bottom centre, front towards -Y; 400 x 300 x 250 mm regular slotted carton, 3 mm single-wall corrugated (estimate)",
                 hooks={"HOOK_lid_front": "front flap hinge edge", "HOOK_inside": "box floor centre"},
                 custom_properties={"p_open": "0 closed (tape strip visible) .. 1 flaps open 125 deg (drivers on the four flap objects, origins at the hinges)"},
                 dims=[{"item": "box", "value": "400 x 300 x 250", "provenance": "generic carton, estimate", "level": "C"}]))


# ------------------------------------------------------------------ wafer map printout
def wafer_map(cx, M):
    coll, root = cx.item("wafer_map_printout", 230)
    SC = 0.5  # printout scale: 300 mm wafer drawn 150 mm across
    Rw = 150.0  # wafer radius (mm)
    best = None
    for px10 in range(180, 330, 5):
        for py10 in range(180, 360, 5):
            px, py = px10 / 10.0, py10 / 10.0
            for ox, oy in ((0, 0), (0.5, 0.5), (0.5, 0), (0, 0.5)):
                dies = []
                for i in range(-20, 21):
                    for j in range(-20, 21):
                        x0, y0 = (i + ox) * px - px / 2, (j + oy) * py - py / 2
                        if all(math.hypot(x, y) <= Rw - 3.0 for x in (x0, x0 + px) for y in (y0, y0 + py)):
                            dies.append(((i + ox) * px, (j + oy) * py))
                if len(dies) == 90:
                    score = abs(px - py) + 0.0
                    if best is None or score < best[0]:
                        best = (score, px, py, ox, oy, dies)
    _, px, py, ox, oy, dies = best
    rnd = random.Random(2026)
    green = set(rnd.sample(range(len(dies)), 9))
    # sheet
    A(L.rbox("wafer_map_printout_sheet", (mm(210), mm(297), mm(0.15)), (0, 0, mm(0.075)), 0, 1, M["paper"]), coll, root)
    ring = L.lathe("wafer_map_printout_outline", [(mm(Rw * SC), mm(0.15)), (mm(Rw * SC + 1.0), mm(0.15)), (mm(Rw * SC + 1.0), mm(0.4)), (mm(Rw * SC), mm(0.4))], 96, xf=Matrix.Translation((0, mm(25), 0)), closed_profile=True, mat=M["ink"])
    A(ring, coll, root)
    A(L.rbox("wafer_map_printout_notch", (mm(4), mm(3), mm(0.25)), (0, mm(25 - Rw * SC + 0.5), mm(0.275)), 0, 1, M["ink"]), coll, root)
    for key, col in (("red", M["red"]), ("green", M["green"])):
        bm = bmesh.new()
        for k, (dx, dy) in enumerate(dies):
            if (k in green) == (key == "green"):
                L.box_bm(bm, (mm((px - 0.8) * SC), mm((py - 0.8) * SC), mm(0.2)), (mm(dx * SC), mm(25 + dy * SC), mm(0.15 + 0.1)), 0, 1)
        A(L.obj_from_bm("wafer_map_printout_dies_" + key, bm, col, False, True, None), coll, root)
    A(L.text_obj("wafer_map_printout_title", "WAFER MAP", mm(14), loc=(0, mm(125), mm(0.17)), extrude=mm(0.1), mat=M["ink"]), coll, root)
    A(L.text_obj("wafer_map_printout_yield", "1 / 10", mm(46), loc=(0, -mm(85), mm(0.17)), extrude=mm(0.12), mat=M["red"]), coll, root)
    A(L.text_obj("wafer_map_printout_note", "GREEN = PASS", mm(8), loc=(0, -mm(120), mm(0.17)), extrude=mm(0.1), mat=M["ink"]), coll, root)
    L.hook_at("hold_top", coll, root, (0, mm(145), 0), size=0.02)
    L.hook_at("map_center", coll, root, (0, mm(25), mm(0.3)), size=0.02)
    cx.done(dict(accuracy="B", origin="bottom centre of the A4 sheet, lying flat (text readable from -Y); die map drawn at 0.5 scale (300 mm wafer -> 150 mm circle)",
                 die_map={"dies": 90, "green": 9, "red": 81, "die_pitch_mm_wafer": [px, py], "grid_offset": [ox, oy], "edge_exclusion_mm": 3.0, "seed": 2026,
                          "green_die_centres_mm_wafer": [list(dies[k]) for k in sorted(green)]},
                 hooks={"HOOK_hold_top": "top edge centre (hold here)", "HOOK_map_center": "wafer centre on the sheet"},
                 dims=[{"item": "sheet", "value": "210 x 297 x 0.15", "provenance": "ISO 216 A4", "level": "A"}, {"item": "wafer", "value": "300 mm diameter (drawn 150 mm)", "provenance": "task brief / SEMI 300 mm wafer", "level": "A"},
                       {"item": "die pitch", "value": "%.1f x %.1f mm on the wafer" % (px, py), "provenance": "searched so that exactly 90 whole dies fit inside the 3 mm edge exclusion", "level": "C"}],
                 notes="Exactly 9 of 90 dies are green (one in ten) and 81 red; geometry only, no image."))


# ------------------------------------------------------------------ paper stack
def paper_stack(cx, M):
    coll, root = cx.item("paper_stack", 240)
    rnd = random.Random(5)
    bm = bmesh.new()
    h = 1.2
    n = 40
    for k in range(n):
        dx, dy = rnd.uniform(-1.6, 1.6), rnd.uniform(-1.6, 1.6)
        rz = rnd.uniform(-0.006, 0.006)
        L.box_bm(bm, (mm(210), mm(297), mm(h - 0.08)), (mm(dx), mm(dy), mm(k * h + h / 2)), 0, 1, rot=Matrix.Rotation(rz, 3, "Z"))
    A(L.obj_from_bm("paper_stack_sheets", bm, M["paper"], False, True, None), coll, root)
    parts = []
    for k in range(16):
        ln = rnd.uniform(120, 185)
        parts.append(L.rbox("l", (mm(ln), mm(1.6), mm(0.1)), (-mm(92 - ln / 2), mm(120 - k * 14), mm(n * h + 0.1)), 0, 1, M["ink"]))
    A(C.join(parts, "paper_stack_text_lines"), coll, root)
    L.hook_at("top", coll, root, (0, 0, mm(n * h)), size=0.02)
    L.hook_at("pick", coll, root, (mm(105), 0, mm(n * h)), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre; 40 bundles (about 400 sheets at 0.12 mm) of A4 with random offsets; 48 mm high",
                 hooks={"HOOK_top": "top sheet centre", "HOOK_pick": "right edge of the top sheet"},
                 dims=[{"item": "A4 sheet", "value": "210 x 297", "provenance": "ISO 216", "level": "A"}, {"item": "stack height", "value": 48, "provenance": "400 sheets x 0.12 mm (80 gsm), estimate", "level": "C"}]))


def main():
    C.reset()
    M = make_mats()
    cx = Ctx()
    for fn in (envelope, wall_calendar, price_balloon, pager, coffee_mug, coffee_machine, clipboard, hard_hat, cardboard_box, wafer_map, paper_stack):
        fn(cx, M)
    meta = dict(
        category="props", asset_family="office_gags", date="2026-10-02", accuracy="C/B (generic office objects, estimated dimensions; A4 and C5 sizes from ISO standards)",
        description="Office gag props: envelope with YEAR-END BONUS label, wall calendar with 12 flipping pages and removable +3 MONTHS label, inflating price balloon, generic pager, mug, coffee machine, clipboard, spare hard hat, cardboard box, wafer-map printout (90 dies, 9 green), paper stack. Each item is an ASSET_ collection with its own root; roots are spread along +X in the file (reset on append).",
        sources=[{"what": "ISO 216 A4 (210 x 297 mm) and ISO 269 C5 envelope (229 x 162 mm), from general knowledge", "url": "not fetched", "accessed": "2026-10-02"},
                 {"what": "other dimensions are generic estimates (each flagged C in its item)", "url": "n/a", "accessed": "2026-10-02"}],
        simplifications=["generic objects with no brand marks; parody text only (YEAR-END BONUS, +3 MONTHS)", "calendar pages are rigid, flip to 172 deg (stand above the binding); no paper bending", "coffee machine has no internal parts", "hard hat suspension is four straps and a band"],
        intended_usage="S3: envelope handed over, calendar pages flying with +3 MONTHS, price balloon inflating; S1/S2 lab: mug, coffee machine, clipboard; S5: wafer-map printout '1 / 10' held by the Manager; pager buzz and spare hard hat as gags.",
    )
    blend = os.path.join(OUT, "office_gags.blend")
    L.write_family(blend, cx.items, meta, PREV, previews=False)
    L.reopen(blend)
    pn = []
    for it in cx.items:
        aid = it["asset_id"]
        coll = bpy.data.collections["ASSET_" + aid]
        pn += L.render_views([coll], PREV, "office_gags_" + aid, L.std_views())
    # extras
    root = bpy.data.objects["ROOT_wall_calendar"]
    L.set_prop(root, "p_flip", 5.5)
    pn += L.render_views([bpy.data.collections["ASSET_wall_calendar"]], PREV, "office_gags_wall_calendar", [{"name": "flipped", "dir": (0.5, -0.8, 0.35), "fit": 1.0}])
    L.set_prop(root, "p_flip", 0.0)
    root = bpy.data.objects["ROOT_price_balloon"]
    L.set_prop(root, "p_inflate_scale", 0.4)
    pn += L.render_views([bpy.data.collections["ASSET_price_balloon"]], PREV, "office_gags_price_balloon", [{"name": "small", "dir": "front", "fit": 1.0}])
    L.set_prop(root, "p_inflate_scale", 1.0)
    root = bpy.data.objects["ROOT_cardboard_box"]
    L.set_prop(root, "p_open", 1.0)
    pn += L.render_views([bpy.data.collections["ASSET_cardboard_box"]], PREV, "office_gags_cardboard_box", [{"name": "open", "dir": "three_quarter", "fit": 0.8}])
    L.set_prop(root, "p_open", 0.0)
    wm = bpy.data.collections["ASSET_wafer_map_printout"]
    pn += L.render_views([wm], PREV, "office_gags_wafer_map_printout", [{"name": "closeup_map", "target": Vector((0, mm(25), 0)), "cam_dir": Vector((0, -0.05, 1)), "dist": 0.2}])
    L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_office.png"), 6, (300, 225))


main()
