"""Two 12 gauge 2-3/4 in shotgun shells: live (crimped) and spent (open petal crimp, dented primer).

Run: Blender -b --python scripts/assets/props/build_shotgun_shells.py -- assets/components/props
Origin: centre of the brass base underside (z = 0), shell stands along +Z. Spent shell root is offset +40 mm in X (reset when appending).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()


def build_shell(aid, spent, xoff):
    coll, root = C.new_asset(aid, accuracy="B")
    root.location = (mm(xoff), 0, 0)
    red = C.principled("MAT_props_shell_hull_red", (0.62, 0.04, 0.035), rough=0.32, coat=0.3)
    brass = C.principled("MAT_props_shell_brass", (0.85, 0.62, 0.22), metallic=1.0, rough=0.3)
    nickel = C.principled("MAT_props_shell_primer", (0.7, 0.7, 0.72), metallic=1.0, rough=0.35)
    dark = C.principled("MAT_props_shell_crimp_line", (0.12, 0.01, 0.01), rough=0.5)
    parts = []

    def put(o):
        C.add(o, coll, root)
        parts.append(o)
        return o
    # brass head (height about 19 mm, rim 22.2 mm: estimate), base underside at z = 0
    prof = [(0, 0), (10.9, 0), (11.1, 0.4), (11.1, 1.9), (10.4, 2.1), (10.3, 2.8), (10.3, 18.8), (9.95, 19.0), (0, 19.0)]
    put(L.lathe("%s_brass_head" % aid, [(mm(r), mm(h)) for r, h in prof], 64, closed_profile=False, mat=brass, sharp_deg=40))
    # primer
    if spent:
        pp = [(0, 0.15), (1.6, 0.05), (3.0, -0.35), (4.0, -0.5), (4.2, -0.1), (4.2, 0.0)]
    else:
        pp = [(0, -0.5), (4.0, -0.5), (4.2, -0.3), (4.2, 0.2)]
    put(L.lathe("%s_primer" % aid, [(mm(r), mm(h)) for r, h in pp], 32, mat=nickel))
    if not spent:
        hp = [(0, 2.0), (9.9, 2.0), (10.0, 18.0), (10.05, 19.5), (10.05, 63.0), (9.4, 67.5), (8.0, 69.6), (6.5, 70.0), (0, 69.6)]
        put(L.lathe("%s_hull" % aid, [(mm(r), mm(h)) for r, h in hp], 64, mat=red, sharp_deg=40))
        for k in range(6):
            a = k * math.pi / 3
            ln = L.rbox("%s_crimp_line_%d" % (aid, k), (mm(6.0), mm(0.5), mm(0.4)), (0, 0, 0), 0, 1, dark)
            ln.data.transform(Matrix.Translation((mm(3.2), 0, mm(69.9))))
            ln.data.transform(Matrix.Rotation(a, 4, "Z"))
            put(ln)
    else:
        # hollow tube with six splayed petals
        hp = [(9.3, 2.0), (9.9, 2.0), (10.0, 19.5), (10.05, 19.5), (10.05, 59.0), (9.3, 59.0), (9.25, 19.0)]
        put(L.lathe("%s_hull" % aid, [(mm(r), mm(h)) for r, h in hp], 64, closed_profile=True, mat=red, sharp_deg=40))
        for k in range(6):
            a = k * math.pi / 3
            path = [Vector((mm(10.0 + 7.5 * (t ** 1.6)), 0, mm(59.0 + 8.0 * t - 1.0 * t * t))) for t in (0, 0.25, 0.5, 0.75, 1.0)]
            bm = bmesh.new()
            prof = [(mm(-5.2), mm(-0.4)), (mm(5.2), mm(-0.4)), (mm(5.2), mm(0.4)), (mm(-5.2), mm(0.4))]
            L.tube_path_bm(bm, path, prof, binormal=Vector((0, 1, 0)), cap=True, scale_fn=lambda t: 1.0 - 0.75 * t)
            bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(a, 3, "Z"), verts=bm.verts)
            put(L.obj_from_bm("%s_petal_%d" % (aid, k), bm, red, True, True, 50))
    top = 69.8 if not spent else 66.0
    L.hook_at("mouth", coll, root, (0, 0, mm(top)), size=0.01)
    L.hook_at("primer", coll, root, (0, 0, 0), rot=(math.pi, 0, 0), size=0.01)
    return coll, root


def main():
    C.reset()
    c1, r1 = build_shell("shotgun_shell_live", False, 0.0)
    c2, r2 = build_shell("shotgun_shell_spent", True, 40.0)
    meta = dict(
        category="props", asset_family="shotgun_shells", date="2026-10-02", accuracy="B",
        description="Two 12 gauge 2-3/4 in shells: live (rolled crimp with six crease lines) and spent (six splayed petals, dented primer).",
        origin="Centre of the brass base underside, shell upright along +Z. Spent root is offset +40 mm in X in the file (reset on append).",
        sources=[{"what": "12 gauge 2-3/4 in (70 mm) shell length", "url": "https://en.wikipedia.org/wiki/12-gauge_shotgun", "accessed": "2026-10-02"},
                 {"what": "SAAMI hull base diameter 0.809 in (20.55 mm), secondary source", "url": "https://www.shotgunworld.com/bbs/viewtopic.php?f=13&t=72561", "accessed": "2026-10-02"}],
        dimensions_mm=[{"item": "live length", "value": 70, "provenance": "2-3/4 in nominal", "level": "A"},
                       {"item": "brass head diameter / rim diameter", "value": "20.6 / 22.2", "provenance": "SAAMI base 0.809 in; rim estimate", "level": "B"},
                       {"item": "brass head height", "value": 19, "provenance": "estimate +-2", "level": "C"},
                       {"item": "spent length with petals", "value": 67, "provenance": "estimate", "level": "C"}],
        hooks={"HOOK_mouth": "open end centre (+Z)", "HOOK_primer": "base centre, local +Z pointing down into the primer"},
        custom_properties={}, materials=["MAT_props_shell_hull_red", "MAT_props_shell_brass", "MAT_props_shell_primer", "MAT_props_shell_crimp_line"],
        simplifications=["no printed label on the hull", "crimp lines are thin boxes on a smooth dome"],
        intended_usage="Loading/ejecting shots when the shotgun breaks open (use HOOK_chamber_L/R of the shotgun).",
    )
    blend = os.path.join(OUT, "shotgun_shells.blend")
    close = [{"name": "closeup_head", "target": Vector((mm(20), 0, mm(10))), "cam_dir": Vector((0.2, -1, 0.6)), "dist": 0.12}]
    items = [dict(asset_id="shotgun_shell_live", coll=c1, root=r1, meta={}), dict(asset_id="shotgun_shell_spent", coll=c2, root=r2, meta={})]
    L.write_family(blend, items, meta, PREV, previews=False)
    L.reopen(blend)
    c1 = bpy.data.collections["ASSET_shotgun_shell_live"]
    c2 = bpy.data.collections["ASSET_shotgun_shell_spent"]
    views = [{"name": "front", "dir": "front", "fit": 1.0}, {"name": "three_quarter", "dir": (0.5, -0.9, 0.55), "fit": 1.0},
             {"name": "top", "dir": "top", "fit": 1.0}, {"name": "closeup_head", "target": Vector((mm(20), 0, mm(12))), "cam_dir": Vector((0.2, -1, 0.5)), "dist": 0.14}]
    pn = L.render_views([c1, c2], PREV, "shotgun_shells", views)
    L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_shells.png"), 2, (450, 338))


main()
