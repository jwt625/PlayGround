"""Build hbm_stack (HBM4-class 16-Hi stack, plus 12-Hi, molded and cutaway variants).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_hbm_stack.py -- assets/components/packaging
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import hbm  # noqa: E402
from pk import MB, MM  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
AID = "hbm_stack"

C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="B")
root["p_layers"] = 16
root["p_variant"] = "exposed_16hi"
src_c = C.sub_collection(coll, "SOURCES")
dims = pk.Dims()
mats = hbm.hbm_materials()


def variant_coll(name, visible):
    c = C.sub_collection(coll, "VARIANT_" + name)
    if not visible:
        c.hide_render = True
        c.hide_viewport = True
    return c


def ubump_row(c, spec, ptsfilter=None, name="ubump"):
    src = hbm.bump_source(src_c, root, name="hbm_stack_src_ubump_" + name)
    pts = hbm.bump_points(spec)
    if ptsfilter:
        pts = [p for p in pts if ptsfilter(p)]
    return pk.scatter("hbm_stack_ubumps_" + name, pts, src, c, root)


# ---- variant: exposed 16-Hi (default) and exposed 12-Hi, molded 16-Hi
specs = {"16hi": hbm.Spec(16), "12hi": hbm.Spec(12)}
variants = {}
for vname, key, kind, vis in (("exposed_16hi", "16hi", "exposed", True), ("molded_16hi", "16hi", "molded", False),
                              ("exposed_12hi", "12hi", "exposed", False)):
    sp = specs[key]
    vc = variant_coll(vname, vis)
    b = hbm.stack_mesh(sp, kind)
    o = b.to_obj("hbm_stack_%s_body" % vname, mats, vc, root)
    ub = ubump_row(vc, sp, name=vname)
    variants[vname] = vc
    C.hook("mount_%s" % vname, vc, root, loc=(0, 0, 0))

# ---- variant: cutaway (section at y = 0, front half removed; looking from -Y onto the cut face)
sp = specs["16hi"]
vc = variant_coll("cutaway_16hi", False)
variants["cutaway_16hi"] = vc
zs_ = pk.z_exaggeration("hbm_stack_cutaway_zscale", vc, root, root)
hbm.section_parts(sp, vc, zs_, src_c, root, "hbm_stack_cutaway", ubump_pts=hbm.bump_points(sp))
C.hook("mount_cutaway_16hi", vc, root, loc=(0, 0, 0))

# ---- hooks and props
C.hook("top_center", coll, root, loc=(0, 0, sp.height * 1.0))
dims.add("cutaway z exaggeration (custom property p_z_exaggeration)", 1.0, "x", "default 1 = true scale; set 6-10 for readable layers; driver on empty hbm_stack_cutaway_zscale", "A")
C.hook("mount_bottom_center", coll, root, loc=(0, 0, 0))

# ---- dimension table
S = "see sources"
dims.add("footprint (base die / package) X", 11.0, "mm", "HBM3/HBM4-class stack footprint, 'about 11 x 11 mm' (project brief); not verified against paywalled JESD270-4; plausible range 10.5-12.5", "B")
dims.add("footprint Y", 11.0, "mm", "same", "B")
dims.add("stack thickness, 16-Hi (HBM4 limit)", 0.775, "mm", "JEDEC relaxed HBM4 height to 775 um (TechPowerUp; TweakTown); stack modeled to this height", "B")
dims.add("stack thickness, 12-Hi (HBM3 limit)", 0.720, "mm", "720 um HBM3 height limit (TechPowerUp)", "B")
dims.add("DRAM core die size", "10.4 x 10.4", "mm", "assumption: 0.3 mm setback each side of the base die; real DRAM die is not square (range 9-11)", "C")
dims.add("DRAM core die thickness 16-Hi", 25.0, "um", "20-25 um quoted for 16-Hi HBM4 at 775 um (Patsnap HBM4 stack configurations); 30-50 um for 12-Hi class", "B")
dims.add("DRAM core die thickness 12-Hi", 30.0, "um", "30-50 um range quoted for HBM3E (Wevolver); lower end", "B")
dims.add("base die thickness", 150.0, "um", "assumption (base die is thicker than core dies for handling); not sourced", "C")
dims.add("top die thickness", 100.0, "um", "assumption (thicker cap die for mechanical stability); not sourced", "C")
dims.add("bond line (microbump + NCF/MUF) 16-Hi", round(specs["16hi"].gap_um, 2), "um", "solved so the stack height meets 775 um; Patsnap quotes about 5 um joint gap at 16-Hi (we get a larger value because base/top die thicknesses are assumed)", "C")
dims.add("bond line 12-Hi", round(specs["12hi"].gap_um, 2), "um", "solved to 720 um", "C")
dims.add("microbump pitch (modeled PHY band)", 55.0, "um", "55 um HBM microbump pitch (FormFactor SWTW2016 Loranger, probing HBM microbumps); HBM4 pitch may be finer (SemiEngineering notes microbumps retained for HBM4)", "B")
dims.add("microbump diameter", 35.0, "um", "assumption (about 0.6 x pitch)", "C")
dims.add("TSV diameter", 6.0, "um", "5-10 um quoted for HBM TSVs (Wevolver HBM3 guide)", "B")
dims.add("TSV pitch (cutaway)", 55.0, "um", "assumed equal to microbump pitch", "C")
dims.add("number of modeled underside microbump sites", len(hbm.bump_points(sp)), "count", "plausible stand-in map: 55 um PHY band 8.8 x 3.6 mm + 110 um power/ground array; real HBM4 has 2048 data I/O plus power, not published", "C")

meta = {
    "title": "HBM4-class stack (16-Hi default, 12-Hi, molded and cutaway variants)",
    "accuracy_level": "B (footprint/height from public figures); internal map C",
    "origin": "bottom center of the footprint at the microbump tips (z = 0); the stack sits on a z = 0 plane as if placed on an interposer pad with bumps landed",
    "sources": [
        {"what": "JEDEC HBM4 package height relaxed to 775 um (HBM3: 720 um)", "url": "https://www.techpowerup.com/320314/jedec-agrees-to-relax-hbm4-package-thickness", "accessed": "2026-10-02"},
        {"what": "HBM4 die thickness 20-25 um at 16-Hi, 775 um limit", "url": "https://eureka.patsnap.com/report-hbm4-stack-configurations-12-hi-and-16-hi-structures-and-reliability", "accessed": "2026-10-02"},
        {"what": "HBM microbump pitch 55 um, probing", "url": "https://www.formfactor.com/wp-content/uploads/S01_02_Loranger_SWTW2016-2.pdf", "accessed": "2026-10-02"},
        {"what": "HBM3 TSV size and die thickness ranges", "url": "https://www.wevolver.com/article/what-is-high-bandwidth-memory-3-hbm3-complete-engineering-guide-2025", "accessed": "2026-10-02"},
        {"what": "HBM4: 2048 I/O, 12/16-Hi, microbumps retained", "url": "https://semiengineering.com/hbm4-sticks-with-microbumps-postponing-hybrid-bonding/", "accessed": "2026-10-02"},
    ],
    "variants": {
        "VARIANT_exposed_16hi": "default: stacked dies with visible bond-line beads (NCF/underfill squeeze-out) and the gold-tan top die",
        "VARIANT_molded_16hi": "molded underfill (MR-MUF-like) flush with the top die; mold compound walls",
        "VARIANT_exposed_12hi": "12-Hi, 720 um stack",
        "VARIANT_cutaway_16hi": "section at y = 0 (front half removed): shows base die, 15 core dies, top die, bond lines, TSV columns, microbump joints, mold",
    },
    "instancing": "microbumps, TSVs and joint bumps are Geometry Nodes instances (NG_pk_scatter) of hidden sources in the SOURCES collection",
    "simplifications": ["no HBM internal circuitry or die seal rings", "no bump map (stand-in); TSVs only in the cutaway variant", "side fillets are uniform beads"],
    "scene_usage": "S4 (16 stacks on the XPU package); S3 for die tower/stack gags if needed",
    "variants_visibility": "only VARIANT_exposed_16hi is visible by default; others have hide_render and hide_viewport set on their collection",
}

# finish: save, then per-variant previews
blend = os.path.join(OUT, AID + ".blend")
bb, tris = pk.eval_stats(variants["exposed_16hi"])
pv = os.path.join(OUT, "previews")
C.save(blend)
V = {}
all_pngs = []
H = sp.height


def views(prefix):
    return [dict(name=prefix + "_front", loc=(0, -26, 6), tgt=(0, 0, 0.4), lens=60, floor=True),
            dict(name=prefix + "_three_quarter", loc=(14, -16, 11), tgt=(0, 0, 0.3), lens=55, floor=True),
            dict(name=prefix + "_top", loc=(0, -0.2, 34), tgt=(0, 0, 0.4), lens=55),
            dict(name=prefix + "_edge_closeup", loc=(5.4, -9.0, 1.7), tgt=(4.6, -4.8, 0.35), lens=85, floor=True),
            dict(name=prefix + "_underside", loc=(0, -6, -22), tgt=(0, 0, 0.1), lens=55)]


per_stats = {}
for vname, vc in variants.items():
    for v2, c2 in variants.items():
        c2.hide_render = (v2 != vname)
        c2.hide_viewport = (v2 != vname)
    bpy.context.view_layer.update()
    if vname == "cutaway_16hi":
        vs = [dict(name="cutaway_section", loc=(0, -14, 2.6), tgt=(0, 0, 0.4), lens=60, floor=True),
              dict(name="cutaway_closeup_layers", loc=(-4.2, -2.4, 0.9), tgt=(-4.6, 0, 0.35), lens=85, floor=True),
              dict(name="cutaway_closeup_tsv", loc=(-4.9, -0.45, 0.45), tgt=(-4.95, 0, 0.36), lens=100, floor=True),
              dict(name="cutaway_three_quarter", loc=(10, -12, 8), tgt=(0, 2, 0.4), lens=55, floor=True)]
        root["p_z_exaggeration"] = 1.0
        pngs = pk.render_views(vc, pv, "hbm_stack", vs)
        root["p_z_exaggeration"] = 8.0
        bpy.context.view_layer.update()
        pngs += pk.render_views(vc, pv, "hbm_stack", [dict(name="cutaway_z8", loc=(-1.5, -9.0, 5.5), tgt=(-3.5, 0, 2.2), lens=70, floor=True)])
        root["p_z_exaggeration"] = 1.0
        bpy.context.view_layer.update()
    elif vname == "exposed_16hi":
        pngs = pk.render_views(vc, pv, "hbm_stack", views("exposed_16hi"))
    else:
        pngs = pk.render_views(vc, pv, "hbm_stack", views(vname)[:3])
    all_pngs += pngs
    per_stats[vname] = pk.eval_stats(vc)[1]
# restore default visibility and save again (render toggles do not change saved file; set explicit)
for v2, c2 in variants.items():
    c2.hide_render = (v2 != "exposed_16hi")
    c2.hide_viewport = (v2 != "exposed_16hi")
C.save(blend)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)],
             "triangles_including_instances_default_variant": tris, "triangles_by_variant": per_stats,
             "dimension_table": dims.rows, "hooks": hooks, "material_slots": mnames, "custom_properties": props,
             "previews": [os.path.relpath(p, OUT) for p in all_pngs],
             "units": "1 Blender unit = 1 m; z up; -Y front"})
pk.write_json(blend, meta)
print("DONE", AID, tris, per_stats)
