"""Build qsfp_dd_module.blend (QSFP-DD 400G FR4-style pluggable, dual-LC receptacle, openable/explodable).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/interconnect/build_qsfp_dd_module.py -- assets/components/interconnect
Variants (front section, outside the cage): VARIANT_type1_flat (flat cap, visible by default) and VARIANT_type2a_finned (11 fins).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import ic_common as I
import ic_modules as MOD

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "qsfp_dd_module"

C.reset()
M = I.Mats()
coll, root = C.new_asset(ASSET, accuracy="A")
info = MOD.build_qsfpdd(coll, root, M)
v_flat = C.sub_collection(coll, "VARIANT_type1_flat")
v_fin = C.sub_collection(coll, "VARIANT_type2a_finned")
for nm, vc in (("qsfp_dd_module_front_cap_flat", v_flat), ("qsfp_dd_module_front_fins", v_fin)):
    o = bpy.data.objects[nm]
    for c in list(o.users_collection):
        c.objects.unlink(o)
    vc.objects.link(o)
I.hide_collection(v_fin, True)

meta = {
    "description": "QSFP-DD pluggable (generic 400G FR4-style: TOSA/ROSA, dual LC receptacle), openable via p_open. Type 1 front section 20+ mm (modelled 22 mm for the Type 2A heat sink length); Type 2A finned variant.",
    "sources": [
        {"what": "QSFP-DD Hardware Specification Rev 5.1 (2020-08-07): Fig 33-36 outline/paddle card, Fig 34/56 Type 2A heat sink, Fig 15/18 receptacles, App A length", "url": "http://www.qsfp-dd.com/wp-content/uploads/2020/08/QSFP-DD-Hardware-rev5.1.pdf", "accessed": "2026-10-02"},
        {"what": "Wentao's teardown photos: Innolight 200G FR4 TOSA (20250122_*), Intel 100G CWDM4 internals (20251219_CPO annotated), Innolight opened module (3DGS/innolight)", "path": "jwt625.github.io/assets/images/2025/", "accessed": "2026-10-02"},
    ],
    "dimension_table": [dict(item=a, value=b, unit=c, source=d, accuracy=e) for (a, b, c, d, e) in MOD.QSFP_DIMS],
    "origin": "root at the bottom centre of the module at datum D (forward stop). -Y = LC receptacle and pull tab, +Y = paddle card (insertion direction); module bottom is z = 0, front section extends to z = -1.6 below and 11.9 above (spec max).",
    "hooks": {
        "HOOK_open": "on the top shell (parent qsfp_dd_module_top_group); drive root[p_open] 0..1 (lifts top shell 30 mm, PCB assembly 12 mm)",
        "HOOK_plug_axis": "local +Y = insertion direction; translate the root along +Y to plug",
        "HOOK_receptacle": "centre of the dual-LC window (mate a duplex LC here)",
        "HOOK_pull_tab_end": "end of the pull loop", "HOOK_card_edge": "paddle card leading edge",
    },
    "custom_properties": {"p_open": "0..1 explode", "p_pull_mm": "0..30 mm pull-tab travel toward the front"},
    "variants": {"VARIANT_type1_flat": "default: flat cap on the front section (11.9 mm above bottom)", "VARIANT_type2a_finned": "hidden: Type 2A extruded heat sink, 11 fins x 0.4 mm"},
    "simplifications": ["boxes/prisms for shell, no ribs/screws; no EMI springs", "latch and elastomeric pull loop are plain strips", "generic passives from a seeded RNG (seed 11)", "DSP/TOSA/ROSA sizes are estimates (level C)", "pads: 2 rows x 19 per side, no traces"],
    "intended_usage": "S1/S2 hardware context, S3 connector-form-factor comparison; assembler scales up for macro.",
}
meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
blend = os.path.join(OUT, ASSET + ".blend")
prev = os.path.join(OUT, "previews")
meta["triangles_unique_meshes"] = I.unique_tris(coll)
C.finish(ASSET, blend, coll, meta, preview_dir=None)
tgt = (0, -30, 4.5)
I.preview_views(coll, prev, ASSET, [("front", (0, -45, 5), (0, -1, 0.15), 170), ("three_quarter", tgt, (0.85, -0.9, 0.7), 150), ("top", tgt, (0, -0.02, 1), 150),
                                   ("closeup_receptacle", (0, -70.2, 4.7), (0.35, -1, 0.25), 38), ("closeup_card_edge", (0, 6.0, 2.8), (0.3, 1, 0.55), 35)])
v_fin.hide_render = False
v_flat.hide_render = True
I.preview_views(coll, prev, ASSET, [("type2a_finned_three_quarter", tgt, (0.85, -0.9, 0.7), 150)])
v_fin.hide_render = True
v_flat.hide_render = False
root["p_open"] = 1.0
root.update_tag()
bpy.context.view_layer.update()
I.preview_views(coll, prev, ASSET, [("exploded_three_quarter", (0, -30, 8), (0.7, -0.9, 0.75), 150), ("exploded_closeup_optics", (0, -40, 8), (0.4, -0.8, 0.9), 50)])
root["p_open"] = 0.0
