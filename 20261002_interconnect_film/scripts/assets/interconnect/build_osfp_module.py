"""Build osfp_module.blend (OSFP Type 1 pluggable, finned top default + flat-top cover variant, openable/explodable).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/interconnect/build_osfp_module.py -- assets/components/interconnect
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ic_common as I
import ic_modules as MOD

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "osfp_module"

C.reset()
M = I.Mats()
coll, root = C.new_asset(ASSET, accuracy="A")
info = MOD.build_osfp(coll, root, M)
# variant collections: flat-top cover (closed-top heat sink) hidden by default
var_flat = C.sub_collection(coll, "VARIANT_flat_top")
import bpy
cover = bpy.data.objects["osfp_module_flat_top_cover"]
coll.objects.unlink(cover)
var_flat.objects.link(cover)
I.hide_collection(var_flat, True)
var_fin = C.sub_collection(coll, "VARIANT_finned_top")   # default (open top, 7 fins); nothing extra, documented

meta = {
    "description": "OSFP Type 1 pluggable transceiver (generic 800G DR8-style, MPO-16 APC receptacle), openable via p_open.",
    "sources": [
        {"what": "OSFP MSA Rev 5.22 (2025-08-09): outline, heat sink, card edge, latch pocket", "url": "https://www.osfpmsa.org/assets/pdf/OSFP_Module_Specification_Rev5_22.pdf", "accessed": "2026-10-02"},
        {"what": "Wentao's teardown photos: Innolight 400G OSFP DR4+ and 800G OSFP PSM8 (20250913_*), Cisco/Finisar 400G DR4, Intel 100G CWDM4", "path": "jwt625.github.io/assets/images/2025/", "accessed": "2026-10-02"},
        {"what": "Wentao's Innolight opened-module photo set (PCB, DSP with thermal pad, fiber stubs, edge connector)", "path": "~/Documents/3DGS/innolight/images/", "accessed": "2026-10-02"},
    ],
    "dimension_table": [dict(item=a, value=b, unit=c, source=d, accuracy=e) for (a, b, c, d, e) in MOD.OSFP_DIMS],
    "origin": "root at the bottom centre of the module at the forward-stop plane (datum B). Front (-Y) is the optical port and pull tab, +Y is the card edge (insertion direction). When fully plugged the root coincides with the cage forward-stop plane on the cage floor.",
    "hooks": {
        "HOOK_open": "on the top shell (parent osfp_module_top_group); marks the lift axis (+Z). Drive root[p_open] 0..1 to lift the top shell (40 mm) and the PCB assembly (16 mm).",
        "HOOK_plug_axis": "at the module mid height on the forward-stop plane; local +Y = insertion direction toward the host connector (translate the root along +Y to plug).",
        "HOOK_receptacle": "centre of the MPO-16 receptacle window; mate a fibre connector here along +Y",
        "HOOK_pull_tab_end": "end of the pull tab loop (parent osfp_module_tab_group)",
        "HOOK_card_edge": "centre of the card-edge leading edge",
    },
    "custom_properties": {
        "p_open": "0..1 explode/open (drivers on osfp_module_top_group.z and osfp_module_pcb_group.z)",
        "p_pull_mm": "0..30 mm pull-tab/latch travel toward the front (driver on osfp_module_tab_group.y)",
    },
    "variants": {"VARIANT_finned_top": "default: open-top heat sink, 7 fins x 1.00 mm, 8 slots x 1.57 mm (spec Fig 3-16 example 2)",
                 "VARIANT_flat_top": "closed-top cover plate over the fins (hidden by default; unhide hide_render/hide_viewport on the collection)"},
    "material_slots": sorted(m.name for m in bpy.data.materials if m.users),
    "simplifications": [
        "Shell halves are built from boxes and prisms (no internal ribs, screws, EMI spring fingers on the module, or label recess).",
        "Latch is represented by side pockets and underside rails with pull-tab loop; no spring mechanics.",
        "PCB shows no traces or vias; passives are generic boxes placed by a seeded RNG (seed 7).",
        "DSP is a generic 15 x 15 mm package; its marking is generic text (no logos).",
    ],
    "intended_usage": "S2 (retimer/DSP pluggable context), S3 (NPO connector comparison), S6 (AOC context); assembler scales up for macro shots.",
    "usage_notes": "Plug the module by translating the ROOT along +Y. Receptacle faces -Y.",
}
blend = os.path.join(OUT, ASSET + ".blend")
prev = os.path.join(OUT, "previews")
C.finish(ASSET, blend, coll, meta, preview_dir=None)
# previews
import ic_common
hide = lambda n, h: setattr(bpy.data.collections[n], "hide_render", h)
specs_closed = [("front", (0, -60, 6.5), (0, -1, 0.15), 200), ("three_quarter", (0, -50, 6.5), (0.85, -0.9, 0.7), 190), ("top", (0, -50, 6.5), (0, -0.02, 1), 190),
                ("closeup_receptacle", (0, -89.8, 6.1), (0.35, -1, 0.25), 45), ("closeup_card_edge", (0, 6.0, 5.0), (0.3, 1, 0.55), 40)]
I.preview_views(coll, prev, ASSET, specs_closed)
# flat-top variant
bpy.data.collections["VARIANT_flat_top"].hide_render = False
I.preview_views(coll, prev, ASSET, [("flat_top_three_quarter", (0, -50, 6.5), (0.85, -0.9, 0.7), 190)])
bpy.data.collections["VARIANT_flat_top"].hide_render = True
# exploded
root["p_open"] = 1.0
root.update_tag()
bpy.context.view_layer.update()
I.preview_views(coll, prev, ASSET, [("exploded_three_quarter", (0, -45, 14), (0.7, -0.9, 0.75), 190), ("exploded_closeup_optics", (0, -55, 14), (0.4, -0.8, 0.9), 70)])
root["p_open"] = 0.0
