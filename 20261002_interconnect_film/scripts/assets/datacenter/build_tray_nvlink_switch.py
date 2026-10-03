"""tray_nvlink_switch: NVL72-style 1U NVLink switch tray (no OSFP; rear copper-cartridge blind-mate blocks, front maintenance panel) (cutaway: lid in its own collection).

Run: Blender -b --python scripts/assets/datacenter/build_tray_nvlink_switch.py -- assets/components/datacenter
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_parts as P
import common as C
import dc_common as D
import bpy

OUT, KV = D.parse_args()
AID = "tray_nvlink_switch"
C.reset()
D.reset_mats()
coll, root = C.new_asset(AID, accuracy="C")
root["p_lid_note"] = "toggle collection LID_cover to hide/show the lid (default: lid hidden = cutaway)"
subs = {"lid": C.sub_collection(coll, "VARIANT_lid_cover"), "rails": C.sub_collection(coll, "RAILS_rack_side")}
parts = P.switch_tray_parts("high", lid=True)
objs = P.instantiate(parts, coll, root, AID, subcolls=subs)
hoses = P.switch_tray_hoses(coll, root, AID)
H, Dm = P.TRAY_H * 0.001, P.SW_D * 0.001
C.hook("tray_handle_L", coll, objs["handles"], (-0.241, -Dm / 2 - 0.024, H / 2))
C.hook("tray_handle_R", coll, objs["handles"], (0.241, -Dm / 2 - 0.024, H / 2))
C.hook("tray_front_center", coll, root, (0, -Dm / 2 - 0.005, H / 2))
C.hook("qd_supply", coll, root, (-0.222, Dm / 2 + 0.027, H / 2))
C.hook("qd_return", coll, root, (0.222, Dm / 2 + 0.027, H / 2))
for i, x in enumerate((-0.140, -0.070, 0.070, 0.140)):
    C.hook("nvlink_rear_%d" % i, coll, root, (x, Dm / 2 + 0.022, H / 2))
C.hook("maintenance_panel", coll, root, (0.115, -Dm / 2 - 0.006, H / 2))
C.hook("power_rear", coll, root, (0, Dm / 2 + 0.020, H / 2))
C.hook("slide_travel_end", coll, root, (0, -Dm / 2 - 0.6, 0))
# lid hidden by default (cutaway); collection stays in the file
subs["lid"].hide_render = True
subs["lid"].hide_viewport = True

blend = os.path.join(OUT, AID + ".blend")
meta = dict(
    sources=[
        dict(what="9 x 1RU NVLink Switch trays per NVL72, each with NVLink Switch ASICs liquid cooled",
             url="https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html", accessed="2026-10-02"),
        dict(what="NVL72 NVLink spine: 5184 copper cables, blind-mate backplane", url="https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component", accessed="2026-10-02"),
        dict(what="Rear blind-mate connection to the copper backplane; front panel carries management only", url="https://www.fibermall.com/blog/nvidia-gb200-superchip.htm", accessed="2026-10-02"),
    ],
    dimensions=[
        D.dim("tray height", 43.0, "mm", "1U = 44.45 mm (EIA-310) minus ~1.5 mm clearance", "B"),
        D.dim("tray width incl. slides", 526.0, "mm", "ORV3 537 mm opening minus clearance (estimate)", "C"),
        D.dim("tray depth", 820.0, "mm", "estimate: same envelope as compute tray (flush fronts in public photos); range 700-900", "C"),
        D.dim("switch ASIC cold plates", "2 x 94 x 94 mm", "mm", "estimate: two NVLink Switch chips per tray (NVIDIA: 4 NVLink Switch chips per tray in some sources; SemiAnalysis/introl 'nine trays each supporting four NVLink Switch chips'): modelled 2 visible plates, simplified", "C"),
        D.dim("rear blind-mate blocks", "4 x 62 x 34 x 22", "mm", "estimate; real connector count differs (one block pair per spine bay column)", "C"),
    ],
    simplifications=["no brand logos", "only 2 ASIC cold plates modelled (sources mention 4 switch chips per tray)",
                     "board components generic; small unlabeled packages near rear blocks stand in for signal conditioning"],
    hooks_doc={"HOOK_nvlink_rear_0..3": "rear blind-mate block faces (x = -140, -70, 70, 140 mm)", "HOOK_maintenance_panel": "front maintenance panel",
               "HOOK_qd_supply/return": "rear coolant QD tips", "HOOK_tray_handle_L/R": "front handle centres"},
    custom_properties_doc={"p_lid_note": "documentation only"},
    origin="bottom centre of tray footprint; front at -y", intended_usage="S1/S2/S6 rack interior and rack previews",
    variants={"cutaway": "default: lid collection hidden", "closed": "unhide VARIANT_lid_cover"},
)
bpy.context.view_layer.update()
C.finish(AID, blend, coll, meta)
meta2 = dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll), custom_properties=D.custom_props(root))
D.extend_meta(blend, meta2)
# previews
pv = os.path.join(OUT, "previews")
vs = [dict(name="front", loc=(0, -0.85, 0.10), target=(0, -0.35, 0.02), lens=40),
      dict(name="three_quarter", loc=(0.75, -0.95, 0.55), target=(0, -0.02, 0.02), lens=40),
      dict(name="top", loc=(0, -0.02, 1.25), target=(0, -0.02, 0.0), lens=40),
      dict(name="chips_closeup", loc=(-0.05, -0.25, 0.20), target=(-0.05, 0.04, 0.02), lens=40),
      dict(name="rear", loc=(0.35, 0.95, 0.30), target=(0, 0.3, 0.02), lens=40)]
D.render_views(coll, pv, AID, vs, floor=True, floor_size=3.0, lights=[((0.4, -0.6, 0.9), 40, 1.0)])
subs["lid"].hide_render = False
subs["lid"].hide_viewport = False
D.render_views(coll, pv, AID, [dict(name="closed_three_quarter", loc=vs[1]["loc"], target=vs[1]["target"], lens=40)],
               floor=True, floor_size=3.0, lights=[((0.4, -0.6, 0.9), 40, 1.0)])
