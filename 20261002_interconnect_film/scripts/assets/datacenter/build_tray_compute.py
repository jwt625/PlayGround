"""tray_compute: NVL72-style 1U liquid-cooled compute tray (cutaway: lid in its own collection).

Run: Blender -b --python scripts/assets/datacenter/build_tray_compute.py -- assets/components/datacenter
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_parts as P
import common as C
import dc_common as D
import bpy

OUT, KV = D.parse_args()
AID = "tray_compute"
C.reset()
D.reset_mats()
coll, root = C.new_asset(AID, accuracy="C")
root["p_lid_note"] = "toggle collection LID_cover to hide/show the lid (default: lid hidden = cutaway)"
subs = {"lid": C.sub_collection(coll, "VARIANT_lid_cover"), "rails": C.sub_collection(coll, "RAILS_rack_side")}
parts = P.compute_tray_parts("high", lid=True)
objs = P.instantiate(parts, coll, root, AID, subcolls=subs)
hoses = P.compute_tray_hoses(coll, root, AID)
H, Dm = P.TRAY_H * 0.001, P.TRAY_D * 0.001
C.hook("tray_handle_L", coll, objs["handles"], (-0.241, -Dm / 2 - 0.024, H / 2))
C.hook("tray_handle_R", coll, objs["handles"], (0.241, -Dm / 2 - 0.024, H / 2))
C.hook("tray_front_center", coll, root, (0, -Dm / 2 - 0.005, H / 2))
C.hook("qd_supply", coll, root, (-0.222, Dm / 2 + 0.027, H / 2))
C.hook("qd_return", coll, root, (0.222, Dm / 2 + 0.027, H / 2))
C.hook("nvlink_rear_L", coll, root, (-0.110, Dm / 2 + 0.022, H / 2))
C.hook("nvlink_rear_R", coll, root, (0.110, Dm / 2 + 0.022, H / 2))
C.hook("power_rear", coll, root, (0, Dm / 2 + 0.020, H / 2))
C.hook("slide_travel_end", coll, root, (0, -Dm / 2 - 0.6, 0))
# lid hidden by default (cutaway); collection stays in the file
subs["lid"].hide_render = True
subs["lid"].hide_viewport = True

blend = os.path.join(OUT, AID + ".blend")
meta = dict(
    sources=[
        dict(what="NVL72 compute tray = 1U, 18 per rack, 2 Bianca boards per tray (1 Grace + 2 Blackwell each), liquid cooled CPU/GPU/CX7",
             url="https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html", accessed="2026-10-02"),
        dict(what="Compute tray is 1U and holds 2 Bianca boards (NVL72); 5184 NVLink copper cables", url="https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component", accessed="2026-10-02"),
        dict(what="EIA-310 rack unit 44.45 mm; ORV3 21 in opening (537 mm) used for width", url="https://www.opencompute.org/ (Open Rack V3 spec, from memory; the OCP product page returned HTTP 403)", accessed="2026-10-02"),
        dict(what="Overall layout, hybrid liquid/air cooling, blind-mate rear connectors", url="https://www.fibermall.com/blog/nvidia-gb200-superchip.htm", accessed="2026-10-02"),
    ],
    dimensions=[
        D.dim("tray height", 43.0, "mm", "1U = 44.45 mm (EIA-310) minus ~1.5 mm clearance", "B"),
        D.dim("tray width incl. slides", 526.0, "mm", "ORV3 537 mm opening minus clearance (estimate)", "C"),
        D.dim("tray depth", 820.0, "mm", "estimate; rack 1068 mm deep, trays end before rear spine (range 780-900)", "C"),
        D.dim("cold plate size", "104 x 104 x 14", "mm", "estimate from package size (range 90-120)", "C"),
        D.dim("board layout", "2 boards x (GPU, CPU, GPU) in a line", "-", "Bianca = 1 Grace + 2 Blackwell per board (NVIDIA docs); geometry estimated", "C"),
        D.dim("fans", "6 x 40 mm", "mm", "estimate for air-cooled remainder (NVIDIA: hybrid cooling)", "C"),
    ],
    simplifications=["no brand logos; no DIMM/SOCAMM, no HBM visible (under cold plates)", "board components generic",
                     "tubing path schematic: supply -> board A chain -> cross -> board B chain -> return"],
    hooks_doc={"HOOK_tray_handle_L/R": "front hot-swap handle centres", "HOOK_tray_front_center": "front-face centre (label/pull point)",
               "HOOK_qd_supply/return": "rear blind-mate coolant QD tips", "HOOK_nvlink_rear_L/R": "rear NVLink blind-mate faces",
               "HOOK_power_rear": "rear power blade face", "HOOK_slide_travel_end": "front end of full slide travel (z=0)"},
    custom_properties_doc={"p_lid_note": "documentation only"},
    origin="bottom centre of tray footprint; front at -y", intended_usage="S6 close-ups, rack trays (rack_nvl72_style reuses the same generator)",
    variants={"cutaway": "default: lid collection VARIANT_lid_cover hidden", "closed": "unhide VARIANT_lid_cover"},
)
bpy.context.view_layer.update()
C.finish(AID, blend, coll, meta)
meta2 = dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll), custom_properties=D.custom_props(root))
D.extend_meta(blend, meta2)
# previews
pv = os.path.join(OUT, "previews")
vs = [dict(name="front", loc=(0, -0.95, 0.10), target=(0, -0.4, 0.02), lens=40),
      dict(name="three_quarter", loc=(0.75, -0.95, 0.55), target=(0, -0.02, 0.02), lens=40),
      dict(name="top", loc=(0, -0.02, 1.25), target=(0, -0.02, 0.0), lens=40),
      dict(name="chips_closeup", loc=(-0.05, -0.32, 0.20), target=(-0.12, -0.05, 0.02), lens=40),
      dict(name="rear", loc=(0.35, 0.95, 0.30), target=(0, 0.3, 0.02), lens=40)]
D.render_views(coll, pv, AID, vs, floor=True, floor_size=3.0, lights=[((0.4, -0.6, 0.9), 40, 1.0)])
subs["lid"].hide_render = False
subs["lid"].hide_viewport = False
D.render_views(coll, pv, AID, [dict(name="closed_three_quarter", loc=vs[1]["loc"], target=vs[1]["target"], lens=40)],
               floor=True, floor_size=3.0, lights=[((0.4, -0.6, 0.9), 40, 1.0)])
