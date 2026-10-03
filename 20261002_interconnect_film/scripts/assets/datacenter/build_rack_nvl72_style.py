"""rack_nvl72_style: liquid-cooled NVL72-style rack (600 x 1068 x 2236 mm), 18 compute + 9 switch trays, 8 power shelves.

Run: Blender -b --python scripts/assets/datacenter/build_rack_nvl72_style.py -- assets/components/datacenter
Variants are collection toggles (see meta.variants): closed, open (doors removed), trays_removed, one_tray_pulled.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_rack as R
import dc_parts as P
import common as C
import dc_common as D
import bpy
from mathutils import Vector

OUT, KV = D.parse_args()
AID = "rack_nvl72_style"
PULLED_SLOT = KV.get("pulled", 13)          # compute tray index 1..18 (bottom=1) shown pulled out in the pulled variant
C.reset()
D.reset_mats()
L = D.lib()
coll, root = C.new_asset(AID, accuracy="B")
root["p_door_front"] = 0.0     # rad, front door opening (0 closed, up to 2.0)
root["p_door_rear"] = 0.0
root["p_pull"] = 0.55          # m, pulled-out tray travel (variant one_tray_pulled)
cc = {k: C.sub_collection(coll, k) for k in
      ("FRAME", "SIDE_PANELS", "DOORS", "POWER_SHELVES", "TRAYS", "VARIANT_pulled_slot_seated", "VARIANT_pulled_slot_pulled")}

# ---- layout in OpenU (1 OU = 48 mm). bottom -> top. Estimate: public sources give the counts (18/9/8), not the order or spare space.
LAYOUT = []
LAYOUT += [("ps", n) for n in range(1, 5)]
LAYOUT += [("c", n) for n in range(5, 15)]
LAYOUT += [("s", n) for n in range(15, 24)]
LAYOUT += [("c", n) for n in range(24, 32)]
LAYOUT += [("blank", n) for n in range(32, 39)]
LAYOUT += [("ps", n) for n in range(39, 43)]

# ---- frame and static hardware
fparts = R.rack_frame_parts(L, n_trays_z=(5, 31))
fo = P.instantiate(fparts, cc["FRAME"], root, AID)
tray_ou = [n for k, n in LAYOUT if k in ("c", "s")]
sup, ret = R.tray_qd_heads(L, tray_ou)
for nm, mb, key in (("qd_heads_supply", sup, "qd_blue"), ("qd_heads_return", ret, "qd_red")):
    o = mb.obj(AID + "_" + nm, [L["nickel"], L[key]], smooth=True)
    D.put(o, cc["FRAME"], root)
# side panels, doors
P.instantiate(R.side_panel_parts(L), cc["SIDE_PANELS"], root, AID + "_side")
for nm, side, y in (("front", -1, -R.RACK_D / 2 - 8), ("rear", 1, R.RACK_D / 2 + 8)):
    hinge = bpy.data.objects.new("HOOK_door_%s_hinge" % nm, None)
    hinge.empty_display_type = "ARROWS"
    hinge.empty_display_size = 0.08
    cc["DOORS"].objects.link(hinge)
    hinge.parent = root
    hinge.location = (-R.RACK_W / 2 + 2 if nm == "front" else R.RACK_W / 2 - 2, y / 1000.0 * 1000 * 0.001 * 1000, R.BASE_H + 8)
    hinge.location = ((-0.298 if nm == "front" else 0.298), y * 0.001, (R.BASE_H + 8) * 0.001)
    P.instantiate(R.door_parts(L, hinge_side=(-1 if nm == "front" else 1)), cc["DOORS"], hinge, AID + "_door_" + nm)
    D.drive(hinge, "rotation_euler", 2, root, "p_door_" + nm, "p" if nm == "front" else "-p")

# ---- trays, shelves, blanks
parts_c_low = P.compute_tray_parts("low", lid=True)
parts_s_low = P.switch_tray_parts("low", lid=True)
parts_ps, ps_d = R.power_shelf_parts(L)
parts_blank, _ = R.blank_panel_parts(L)
cn = sn = pn = bn = 0
pulled = None
for kind, n in LAYOUT:
    z = (R.ou_z(n) + 2.5) * 0.001
    if kind in ("c", "s"):
        Dm = (P.TRAY_D if kind == "c" else P.SW_D)
        yc = (R.FRONT_Y + Dm / 2) * 0.001
        if kind == "c":
            cn += 1
            tid = "c%02d" % cn
        else:
            sn += 1
            tid = "s%02d" % sn
        is_pulled = (kind == "c" and cn == PULLED_SLOT)
        target = cc["VARIANT_pulled_slot_seated"] if is_pulled else cc["TRAYS"]
        e = bpy.data.objects.new("%s_tray_%s" % (AID, tid), None)
        e.empty_display_size = 0.05
        target.objects.link(e)
        e.parent = root
        e.location = (0, yc, z)
        P.instantiate(parts_c_low if kind == "c" else parts_s_low, target, e, "%s_tray_%s" % (AID, tid),
                      key=("c_low" if kind == "c" else "s_low"))
        if is_pulled:
            pulled = (n, yc, z)
    elif kind == "ps":
        pn += 1
        e = bpy.data.objects.new("%s_psu_shelf_%d" % (AID, pn), None)
        cc["POWER_SHELVES"].objects.link(e)
        e.parent = root
        e.location = (0, (R.FRONT_Y + ps_d / 2) * 0.001, z)
        P.instantiate(parts_ps, cc["POWER_SHELVES"], e, "%s_psu_shelf_%d" % (AID, pn), key="ps")
    else:
        bn += 1
        e = bpy.data.objects.new("%s_blank_%d" % (AID, bn), None)
        cc["TRAYS"].objects.link(e)
        e.parent = root
        e.location = (0, (R.FRONT_Y - 1.5) * 0.001, z)
        P.instantiate(parts_blank, cc["TRAYS"], e, "%s_blank_%d" % (AID, bn), key="blank")

# ---- pulled tray (high detail, lid off) driven by p_pull
n, yc, z = pulled
pe = bpy.data.objects.new("HOOK_pulled_tray", None)
pe.empty_display_type = "ARROWS"
pe.empty_display_size = 0.1
cc["VARIANT_pulled_slot_pulled"].objects.link(pe)
pe.parent = root
pe.location = (0, yc, z)
D.drive(pe, "location", 1, root, "p_pull", "%.5f - p" % yc)
parts_c_high = P.compute_tray_parts("high", lid=False)
pobjs = P.instantiate(parts_c_high, cc["VARIANT_pulled_slot_pulled"], pe, AID + "_tray_pulled", key="c_high")
P.compute_tray_hoses(cc["VARIANT_pulled_slot_pulled"], pe, AID + "_tray_pulled")
# rack-side slide members stay with the rack
st = bpy.data.objects.new(AID + "_tray_pulled_slot_static", None)
cc["VARIANT_pulled_slot_pulled"].objects.link(st)
st.parent = root
st.location = (0, yc, z)
for k in ("slide_outer_L", "slide_outer_R"):
    pobjs[k].parent = st
pobjs["slide_outer_L"].matrix_parent_inverse = st.matrix_world.inverted() if False else pobjs["slide_outer_L"].matrix_parent_inverse
# ---- hooks
y0 = R.FRONT_Y * 0.001
zc = lambda n_: (R.ou_z(n_) + 24) * 0.001
C.hook("cartridge_bay_L", cc["FRAME"], fo["cartridge_bay_L"], (-0.110, 0.340, (zc(5) + zc(31)) / 2), (0, 0, 0))
C.hook("cartridge_bay_R", cc["FRAME"], fo["cartridge_bay_R"], (0.110, 0.340, (zc(5) + zc(31)) / 2), (0, 0, 0))
C.hook("busbar_base", cc["FRAME"], fo["busbar"], (0, 0.338, 0.19))
C.hook("busbar_top", cc["FRAME"], fo["busbar"], (0, 0.338, 2.176))
C.hook("manifold_supply_inlet", cc["FRAME"], fo["manifold_supply"], (-0.222, 0.549, (R.BASE_H + 60) * 0.001))
C.hook("manifold_return_outlet", cc["FRAME"], fo["manifold_return"], (0.222, 0.549, (R.BASE_H + 60) * 0.001))
C.hook("rack_front_center", cc["FRAME"], root, (0, -0.534, 1.1))
C.hook("rack_top_center", cc["FRAME"], root, (0, 0, R.RACK_H * 0.001))
for i in (5, 14, 15, 23, 24, 31):
    C.hook("tray_front_ou%d" % i, cc["FRAME"], root, (0, y0 - 0.025, zc(i)))
# defaults: open variant
D.set_collection_visible(cc["VARIANT_pulled_slot_pulled"], False)
D.set_collection_visible(cc["DOORS"], False)
D.set_collection_visible(cc["SIDE_PANELS"], False)
bpy.context.view_layer.update()

def variant(name):
    on = dict(closed=("SIDE_PANELS", "DOORS", "TRAYS", "VARIANT_pulled_slot_seated"),
              open=("TRAYS", "VARIANT_pulled_slot_seated"),
              trays_removed=("SIDE_PANELS",),
              one_tray_pulled=("TRAYS", "VARIANT_pulled_slot_pulled")).get(name)
    for k in ("SIDE_PANELS", "DOORS", "TRAYS", "VARIANT_pulled_slot_seated", "VARIANT_pulled_slot_pulled"):
        D.set_collection_visible(cc[k], k in on)

blend = os.path.join(OUT, AID + ".blend")
src_nv = "https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html"
meta = dict(
    sources=[
        dict(what="Rack external 600 x 1068 x 2236 mm (+132 mm optional extension frame to 1200 mm deep), ~1.36 t, 48U MGX ORV3 rack, 18 compute + 9 switch trays, 8 x 33 kW power shelves (6 x 5.5 kW PSUs each)",
             url="https://www.supermicro.com/en/products/system/gpu/48u/srs-gb200-nvl72", accessed="2026-10-02"),
        dict(what="Same dimensions via search summaries of OCP / Cheval ORV3 NVIDIA MGX rack listing (the listing page itself returned HTTP 403)",
             url="https://www.opencompute.org/products/525/cheval-group-open-rack-v3-nvidia-mgx-rack-for-gb200-nvl72", accessed="2026-10-02"),
        dict(what="18 x 1RU compute trays and 9 x 1RU NVLink Switch trays; CPU/GPU/CX7/NVSwitch liquid cooled, rest air", url=src_nv, accessed="2026-10-02"),
        dict(what="Compute tray 1U with 2 Bianca boards; 5184 NVLink copper cables", url="https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component", accessed="2026-10-02"),
        dict(what="Busbar at rear, blind-mate liquid cooling nozzles, rack manifolds on either side (ORV3 MGX rack)", url="https://www.servethehome.com/liteon-shows-nvidia-gb200-nvl72-rack-at-ocp-summit-2024/", accessed="2026-10-02"),
        dict(what="Power shelves at top and bottom feeding a vertical rear busbar; trays blind-mate onto the copper backplane", url="https://www.fibermall.com/blog/nvidia-gb200-superchip.htm", accessed="2026-10-02"),
        dict(what="EIA-310 1U = 44.45 mm, 19 in; Open Rack OpenU 48 mm, 21 in opening (from OCP ORV3 spec knowledge; spec PDF not re-read)", url="https://www.opencompute.org/", accessed="2026-10-02"),
    ],
    dimensions=[
        D.dim("external width", 600, "mm", "Supermicro / OCP MGX ORV3 listing", "B"),
        D.dim("external depth (frame)", 1068, "mm", "Supermicro / OCP MGX ORV3 listing; +132 mm extension frame optional (not modelled)", "B"),
        D.dim("external height", 2236, "mm", "Supermicro / OCP MGX ORV3 listing; modelled incl. casters", "B"),
        D.dim("tray count", "18 compute + 9 switch + 8 power shelves", "-", "NVIDIA docs (trays), Supermicro (shelves)", "A"),
        D.dim("tray height", 43, "mm", "1U = 44.45 mm (EIA-310) less clearance, in 48 mm OpenU slots", "B"),
        D.dim("slot pitch", 48, "mm", "OpenU (ORV3); the '48U' nominal does not fit 2236 mm at 44.45 or 48 mm pitch: 42 OU modelled", "C"),
        D.dim("OU order bottom->top", "4 PS, 10 compute, 9 switch, 8 compute, 7 blank, 4 PS", "-", "counts public; order of compute vs switch (10/9/8) from public photos recollection; blanks fill the unexplained spare height", "C"),
        D.dim("tray width / depth", "526 x 820", "mm", "estimates (ORV3 537 mm opening)", "C"),
        D.dim("base + casters", 160, "mm", "estimate", "C"),
        D.dim("busbar", "2 x 22 x 28 mm bars, centre rear", "mm", "estimate", "C"),
        D.dim("manifolds", "2 x 44 mm OD vertical, x = +-222 mm, rear", "mm", "estimate; supply left, return right", "C"),
        D.dim("cartridge bays", "2 x 150 x 120 mm x 1.3 m, flanking the busbar", "mm", "estimate; empty bay frames only", "C"),
    ],
    simplifications=["trays are shells (lid on) except the pulled tray (high detail, lid off)", "no cable cartridges (other agent's asset mounts at HOOK_cartridge_bay_L/R)",
                     "no rack-level CDU, PDUs or fire/sensor hardware", "no logos; door has no brand plate",
                     "front door is a generic perforated door (NVL72 racks are often shipped without front doors)"],
    variants={
        "closed": "enable SIDE_PANELS, DOORS, TRAYS, VARIANT_pulled_slot_seated",
        "open (default saved state; doors removed)": "enable TRAYS, VARIANT_pulled_slot_seated; disable DOORS, SIDE_PANELS, VARIANT_pulled_slot_pulled",
        "trays_removed": "enable only SIDE_PANELS (or none) plus FRAME and POWER_SHELVES",
        "one_tray_pulled": "enable TRAYS and VARIANT_pulled_slot_pulled; disable VARIANT_pulled_slot_seated (tray c%02d, OU %d pulled; p_pull in metres)" % (PULLED_SLOT, pulled[0]),
    },
    hooks_doc={"HOOK_cartridge_bay_L/R": "front-bottom... centre of the empty bay frames at the tray-rear connector plane (y = +340 mm); bay envelope 150 x 120 mm x ~1.3 m",
               "HOOK_busbar_base/top": "busbar ends", "HOOK_manifold_supply_inlet / return_outlet": "rack-level coolant ports at bottom rear",
               "HOOK_pulled_tray": "animated: y location driven by p_pull", "HOOK_door_front_hinge / rear_hinge": "rotation z driven by p_door_front / p_door_rear (rad)",
               "HOOK_tray_front_ou*": "front face points in front of selected slots", "HOOK_rack_front_center, HOOK_rack_top_center": "framing aids"},
    custom_properties_doc={"p_door_front": "front door opening angle, rad (0..2)", "p_door_rear": "rear door opening angle, rad (0..2)",
                           "p_pull": "pulled tray travel, m (0..0.6)"},
    origin="bottom centre of footprint, floor at z=0, front at -y", intended_usage="S1 rack shots, S6 rack context, data hall row stand-in (use rack_stub for many racks)",
    layout_ou=[[k, n] for k, n in LAYOUT],
)
D.use_visible_bbox()
C.finish(AID, blend, coll, meta)
D.extend_meta(blend, dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll), custom_properties=D.custom_props(root)))

pv = os.path.join(OUT, "previews")
light = [((1.8, -3.5, 3.0), 60, 3.0), ((-2.0, 3.5, 2.5), 60, 3.0)]
variant("open")
vs = [dict(name="front", loc=(0, -5.2, 1.25), target=(0, 0, 1.15), lens=50),
      dict(name="three_quarter", loc=(3.4, -3.8, 1.9), target=(0, 0, 1.1), lens=40),
      dict(name="top", loc=(0, -0.05, 6.5), target=(0, 0, 1.0), lens=40),
      dict(name="rear", loc=(-2.8, 3.6, 1.8), target=(0, 0.3, 1.1), lens=40),
      dict(name="front_closeup", loc=(0.5, -1.4, 1.3), target=(0, -0.5, 1.0), lens=40)]
D.render_views(coll, pv, AID, vs, floor=True, floor_size=12, lights=light, exposure=0.0)
for vn in ("closed", "trays_removed", "one_tray_pulled"):
    variant(vn)
    D.render_views(coll, pv, AID, [dict(name="%s_three_quarter" % vn, loc=(3.4, -3.8, 1.9), target=(0, -0.2, 1.1), lens=40)],
                   floor=True, floor_size=12, lights=light)
variant("open")
D.render_views(coll, pv, AID, [dict(name="rear_closeup", loc=(0.9, 1.5, 1.3), target=(0, 0.4, 1.0), lens=40)], floor=True,
               floor_size=12, lights=light)
