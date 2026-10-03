"""tray_dummy_layouts: three board-dressing layouts for the S2 rack-interior stack (new 2026-10-02):
  tray_dummy_gb300_compute, tray_dummy_nvlink_switch, tray_dummy_helios_compute.
Each ASSET_<id> is ONE joined mesh (linked-duplicate friendly) at real size (feature mm), origin = tray centre at the board top surface
(x = 0, y = 0 mid-depth, z = 0 board top; -y front, +y backplane), so it drops onto HOOK_tray<j>_surface of rack_interior_tray_stack
with scale = the stack's detail scale (4). Free area used: x in +-245 mm, y in -183..-8 mm (front 3/4 of the board; rear 1/4 is the
connector and retimer zone). Plus hose runs along both side walls from the rear QD pair.
Component inventory follows the public tray lists (references/datacenter/SOURCES_s02_dummy_parts.json); positions are plausible generic (C).

Run: Blender -b --python scripts/assets/datacenter/build_tray_dummy_layouts.py -- assets/components/datacenter
"""
import json
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_board_parts_lib as B
import common as C
import dc_common as D
from dc_common import MB
import bpy

OUT, KV = D.parse_args()
C.reset()
D.reset_mats()
M = B.mats()
PI = math.pi
mm = 0.001
BUILT = {k: fn(M) for k, fn in B.PARTS.items()}


class Layout:
    def __init__(self):
        self.mb = MB()
        self.mats = []
        self.items = []

    def put(self, part, x, y, rot=0.0, z=0.0, sc=1.0, zs=1.0):
        pm, ms = BUILT[part]
        mmap = []
        for m in ms:
            if m not in self.mats:
                self.mats.append(m)
            mmap.append(self.mats.index(m))
        B.xform(pm, rot, (x * mm, y * mm, z * mm), mmap, self.mb, sc, zs)
        self.items.append(part)

    def hose(self, pts, r=4.0, m=None):
        if m not in self.mats:
            self.mats.append(m)
        self.mb.tube([(p[0] * mm, p[1] * mm, p[2] * mm) for p in pts], r * mm, 10, self.mats.index(m))
        self.items.append("hose")


def common_rear(Ly, plates):
    """QD pair at both rear corners and side-wall hose runs to the cold plates (plates: list of (x, y))."""
    for sx in (-1, 1):
        Ly.put("qd_pair", sx * 236, 183, 0.0)
        # hose from the QD stubs forward along the wall, then across to the nearest plate barb height z = 8
        py = [p for p in plates if (p[0] * sx) > 0]
        yend = py[0][1] if py else -85
        Ly.hose([(sx * 236, 170, 12), (sx * 244, 150, 12), (sx * 244, yend + 25, 10), (sx * 232, yend + 18, 8)], 4.0, M["hose"])


def gb300():
    Ly = Layout()
    for i in range(8):
        Ly.put("fan_module", (i - 3.5) * 42.0, -158, 0, 0, zs=0.6)
    for sx in (-1, 1):
        Ly.put("e1s_bank", sx * 222, -120, 0)
        Ly.put("coldplate_manifold", sx * 105, -92, 0)
        for k in range(3):
            Ly.put("socamm_module", sx * 105, -43 + k * 15, 0)
        Ly.put("inductor_bank", sx * 185, -22, 0)
    Ly.put("nic_dpu_card", 0, -93, PI / 2)
    Ly.put("vrm_block", 0, -30, 0)
    Ly.put("cap_bank", 56, -23, 0)
    Ly.put("cap_bank", -56, -23, 0)
    Ly.put("bmc_card", -190, -50, 0)
    Ly.put("coin_cell", 190, -50, 0)
    Ly.put("pcie_slot", 0, -12, 0)
    common_rear(Ly, [(-105, -92), (105, -92)])
    return Ly


def nvl_switch():
    Ly = Layout()
    for i in range(4):
        Ly.put("fan_module", (i - 1.5) * 48.0, -158, 0, 0, zs=0.6)
    for sx in (-1, 1):
        Ly.put("coldplate_manifold", sx * 80, -92, 0)
        Ly.put("heatsink_fin_stack", sx * 175, -92, 0)
        Ly.put("vrm_block", sx * 95, -30, 0)
        Ly.put("vrm_block", sx * 185, -30, 0)
        Ly.put("cap_bank", sx * 40, -22, 0)
        Ly.put("bmc_card", sx * 205, -158, 0)
    Ly.put("inductor_bank", 0, -45, 0)
    Ly.put("pcie_slot", 0, -12, 0)
    common_rear(Ly, [(-80, -92), (80, -92)])
    return Ly


def helios():
    Ly = Layout()
    for i in range(4):
        Ly.put("fan_module", (i - 1.5) * 42.0, -158, 0, 0, zs=0.6)
    for sx in (-1, 1):
        Ly.put("nic_dpu_card", sx * 165, -125, 0)
        Ly.put("e1s_bank", sx * 228, -120, 0)
    for k, x in enumerate((-120, -60, 0, 60, 120)):
        Ly.put("coldplate_manifold", x, -90, PI / 2, sc=0.62)
    for sx in (-1, 1):
        for k in range(4):
            Ly.put("dimm_module", sx * 66, -52 + k * 9, 0)
        Ly.put("vrm_block", sx * 200, -30, 0, sc=0.8)
        Ly.put("cap_bank", sx * 150, -20, 0)
    Ly.put("pcie_slot", 0, -12, 0)
    Ly.put("bmc_card", 0, -38, 0)
    common_rear(Ly, [(-120, -90), (120, -90)])
    return Ly


LAYOUTS = {"tray_dummy_gb300_compute": (gb300, "GB300 compute-tray style: 8 fans, E1.S banks at the sides, two cold-plate boards, 6 SOCAMM-style modules, DPU/NIC card, VRMs, BMC card, CR2032"),
           "tray_dummy_nvlink_switch": (nvl_switch, "NVLink switch-tray style: 4 fan modules, two switch cold plates, fin-stack heat sinks, 4 VRM patches, BMC cards"),
           "tray_dummy_helios_compute": (helios, "Helios-style compute tray: 6 fans, two OCP NIC 3.0 cards, E1.S banks, one CPU + four GPU cold plates, DIMM banks, VRMs")}
meta_l = {}
for aid, (fn, desc) in LAYOUTS.items():
    Ly = fn()
    coll, root = C.new_asset(aid, accuracy="C")
    o = Ly.mb.obj(aid + "_dressing", Ly.mats, bevel=(0.0003, 1))
    C.add(o, coll, root)
    C.hook("board_top_origin", coll, root, (0, 0, 0))
    root["p_detail_scale_hint"] = 4.0
    meta_l[aid] = dict(description=desc, parts=sorted(set(Ly.items)), part_counts={k: Ly.items.count(k) for k in sorted(set(Ly.items))})
bpy.context.view_layer.update()
for aid in LAYOUTS:
    bb = C.bbox_mm(bpy.data.collections["ASSET_" + aid])
    meta_l[aid]["size_mm"] = [round(bb[1][k] - bb[0][k], 1) for k in range(3)]
    meta_l[aid]["bbox_mm"] = [[round(v, 1) for v in bb[0]], [round(v, 1) for v in bb[1]]]
    meta_l[aid]["triangles"] = C.count_tris(bpy.data.collections["ASSET_" + aid])
blend = os.path.join(OUT, "tray_dummy_layouts.blend")
meta = dict(asset_id="tray_dummy_layouts", family=list(LAYOUTS), layouts=meta_l,
            origin="tray centre at the board top surface: x = 0, y = 0 (mid-depth), z = 0 board top; -y front, +y backplane; same frame as HOOK_tray<j>_surface of rack_interior_tray_stack",
            usage="instantiate per tray (linked duplicate of the mesh), scale = rack_interior_tray_stack p_detail_scale (4), place at HOOK_tray<j>_surface",
            accuracy="C (inventory from public component lists; positions and sizes generic, parts see dc_board_parts.json)",
            sources="references/datacenter/SOURCES_s02_dummy_parts.json", material_slots=D.mat_slots(),
            hooks={"HOOK_board_top_origin": "each layout root"}, custom_properties={"p_detail_scale_hint": 4.0},
            simplifications=["fan modules drawn at 60 percent height (24 mm instead of 40 mm) so the board stays visible from the low interior camera", "one joined mesh per layout (parts of dc_board_parts.blend baked in)", "hoses are straight polylines, not routed physically"])
C.write_meta(os.path.splitext(blend)[0] + ".json", meta)
C.save(blend)
# previews: each layout lying on a green board
pcb = D.lib()["pcb"]
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -0.0005))
pl = bpy.context.active_object
pl.scale = (0.54, 0.4, 1)
pl.name = "PREV_board"
pl.data.materials.append(pcb)
tmp = bpy.data.collections.new("PREV_ALL")
bpy.context.scene.collection.children.link(tmp)
for aid in LAYOUTS:
    tmp.children.link(bpy.data.collections["ASSET_" + aid])
tmp.objects.link(pl)
for k, aid in enumerate(LAYOUTS):
    bpy.data.objects["ROOT_" + aid].location = (0, -0.0, 0)
    bpy.data.collections["ASSET_" + aid].hide_render = True
bpy.context.view_layer.update()
pv = os.path.join(OUT, "previews")
for aid in LAYOUTS:
    for a2 in LAYOUTS:
        bpy.data.collections["ASSET_" + a2].hide_render = (a2 != aid)
        for ob in bpy.data.collections["ASSET_" + a2].objects:
            ob.hide_render = (a2 != aid)
    D.render_views(bpy.data.collections["ASSET_" + aid], pv, aid, [
        dict(name="top", loc=(0, -0.01, 0.7), target=(0, 0, 0), lens=34),
        dict(name="three_quarter", loc=(0.15, -0.45, 0.35), target=(0, -0.05, 0), lens=32)],
        floor=False, sun=0.4, world=0.35, lights=[((0.0, -0.3, 0.6), 25, 0.6)], clip=(0.01, 20))
print("DONE layouts")
