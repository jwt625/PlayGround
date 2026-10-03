"""nvl72_backplane_cartridges: stylised NVL72 copper-cable cartridge band for the S2 rack-interior backplane (new 2026-10-02).
One band = 4 vertical cartridge modules side by side (each 104 x 14 x 112 mm real size, 108 mm pitch) with a dark frame, a window showing a
bundle of 56 silver twinax cables bowing between black end connector blocks with gold contact strips, a blank label plate and a pull handle.
DEVIATION from the real system: the real cable cartridges sit at the rack REAR; this band is shown on the backplane FRONT face (stylised) so
the cartridges are visible in the interior camera rise. Origin: bottom centre of the band footprint; +y against the backplane, -y front.
Run: Blender -b --python scripts/assets/datacenter/build_nvl72_backplane_cartridges.py -- assets/components/datacenter
"""
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
AID = "nvl72_backplane_cartridges"
C.reset()
D.reset_mats()
L = D.lib()
mm = 0.001
mats = [L["dark_steel"], L["nickel"], L["black"], L["gold"], L["label"], L["alu"], L["copper"]]
b = MB()
W, Dp, H, PITCH = 104.0, 14.0, 112.0, 108.0
for k in range(4):
    cx = (k - 1.5) * PITCH
    b.boxmm((cx, Dp / 2 - 0.5, 1.5), (W, Dp - 1, 3), 0)               # bottom rail
    b.boxmm((cx, Dp / 2 - 0.5, H - 1.5), (W, Dp - 1, 3), 0)           # top rail
    for s in (-1, 1):
        b.boxmm((cx + s * (W / 2 - 1.5), Dp / 2 - 0.5, H / 2), (3, Dp - 1, H), 0)
    b.boxmm((cx, Dp - 0.5, H / 2), (W - 4, 1, H - 4), 0)              # back plate
    for z, m in ((8, 2), (H - 8, 2)):                                   # end connector blocks
        b.boxmm((cx, 6.5, z), (W - 10, 9, 10), m)
        b.boxmm((cx, 1.6, z), (W - 14, 0.6, 2.5), 3)
    for i in range(56):
        x = cx + (i - 27.5) * 1.6
        bow = 3.0 * math.sin(math.pi * ((i * 7) % 13) / 12.0) + 1.0
        zs = (13, H / 2, H - 13)
        pts = [(x * mm, (6.5 - 2.0) * mm, zs[0] * mm), (x * mm, (6.5 - 2.0 - bow) * mm, zs[1] * mm), (x * mm, (6.5 - 2.0) * mm, zs[2] * mm)]
        b.tube(pts, 0.7 * mm, 5, 1 if i % 4 else 6)
    b.boxmm((cx, -0.4, H / 2 - 18), (40, 0.8, 10), 4)                  # blank label plate
    b.boxmm((cx, -2.0, H - 5.5), (30, 3.0, 2.2), 5)                     # pull handle
    for s in (-1, 1):
        b.boxmm((cx + s * 13, -0.8, H - 5.5), (2.0, 2.4, 2.2), 5)
coll, root = C.new_asset(AID, accuracy="C")
o = b.obj(AID + "_band", mats, bevel=(0.0003, 1))
C.add(o, coll, root)
C.hook("backplane_contact_center", coll, root, (0, Dp, H / 2))
bpy.context.view_layer.update()
D.use_visible_bbox()
meta = dict(
    sources=[dict(what="4 cable cartridges in an NVL72 rack, >5,000 copper cables (Lenovo Press GB300 guide, SemiAnalysis); NVLink switch tray rear has 'cable cartridge connectors' (Lenovo GB300 NVL72 user guide)",
                  url="https://pubs.lenovo.com/gb300-nvl72/gb300-nvl72_user_guide.pdf", accessed="2026-10-02"),
             dict(what="existing library asset assets/components/interconnect/nvl72_copper_cartridge_wall (cartridge sizes were estimates there too)", url="local", accessed="2026-10-02")],
    dimensions=[D.dim("module", "104 x 14 x 112", "mm", "estimate (range 60-150 x 10-30 x 80-200)", "C"), D.dim("modules per band", 4, "-", "design", "-"),
                D.dim("cables per module", 56, "-", "decorative; not the real count (>5,000 per rack is sourced; per module unknown)", "C")],
    origin="bottom centre of the band footprint; +y against the backplane, -y front (stylised front-face placement)",
    hooks={"HOOK_backplane_contact_center": "centre of the back face"},
    simplifications=["real cartridges are at the rack rear; here on the backplane front", "cable count and routing decorative", "no text"],
    material_slots=D.mat_slots(), intended_usage="rack_interior_tray_stack backplane, one band per tray gap at detail scale 4")
blend = os.path.join(OUT, AID + ".blend")
C.finish(AID, blend, coll, meta, preview_dir=os.path.join(OUT, "previews"), views=("front", "three_quarter"), res=(900, 675))
