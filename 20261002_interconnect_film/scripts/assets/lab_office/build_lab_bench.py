"""ESD lab bench: laminate top, steel frame, lower shelf, riser back panel with power strip and outlets, monitor arm, task light,
drawer unit, ESD mat, cable troughs. Parametric width: -- <out_dir> [width_mm]  (default 1800).
Origin: bottom centre of the footprint, worktop top surface at z = 900 mm, -Y = front (operator side).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
from lo_common import C, M, S

args = C.argv_after_dashes()
OUT = args[0]
W = float(args[1]) if len(args) > 1 else 1800.0
AID = "lab_bench"
L.begin(AID, "B")
root = S.root

D = 750.0
TOP_Z = 900.0
TOP_T = 38.0
LEG = 50.0
RISE_Z = 1500.0
root["p_width_mm"] = W
root["p_worktop_height_mm"] = TOP_Z
root["p_drawers_open"] = 0.0   # informational; animate the drawer_N objects along -Y

m_fr = M("frame_powder")
m_top = M("laminate_esd")
m_steel = M("zinc")
m_dark = M("abs_dark")
lx = W / 2 - 45.0   # leg centre x
ly = D / 2 - 45.0   # leg centre y

# ---------------------------------------------------------------- worktop
L.box("worktop", (W, D, TOP_T), (0, 0, TOP_Z - TOP_T / 2), m_top, r=4.0, seg=3)
L.box("worktop_edge_band", (W - 2, 3.0, TOP_T - 6.0), (0, -D / 2 - 0.4, TOP_Z - TOP_T / 2), M("alu_dark"), r=0.8)  # front edge trim

# ---------------------------------------------------------------- frame
leg_h = TOP_Z - TOP_T - 20.0
for sx in (-1, 1):
    for sy in (-1, 1):
        tall = sy > 0
        h = (RISE_Z - 20.0) if tall else leg_h
        n = "leg_%s%s" % ("l" if sx < 0 else "r", "b" if tall else "f")
        L.box(n, (LEG, LEG, h), (sx * lx, sy * ly, 20.0 + h / 2), m_fr, r=3.0)
        L.cyl(n + "_foot", 22.0, 20.0, (sx * lx, sy * ly, 10.0), M("rubber"), axis="z", seg=20, bev=2.0)
        L.cyl(n + "_leveler", 8.0, 14.0, (sx * lx, sy * ly, 24.0), M("steel"), axis="z", seg=12)
# apron rails under the top and lower shelf rails
Z_APR = TOP_Z - TOP_T - 35.0
Z_SH = 260.0
for z, nm in ((Z_APR, "top"), (Z_SH - 45.0, "shelf")):
    for sy in (-1, 1):
        L.box("rail_%s_%s" % (nm, "f" if sy < 0 else "b"), (W - 2 * LEG - 90.0, 25.0, 60.0 if nm == "top" else 40.0), (0, sy * ly, z), m_fr, r=2.0)
    for sx in (-1, 1):
        L.box("rail_%s_%s" % (nm, "l" if sx < 0 else "r"), (25.0, D - 2 * LEG - 90.0, 60.0 if nm == "top" else 40.0), (sx * lx, 0, z), m_fr, r=2.0)
L.box("lower_shelf", (W - 2 * LEG - 50.0, D - 2 * LEG - 20.0, 22.0), (0, 0, Z_SH - 22.0), M("laminate_white"), r=2.0)
L.box("shelf_lip_front", (W - 2 * LEG - 50.0, 6.0, 40.0), (0, -D / 2 + LEG + 12.0, Z_SH - 5.0), m_steel, r=1.0)
# gussets
for sx in (-1, 1):
    for sy in (-1, 1):
        L.box("gusset", (60.0, 60.0, 4.0), (sx * (lx - 40.0), sy * (ly - 40.0), Z_APR - 28.0), m_fr, r=0.6)

# ---------------------------------------------------------------- riser back panel, rails, power strip, outlets
PNL_Z0, PNL_Z1 = TOP_Z + 60.0, RISE_Z - 70.0
L.box("riser_panel", (W - 2 * LEG - 10.0, 3.0, PNL_Z1 - PNL_Z0), (0, ly + 5.0, (PNL_Z0 + PNL_Z1) / 2), M("frame_grey"), r=1.0)
for z in (PNL_Z0 + 60.0, PNL_Z1 - 60.0):
    L.box("riser_rail", (W - 2 * LEG - 10.0, 20.0, 20.0), (0, ly - 6.0, z), M("alu_dark"), r=1.5)
# power strip (surface strip, 10 NEMA-5-15 style outlets, switch, LED, breaker)
SL = min(W - 500.0, 1200.0)
SZ = PNL_Z0 + 150.0
sx0 = -(W / 2) + 600.0 if W > 1700 else -SL / 2 + 100.0
sy_ = ly - 6.0
L.box("power_strip_body", (SL, 48.0, 56.0), (sx0 + SL / 2 - 300.0 + 300.0 - SL / 2 + SL / 2 - SL / 2, sy_ - 24.0 + 24.0, SZ), M("alu_dark"), r=4.0)
ps_x0 = sx0
slots = []
for i in range(10):
    x = ps_x0 - SL / 2 + 110.0 + i * ((SL - 220.0) / 9.0) if False else ps_x0 + (-SL / 2 + 80.0) + i * ((SL - 260.0) / 9.0)
    slots.append(((30.0, 1.0, 30.0), (x, sy_ - 49.0 + 25.0 - 24.0, SZ)))
L.multi_box("power_strip_faces", [((30.0, 1.2, 30.0), (ps_x0 + (-SL / 2 + 80.0) + i * ((SL - 260.0) / 9.0), sy_ - 25.0, SZ)) for i in range(10)], M("plastic_white"), r=2.0)
prongs = []
for i in range(10):
    x = ps_x0 + (-SL / 2 + 80.0) + i * ((SL - 260.0) / 9.0)
    prongs += [((1.6, 1.0, 8.0), (x - 6.0, sy_ - 25.9, SZ + 3.0)), ((1.6, 1.0, 8.0), (x + 6.0, sy_ - 25.9, SZ + 3.0)), ((3.0, 1.0, 3.0), (x, sy_ - 25.9, SZ - 7.0))]
L.multi_box("power_strip_slots", prongs, M("abs_black"))
L.box("power_strip_switch", (30.0, 6.0, 22.0), (ps_x0 + SL / 2 - 60.0, sy_ - 27.0, SZ + 6.0), M("key_red"), r=2.0)
L.cyl("power_strip_led", 3.0, 1.0, (ps_x0 + SL / 2 - 60.0, sy_ - 27.5, SZ - 14.0), M("led_green"), axis="y", seg=12)
L.box("power_strip_breaker", (16.0, 8.0, 10.0), (ps_x0 + SL / 2 - 110.0, sy_ - 28.0, SZ - 16.0), M("abs_black"), r=1.0)
# supply cord from the strip end down the right rear leg
cord = L.catmull([(ps_x0 + SL / 2, sy_ - 5.0, SZ), (ps_x0 + SL / 2 + 50, sy_ - 5.0, SZ - 40), (lx - 40, sy_ - 20.0, TOP_Z + 30.0), (lx - 40, sy_ - 40.0, TOP_Z - 80.0), (lx - 40, sy_ - 50.0, 520.0)], 6)
L.tube("power_cord", cord, 4.0, (0, 0, 0), M("abs_black"), seg=8)
# wall-box duplex outlet on the panel (left)
dx = -W / 2 + 280.0
L.box("duplex_plate", (70.0, 3.0, 115.0), (dx, ly + 3.0 - 6.0, PNL_Z0 + 150.0), M("plastic_white"), r=1.5)
L.multi_box("duplex_faces", [((26.0, 1.2, 26.0), (dx, ly - 4.7, PNL_Z0 + 150.0 + dz)) for dz in (-27.0, 27.0)], M("plastic_white"), r=1.5)
L.multi_box("duplex_slots", [((1.6, 1.0, 7.0), (dx + sx * 5.5, ly - 5.4, PNL_Z0 + 150.0 + dz + 3.0)) for dz in (-27.0, 27.0) for sx in (-1, 1)] + [((3.0, 1.0, 3.0), (dx, ly - 5.4, PNL_Z0 + 150.0 + dz - 6.0)) for dz in (-27.0, 27.0)], M("abs_black"))
# ground point (common point) screw on the panel
L.cyl("ground_point", 6.0, 6.0, (dx + 90.0, ly - 4.0, PNL_Z0 + 150.0), M("chrome"), axis="y", seg=20, bev=0.8)

# ---------------------------------------------------------------- task light (LED bar on two arms from the riser top)
LZ = RISE_Z - 40.0
for sx in (-1, 1):
    L.box("light_arm_%s" % ("l" if sx < 0 else "r"), (20.0, 220.0, 12.0), (sx * 420.0, ly - 120.0, LZ), M("alu_dark"), r=2.0)
LB = min(1100.0, W - 600.0)
L.box("light_housing", (LB, 70.0, 36.0), (0, ly - 255.0, LZ - 20.0), M("case_light"), r=8.0, seg=3)
L.box("light_diffuser", (LB - 30.0, 50.0, 2.0), (0, ly - 255.0, LZ - 39.5), M("light_panel"), r=0.5, seg=1)
# ---------------------------------------------------------------- monitor arm (clamp, post, two-link arm, VESA head) + generic monitor
MXc = lx - 140.0
post_h = 420.0
L.box("arm_clamp_top", (70.0, 80.0, 12.0), (MXc, ly - 60.0, TOP_Z + 6.0), M("alu_dark"), r=2.0)
L.box("arm_clamp_bottom", (70.0, 80.0, 12.0), (MXc, ly - 60.0, TOP_Z - TOP_T - 6.0), M("alu_dark"), r=2.0)
L.box("arm_clamp_screw", (14.0, 14.0, TOP_T + 40.0), (MXc, ly - 60.0 + 25.0, TOP_Z - TOP_T / 2), M("steel"), r=2.0)
post = L.cyl("arm_post", 18.0, post_h, (MXc, ly - 60.0, TOP_Z + 12.0 + post_h / 2), M("alu_dark"), axis="z", seg=24, bev=1.5)
J1 = (MXc, ly - 60.0, TOP_Z + post_h)
L.box("arm_link_1", (26.0, 330.0, 26.0), (J1[0], J1[1] - 150.0, J1[2] + 6.0), M("alu_dark"), r=5.0, seg=3)
L.cyl("arm_joint_1", 20.0, 40.0, (J1[0], J1[1], J1[2] + 6.0), M("alu_dark"), axis="z", seg=24, bev=1.5)
J2 = (J1[0], J1[1] - 300.0, J1[2] + 6.0)
L.cyl("arm_joint_2", 18.0, 40.0, J2, M("alu_dark"), axis="z", seg=24, bev=1.5)
L.box("arm_link_2", (24.0, 230.0, 24.0), (J2[0] + 0.0, J2[1] - 105.0, J2[2] + 3.0), M("alu_dark"), r=5.0, seg=3)
J3 = (J2[0], J2[1] - 210.0, J2[2] + 3.0)
L.cyl("arm_head_pivot", 16.0, 36.0, J3, M("alu_dark"), axis="z", seg=24, bev=1.5)
L.box("arm_vesa_plate", (100.0, 6.0, 100.0), (J3[0], J3[1] - 18.0, J3[2]), M("abs_black"), r=1.5)
# generic 24 in monitor (no brand): 531 x 299 active, bezel 8 mm, depth 18 mm
mcz = J3[2]
L.box("monitor_body", (545.0, 18.0, 322.0), (J3[0], J3[1] - 36.0, mcz), M("abs_black"), r=4.0)
L.box("monitor_glass", (531.0, 1.0, 299.0), (J3[0], J3[1] - 45.5, mcz), M("glass_dark"))
L.box("monitor_back_hump", (240.0, 24.0, 160.0), (J3[0], J3[1] - 14.0, mcz), M("abs_dark"), r=8.0, seg=3)

# ---------------------------------------------------------------- drawer unit (right end, hangs under the worktop)
DW, DD, DH = 450.0, 600.0, 540.0
dcx = lx - 40.0 - DW / 2
dcz = TOP_Z - TOP_T - 12.0 - DH / 2
L.box("drawer_carcass", (DW, DD, DH), (dcx, 0, dcz), m_fr, r=2.0)
y_front = -DD / 2
dh = (DH - 40.0) / 3.0
handle_mat = M("alu")
for i in range(3):
    zc = dcz + DH / 2 - 14.0 - dh / 2 - i * (dh + 6.0)
    g, e = L.push_group("drawer_%d" % (i + 1), (dcx, y_front - 9.0, zc), sub=False)
    L.box("front", (DW - 16.0, 18.0, dh), (0, 0, 0), M("frame_grey"), r=2.0)
    L.box("handle_bar", (DW * 0.55, 14.0, 12.0), (0, -17.0, dh / 2 - 22.0), handle_mat, r=3.0, seg=2)
    L.box("handle_post_l", (8.0, 12.0, 8.0), (-DW * 0.25, -11.0, dh / 2 - 22.0), handle_mat, r=1.0)
    L.box("handle_post_r", (8.0, 12.0, 8.0), (DW * 0.25, -11.0, dh / 2 - 22.0), handle_mat, r=1.0)
    L.box("label_holder", (90.0, 1.0, 24.0), (0, -9.5, -dh / 2 + 28.0), M("paper"), r=0.3, seg=1)
    L.box("body", (DW - 40.0, DD - 70.0, dh - 20.0), (0, (DD - 70.0) / 2 + 9.0, 0), M("zinc"), r=2.0)
    L.hook("drawer_%d_handle" % (i + 1), (0, -24.0, dh / 2 - 22.0), parent=e, note="grab point; slide the drawer_%d empty along -Y (max travel about 450 mm)" % (i + 1))
    L.pop_group()
L.cyl("drawer_lock", 6.0, 3.0, (dcx + DW / 2 - 30.0, y_front - 19.0, dcz + DH / 2 - 14.0 - dh / 2 + dh * 0.3), M("chrome"), axis="y", seg=16)

# ---------------------------------------------------------------- ESD mat, snaps, ground cord, wrist strap jack
MATW, MATD = W - 700.0, 540.0
mx = -120.0
my = -D / 2 + 70.0 + MATD / 2
L.box("esd_mat", (MATW, MATD, 2.4), (mx, my, TOP_Z + 1.2), M("esd_mat"), r=0.8)
for k, off in enumerate((-60.0, 60.0)):
    L.cyl("esd_snap_%d" % (k + 1), 5.0, 4.0, (mx + MATW / 2 - 30.0, my + off, TOP_Z + 4.4), M("chrome"), axis="z", seg=16, bev=0.6)
snapx, snapy = mx + MATW / 2 - 30.0, my + 60.0
gc = L.catmull([(snapx, snapy, TOP_Z + 6.0), (snapx + 40, snapy + 20, TOP_Z + 30.0), (snapx + 90, snapy + 40, TOP_Z + 8.0), (lx - 60, ly - 80.0, TOP_Z + 3.0), (lx - 60, ly - 100.0, TOP_Z - TOP_T - 20.0)], 8)
L.tube("esd_ground_cord", gc, 2.0, (0, 0, 0), M("abs_black"), seg=8)
# 4 mm banana jack common point ground at the front apron, wrist strap
jx = lx - 160.0
L.box("ground_plate", (40.0, 3.0, 40.0), (jx, -D / 2 - 1.5, TOP_Z - TOP_T / 2), M("alu_dark"), r=1.0)
L.cyl("ground_banana_jack", 6.0, 12.0, (jx, -D / 2 - 8.0, TOP_Z - TOP_T / 2), M("key_green"), axis="y", seg=16, bev=0.8)

# ---------------------------------------------------------------- cable troughs
tw = W - 2 * LEG - 160.0
TZ = 700.0
TY = ly - 70.0
wires = [((tw, 4.0, 4.0), (0, TY + sy * 50.0, TZ + sz * 30.0)) for sy in (-1, 1) for sz in (-1, 0, 1) if not (sz == 0 and sy == 1)]
wires += [((tw, 4.0, 4.0), (0, TY + sy * 17.0, TZ - 30.0)) for sy in (-1, 1)]
wires += [((4.0, 104.0, 4.0), (-tw / 2 + i * 50.0, TY, TZ - 30.0)) for i in range(int(tw // 50) + 1)]
wires += [((4.0, 4.0, 60.0), (-tw / 2 + i * 150.0, TY + sy * 50.0, TZ)) for i in range(int(tw // 150) + 1) for sy in (-1, 1)]
L.multi_box("cable_trough_wire_basket", wires, M("steel"))
for sx in (-1, 1):
    L.box("trough_bracket_%s" % ("l" if sx < 0 else "r"), (6.0, 120.0, 12.0), (sx * (tw / 2 - 50.0), TY, TZ - 38.0), M("alu_dark"), r=1.0)
# vertical cable duct (cover) on the left rear leg and worktop grommets
L.box("vertical_cable_duct", (60.0, 38.0, TOP_Z - 120.0 - 340.0), (-lx + 60.0, ly - 40.0, 340.0 + (TOP_Z - 120.0 - 340.0) / 2), M("frame_grey"), r=2.0)
for sx in (-1, 1):
    L.cyl("grommet_%s" % ("l" if sx < 0 else "r"), 32.0, 3.0, (sx * (W / 2 - 150.0), ly - 120.0, TOP_Z + 1.5), M("abs_black"), axis="z", seg=24, bev=0.6)
    L.cyl("grommet_hole_%s" % ("l" if sx < 0 else "r"), 22.0, 3.2, (sx * (W / 2 - 150.0), ly - 120.0, TOP_Z + 1.6), M("glass_dark"), axis="z", seg=24)

# ---------------------------------------------------------------- hooks
L.hook("worktop_center", (0, 0, TOP_Z), note="centre of the worktop surface")
L.hook("scope_slot", (-300.0, 20.0, TOP_Z), note="bench_oscilloscope origin goes here (scope sits on its feet; scope centre y about 0 to +20)")
L.hook("clutter_left", (-W / 2 + 250.0, -100.0, TOP_Z), note="spot for mug/notebook")
L.hook("clutter_right", (W / 2 - 400.0, -100.0, TOP_Z), note="spot for DMM/PSU")
L.hook("esd_mat_snap", (snapx, snapy, TOP_Z + 6.0), note="ESD mat ground snap")
L.hook("monitor_screen_center", (J3[0], J3[1] - 46.5, mcz), note="monitor glass centre (generic dark glass, no image slot)")
L.hook("operator_stand", (0, -D / 2 - 450.0, 0), note="floor position for a standing operator in front of the bench")

# ---------------------------------------------------------------- dimensions
L.dim("worktop height", TOP_Z, "mm", "task spec 0.9 m; catalogue ESD benches are adjustable 30-36 in (762-914 mm), e.g. Pro-Line / Cisco-Eagle listings (search 2026-10-02)", "B")
L.dim("width (default)", W, "mm", "task spec 1.5-1.8 m; catalogue sizes 72 in (1829 mm) and 90 in", "B")
L.dim("depth", D, "mm", "task spec 0.75 m; catalogue 30 in (762 mm) and 36 in (914 mm)", "B")
L.dim("worktop thickness", TOP_T, "mm", "estimate; ESD laminate benches commonly 1.25-1.5 in", "C")
L.dim("leg / rail tube", "50 x 50", "mm", "estimate", "C")
L.dim("riser top height", RISE_Z, "mm", "estimate", "C")
L.dim("lower shelf top height", Z_SH, "mm", "estimate", "C")
L.dim("drawer unit", "450 x 600 x 540", "mm", "estimate", "C")
L.dim("ESD mat", "%.0f x %.0f x 2.4" % (MATW, MATD), "mm", "estimate; typical ESD mat 2 mm thick dissipative rubber with 10 mm snap studs", "C")
L.dim("power strip", "%.0f x 48 x 56, 10 outlets" % SL, "mm", "estimate; catalogue ESD benches list 12-outlet strips with switch and breaker", "C")
L.dim("monitor", "24 in class, 531 x 299 active", "mm", "computed from 24 in 16:9", "A")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="typical ESD workbench sizes, laminate top, power strip, light", url="https://www.zoro.com/pro-line-bolted-workbench-with-riser-esd-laminate-72-in-w-30-in-to-36-in-height-5-000-lb-straight-tshd7230esd-l14/i/G9905183/ and https://www.cisco-eagle.com/product/156877/industrial-workbench-72w-x-36d-x-30h-esd-laminate-top", access_date="2026-10-02")],
    simplifications=["no cable contents in the trough", "monitor is a generic dark panel without an image slot", "drawer interiors are blocks", "light is an emissive diffuser, not a Blender light"],
    custom_properties={"p_width_mm": "build-time width (rebuild with a second argument to change)", "p_worktop_height_mm": "900", "p_drawers_open": "informational"},
    usage="S1 and S2 bench for the scope; scope origin on HOOK_scope_slot; clutter items on HOOK_clutter_left/right."))

# preview: append the bench scope (preview only, blend already saved)
sc_blend = os.path.join(OUT, "bench_oscilloscope.blend")
with bpy.data.libraries.load(sc_blend, link=False) as (src, dst):
    dst.collections = ["ASSET_bench_oscilloscope"]
col = dst.collections[0]
S.top_coll.children.link(col)
for o in col.all_objects:
    if o.name == "ROOT_bench_oscilloscope":
        o.location = (-0.3, 0.02, TOP_Z / 1000.0)
# put materials' images: plug eye
for mt in bpy.data.materials:
    if mt.name.startswith("MAT_lab_office_screen") and mt.use_nodes and "SCREEN_IMAGE" in mt.node_tree.nodes:
        L.plug_image(mt, L.tex_paths("eye_s1", 120)[119])
        break
# render bbox: bench only
views = [("front", dict(loc=(0, -3.4, 1.2), target=(0, 0, 0.8), lens=40)),
         ("three_quarter", dict(loc=(2.6, -2.8, 2.2), target=(0, 0, 0.8), lens=40)),
         ("top", dict(loc=(0, -0.2, 4.2), target=(0, -0.05, 0.9), lens=40)),
         ("back", dict(loc=(-1.8, 2.8, 1.6), target=(0, 0.2, 0.8), lens=40)),
         ("closeup_drawers", dict(loc=(1.5, -1.3, 0.75), target=(0.7, -0.2, 0.55), lens=40)),
         ("closeup_riser", dict(loc=(-0.3, -0.6, 1.5), target=(-0.3, 0.3, 1.2), lens=40))]
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=True, sun=1.8, world=0.6)
