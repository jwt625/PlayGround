"""rack_gb300_nvl72: GB300 NVL72-style liquid-cooled compute rack, true scale (600 W x 1068 D x 2236 H mm).

Sources and grades: research/racks_refs.md section 2 (Lenovo GB300 NVL72 user guide Figs. 2-12; CC BY GB200 photo for colours).
Frame: origin = floor centre of the footprint, FRONT = +X, width along Y (+Y = viewer's right seen from the front), Z up.
Run (one Blender process, slot-capped):
  scripts/bslot.sh -b --python scripts/assets/datacenter/build_rack_gb300_nvl72.py -- assets/components/datacenter [previews=1]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import rk_lib as K  # noqa: E402
import rk_cabling as CB  # noqa: E402
from rk_lib import MB, Part, mm  # noqa: E402

OUT, KV = K.parse_args()
AID = "rack_gb300_nvl72"
PREV = os.path.join(os.path.dirname(os.path.abspath(OUT)), "..", "..", "outputs", "assets", "datacenter")
PREV = os.path.normpath(KV.get("prev", PREV))

# ------------------------------------------------------------------ dimensions (mm)
W, D, H = 600.0, 1068.0, 2236.0          # Lenovo UG p. 6 (A)
U1 = 82.0                                # U1 bottom above floor (C: 2236 - 48U - top frame)
UH = 44.45                               # EIA-310 1U
XF = D / 2                               # front plane x = +534
RAIL_X = 480.0                           # front rail face (C)
TRAY_FACE = 492.0                        # tray faceplate front (C)
TRAY_D = 880.0                           # tray depth (C: Lenovo Fig. 4 aspect x 440 mm)
TRAY_REAR = TRAY_FACE - TRAY_D           # -388
BODY_W = 440.0                           # 19 in chassis body (B)
EAR_W = 482.6                            # 19 in with ears (A)
TH = 43.0                                # tray height in a 44.45 slot

# U layout measured on Lenovo Fig. 11 (B, +-1U), 6-shelf configuration, bottom -> top
LAYOUT = ([("open", n) for n in range(1, 7)] + [("ps", n) for n in range(7, 10)] + [("drip", 10)] +
          [("c", n) for n in range(11, 19)] + [("s", n) for n in range(19, 28)] + [("c", n) for n in range(28, 38)] +
          [("open", 38)] + [("ps", n) for n in range(39, 42)] + [("open", n) for n in range(42, 45)] +
          [("mgmt", n) for n in range(45, 47)] + [("open", n) for n in range(47, 49)])


def uz(n):
    return U1 + (n - 1) * UH


def m(x):
    return x * 0.001


K.reset()
L = K.palette()
coll, root = K.new_asset(AID, accuracy="B")
cc = {k: K.sub(coll, k) for k in ("FRAME", "SIDE_PANELS", "TRAYS", "POWER", "REAR", "LIQUID", "CABLES", "BLANKS",
                                   "REAR_CARTRIDGE_DETAIL", "VARIANT_seated_slots", "VARIANT_pulled_trays",
                                   "FRONT_CABLING")}


# ------------------------------------------------------------------ helpers in tray-local mm (face at x = 0, body toward -x)
def fbox(mb, y0, y1, z0, z1, x0, x1, mi=0):
    mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)


def osfp_cage(mb, y, z, mi_cage, mi_hole, w=23.9, h=14.3, out=1.5):    # opening 22.7 x 13.1 mm > OSFP 22.58 x 13.0
    K.port(mb, y, z, w, h, 0.6, 9.0, out, mi_cage, mi_hole)


def rj45(mb, y, z, mi_shell, mi_hole, mi_led=None):
    K.port(mb, y, z, 16.0, 13.6, 1.6, 6.0, 1.2, mi_shell, mi_hole)
    if mi_led is not None:
        fbox(mb, y - 6.5, y - 4.5, z + 4.6, z + 6.2, -1, 1.5, mi_led)
        fbox(mb, y + 4.5, y + 6.5, z + 4.6, z + 6.2, -1, 1.5, mi_led + 1)


def ears(mb, mi_ear, mi_handle):
    for s in (-1, 1):
        y0, y1 = sorted((s * BODY_W / 2, s * EAR_W / 2))
        fbox(mb, y0, y1, 0.5, TH - 0.5, -2.5, 0.5, mi_ear)
        yc = s * (BODY_W / 2 + 10.5)
        # vertical latch lever
        fbox(mb, yc - 3, yc + 3, 5, TH - 5, 0.5, 6.5, mi_handle)
        fbox(mb, yc - 3, yc + 3, TH - 9, TH - 5, 0.5, 11, mi_handle)


def tray_body(mb, mi):
    fbox(mb, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -TRAY_D, -2.0, mi)


# ------------------------------------------------------------------ compute tray (Lenovo Fig. 2 front, Fig. 3 rear)
def compute_tray():
    mats = [L["silver"], L["silver_dk"], L["nickel"], L["port_hole"], L["black"], L["led_g"], L["led_a"], L["rj45"],
            L["led_b"]]
    fr = MB()
    fbox(fr, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -2.0, 0.0, 0)            # faceplate
    ears(fr, 4, 1)
    # left lower: OSFP card (2 x 800G OSFP) + RJ-45 OS mgmt
    fbox(fr, -210, -126, 2, 20.5, -1, 0.6, 1)
    for y in (-190, -144):                                                     # measured on Fig. 2 (audit v4)
        osfp_cage(fr, y, 11.2, 2, 3)
    rj45(fr, -108, 11, 7, 3, 5)
    # control panel (centre): buttons, RJ-45 BMC, USB, micro-USB, mDP, LEDs
    fbox(fr, 44, 77, 2, 41, -1, 0.5, 4)
    for i, yy in enumerate((50, 60, 70)):
        fr.cyl((m(1.2), m(yy), m(36)), m(2.6), m(2.0), "x", 12, 1)
    rj45(fr, 54, 22, 7, 3)
    K.port(fr, 54, 8.5, 12.5, 5.6, 0.8, 5.0, 1.0, 2, 3)                           # USB 3.0 (recessed)
    fbox(fr, 64, 73, 26, 30, -2, 1.0, 2)                                          # micro-USB
    fbox(fr, 64, 73, 7, 13, -2, 1.0, 2)                                           # mini-DP
    fbox(fr, 68, 71, 20, 22, -1, 1.2, 5)                                          # NVL LED
    fbox(fr, 68, 71, 16, 18, -1, 1.2, 6)                                          # fault LED (amber, off-ish)
    # right upper: BlueField-3 bracket (RJ-45 + 2 QSFP)
    fbox(fr, 82, 210, 22.5, 41, -1, 0.6, 1)
    rj45(fr, 120, 31.5, 7, 3)
    for y in (156, 178):
        osfp_cage(fr, y, 31.8, 2, 3, w=19.6, h=10.5)                           # opening 18.4 x 9.3 > QSFP112 18.35 x 8.5
    # right lower: OSFP card (2 x 800G OSFP)
    fbox(fr, 82, 210, 2, 20.5, -1, 0.6, 1)
    for y in (114, 160):
        osfp_cage(fr, y, 11.2, 2, 3)
    fr.box((m(1.0), m(196), m(32)), (m(1), m(3), m(2.5)), 8)                       # ID LED
    # E1.S hot-swap bays (8)
    eb = MB()
    for i in range(8):
        y0 = -84 + i * 15.75
        fbox(eb, y0 + 0.8, y0 + 14.95, 2, 41, -1.5, 1.2, 0)
        fbox(eb, y0 + 3.0, y0 + 12.7, 6, 34, 1.2, 2.4, 1)                           # latch bar
        fbox(eb, y0 + 6.5, y0 + 9.2, 37, 39.5, 1.2, 1.8, 2)                          # activity LED
    # perforated vent areas (PCIe blank bracket upper left, small vents)
    pv = MB()
    fbox(pv, -208, -88, 22.5, 41, -0.5, 0.9, 0)
    fbox(pv, 78, 81, 2, 41, -0.5, 0.9, 0)
    # body and rear
    bd = MB()
    tray_body(bd, 0)
    rr = rear_connectors()
    return [Part("front", fr, mats, bevel=None), Part("e1s", eb, [L["silver_dk"], L["grey"], L["led_g"]]),
            Part("vents", pv, [L["perf_hex"]], uv_tile=0.006), Part("body", bd, [L["silver_dk"]]),
            Part("rear", rr, [L["black"], L["gold"], L["nickel"], L["qd_blue"], L["qd_red"], L["copper"], L["dark_steel"]])]


def rear_connectors():
    """Tray rear (Lenovo Fig. 3/8): return QD (+Y), 4 cartridge connector blocks, busbar connector, supply QD (-Y)."""
    rr = MB()
    xr = -TRAY_D
    fbox(rr, -BODY_W / 2, BODY_W / 2, 1, TH - 1, xr - 1.5, xr, 6)                    # rear plate
    for yc in (151, 70, -70, -141):
        fbox(rr, yc - 32, yc + 32, 8, 34, xr - 22, xr - 1.5, 0)                      # housing
        for dy in (-15, 15):
            fbox(rr, yc + dy - 12, yc + dy + 12, 12, 26, xr - 22.6, xr - 21, 1)       # contact field
        for dy in (-27, 27):
            rr.cyl((m(xr - 26), m(yc + dy), m(30)), m(2.2), m(10), "x", 8, 2)        # guide pins
    fbox(rr, -11, 11, 6, 37, xr - 26, xr - 1.5, 0)                                  # busbar clip housing
    for dy in (-5, 5):
        fbox(rr, dy - 2.5, dy + 2.5, 9, 34, xr - 34, xr - 24, 5)                      # copper fingers
    for yc, mi in ((205, 4), (-205, 3)):
        rr.cyl((m(xr - 16), m(yc), m(21.5)), m(9.5), m(30), "x", 16, 2)
        rr.cyl((m(xr - 8), m(yc), m(21.5)), m(12), m(8), "x", 16, mi)
    return rr


# ------------------------------------------------------------------ NVLink switch tray (Fig. 7 front, Fig. 8 rear)
def switch_tray():
    mats = [L["silver"], L["light_grey"], L["black"], L["nickel"], L["port_hole"], L["led_g"], L["led_a"], L["rj45"]]
    fr = MB()
    fbox(fr, -BODY_W / 2, -82, 0.5, TH - 0.5, -2.0, 0.0, 0)
    fbox(fr, -82, BODY_W / 2, 0.5, TH - 0.5, -2.0, 0.0, 1)                          # plain panel (light grey in Fig. 11)
    fbox(fr, -82, -80, 2, TH - 2, -2.0, 0.4, 0)
    ears(fr, 2, 3)
    fbox(fr, -212, -90, 2.5, 41, -1, 0.4, 2)                                        # recessed control area
    for i, yy in enumerate((-200, -189, -178)):
        fr.cyl((m(1.0), m(yy), m(34)), m(2.6), m(1.6), "x", 12, 3)
    fbox(fr, -168, -165, 33, 35.5, -0.5, 1.0, 5)                                    # status LED
    for k in range(4):
        fbox(fr, -157 + k * 5, -154 + k * 5, 33, 35.5, -0.5, 1.0, 5 if k < 3 else 6)  # NVLink LEDs 1-4
    K.port(fr, -109, 34, 13.0, 6.0, 0.8, 5.0, 1.0, 3, 4)                           # USB 2.0 (recessed)
    for k, yy in enumerate((-198, -175, -152, -129, -106)):                         # 5 x RJ-45
        rj45(fr, yy, 13, 7, 4)
    bd = MB()
    tray_body(bd, 0)
    return [Part("front", fr, mats), Part("body", bd, [L["silver_dk"]]),
            Part("rear", rear_connectors(), [L["black"], L["gold"], L["nickel"], L["qd_blue"], L["qd_red"], L["copper"],
                                             L["dark_steel"]])]


# ------------------------------------------------------------------ power shelf (Fig. 9 front, Fig. 10 rear)
PS_D = 700.0


def power_shelf():
    mats = [L["silver"], L["silver_dk"], L["black"], L["led_g"], L["led_a"], L["rj45"], L["port_hole"], L["grey"]]
    fr = MB()
    fbox(fr, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -2.0, 0.0, 1)
    for s in (-1, 1):                                                               # ears with round holes look
        y0, y1 = sorted((s * BODY_W / 2, s * EAR_W / 2))
        fbox(fr, y0, y1, 3, TH - 3, -2.5, 0.5, 0)
        fr.cyl((m(0.8), m(s * (BODY_W / 2 + 11)), m(TH / 2)), m(4.5), m(1.0), "x", 12, 2)
    fbox(fr, -219, -188, 1.5, TH - 1.5, -1, 0.6, 2)                                 # PMC module
    rj45(fr, -206, 26, 5, 6)
    fr.box((m(0.9), m(-212), m(10)), (m(1), m(3), m(3)), 3)
    fr.cyl((m(0.9), m(-202), m(10)), m(1.8), m(1.2), "x", 8, 1)
    fan = MB()
    pw = (BODY_W - 33) / 6
    for i in range(6):
        y0 = -186 + i * pw
        fbox(fr, y0 + 0.8, y0 + pw - 0.8, 1.5, TH - 1.5, -1, 0.8, 0)               # PSU face
        fbox(fr, y0 + 4, y0 + 13, 4, TH - 4, 0.8, 2.6, 7)                           # handle grip
        fbox(fr, y0 + pw - 7, y0 + pw - 4, 30, 33, 0.6, 1.6, 3)                     # power LED
        fbox(fr, y0 + pw - 7, y0 + pw - 4, 24, 27, 0.6, 1.6, 3)                     # health LED
        yc = y0 + pw / 2 + 4
        fan.cyl((m(1.0), m(yc), m(TH / 2)), m(17), m(1.2), "x", 24, 0)              # fan grille disc
        fan.ring((m(1.8), m(yc), m(TH / 2)), m(17.5), m(15.5), m(1.2), "x", 24, 1)
        fan.ring((m(1.8), m(yc), m(TH / 2)), m(9.5), m(8.0), m(1.2), "x", 20, 1)
        fan.cyl((m(1.9), m(yc), m(TH / 2)), m(4.5), m(1.2), "x", 12, 1)
        for k in range(4):
            a = 0.785 * (2 * k + 1)
            import math
            fan.box((m(1.9), m(yc + 8 * math.cos(a)), m(TH / 2 + 8 * math.sin(a))), (m(1.2), m(1.4), m(1.4)), 1)
    bd = MB()
    fbox(bd, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -PS_D, -2, 0)
    rr = MB()
    xr = -PS_D
    fbox(rr, -12, 12, 6, 37, xr - 24, xr, 0)                                        # busbar connector
    fbox(rr, 120, 190, 6, 37, xr - 40, xr, 1)                                       # power whip connector
    fbox(rr, -120, -100, 14, 28, xr - 6, xr, 2)                                     # RJ-45 x2
    fbox(rr, -95, -75, 14, 28, xr - 6, xr, 2)
    return [Part("front", fr, mats), Part("fans", fan, [L["black"], L["dark_steel"]]), Part("body", bd, [L["silver_dk"]]),
            Part("rear", rr, [L["copper"], L["black"], L["rj45"]])]


# ------------------------------------------------------------------ management switch (Lenovo Fig. 5: 48 RJ-45 + 4 QSFP28)
def mgmt_switch():
    mats = [L["black"], L["rj45"], L["port_hole"], L["led_g"], L["nickel"], L["grey"]]
    fr = MB()
    fbox(fr, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -2, 0, 0)
    for s in (-1, 1):
        y0, y1 = sorted((s * BODY_W / 2, s * EAR_W / 2))
        fbox(fr, y0, y1, 1, TH - 1, -2.5, 0.5, 5)
    for g in range(3):
        for c in range(8):
            for r in range(2):
                y = -196 + g * 104 + c * 12.4
                z = 12 + r * 17
                fbox(fr, y - 5.6, y + 5.6, z - 7, z + 7, -3, 0.8, 1)
                fbox(fr, y - 4.6, y + 4.6, z - 5.5, z + 4.0, -2, 1.0, 2)
    for k in range(4):
        osfp_cage(fr, 128 + k * 21, 21, 4, 2, w=19.6, h=10.5)
    for k in range(4):
        fbox(fr, 210, 213, 10 + k * 7, 12.5 + k * 7, -0.5, 1.0, 3)
    bd = MB()
    fbox(bd, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -460, -2, 0)
    return [Part("front", fr, mats), Part("body", bd, [L["dark_steel"]])]


def drip_pan():
    mb = MB()
    fbox(mb, -BODY_W / 2, BODY_W / 2, 0, 3, -TRAY_D, 0, 0)
    fbox(mb, -BODY_W / 2, BODY_W / 2, 0, 24, -2, 0, 0)
    for s in (-1, 1):
        fbox(mb, s * BODY_W / 2 - 1, s * BODY_W / 2 + 1, 0, 24, -TRAY_D, 0, 0)
    return [Part("pan", mb, [L["silver"]])]


# ------------------------------------------------------------------ rack frame (black)
def frame_parts():
    fr = MB()
    hw, hd = W / 2, D / 2
    # plinth
    fr.box2((m(-hd), m(-hw), 0), (m(hd), m(hw), m(U1 - 4)), 0)
    # corner posts
    for sx in (-1, 1):
        for sy in (-1, 1):
            fr.box((m(sx * (hd - 22)), m(sy * (hw - 22)), m((U1 + H) / 2)), (m(44), m(44), m(H - U1)), 0)
    # top frame + roof
    fr.box2((m(-hd), m(-hw), m(H - 22)), (m(hd), m(hw), m(H)), 0)
    # side mid rails (front-to-back) top and bottom
    for sy in (-1, 1):
        for z in (U1 + 10, H - 40):
            fr.box((0, m(sy * (hw - 22)), m(z)), (m(D - 44), m(40), m(20)), 0)
    # front outer frame stiles (with round holes look via bevel-less boxes)
    for sy in (-1, 1):
        fr.box((m(XF - 30), m(sy * (hw - 50)), m((U1 + H) / 2)), (m(14), m(30), m(H - U1 - 20)), 0)
        fr.box((m(-XF + 30), m(sy * (hw - 50)), m((U1 + H) / 2)), (m(14), m(30), m(H - U1 - 20)), 0)
    # inner depth rails carrying the slides
    for sy in (-1, 1):
        for xx in (RAIL_X - 120, 0, -380):
            fr.box((m(xx), m(sy * 236), m((U1 + H - 22) / 2)), (m(30), m(10), m(H - 22 - U1)), 0)
    # leveling feet / caster pockets
    ft = MB()
    import math
    for sx in (-1, 1):
        for sy in (-1, 1):
            ft.cyl((m(sx * 440), m(sy * 230), m(8)), m(26), m(16), "z", 16, 0)
    # EIA rails (front and rear) with square-hole texture
    rl = MB()
    for xx in (RAIL_X - 8, -420):
        for sy in (-1, 1):
            rl.box((m(xx), m(sy * 233), m((U1 + uz(49)) / 2)), (m(16), m(17), m(uz(49) - U1)), 0)
    # roof brush cutouts
    rf = MB()
    for sy in (-1, 1):
        rf.box((m(-300), m(sy * 150), m(H + 0.6)), (m(240), m(120), m(1.2)), 0)
    sp = MB()
    for sy in (-1, 1):
        sp.box((0, m(sy * (hw - 1.5)), m((U1 + H - 22) / 2)), (m(D - 10), m(3), m(H - U1 - 26)), 0)
    return ([Part("frame", fr, [L["frame"]], bevel=0.0012), Part("feet", ft, [L["nickel"]]),
             Part("rails", rl, [L["rail"]], uv_tile=mm(UH / 3)), Part("roof_brush", rf, [L["black"]])],
            [Part("side_panels", sp, [L["panel"]])])


# ------------------------------------------------------------------ rear: busbar, cable cartridges, manifolds, couplings
def rear_parts(z_lo, z_hi):
    parts = []
    # busbar (copper pair inside black shroud), full height U1..U48
    bb = MB()
    zb0, zb1 = uz(1), uz(49)
    for dy in (-12, 12):
        bb.box2((m(-440), m(dy - 9), m(zb0)), (m(-415), m(dy + 9), m(zb1)), 0)
    bb.box2((m(-445), m(-30), m(zb0)), (m(-441), m(30), m(zb1)), 1)
    for s in (-1, 1):
        bb.box2((m(-445), m(s * 30 - 2), m(zb0)), (m(-412), m(s * 30 + 2), m(zb1)), 1)
    parts.append(Part("busbar", bb, [L["copper"], L["black"]]))
    # cable cartridges flanking the busbar (cable-bundle texture) over the tray zone
    ct = MB()
    for s in (-1, 1):
        ct.box2((m(-470), m(s * 36), m(z_lo)), (m(-412), m(s * 188), m(z_hi)), 0)
    ct.box2((m(-470), m(-36), m(z_lo)), (m(-447), m(36), m(z_hi)), 0)                 # centre column covers the busbar (Fig. 12)
    parts.append(Part("cartridges", ct, [L["cables"]], uv_tile=0.04))
    # cartridge frame: 2 side rails, 9 bolted cross bars, 5 column dividers (Lenovo Fig. 12)
    cf = MB()
    for s in (-1, 1):
        cf.box2((m(-492), m(s * 212), m(z_lo - 20)), (m(-470), m(s * 246), m(z_hi + 20)), 0)
        for zz in (z_lo + (z_hi - z_lo) * 0.18, z_lo + (z_hi - z_lo) * 0.82):         # side latches
            cf.box2((m(-500), m(s * 222), m(zz - 30)), (m(-492), m(s * 240), m(zz + 30)), 1)
    bolts = MB()
    for k in range(9):
        zz = z_lo + (z_hi - z_lo) * k / 8.0
        cf.box2((m(-490), m(-246), m(zz - 14)), (m(-472), m(246), m(zz + 14)), 0)
        for j in range(7):
            yy = -210 + j * 70
            bolts.cyl((m(-492), m(yy), m(zz)), m(4.5), m(4), "x", 10, 0)
    for j in range(1, 5):
        yy = -228 + j * 91.2
        if abs(yy) > 40:
            cf.box2((m(-474), m(yy - 2), m(z_lo)), (m(-470), m(yy + 2), m(z_hi)), 1)
    parts.append(Part("cartridge_frame", cf, [L["dark_steel"], L["black"]], bevel=0.001))
    parts.append(Part("cartridge_bolts", bolts, [L["nickel"]]))
    return parts


def liquid_parts(tray_us):
    parts = []
    mf = MB()
    zb0, zb1 = uz(1) + 20, uz(48)
    import math
    for s, mi in ((1, 1), (-1, 2)):                                                 # +Y = return (red), -Y = supply (blue)
        mf.cyl((m(-455), m(s * 262), m((zb0 + zb1) / 2)), m(25), m(zb1 - zb0), "z", 20, 0)
        for n in tray_us:
            z = uz(n) + TH / 2
            mf.cyl((m(-455 + 26), m(s * 262), m(z)), m(7), m(26), "x", 10, 0)       # manifold QD stub
            mf.cyl((m(-455 + 40), m(s * 262), m(z)), m(10), m(6), "x", 12, mi)
    parts.append(Part("manifolds", mf, [L["steel"], L["qd_red"], L["qd_blue"]], smooth=True))
    hs = MB()
    for s in (1, -1):
        for n in tray_us:
            z = m(uz(n) + 21.5)
            a = (m(TRAY_REAR - 30), m(s * 205), z)
            b = (m(-420), m(s * 262), z)
            hs.tube([a, (m(TRAY_REAR - 48), m(s * 225), z), (m(-412), m(s * 255), z), b], m(6), 8, 0)
    parts.append(Part("tray_hoses", hs, [L["hose"]], smooth=True))
    # rack-level manifold couplings, 2 x 2 at the bottom rear (Lenovo Fig. 12 item 3): quick-disconnect bodies on a grey
    # mounting plate, facility hoses leaving rearward and down through the plinth (sizes C: body 56 mm OD, hose 44 mm OD)
    cp = MB()      # 0 steel body (matte), 1 red sleeve, 2 blue sleeve, 3 grey plate, 4 hose, 5 dark steel
    hs2 = MB()
    for s, mi in ((1, 1), (-1, 2)):
        for j, zc in enumerate((uz(2), uz(4) + 10)):
            yc = s * (95 + 70 * j)
            dx = 30.0                                                                                  # kept inside the 1068 mm footprint
            cp.box2((m(-450 + dx), m(yc - 48), m(zc - 48)), (m(-440 + dx), m(yc + 48), m(zc + 48)), 3)          # mounting plate
            cp.cyl((m(-470 + dx), m(yc), m(zc)), m(28), m(40), "x", 20, 0)                                  # female body
            for k in range(4):                                                                       # knurled sleeve
                cp.cyl((m(-497 + dx - k * 5), m(yc), m(zc)), m(31 if k % 2 == 0 else 29.5), m(5), "x", 20, mi)
            cp.cyl((m(-528 + dx), m(yc), m(zc)), m(22), m(18), "x", 16, 5)                                 # male nipple / hex
            hs2.tube([(m(-537 + dx), m(yc), m(zc)), (m(-520), m(yc), m(zc - 8)), (m(-512), m(yc + s * 10), m(zc - 50)),
                      (m(-510), m(yc + s * 10), m(2))], m(21), 14, 0)                                     # facility hose, down through the plinth
        cp.tube([(m(-455), m(s * 262), m(uz(1) + 25)), (m(-455), m(s * 200), m(uz(2))), (m(-450), m(s * 150), m(uz(2)))],
                m(22), 12, 0)
    parts.append(Part("manifold_couplings", cp, [L["dark_steel"], L["qd_red"], L["qd_blue"], L["grey"], L["hose"], L["steel"]],
                      smooth=True))
    parts.append(Part("facility_hoses", hs2, [L["hose"]], smooth=True))
    return parts


# ------------------------------------------------------------------ assemble
fparts, sparts = frame_parts()
K.instantiate(fparts, cc["FRAME"], root, AID + "_frame")
K.instantiate(sparts, cc["SIDE_PANELS"], root, AID + "_side")

proto = dict(c=compute_tray(), s=switch_tray(), ps=power_shelf(), mgmt=mgmt_switch(), drip=drip_pan())
counts = dict(c=0, s=0, ps=0, mgmt=0, drip=0)
tray_us = []
tray_index = []
for kind, n in LAYOUT:
    if kind == "open":
        continue
    counts[kind] += 1
    z = m(uz(n) + 0.7)
    tag = {"c": "compute", "s": "nvswitch", "ps": "power_shelf", "mgmt": "mgmt_switch", "drip": "drip_pan"}[kind]
    name = "%s_%s_%02d" % (AID, tag, counts[kind])
    target = cc["POWER"] if kind == "ps" else cc["TRAYS"]
    e = K.empty(name, target, root, (m(TRAY_FACE), 0, z))
    K.instantiate(proto[kind], target, e, name, key=kind)
    if kind in ("c", "s"):
        tray_us.append(n)
        tray_index.append(dict(kind=tag, n=counts[kind], u=n))
        K.hook("%s_%02d_front" % (tag, counts[kind]), cc["TRAYS"], root, (m(TRAY_FACE + 3), 0, m(uz(n) + UH / 2)))

z_lo, z_hi = uz(11), uz(38)
K.instantiate(rear_parts(z_lo, z_hi), cc["REAR"], root, AID + "_rear")

# ---- blank panels for the open front bays (toggle; Lenovo Fig. 11 shows the bays open, so hidden by default)
bl = MB()
for kind, n in LAYOUT:
    if kind == "open":
        fbox(bl, -EAR_W / 2, EAR_W / 2, uz(n) + 0.4, uz(n) + UH - 0.4, TRAY_FACE - 2.0, TRAY_FACE, 0)
        for s_ in (-1, 1):
            bl.cyl((m(TRAY_FACE + 0.6), m(s_ * 232.5), m(uz(n) + UH / 2)), m(3.5), m(1.2), "x", 10, 1)
K.instantiate([Part("blank_panels", bl, [L["panel"], L["nickel"]])], cc["BLANKS"], root, AID + "_blank")
K.set_visible(cc["BLANKS"], False)

# ---- reused from the interconnect film (read-only append): NVL72 cable-cartridge bands on the rear cartridge bays
reused = []
band = K.append_mesh("nvl72_backplane_cartridges.blend", "nvl72_backplane_cartridges_band")
if band is not None:
    reused.append("nvl72_backplane_cartridges_band (interconnect film assets/components/datacenter/nvl72_backplane_cartridges.blend): 8 bands, one per cartridge bay between the cross bars, scaled 0.95 in width, facing the rack rear")
    for k in range(8):
        zc = z_lo + (k + 0.5) * (z_hi - z_lo) / 8
        o = bpy.data.objects.new("%s_cartridge_band_%d" % (AID, k + 1), band)
        cc["REAR_CARTRIDGE_DETAIL"].objects.link(o)
        o.parent = root
        o.location = (m(-484), 0, m(zc - 56))
        o.rotation_euler = (0, 0, -math.pi / 2)
        o.scale = (0.95, 1, 1)

# ---- pulled-out tray variants (toggle): one compute tray and one NVLink switch tray slid out 0.55 m with the lid off,
#      dressed with the interconnect film's GB300 / NVLink switch tray layouts (uniformly scaled 0.8 to fit the 440 mm body)
PULL = 0.55
pulled_spec = [("c", 14, "tray_dummy_layouts.blend", "tray_dummy_gb300_compute_dressing"),
               ("s", 5, "tray_dummy_layouts.blend", "tray_dummy_nvlink_switch_dressing")]
open_pan = MB()
fbox(open_pan, -BODY_W / 2, BODY_W / 2, 0.5, 2.5, -TRAY_D, -2.0, 0)
for s_ in (-1, 1):
    fbox(open_pan, s_ * BODY_W / 2 - 1.5 * s_, s_ * BODY_W / 2, 0.5, TH - 0.5, -TRAY_D, -2.0, 0)
fbox(open_pan, -BODY_W / 2, BODY_W / 2, 0.5, TH - 0.5, -TRAY_D, -TRAY_D + 1.5, 0)
fbox(open_pan, -BODY_W / 2 + 6, BODY_W / 2 - 6, 2.5, 4.1, -TRAY_D + 20, -30, 1)                    # main board
pulled_hooks = []
for kind, idx, bfile, oname in pulled_spec:
    tag = {"c": "compute", "s": "nvswitch"}[kind]
    seated = bpy.data.objects.get("%s_%s_%02d" % (AID, tag, idx))
    n_u = [t["u"] for t in tray_index if t["kind"] == tag and t["n"] == idx][0]
    # move the seated tray (empty + children) into VARIANT_seated_slots so the two variants toggle cleanly
    for ob in [seated] + list(seated.children):
        for c_ in list(ob.users_collection):
            c_.objects.unlink(ob)
        cc["VARIANT_seated_slots"].objects.link(ob)
    hk = K.hook("pulled_%s_%02d" % (tag, idx), cc["VARIANT_pulled_trays"], root, (m(TRAY_FACE) + PULL, 0, m(uz(n_u) + 0.7)),
                note="pulled tray root; animate location.x between %.3f (seated) and %.3f (pulled)" % (m(TRAY_FACE), m(TRAY_FACE) + PULL))
    pulled_hooks.append(hk.name)
    parts_ = [p for p in proto[kind] if p.name != "body"]
    K.instantiate(parts_, cc["VARIANT_pulled_trays"], hk, "%s_pulled_%s" % (AID, tag), key=kind)
    K.instantiate([Part("open_pan", open_pan, [L["dark_steel"], L["pcb"]])], cc["VARIANT_pulled_trays"], hk, "%s_pulled_%s" % (AID, tag),
                  key="open_pan")
    me = K.append_mesh(bfile, oname)
    if me is not None:
        o = bpy.data.objects.new("%s_pulled_%s_dressing" % (AID, tag), me)
        cc["VARIANT_pulled_trays"].objects.link(o)
        o.parent = hk
        o.location = (m(-330), 0, m(4.1))
        o.rotation_euler = (0, 0, math.pi / 2)
        o.scale = (0.8, 0.8, 0.8)
        reused.append("%s (interconnect film tray_dummy_layouts.blend): dressing of the pulled %s tray, scale 0.8" % (oname, tag))
K.set_visible(cc["VARIANT_pulled_trays"], False)
K.instantiate(liquid_parts(tray_us), cc["LIQUID"], root, AID + "_liquid")

# power whips: upper shelves exit through the roof, lower shelves through the plinth
for k, n in enumerate([n for kind, n in LAYOUT if kind == "ps"]):
    z = m(uz(n) + 21)
    up = n > 20
    y = m(155 + (k % 3) * 12)
    pts = [(m(TRAY_FACE - PS_D - 40), y, z), (m(-300), y + m(20), z + (m(60) if up else -m(60))),
           (m(-280), y + m(30), m(H - 40) if up else m(U1 + 30)), (m(-280), y + m(30), m(H + 140) if up else m(12))]
    K.cable("%s_power_whip_%d" % (AID, k + 1), pts, m(11), L["cable_black"], cc["CABLES"], root)

# ---- front cabling (v4): CX-8 OSFP modules + MPO-12/APC + SMF jumpers to the side bundles and up to the overhead
#      fibre raceway drops; BlueField-3 QSFP112; slim Cat6 to the two in-rack management switches (section 12)
MPO_GREEN = K.M("mpo_apc_green", (0.05, 0.45, 0.12), 0.0, 0.4)
mods, fib, cat, strap = MB(), MB(), MB(), MB()
MI = dict(module=0, dark=1, mpo=2, boot=3, tab=4, cap=5, plug=6, cat6_boot=7)
mods_mats = [L["silver"], L["black"], MPO_GREEN, L["black"], L["cable_yellow"], L["rubber"], L["cable_blue"], L["cable_blue"]]
CAPPED_TRAYS = (5, 16)                         # compute trays shown with OSFP dust caps (not yet cabled)
PULL_TRAYS = {("compute", 14), ("nvswitch", 5)}  # pull-out variant trays: capped, unplugged, caps parented to the tray (seated + pulled)
FL = CB.Bundle("fiber_left", m(587), m(-258), m(3.3), m(CB.FIBER_R))
FR = CB.Bundle("fiber_right", m(587), m(258), m(3.3), m(CB.FIBER_R))
CL = CB.Bundle("cat6_left", m(632), m(-262), m(4.5), m(CB.CAT6_R))
CR = CB.Bundle("cat6_right", m(632), m(262), m(4.5), m(CB.CAT6_R))
fx0 = m(TRAY_FACE)
n_osfp = n_osfp_cap = n_qsfp = 0
for t in tray_index:
    zt = m(uz(t["u"]) + 0.7)
    P = lambda y, z: (fx0, m(y), zt + m(z))
    if (t["kind"], t["n"]) in PULL_TRAYS:
        continue
    if t["kind"] == "compute":
        for y in (-190, -144, 114, 160):
            if t["n"] in CAPPED_TRAYS:
                CB.osfp_cap(mods, P(y, 11.2), MI)
                n_osfp_cap += 1
            else:
                e = CB.osfp_module(mods, P(y, 11.2), MI)
                (FL if y < 0 else FR).add(e, e[2])
                n_osfp += 1
        e = CB.qsfp_module(mods, P(156, 31.8), MI)                      # BlueField-3 port 1 (front-end network)
        FR.add(e, e[2])
        n_qsfp += 1
        CB.qsfp_cap(mods, P(178, 31.8), MI)                              # port 2 capped
        for y, z, bnd in ((-108, 11, CL), (54, 22, CR), (120, 31.5, CR)):  # OS mgmt, BMC, BF3 OOB
            e = CB.rj45_plug(mods, P(y, z), MI)
            bnd.add(e, e[2])
    else:
        for y in (-175, -129):                       # NVLink switch tray: 2 of the 5 RJ-45 cabled (which jacks = design choice)
            e = CB.rj45_plug(mods, P(y, 13), MI)
            CL.add(e, e[2])
z_fib_top = m(H + 25)                                                    # bundles turn back over the roof here
DROP_LOCAL = (0.10, 0.20, m(H + 184))                                   # up into the raceway drop above the roof (bottom 2.30 m)


def fiber_roof(bnd):
    sgn = 1 if bnd.by > 0 else -1
    def f(k, top, end):
        dx, dy = top[0] - bnd.bx, top[1] - bnd.by
        yd = sgn * DROP_LOCAL[1] + dy
        ctrl = [top, (top[0] - 0.035, top[1], top[2] + 0.04), (top[0] - 0.12, (top[1] + yd) / 2, top[2] + 0.055),
                (0.16 + dx, yd, top[2] + 0.055), (DROP_LOCAL[0] + dx, yd, top[2] + 0.095), (DROP_LOCAL[0] + dx, yd, DROP_LOCAL[2])]
        return K.smooth_path(ctrl, 5)[1:]
    return f


FL.build(fib, strap, z_fib_top, fan_out=fiber_roof(FL), fwd=0.03)
FR.build(fib, strap, z_fib_top, fan_out=fiber_roof(FR), fwd=0.03)
mgmt_u = [n for kind, n in LAYOUT if kind == "mgmt"]                    # U45 (left bundle), U46 (right bundle)


def mgmt_port(unit_idx, k):
    g, c, r = (k // 16) % 3, (k % 16) // 2, k % 2
    y = -196 + g * 104 + c * 12.4
    z = uz(mgmt_u[unit_idx]) + 0.7 + 12 + r * 17
    return (fx0, m(y), m(z))


def fan(unit_idx):
    def f(k, top, end):
        px, py, pz = mgmt_port(unit_idx, k)
        e = CB.rj45_plug(mods, (px, py, pz), MI)
        ctrl = [top, (top[0], top[1], top[2] + 0.04), (m(588), (top[1] + py) / 2, pz - 0.012), (m(548), py, pz), e]
        return K.smooth_path(ctrl, 5)[1:]
    return f


z_cat_top = m(uz(mgmt_u[0]) - 70)
CL.build(cat, strap, z_cat_top, fan_out=fan(0), fwd=0.09)
CR.build(cat, strap, z_cat_top, fan_out=fan(1), fwd=0.09)
# pull-out trays: caps on every optical port (no cables), as a small mesh on the tray itself, once on the seated tray
# (VARIANT_seated_slots) and once on the pulled hook (VARIANT_pulled_trays), so both variants stay consistent
for kind_, n_, parent_name, coll_key in (("compute", 14, "%s_compute_14" % AID, "VARIANT_seated_slots"),
                                         ("compute", 14, "HOOK_pulled_compute_14", "VARIANT_pulled_trays")):
    pc = MB()
    for y in (-190, -144, 114, 160):
        CB.osfp_cap(pc, (0, m(y), m(11.2)), MI)
    for y in (156, 178):
        CB.qsfp_cap(pc, (0, m(y), m(31.8)), MI)
    par = bpy.data.objects.get(parent_name)
    if par is not None:
        K.instantiate([Part("caps", pc, mods_mats)], cc[coll_key], par, "%s_%s" % (parent_name, "caps"), key="pull_caps")
mins = [b.min_radius for b in (FL, FR, CL, CR)]
print("BEND_RADIUS_MIN_MM", [round(r * 1000, 1) for r in mins])
K.instantiate([Part("modules_connectors", mods, mods_mats), Part("fiber_jumpers", fib, [L["cable_yellow"]], smooth=True),
               Part("cat6_patch", cat, [L["cable_blue"]], smooth=True), Part("velcro_straps", strap, [L["black"]])],
              cc["FRONT_CABLING"], root, AID + "_front")
cabling_meta = dict(min_bend_radius_mm={b.name: round(b.min_radius * 1000, 1) for b in (FL, FR, CL, CR)},
                    pull_out_trays="compute 14 and NVLink switch 5: capped / unplugged so the pulled-tray variant stays consistent",
                    osfp_populated=n_osfp, osfp_dust_caps=n_osfp_cap, qsfp112_populated=n_qsfp, qsfp112_dust_caps=n_qsfp,
                    fibers=len(FL.members) + len(FR.members), cat6=len(CL.members) + len(CR.members),
                    fiber_bundles={"left": len(FL.members), "right": len(FR.members)},
                    fiber_vertical_bundles_xy_m=[[0.587, -0.258], [0.587, 0.258]], fiber_turn_back_z_m=round(z_fib_top, 3),
                    fiber_end_rack_local_m=[[DROP_LOCAL[0], -DROP_LOCAL[1], round(DROP_LOCAL[2], 3)], [DROP_LOCAL[0], DROP_LOCAL[1], round(DROP_LOCAL[2], 3)]],
                    note="bundles rise at the front sides, turn back over the roof and end inside the overhead raceway drops (hall_layout overhead.raceway_drops kind gb300_front_left/right)")

# hooks
K.hook("rack_front_center", cc["FRAME"], root, (m(XF), 0, m(1100)))
K.hook("rack_top_center", cc["FRAME"], root, (0, 0, m(H)))
K.hook("busbar_top", cc["REAR"], root, (m(-428), 0, m(uz(49))))
K.hook("manifold_couplings", cc["LIQUID"], root, (m(-500), 0, m(uz(3))))
bpy.context.view_layer.update()

# ------------------------------------------------------------------ save, export, meta
os.makedirs(OUT, exist_ok=True)
blend = os.path.join(OUT, AID + ".blend")
glb = os.path.join(OUT, AID + ".glb")
bb = K.bbox(coll)
meta = dict(
    asset_id=AID, date="2026-10-10", units="m",
    frame="origin floor centre of footprint; FRONT = +X; width along Y (+Y = right seen from the front); Z up. glTF: Y-up (Blender +Z -> glTF +Y, Blender +Y -> glTF -Z)",
    size_m=[round(bb[1][i] - bb[0][i], 4) for i in range(3)], bbox_m=bb, triangles=K.count_tris(coll),
    layout_bottom_to_top=[[k, n] for k, n in LAYOUT if k != "open"],
    counts=dict(compute_trays=counts["c"], nvlink_switch_trays=counts["s"], power_shelves=counts["ps"],
                mgmt_switches=counts["mgmt"], drip_pan=counts["drip"]),
    trays=tray_index,
    sources="research/racks_refs.md section 2 (Lenovo GB300 NVL72 UG Figs. 2-12; CC BY GB200 photo for colours) and section 6 (front ports, NVIDIA MMS4A20 flat-top OSFP, fibre)",
    accuracy=dict(envelope="A", u_layout="B (measured on Lenovo Fig. 11, +-1U, 6-shelf config)", tray_fronts="A inventory, B positions",
                  front_cabling="A port inventory and module type; B port positions (Fig. 2 measured); C module protrusion, connector and cable sizes; design: which ports are cabled or capped, dressing route",
                  tray_depth="C (880 mm, range 800-950)", rear="A inventory, B positions", plinth="C (82 mm)"),
    collections=dict(FRAME="posts, plinth, roof, EIA rails", SIDE_PANELS="side skins (hide for cutaways)", TRAYS="compute, NVLink switch, mgmt switch, drip pan",
                     POWER="6 power shelves", REAR="busbar, cable cartridges + frame", LIQUID="side manifolds, per-tray QD hoses, rack-level couplings",
                     CABLES="power whips (upper 3 end at z = H + 140 mm under the overhead busway tap-off boxes; lower 3 go down through the plinth)",
                     BLANKS="1U blank panels for the open front bays (hidden by default; Fig. 11 shows open bays)",
                     REAR_CARTRIDGE_DETAIL="reused interconnect-film cartridge bands on the rear cartridge bays (visible by default)",
                     VARIANT_seated_slots="compute tray 14 and NVLink switch tray 5 in their seated position (visible by default)",
                     FRONT_CABLING="v4: CX-8 OSFP modules (flat top, yellow pull tab) + MPO-12/APC + 3.0 mm SMF jumpers in 2 strapped side bundles up to the overhead raceway drops; BF3 QSFP112 (port 1 cabled, port 2 capped); OSFP dust caps on compute trays 5 and 16; slim Cat6 to the in-rack management switches (visible by default)",
                     VARIANT_pulled_trays="the same two trays slid out 0.55 m, lid off, interconnect-film tray dressings (hidden by default; show this and hide VARIANT_seated_slots)"),
    reused_from_interconnect_film=reused, pulled_tray_hooks=pulled_hooks, front_cabling=cabling_meta,
    simplifications=["tray internals not modelled (closed trays)", "no logos or text",
                     "8-shelf configuration not modelled (Lenovo: 6 or 8 shelves; NVIDIA RA: 8)", "no front/rear doors (open frame as in Lenovo Fig. 11)"],
    hooks=K.hooks_of(coll),
)
K.save_blend(blend)
K.export_glb(coll, glb)
K.write_json(os.path.join(OUT, AID + ".json"), meta)
print("ASSET", AID, "tris", meta["triangles"], "size_m", meta["size_m"])

# ------------------------------------------------------------------ previews
if int(KV.get("previews", 1)):
    os.makedirs(PREV, exist_ok=True)
    K.preview_setup(res=(720, 900), samples=16, world=0.25, exposure=-0.3)
    K.floor(20)
    K.light("key", "AREA", (3.4, -1.8, 3.0), 220, 2.0, (0, 0, 1.1))
    K.light("fill", "AREA", (2.8, 2.4, 1.4), 90, 2.0, (0, 0, 1.0))
    K.light("rear", "AREA", (-3.2, 1.6, 2.8), 220, 2.0, (0, 0, 1.1))
    K.light("rear_fill", "AREA", (-2.8, -2.2, 1.4), 70, 2.0, (0, 0, 1.0))
    V = lambda n: os.path.join(PREV, "%s_%s.png" % (AID, n))
    K.render_view(V("front"), (5.0, 0, 1.15), (0, 0, 1.15), ortho=2.5)
    K.render_view(V("rear"), (-5.0, 0, 1.15), (0, 0, 1.15), ortho=2.5)
    K.render_view(V("three_quarter"), (2.9, -2.0, 1.7), (0, 0, 1.1), lens=32)
    K.render_view(V("three_quarter_rear"), (-2.8, 2.0, 1.6), (0, 0, 1.1), lens=32)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 960, 640
    K.render_view(V("front_closeup"), (1.25, 0.25, 1.32), (0.45, 0, 1.16), lens=40)
    K.render_view(V("cable_dressing_left"), (1.05, -0.62, 1.45), (0.56, -0.25, 1.25), lens=35)
    K.render_view(V("cable_fanout_top"), (1.25, 0.55, 2.45), (0.55, 0.18, 2.1), lens=35)
    K.render_view(V("ports_macro"), (0.78, 0.05, 1.42), (0.5, 0.12, 1.36), lens=50)
    K.render_view(V("rear_closeup"), (-1.3, -0.35, 1.0), (-0.45, 0, 0.75), lens=35)
    # variants: pulled trays (seated copies hidden) + blank panels, and the rear coupling detail
    K.set_visible(cc["VARIANT_pulled_trays"], True)
    K.set_visible(cc["VARIANT_seated_slots"], False)
    K.set_visible(cc["BLANKS"], True)
    K.render_view(V("variant_pulled_trays"), (1.75, -0.95, 1.75), (0.75, 0, 1.25), lens=35)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 900, 1125
    K.render_view(V("variant_blanks_front"), (5.0, 0, 1.15), (0, 0, 1.15), ortho=2.5)
    K.set_visible(cc["VARIANT_pulled_trays"], False)
    K.set_visible(cc["VARIANT_seated_slots"], True)
    K.set_visible(cc["BLANKS"], False)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 960, 640
    K.render_view(V("rear_couplings"), (-1.45, -0.55, 0.55), (-0.5, 0, 0.22), lens=35)
