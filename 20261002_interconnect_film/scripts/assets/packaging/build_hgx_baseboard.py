"""Build hgx_baseboard: (1) HGX-style GPU baseboard at real size, (2) the S4 bare board region with one XPU package socket.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_hgx_baseboard.py -- assets/components/packaging
Deterministic (seeded). Root custom property p_density (0..1) thins the random passive fields through a Geometry Nodes driver.
Frame: X right, Y up (-Y front); z = 0 at the PCB bottom; board top at z = 3.2 mm. HGX: connector edge at +Y.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import pcb_gen  # noqa: E402
import parts  # noqa: E402
from pk import MB, MM, PI  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
AID = "hgx_baseboard"
P = AID + "_"
mat = pk.mat
C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="B")
root["p_density"] = 1.0
src = C.sub_collection(coll, "SOURCES")
cH = C.sub_collection(coll, "VARIANT_hgx_board")           # bare HGX board with populations (no modules)
cMods = C.sub_collection(coll, "VARIANT_hgx_oam_modules")  # OAM modules + heatsinks, NVSwitch heatsinks
cS = C.sub_collection(coll, "VARIANT_s4_board")
cSp = C.sub_collection(coll, "VARIANT_s4_socket_pads")
cXpu = C.sub_collection(coll, "VARIANT_s4_xpu_installed")
for c_ in (cS, cSp, cXpu):
    c_.hide_render = c_.hide_viewport = True
dims = pk.Dims()
PCB_T = 3.2

# ------------------------------------------------------------------------------------------------ helpers
REGISTRY = {
    "mlcc0402": lambda: parts.mlcc("0402"), "mlcc0603": lambda: parts.mlcc("0603"), "mlcc0805": lambda: parts.mlcc("0805"),
    "mlcc1206": lambda: parts.mlcc("1206"), "res0402": lambda: parts.chip_resistor("0402"), "res0603": lambda: parts.chip_resistor("0603"),
    "ferrite0603": parts.ferrite_bead, "ind7": lambda: parts.power_inductor(7.0, 7.0, 3.5), "ind10": lambda: parts.power_inductor(10.0, 10.0, 5.0),
    "tantB": lambda: parts.tantalum("B"), "tantD": lambda: parts.tantalum("D"), "polyD": lambda: parts.tantalum("D", polymer=True),
    "alu": parts.alu_electrolytic, "drmos": parts.drmos_5x6, "dcdc": parts.dcdc_controller_4x4, "toroid": parts.toroid,
    "flyback": parts.flyback_transformer, "tp_pad": parts.test_point_pad, "tp_loop": parts.test_point_loop,
    "led_g": lambda: parts.led_0603("green"), "led_r": lambda: parts.led_0603("red"),
    "screw": parts.screw_m3, "standoff": parts.standoff_m3_hex, "washer": parts.washer_m3,
    "bolt_m6": lambda: parts.screw_m3(length=16.0, head_d=10.0, head_h=4.0, r=3.0),
    "pwr_conn": parts.power_connector_2x3, "b2b": parts.b2b_receptacle,
}
SIZE = {"mlcc0402": (1.0, 0.5), "mlcc0603": (1.6, 0.8), "mlcc0805": (2.0, 1.25), "mlcc1206": (3.2, 1.6), "res0402": (1.0, 0.5),
        "res0603": (1.6, 0.8), "ferrite0603": (1.6, 0.8), "ind7": (7.0, 7.0), "ind10": (10.0, 10.0), "tantB": (3.5, 2.8), "tantD": (7.3, 4.3),
        "polyD": (7.3, 4.3), "alu": (7.0, 7.0), "drmos": (5.0, 6.0), "dcdc": (4.0, 4.0), "toroid": (12.0, 12.0), "flyback": (13.0, 11.0),
        "tp_pad": (1.6, 1.6), "tp_loop": (5.0, 2.0), "led_g": (1.6, 0.8), "led_r": (1.6, 0.8), "screw": (5.6, 5.6), "standoff": (6.4, 6.4),
        "washer": (7.0, 7.0), "bolt_m6": (10.0, 10.0), "pwr_conn": (13.2, 9.0), "b2b": (14.4, 4.0)}
SRC = {}


def part_src(name):
    if name not in SRC:
        mb, mn = REGISTRY[name]()
        SRC[name] = pk.source(mb.to_obj(P + "src_" + name, [mat(m) for m in mn], src, root))
    return SRC[name]


class Occ:
    def __init__(self):
        self.r = []

    def block(self, x, y, w, d, m=0.0):
        self.r.append((x - w / 2 - m, y - d / 2 - m, x + w / 2 + m, y + d / 2 + m))

    def free(self, x, y, w, d, m=0.4):
        x0, y0, x1, y1 = x - w / 2 - m, y - d / 2 - m, x + w / 2 + m, y + d / 2 + m
        for a, b, c, e in self.r:
            if x0 < c and x1 > a and y0 < e and y1 > b:
                return False
        return True


class Pop:
    """Population of instanced parts. tag 'fixed' parts always render; tag 'rand' parts are thinned by root p_density."""

    def __init__(self):
        self.d = {}

    def add(self, part, x, y, rz=0.0, tag="fixed"):
        self.d.setdefault((part, tag), []).append((x, y, rz))

    def count(self):
        return {k[0] + "/" + k[1]: len(v) for k, v in self.d.items()}

    def flush(self, c, z, seed=7):
        rr = random.Random(seed)
        for (part, tag), pts in self.d.items():
            s = part_src(part)
            P3 = [(x, y, z) for (x, y, _) in pts]
            rzs = [r for (_, _, r) in pts]
            nm = "%spop_%s_%s_%s" % (P, part, tag, c.name[-5:])
            if tag == "rand":
                pk.scatter_density(nm, P3, s, c, root, root, rz=rzs, seed=rr.randint(0, 10 ** 6))
            else:
                pk.scatter(nm, P3, s, c, root, rz=rzs)


def rotp(cx, cy, rz, lx, ly):
    return cx + lx * math.cos(rz) - ly * math.sin(rz), cy + lx * math.sin(rz) + ly * math.cos(rz)


def place_random(pop, occ, rng, part, n, region, rots=(0.0, PI / 2), margin=0.5, tries=40):
    w0, d0 = SIZE[part]
    placed = 0
    t = 0
    while placed < n and t < n * tries:
        t += 1
        x, y = rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])
        rz = rng.choice(rots)
        w, d = (w0, d0) if abs(math.sin(rz)) < 0.5 else (d0, w0)
        if x - w / 2 < region[0] or x + w / 2 > region[2] or y - d / 2 < region[1] or y + d / 2 > region[3]:
            continue
        if occ.free(x, y, w, d, margin):
            occ.block(x, y, w, d)
            pop.add(part, x, y, rz, "rand")
            placed += 1
    return placed


def mlcc_group(pop, occ, rng, region, size="mlcc0603"):
    w, d = SIZE[size]
    nx, ny = rng.randint(4, 9), rng.randint(2, 5)
    px, py = w + 0.8, d + 0.9
    for _ in range(40):
        if region[2] - region[0] < nx * px + 1 or region[3] - region[1] < ny * py + 1:
            return 0
        x0 = rng.uniform(region[0], region[2] - nx * px)
        y0 = rng.uniform(region[1], region[3] - ny * py)
        if occ.free(x0 + nx * px / 2, y0 + ny * py / 2, nx * px, ny * py, 0.8):
            occ.block(x0 + nx * px / 2, y0 + ny * py / 2, nx * px, ny * py)
            for j in range(ny):
                for i in range(nx):
                    pop.add(size, x0 + (i + 0.5) * px, y0 + (j + 0.5) * py, 0.0, "rand")
            return nx * ny
    return 0


def vrm_cluster(pop, occ, cx, cy, rz, phases=6, pitch=12.5):
    """Multiphase power-stage cluster: per phase a DrMOS, a shielded 10x10 inductor and output MLCCs; polymer bulk caps, an electrolytic and a controller."""
    for p in range(phases):
        lx = (p - (phases - 1) / 2) * pitch
        for (part, ly) in (("drmos", -9.0), ("ind10", 2.0)):
            x, y = rotp(cx, cy, rz, lx, ly)
            pop.add(part, x, y, rz)
        for k in range(6):
            x, y = rotp(cx, cy, rz, lx + (k % 3 - 1) * 2.6, 11.0 + (k // 3) * 1.8)
            pop.add("mlcc0805", x, y, rz + PI / 2)
        for k in range(2):
            x, y = rotp(cx, cy, rz, lx + (k - 0.5) * 3.0, -15.5)
            pop.add("mlcc0603", x, y, rz)
    for k in range(3):
        x, y = rotp(cx, cy, rz, (k - 1) * (phases * pitch / 3.2), -19.5)
        pop.add("polyD", x, y, rz)
    x, y = rotp(cx, cy, rz, -phases * pitch / 2 - 5.0, -2.0)
    pop.add("alu", x, y, 0.0)
    x, y = rotp(cx, cy, rz, phases * pitch / 2 + 5.0, -2.0)
    pop.add("dcdc", x, y, rz)
    w, d = phases * pitch + 14.0, 36.0
    ww, dd = (w, d) if abs(math.sin(rz)) < 0.5 else (d, w)
    ox, oy = rotp(cx, cy, rz, 0.0, -3.0)
    occ.block(ox, oy, ww, dd)


def fill_passives(pop, occ, rng, region, n_groups, small):
    for _ in range(n_groups):
        mlcc_group(pop, occ, rng, region, rng.choice(["mlcc0402", "mlcc0603", "mlcc0603", "mlcc0805"]))
    for part, n in small:
        place_random(pop, occ, rng, part, n, region)


SMALL = [("flyback", 6), ("toroid", 8), ("ind7", 14), ("tantB", 20), ("tantD", 6), ("alu", 8), ("mlcc1206", 24), ("mlcc0805", 60),
         ("mlcc0603", 160), ("mlcc0402", 220), ("res0402", 140), ("res0603", 60), ("ferrite0603", 30), ("tp_pad", 24), ("tp_loop", 6),
         ("dcdc", 6), ("drmos", 10)]

# ------------------------------------------------------------------------------------------------ S4 board
S4W, S4D = 260.0, 200.0
b4 = pcb_gen.PCB(cS, root, "hgx_baseboard_s4", S4W, S4D, PCB_T, "mask_green")
for sx in (-1, 1):
    for sy in (-1, 1):
        b4.hole(sx * 123.0, sy * 93.0, 3.2)
b4.fiducials(8.0)
rng = random.Random(4242)
pop4, occ4 = Pop(), Occ()
FR_O, FR_I = (200.0, 120.0), (180.4, 100.4)
occ4.block(0, 0, FR_O[0] + 22.0, FR_O[1] + 22.0)
for sx in (-1, 1):
    for sy in (-1, 1):
        occ4.block(sx * 123.0, sy * 93.0, 12.0, 12.0)
        pop4.add("standoff", sx * 123.0, sy * 93.0, 0.0)
        pop4.add("screw", sx * 123.0, sy * 93.0, 0.0)
        pop4.add("washer", sx * 123.0, sy * 93.0, 0.0)
for (nm, x) in (("pwr_conn", -95.0), ("b2b", -35.0), ("b2b", 35.0), ("pwr_conn", 95.0)):
    pop4.add(nm, x, -92.0, 0.0)
    occ4.block(x, -92.0, SIZE[nm][0] + 2, SIZE[nm][1] + 2)
for (cx, cy, rz) in ((-114.0, -48.0, PI / 2), (-114.0, 36.0, PI / 2), (114.0, -48.0, PI / 2), (114.0, 36.0, PI / 2),
                     (-55.0, 79.0, 0.0), (55.0, 79.0, 0.0), (-55.0, -77.0, 0.0), (55.0, -77.0, 0.0)):
    vrm_cluster(pop4, occ4, cx, cy, rz, phases=5 if abs(rz) < 0.1 else 6)
fill_passives(pop4, occ4, rng, (-128.0, -98.0, 128.0, 98.0), 26, SMALL)
for (x, y) in [(sx * (FR_O[0] / 2 + 6.0), sy * (FR_O[1] / 2 + 6.0)) for sx in (-1, 1) for sy in (-1, 1)]:
    pop4.add("bolt_m6", x, y, 0.0)
b4.text("S4 BOARD REGION", -118.0, 96.5, 3.0, align="LEFT")
b4.text("XPU0", 0.0, 66.0, 5.0)
b4.outline(-FR_O[0] / 2 - 3, -FR_O[1] / 2 - 3, FR_O[0] / 2 + 3, FR_O[1] / 2 + 3, 0.3)
b4.finish()
mb = MB()
mb.ring_rect(FR_O[0], FR_O[1], FR_I[0], FR_I[1], PCB_T, PCB_T + 4.0, rad_o=3.0, chamfer=0.4, seg=4, mi=(0, 0, 0))
mb.to_obj(P + "s4_socket_frame", [mat("plastic_black")], cS, root)
mb = MB()
mb.box((0, 0, PCB_T + 0.05), (FR_I[0] + 0.4, FR_I[1] + 0.4, 0.1), mi=0)
mb.to_obj(P + "s4_socket_seat", [mat("substrate_land")], cS, root)
mb = MB()
mb.ring_rect(FR_O[0] + 16.0, FR_O[1] + 16.0, FR_O[0] - 14.0, FR_O[1] - 14.0, PCB_T + 4.0, PCB_T + 6.5, rad_o=5.0, chamfer=0.5, seg=4, mi=(0, 0, 0))
mb.to_obj(P + "s4_retention_frame", [mat("steel")], cS, root)
pop4.flush(cS, PCB_T, seed=11)
mb = MB()
mb.lathe([(0, 0), (0.33, 0), (0.33, 0.05), (0, 0.05)], seg=8, mi=0, smooth=False)
padsrc = pk.source(mb.to_obj(P + "src_lga_pad", [mat("gold_enig")], src, root))
cells = [((i - 87) * 1.0, (j - 47) * 1.0, PCB_T + 0.1) for j in range(95) for i in range(175)
         if not (abs(i - 87) < 12 and abs(j - 47) < 12)]
pk.scatter(P + "s4_lga_pads", cells, padsrc, cSp, root)
C.hook("xpu_origin", cS, root, loc=(0, 0, (PCB_T + 2.0) * MM))
C.hook("socket_center", cS, root, loc=(0, 0, PCB_T * MM))

# ------------------------------------------------------------------------------------------------ HGX baseboard
HW, HL = 416.0, 553.0
bh = pcb_gen.PCB(cH, root, "hgx_baseboard_hgx", HW, HL, PCB_T, "mask_green")


def hx(x):
    return x - 208.13


def hy(y):
    return 276.5 - y


HOLES = [(12.87, 10.5), (405.5, 10.5), (118.0, 112.25), (298.0, 112.25), (20.5, 158.0), (395.5, 158.0), (123.0, 275.64), (208.0, 275.64),
         (293.0, 275.64), (122.25, 439.0), (293.75, 439.0), (9.5, 458.0), (405.5, 446.2), (10.5, 542.5), (123.0, 542.5), (293.0, 542.5),
         (405.5, 542.5)]
for (x, y) in HOLES:
    bh.hole(hx(x), hy(y), 3.2)
bh.fiducials(8.0)
popH, occH = Pop(), Occ()
rngH = random.Random(1717)
OAM_X = [(i - 1.5) * 103.0 for i in range(4)]
OAM_Y = {"A": 94.0, "B": -191.0}
sites = []
for rowk, yy in OAM_Y.items():
    for i, xx in enumerate(OAM_X):
        sites.append((rowk, i, xx, yy))
for (rowk, i, xx, yy) in sites:
    occH.block(xx, yy, 103.0, 166.0)
    bh.outline(xx - 51.5, yy - 83.0, xx + 51.5, yy + 83.0, 0.3)
mb = MB()
mbn = MB()
for (rowk, i, xx, yy) in sites:
    for sgn in (-1, 1):
        mb.box((xx, yy + sgn * 51.0, PCB_T + 2.5), (68.0, 11.0, 5.0), mi=0, bevel=0.3, seg=1)
        for sx in (-45.0, 45.0):
            mbn.lathe([(0, PCB_T), (3.0, PCB_T), (3.0, PCB_T + 1.0), (0, PCB_T + 1.0)], c=(xx + sx, yy + sgn * 51.0, 0), seg=16, mi=0, smooth=False)
mb.to_obj(P + "oam_mirror_mezz_connectors", [mat("plastic_black")], cH, root)
mbn.to_obj(P + "oam_smt_nuts", [mat("steel")], cH, root)
NVX = OAM_X
NVY = -48.0
for xx in NVX:
    occH.block(xx, NVY, 64.0, 64.0)
YC = 276.5
conn_types = [(0.0, "guide"), (26.25, "radsok"), (40.80, "airmax23"), (58.0, "airmax22"), (72.8, "pwrmax"), (83.5, "radsok"),
              (109.0, "exa68"), (127.0, "exa48"), (145.0, "exa48"), (162.0, "exa48"),
              (254.0, "exa48"), (271.0, "exa48"), (289.0, "exa48"), (307.0, "exa48"), (324.0, "radsok"), (343.2, "pwrmax"),
              (358.0, "airmax22"), (375.2, "airmax23"), (389.75, "guide")]
CONN_DIM = {"exa48": (16.5, 22.0, 9.5, "plastic_black"), "exa68": (16.5, 24.0, 9.5, "plastic_black"), "airmax23": (13.0, 20.0, 11.0, "plastic_black"),
            "airmax22": (9.0, 20.0, 11.0, "plastic_black"), "pwrmax": (16.0, 20.0, 14.0, "plastic_white"), "radsok": (6.0, 16.0, 6.0, "steel"),
            "guide": (6.0, 18.0, 6.0, "steel")}
mbc = {k: MB() for k in set(v[3] for v in CONN_DIM.values())}
for (x, t) in conn_types:
    w, d, h, mname = CONN_DIM[t]
    cx_, cy_ = hx(x), YC + 12.4 - d / 2
    if t in ("radsok", "guide"):
        mbc[mname].cyl((cx_, cy_, PCB_T + 3.0), 3.0, d, axis="Y", seg=16, mi=0)
    else:
        mbc[mname].box((cx_, cy_, PCB_T + h / 2), (w, d, h), mi=0, bevel=0.3, seg=1)
    occH.block(cx_, cy_, w + 1.0, d + 1.0)
for mname, mbx in mbc.items():
    mbx.to_obj(P + "edge_connectors_" + mname, [mat(mname)], cH, root)
mb = MB()
mbm = MB()
RT_Y = 243.0
for i in range(8):
    x = (i - 3.5) * 46.0
    mb.box((x, RT_Y, PCB_T + 0.25), (15.0, 15.0, 0.5), mi=0)
    mbm.box((x, RT_Y, PCB_T + 0.5 + 0.45), (14.9, 14.9, 0.9), mi=0, bevel=0.15, seg=1)
    occH.block(x, RT_Y, 17.0, 17.0)
    bh.text("U%d" % (i + 1), x, RT_Y + 10.5, 2.2)
mb.to_obj(P + "retimer_substrates", [mat("substrate")], cH, root)
mbm.to_obj(P + "retimers_mold", [mat("mold_black")], cH, root)
for k, x in enumerate((-150.0, 150.0)):
    occH.block(x, 243.0, 27.0, 27.0)
    mbq = MB()
    mbq.box((x, 243.0, PCB_T + 0.9), (25.0, 25.0, 1.8), mi=0, bevel=0.3, seg=1)
    mbq.to_obj(P + ("fpga" if k == 0 else "bmc") + "_package", [mat("mold_black")], cH, root)
for xx in (-150.0, -50.0, 50.0, 150.0):
    vrm_cluster(popH, occH, xx, 205.0, 0.0, phases=5)
for xx in (-103.0, 0.0, 103.0):
    vrm_cluster(popH, occH, xx, -48.0, PI / 2, phases=5)
fill_passives(popH, occH, rngH, (-206.0, 180.0, 206.0, 266.0), 30, [(n, int(c * 0.8)) for n, c in SMALL])
fill_passives(popH, occH, rngH, (-206.0, -106.0, 206.0, 8.0), 20, [(n, int(c * 0.6)) for n, c in SMALL])
for k in range(6):
    popH.add("led_g" if k % 2 else "led_r", hx(10.0 + k * 3.0), 268.0, 0.0)
for (x, y) in HOLES:
    popH.add("washer", hx(x), hy(y), 0.0)
bh.text("HGX-STYLE GPU BASEBOARD  553 X 416", 0.0, -272.0, 4.0)
for (rowk, i, xx, yy) in sites:
    bh.text("OAM%d" % ((0 if rowk == "A" else 4) + i), xx - 30.0, yy + (-76.0 if rowk == "B" else 76.0), 3.0)
for i, xx in enumerate(NVX):
    bh.text("NVS%d" % i, xx, NVY - 36.0, 3.0)
bh.finish()
popH.flush(cH, PCB_T, seed=23)

ZTOP = PCB_T + 145.54                 # HGX envelope: heatsink top 145.54 mm above the PCB top (Fig. 3)
for (rowk, i, xx, yy) in sites:
    ang = PI if rowk == "A" else 0.0
    z0 = PCB_T + 5.0                  # Mirror Mezz stack height 5 mm (OAM spec)
    idx = (0 if rowk == "A" else 4) + i
    mbo = MB()
    mbo.box((0, 0, z0 + 1.2), (102.0, 165.0, 2.4), mi=0, bevel=0.2, seg=1)
    for sx in (-45.0, 45.0):
        for sy in (-51.0, 51.0):
            mbo.lathe([(0, z0 + 2.4), (3.0, z0 + 2.4), (3.0, z0 + 2.404), (0, z0 + 2.404)], c=(sx, sy, 0), seg=14, mi=1, smooth=False)
    mbo.box((-51.0 + 2.0, -82.5 + 16.0, z0 + 2.405), (4.0, 12.0, 0.01), mi=2)
    mbo.rotate_z(ang)
    mbo.to_obj(P + "oam_module_%d_pcb" % idx, [mat("mask_green"), mat("steel"), mat("gold_enig")], cMods, root, loc=(xx, yy, 0))
    mbh = MB()
    zb0 = z0 + 2.4 + 2.0
    mbh.box((0, 0, zb0 + 3.0), (100.0, 163.0, 6.0), mi=0, bevel=0.4, seg=1)
    nfin = 45
    fh = ZTOP - (zb0 + 6.0)
    for k in range(nfin):
        mbh.box(((k - (nfin - 1) / 2) * 2.15, 0, zb0 + 6.0 + fh / 2), (0.8, 160.0, fh), mi=0)
    mbh.rotate_z(ang)
    mbh.to_obj(P + "oam_module_%d_heatsink" % idx, [mat("aluminum")], cMods, root, loc=(xx, yy, 0))
for i, xx in enumerate(NVX):
    mbs = MB()
    mbs.box((0, 0, PCB_T + 0.8), (47.5, 47.5, 1.6), mi=0)
    mbs.box((0, 0, PCB_T + 1.6 + 1.8), (40.0, 40.0, 3.6), mi=1, bevel=0.3, seg=1)
    mbs.to_obj(P + "nvswitch_%d_package" % i, [mat("substrate"), mat("nickel")], cMods, root, loc=(xx, NVY, 0))
    mbf = MB()
    zb = PCB_T + 5.2
    mbf.box((0, 0, zb + 2.0), (62.0, 62.0, 4.0), mi=0, bevel=0.3, seg=1)
    for k in range(28):
        mbf.box(((k - 13.5) * 2.2, 0, zb + 4.0 + 17.5), (0.8, 60.0, 35.0), mi=0)
    mbf.to_obj(P + "nvswitch_%d_heatsink" % i, [mat("aluminum")], cMods, root, loc=(xx, NVY, 0))
C.hook("pcie_connector_edge", cH, root, loc=(0, (YC + 12.4) * MM, PCB_T * MM))
C.hook("board_center_top", coll, root, loc=(0, 0, PCB_T * MM))
for (rowk, i, xx, yy) in sites:
    C.hook("oam_site_%d" % ((0 if rowk == "A" else 4) + i), cH, root, loc=(xx * MM, yy * MM, (PCB_T + 5.0) * MM), rot=(0, 0, PI if rowk == "A" else 0.0))

# ------------------------------------------------------------------------------------------------ xpu append for the S4 variant
xblend = os.path.join(OUT, "xpu_package_rubin_style.blend")
xpu_ok = False
if os.path.exists(xblend):
    try:
        with bpy.data.libraries.load(xblend, link=False) as (df, dt):
            dt.collections = ["ASSET_xpu_package_rubin_style"]
        xc = dt.collections[0]
        cXpu.children.link(xc)
        xroot = [o for o in xc.objects if o.name.startswith("ROOT_xpu")][0]
        xroot.parent = bpy.data.objects["HOOK_xpu_origin"]
        xpu_ok = True
    except Exception as e:                                           # noqa: BLE001
        print("XPU append failed", e)

# ------------------------------------------------------------------------------------------------ dimensions / meta
dims.add("HGX baseboard size", "553.0 x 416.0 (565.4 with connectors)", "mm", "OCP HGX Form Factor R1 spec v0.1, Table 1/8 and Fig. 2 (553.26 x 416.26 max outline)", "A")
dims.add("heatsink envelope height above PCB top", 145.54, "mm", "HGX spec Fig. 3 ('145.54 MAX'); stiffener bottom 7.59 mm above PCB top", "A")
dims.add("PCB thickness", PCB_T, "mm", "not in the spec; typical 20+ layer baseboard", "C")
dims.add("captive screw / mounting hole count and positions", 17, "count", "HGX spec Fig. 14 coordinates (read from the figure, +-1.5 mm); hole diameter 3.2 mm assumed", "B")
dims.add("connector A1 x positions", "0, 26.25, 40.80, 58.00, 72.80, 83.50, 109, 127, 145, 162 | 254, 271, 289, 307, 324, 343.2, 358, 375.2, 389.75", "mm", "HGX spec Fig. 14; connector type assignment to positions is illustrative (spec Table 2: 1 ExaMAX 6x8, 7 ExaMAX 4x8, 2+2 AirMax, 2 PwrMAX, 4 guide pins, 4 RadSok)", "B")
dims.add("connector protrusion beyond the PCB edge", 12.4, "mm", "565.4 - 553.0 (spec)", "A")
dims.add("OAM module size", "102 x 165", "mm", "OCP OAM spec v1.0 section 5/6: 102 x 165 mm; connector pitch 102 mm; mount holes 90 x 102 mm (4 x dia 3.9)", "A")
dims.add("OAM site pitch / layout", "103 mm pitch, 4 per row, 2 rows, rows rotated 180 deg", "mm", "OAM spec Fig. 54 (415 mm row width); HGX layout spreads the rows to fit the NVSwitch band", "B")
dims.add("OAM keep-out zone", "103 x 166", "mm", "OAM spec 6.5.2", "A")
dims.add("Mirror Mezz stack height", 5.0, "mm", "OAM spec (Molex 209311-1115, 5 mm)", "A")
dims.add("OAM row Y positions", "A: +94, B: -191 (rear row rotated)", "mm", "chosen to fit 2 x 165 mm rows plus a 119 mm NVSwitch band and a 99 mm connector zone in 553 mm", "C")
dims.add("NVSwitch package / lid / heatsink", "47.5 x 47.5 / 40 x 40 / 62 x 62", "mm", "estimates (range 40-55 mm package)", "C")
dims.add("retimers", "8 x 15 x 15 mm", "mm", "one per GPU PCIe x16 link; size per narrowcom_retimer_chip", "C")
dims.add("S4 board region", "260 x 200 x 3.2", "mm", "scaled from the crude v0.2 S4 board (21 x 16 units vs 16.4 x 10.4 package) to the 180 x 100 mm XPU package", "C")
dims.add("S4 socket frame outer/inner", "200 x 120 / 180.4 x 100.4", "mm", "package 180 x 100 mm plus 0.2 mm clearance", "C")
dims.add("instanced passive counts HGX", popH.count(), "count", "tag rand entries are thinned by p_density", "A")
dims.add("instanced passive counts S4", pop4.count(), "count", "tag rand entries are thinned by p_density", "A")
meta = {
    "title": "HGX-style GPU baseboard and S4 board region", "accuracy_level": "A for outline/holes/OAM sizes, B/C for placement and details",
    "origin": "board bottom center at z = 0 (PCB top at 3.2 mm); HGX connector edge at +Y",
    "taken_from_hgx_pdf": ["553.0 x 416.0 mm board, 565.4 with connectors, 145.54 mm heatsink envelope", "17 mounting hole coordinates and connector A1 x positions (Fig. 14)", "connector counts/types (Table 2), 8 GPUs + 4 NVSwitch (Table 1)"],
    "taken_from_oam_spec": ["102 x 165 mm module, 102 mm connector pitch, 90 x 102 mm mount holes, 5 mm Mirror Mezz stack, 103 x 166 mm KOZ, 103 mm module pitch, rows opposite-oriented"],
    "variants": {"VARIANT_hgx_board": "bare HGX baseboard with populations (default visible)", "VARIANT_hgx_oam_modules": "8 OAM modules with heatsinks, 4 NVSwitch packages with heatsinks (default visible; hide to see the board)",
                 "VARIANT_s4_board": "S4 bare board region with socket frame, retention frame, VRM clusters, MLCC clusters, inductors, transformers, coils, connectors, bolts (hidden by default)",
                 "VARIANT_s4_socket_pads": "LGA contact field, 16.3k instances (hidden)", "VARIANT_s4_xpu_installed": "xpu_package_rubin_style appended (hidden); parent HOOK_xpu_origin (appended ok: %s)" % xpu_ok},
    "density": "root custom property p_density (0..1) drives a Geometry Nodes Delete Geometry threshold on the random passive fields (tag rand); structured VRM clusters, connectors and bolts are fixed. Placement is seeded (HGX seed 1717, S4 seed 4242)",
    "sources": [{"file": "references/Open-Compute-Specification-HGX-Baseboard-Contribution R1 V0.1.pdf", "used_for": "outline, holes, connectors, heatsink envelope"},
                {"file": "references/OAM Spec v1.0.zip (spec PDF, extracted to scratch only)", "used_for": "module size, pitch, stack height, KOZ"}],
    "simplifications": ["OAM modules are PCB + heatsink only (no ASIC/HBM)", "heatsink fins are plain extrusions", "connector bodies are blocks", "no traces on the baseboard (see pcb_generator)", "the OAM grounding pads (8 x 42 mm) are not modeled", "board outline has no notches"],
    "scene_usage": "S4 (VARIANT_s4_board + xpu package), establishing shots",
}
blend = os.path.join(OUT, AID + ".blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
pngs = []


def vis(h=False, m=False, s4=False, pads=False, xpu=False):
    for c_, on in ((cH, h), (cMods, m), (cS, s4), (cSp, pads), (cXpu, xpu)):
        c_.hide_render = c_.hide_viewport = not on
    bpy.context.view_layer.update()


def R(views, pre=""):
    pngs.extend(pk.render_views(coll, pv, AID, [dict(v, name=pre + v["name"]) for v in views]))


vis(h=True, m=True)
R([dict(name="three_quarter", loc=(520, -820, 600), tgt=(0, 0, 40), lens=50, floor=True),
   dict(name="front", loc=(0, -1100, 250), tgt=(0, 0, 70), lens=50, floor=True),
   dict(name="top", loc=(0, -1, 1150), tgt=(0, 0, 0), lens=50)], "hgx_")
vis(h=True)
R([dict(name="top", loc=(0, -1, 1000), tgt=(0, 0, 0), lens=50),
   dict(name="closeup_connectors", loc=(-90, 130, 90), tgt=(-80, 250, 4), lens=50, floor=True),
   dict(name="closeup_nvswitch_band", loc=(0, -190, 120), tgt=(0, -50, 4), lens=50, floor=True)], "hgx_bare_")
vis(s4=True)
R([dict(name="top", loc=(0, -1, 470), tgt=(0, 0, 0), lens=50),
   dict(name="three_quarter", loc=(240, -330, 260), tgt=(0, 0, 0), lens=50, floor=True),
   dict(name="closeup_vrm_cluster", loc=(-90, -75, 70), tgt=(-114, -48, 4), lens=50, floor=True),
   dict(name="closeup_connectors", loc=(-40, -140, 40), tgt=(-35, -92, 4), lens=50, floor=True)], "s4_")
root["p_density"] = 0.35
pk.refresh_drivers(root)
R([dict(name="top_density_035", loc=(0, -1, 470), tgt=(0, 0, 0), lens=50)], "s4_")
root["p_density"] = 1.0
pk.refresh_drivers(root)
if xpu_ok:
    vis(s4=True, xpu=True)
    R([dict(name="three_quarter_with_xpu", loc=(240, -330, 260), tgt=(0, 0, 0), lens=50, floor=True)], "s4_")
vis(s4=True)
bb4, tris4 = pk.eval_stats(cS)
vis(h=True, m=True)
C.save(blend)
bb, tris = pk.eval_stats(coll)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm_hgx_default": bb, "triangles_hgx_default": tris, "triangles_s4_variant": tris4, "bbox_mm_s4": bb4,
             "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)], "dimension_table": dims.rows,
             "hooks": [h["name"] for h in hooks], "material_slots": mnames, "custom_properties": props,
             "previews": [os.path.relpath(p, OUT) for p in pngs]})
pk.write_json(blend, meta)
print("DONE", AID, tris, tris4, xpu_ok)
