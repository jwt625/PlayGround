"""Dummy board components for the S2 rack-interior shots (new 2026-10-02). Shared by build_dc_board_parts.py and
build_tray_dummy_layouts.py. Every builder returns (MB, [materials], meta) in feature millimetres converted to metres:
origin = bottom centre of the part footprint on the board top surface (z = 0), +y = back (towards the backplane), -y = front.
All sizes are real-size values (apply the stack's detail scale at assembly). Accuracy per part is in PART_META.
No logos or trademarks on any model (parts carry no text).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402,F401
import dc_common as D  # noqa: E402
from dc_common import MB  # noqa: E402


def mats():
    L = D.lib()
    return dict(
        L,
        choke=D.M("choke_ferrite", (0.05, 0.05, 0.055), 0.6, 0.35),
        cap_poly=D.M("cap_polymer_tan", (0.62, 0.42, 0.14), 0.2, 0.4),
        cap_can=D.M("cap_can_blue", (0.04, 0.09, 0.3), 0.3, 0.35),
        cap_top=D.M("cap_top_silver", (0.75, 0.76, 0.78), 1.0, 0.3),
        pcb_black=D.M("pcb_black", (0.012, 0.012, 0.014), 0.0, 0.4),
        pcb_blue=D.M("pcb_card_blue", (0.03, 0.12, 0.3), 0.0, 0.35),
        fin=D.M("heatsink_alu", (0.55, 0.57, 0.6), 1.0, 0.35),
        cold=D.M("coldplate_copper", (0.9, 0.5, 0.25), 1.0, 0.3),
        fanbody=D.M("fan_frame", (0.03, 0.03, 0.034), 0.0, 0.5),
        fanrot=D.M("fan_rotor", (0.07, 0.07, 0.08), 0.0, 0.5),
    )


def _sink(mb, cx, cy, w, d, h, nfin, base=2.5, m_base=0, m_fin=0, axis_x=True):
    mb.boxmm((cx, cy, base / 2), (w, d, base), m_base)
    for i in range(nfin):
        if axis_x:
            x = cx - w / 2 + (i + 0.5) * w / nfin
            mb.boxmm((x, cy, base + (h - base) / 2), (w / nfin * 0.42, d, h - base), m_fin)
        else:
            y = cy - d / 2 + (i + 0.5) * d / nfin
            mb.boxmm((cx, y, base + (h - base) / 2), (w, d / nfin * 0.42, h - base), m_fin)


def vrm_block(M):
    """Multiphase VRM patch: 2 rows of 4 molded chokes, 8 power stages between them, 2 rows of polymer caps (60 x 50 x 7 mm)."""
    b = MB()
    for r, y in enumerate((-13.0, 13.0)):
        for i in range(4):
            x = (i - 1.5) * 14.0
            b.boxmm((x, y, 3.2), (12, 12, 6.4), 0)
            b.boxmm((x, y + (3.5 if y < 0 else -3.5), 6.5), (10, 1.2, 0.3), 3)       # copper terminal strip on top
    for i in range(8):
        b.boxmm(((i - 3.5) * 7.0, 0, 0.5), (5, 5, 1.0), 1)
    for r, y in enumerate((-22.0, 22.0)):
        for i in range(8):
            b.boxmm(((i - 3.5) * 7.0, y, 1.6), (4.5, 6.5, 3.2), 2)
    return b, [M["choke"], M["mold"], M["cap_poly"], M["copper"]]


def inductor_bank(M):
    """Row of 6 large molded chokes (14 x 14 x 8 mm each, 16 mm pitch)."""
    b = MB()
    for i in range(6):
        x = (i - 2.5) * 16.0
        b.boxmm((x, 0, 4.0), (14, 14, 8), 0)
        b.boxmm((x, 0, 8.05), (9, 9, 0.2), 1)
        for s in (-1, 1):
            b.boxmm((x + s * 6.8, 0, 1.2), (1.6, 10, 2.4), 2)
    return b, [M["choke"], M["black"], M["copper"]]


def cap_bank(M):
    """3 x 6 array of aluminium electrolytic / polymer cans, 8 mm dia x 11 mm tall, 9.5 mm pitch."""
    s = 0.001
    b = MB()
    for r in range(3):
        for i in range(6):
            x, y = (i - 2.5) * 9.5 * s, (r - 1) * 9.5 * s
            b.cyl((x, y, 5.5 * s), 0.004, 0.011, "z", 14, 0)
            b.cyl((x, y, 11.1 * s), 0.0034, 0.0003, "z", 14, 1)
    return b, [M["cap_can"], M["cap_top"]]


def heatsink_fin_stack(M):
    """Extruded-fin heat sink, 60 x 60 x 24 mm, 22 fins (fin 1 mm, pitch 2.7 mm) on a 2.5 mm base, with a retaining clip bar."""
    b = MB()
    _sink(b, 0, 0, 60, 60, 24, 22, 2.5, 0, 0, True)
    b.boxmm((0, 0, 24.4), (66, 3, 0.8), 1)
    return b, [M["fin"], M["steel"]]


def socamm_module(M):
    """SOCAMM-style LPDDR module lying flat: 14 x 90 mm PCB with 4 packages (2 per side shown on top face) and a screw tab."""
    b = MB()
    b.boxmm((0, 0, 1.5), (90, 14, 0.8), 0)                       # card PCB (long axis x)
    for i in range(4):
        b.boxmm(((i - 1.5) * 19.5, 0, 2.4), (14, 11, 1.0), 1)    # LPDDR packages
        b.boxmm(((i - 1.5) * 19.5, 0, 2.95), (6, 5, 0.1), 3)     # marking window (blank)
    b.boxmm((-43.5, 0, 2.2), (4, 8, 1.2), 2)                     # screw tab / retention
    b.boxmm((0, 0, 0.7), (84, 12, 1.4), 4)                       # compression connector body under the card
    return b, [M["pcb_black"], M["mold"], M["gold"], M["steel"], M["black"]]


def dimm_module(M):
    """DDR5 RDIMM standing in its socket: 133.35 x 31.25 mm card (JEDEC outline), 8 DRAM packages on the visible face, latches."""
    b = MB()
    b.boxmm((0, 0, 3.0 + 31.25 / 2), (133.35, 1.3, 31.25), 0)
    b.boxmm((0, 0, 3.0 / 2), (136, 7.5, 3.0), 1)                  # socket body
    for i in range(8):
        b.boxmm(((i - 3.5) * 14.2 + 4.0, -1.0, 3.0 + 16.5), (11.5, 0.9, 12.5), 2)
    b.boxmm((0, -1.0, 3.0 + 6.0), (14, 0.9, 6), 2)                # RCD / buffer
    for s in (-1, 1):
        b.boxmm((s * 69.0, 0, 8.0), (3.0, 6.0, 12.0), 3)          # latches
    return b, [M["pcb_black"], M["black"], M["mold"], M["alu"]]


def nic_dpu_card(M):
    """OCP NIC 3.0 small-form-factor style card lying flat: 76 x 115 mm PCB, ASIC heat sink, 2 cages at the front (-y)."""
    b = MB()
    b.boxmm((0, 0, 1.0), (76, 115, 1.6), 0)
    _sink(b, 0, 8, 46, 46, 16, 16, 2.0, 1, 1, False)
    for s in (-1, 1):
        b.boxmm((s * 17, -48.5, 7.5), (20, 22, 13), 2)           # cages (belly-to-belly style block)
        b.boxmm((s * 17, -59.6, 7.5), (17, 0.4, 10), 3)          # dark port opening
    for i in range(6):
        b.boxmm((-30 + i * 12, 40, 2.6), (7, 5, 2.4), 3)         # misc VRM / PHY blocks
    b.boxmm((0, 55.5, 2.2), (66, 4, 3.4), 4)                     # edge connector housing (back)
    return b, [M["pcb_blue"], M["fin"], M["steel"], M["black"], M["gold"]]


def coldplate_manifold(M):
    """Cold plate over a processor / GPU: 100 x 70 x 12 mm plate with two hose barbs and a stub manifold bar (barbs +x)."""
    b = MB()
    b.boxmm((0, 0, 6.0), (100, 70, 12), 0)
    b.boxmm((0, 0, 12.3), (86, 56, 0.6), 1)                      # cover plate
    for k, sy in enumerate((-18, 18)):
        b.cyl((57 * 0.001, sy * 0.001, 8.0 * 0.001), 0.0045, 0.014, "x", 12, 1)
        b.cyl((66 * 0.001, sy * 0.001, 8.0 * 0.001), 0.0058, 0.004, "x", 12, 2 + k)
    for sx in (-40, 40):
        for sy in (-27, 27):
            b.cyl((sx * 0.001, sy * 0.001, 13.2 * 0.001), 0.0035, 0.002, "z", 8, 4)     # spring screws
    return b, [M["cold"], M["nickel"], M["qd_blue"], M["qd_red"], M["steel"]]


def qd_pair(M):
    """Rear quick-disconnect pair: blue supply and red return (14 mm bodies, 10 mm hose stubs), 40 mm centre distance."""
    b = MB()
    for k, x in enumerate((-20, 20)):
        b.cyl((x * 0.001, 14 * 0.001, 12 * 0.001), 0.0085, 0.028, "y", 16, 0)
        b.cyl((x * 0.001, 6 * 0.001, 12 * 0.001), 0.0105, 0.01, "y", 16, 1 + k)
        b.cyl((x * 0.001, -4 * 0.001, 12 * 0.001), 0.005, 0.020, "y", 12, 3)
    b.boxmm((0, 4, 3), (60, 20, 6), 4)                            # mounting block
    return b, [M["nickel"], M["qd_blue"], M["qd_red"], M["hose"], M["dark_steel"]]


def fan_module(M):
    """1U dual-rotor fan module: 40 mm x 56 mm per the Lenovo GB300 guide ('40 mm x 56 mm dual-rotor'); read as a 40 x 40 mm frame, 56 mm deep (airflow along y)."""
    b = MB()
    W, Dp, H = 40.0, 56.0, 40.0
    for sx in (-1, 1):
        b.boxmm((sx * (W / 2 - 1.2), 0, H / 2), (2.4, Dp, H), 0)
    for sz in (-1, 1):
        b.boxmm((0, 0, H / 2 + sz * (H / 2 - 1.2)), (W, Dp, 2.4), 0)
    for yy in (-Dp / 4, Dp / 4):
        b.cyl((0, yy * 0.001, H / 2 * 0.001), 0.003 * 3, 0.008, "y", 14, 1)
        for k in range(7):
            a = 2 * math.pi * k / 7
            c, s = math.cos(a), math.sin(a)
            # blade: thin box approximated by a rotated quad prism (flat quad, 1.2 mm thick)
            x0, z0 = H / 2 + 0, H / 2
            pts = [(c * 5 - s * -1.5, s * 5 + c * -1.5), (c * 17 - s * -4.5, s * 17 + c * -4.5),
                   (c * 17 - s * 4.5, s * 17 + c * 4.5), (c * 5 - s * 1.5, s * 5 + c * 1.5)]
            lo = [(p[0] * 0.001, (yy - 0.6) * 0.001, (z0 + p[1]) * 0.001) for p in pts]
            hi = [(p[0] * 0.001, (yy + 0.6) * 0.001, (z0 + p[1]) * 0.001) for p in pts]
            b._add(lo + hi, [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)], 1)
    return b, [M["fanbody"], M["fanrot"]]


def pcie_slot(M):
    """PCIe x16 slot housing (CEM x16 length about 89 mm, estimate for height 11 mm), contact comb in the slot."""
    b = MB()
    b.boxmm((0, 0, 5.5), (89, 7.6, 11), 0)
    b.boxmm((0, 0, 11.05), (80, 1.4, 0.3), 1)
    return b, [M["black"], M["gold"]]


def bmc_card(M):
    """BMC mezzanine card: 50 x 34 mm PCB, BGA controller, 2 flash chips, pin header, standoffs."""
    b = MB()
    b.boxmm((0, 0, 4.0), (50, 34, 1.4), 0)
    b.boxmm((-8, 2, 5.3), (14, 14, 1.2), 1)
    for i in range(2):
        b.boxmm((12, -8 + i * 12, 5.1), (8, 6, 0.9), 1)
    b.boxmm((0, -15, 5.4), (30, 2.5, 8.0 - 3.0), 2)
    for sx in (-22, 22):
        for sy in (-14, 14):
            b.cyl((sx * 0.001, sy * 0.001, 1.8 * 0.001), 0.0025, 0.0036, "z", 8, 3)
    return b, [M["pcb_black"], M["mold"], M["black"], M["brass"]]


def coin_cell(M):
    """CR2032 holder with coin cell: 20 mm dia x 3.2 mm cell (CR2032 standard outline), holder 25 x 22 x 5 mm."""
    b = MB()
    b.boxmm((0, 0, 2.5), (25, 22, 5), 0)
    b.cyl((0, 0, 0.0066), 0.010, 0.0032, "z", 24, 1)
    return b, [M["black"], M["cap_top"]]


def e1s_bank(M):
    """Bank of 4 E1.S-style drives on edge, long axis y: 118.75 x 33.75 x 9.5 mm each (EDSFF E1.S 9.5 mm outline, from memory of SFF-TA-1006: estimate), 11 mm pitch."""
    b = MB()
    for i in range(4):
        x = (i - 1.5) * 11.0
        b.boxmm((x, 0, 3.0 + 33.75 / 2), (9.5, 118.75, 33.75), 0)
        b.boxmm((x, -59.6, 3.0 + 33.75 / 2), (7, 1.2, 28), 1)       # front bezel mark
        b.boxmm((x, -50, 3.0 + 33.75 + 0.1), (6, 40, 0.3), 2)       # label strip
    b.boxmm((0, 62, 2.0), (46, 6, 4.0), 3)                          # backplane connector row
    return b, [M["alu"], M["led_g"], M["label"], M["black"]]


PARTS = dict(vrm_block=vrm_block, inductor_bank=inductor_bank, cap_bank=cap_bank, heatsink_fin_stack=heatsink_fin_stack,
             socamm_module=socamm_module, dimm_module=dimm_module, nic_dpu_card=nic_dpu_card,
             coldplate_manifold=coldplate_manifold, qd_pair=qd_pair, fan_module=fan_module, pcie_slot=pcie_slot,
             bmc_card=bmc_card, coin_cell=coin_cell, e1s_bank=e1s_bank)

# (accuracy, sizes provenance)
PART_META = {
    "vrm_block": ("C", "60 x 50 x 7 mm: estimate (range 40-90 x 30-70); chokes 12 mm and cap sizes typical, not measured"),
    "inductor_bank": ("C", "6 x 14 x 14 x 8 mm molded chokes: estimate (range 10-20 mm)"),
    "cap_bank": ("C", "8 mm dia x 11 mm cans, 9.5 mm pitch: typical electrolytic/polymer can outline (estimate, range 6-10 dia)"),
    "heatsink_fin_stack": ("C", "60 x 60 x 24 mm, 22 fins: estimate (range 40-80 x 15-30)"),
    "socamm_module": ("C", "14 x 90 mm card: recollection of the public JEDEC SOCAMM2 figures, not re-read (range 12-16 x 85-95); thickness and package sizes estimated"),
    "dimm_module": ("B", "133.35 x 31.25 mm: standard JEDEC DIMM outline (recollected, not re-read); component sizes estimated"),
    "nic_dpu_card": ("C", "76 x 115 mm: OCP NIC 3.0 small-form-factor outline recollection (not re-read; range +-5); heat sink / cages generic"),
    "coldplate_manifold": ("C", "100 x 70 x 12 mm: estimate (range 60-130 x 50-100); barbs 9 mm"),
    "qd_pair": ("C", "14 mm bodies, 40 mm spacing: estimate (range 10-20 mm; the Lenovo guide states a rear coolant supply and return on the NVLink switch tray, no sizes)"),
    "fan_module": ("B", "40 mm x 56 mm dual-rotor from the Lenovo GB300 NVL72 user guide; the 3rd dimension (40 mm frame) is an assumption"),
    "pcie_slot": ("C", "89 mm x16 length from the PCIe CEM; height/width estimated"),
    "bmc_card": ("C", "50 x 34 mm: estimate (range 40-70 x 25-45); the Lenovo guide lists a BMC card but gives no size"),
    "coin_cell": ("A", "CR2032 20 mm x 3.2 mm standard cell outline; holder estimated (the Lenovo guide lists a CR2032 CMOS battery)"),
    "e1s_bank": ("C", "E1.S 118.75 x 33.75 x 9.5 mm recollection of EDSFF SFF-TA-1006 (not re-read; range +-3); 4 on edge per bank (Lenovo: up to 8 E1.S bays at the front)"),
}


def xform(mb, rot_z=0.0, off=(0, 0, 0), mmap=None, out=None, sc=1.0, zs=1.0):
    """Return a copy of mb rotated about z (radians) and offset (metres); mmap maps local material index -> layout index."""
    out = out or MB()
    c, s = math.cos(rot_z), math.sin(rot_z)
    o = len(out.v)
    for x, y, z in mb.v:
        x, y = x * sc, y * sc
        out.v.append((c * x - s * y + off[0], s * x + c * y + off[1], z * zs + off[2]))
    for f in mb.f:
        out.f.append(tuple(o + i for i in f))
    out.mi.extend((mmap[m] if mmap else m) for m in mb.mi)
    return out
