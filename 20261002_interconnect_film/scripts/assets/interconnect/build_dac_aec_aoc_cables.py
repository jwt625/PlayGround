"""Build dac_aec_aoc_cables.blend: DAC (OSFP, 8 twinax pairs), AEC (QSFP-DD, retimer paddle card), AOC (OSFP, 8-fibre ribbon) + cross-sections.

Each assembly: plug A (openable, p_open_<type>), U-bend cable, plug B (closed). Cross-sections are true size with x20 magnified copies.
Run: Blender -b --python scripts/assets/interconnect/build_dac_aec_aoc_cables.py -- assets/components/interconnect
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
import ic_common as I
import ic_cables as K
import ic_modules as MOD
from ic_common import _v

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "dac_aec_aoc_cables"

C.reset()
M = I.Mats()
coll, root = C.new_asset(ASSET, accuracy="B")
specs = [("dac", "osfp", 7.0, "jacket_black", "tab_blue", 0.0), ("aec", "qsfpdd", 5.6, "jacket_grey", "tab_orange", 420.0), ("aoc", "osfp", 3.0, "jacket_aqua", "tab_orange", 840.0)]
R_BEND, L1 = 120.0, 160.0
lengths = {}
for mode, kind, od, jm, tab, xo in specs:
    sub = C.sub_collection(coll, "VARIANT_%s_%s" % (mode, kind))
    ga = I.empty("%s_assembly" % mode, sub, root, loc_mm=(xo, 0, 0))
    I.add_prop(root, "p_open_%s" % mode, 0.0, 0.0, 1.0, "0 closed, 1 opened: plug A top shell and card assembly lift (%s)" % mode)
    top, pcb, parts = K.build_plug(kind, mode, sub, ga, M, "%s_a" % mode, True, od, tab_mat=tab, seed=3)
    MOD.drivers_open(root, top, pcb, top_lift=40.0 if kind == "osfp" else 30.0, pcb_lift=16.0 if kind == "osfp" else 12.0) if False else None
    for g_, lift in ((top, 40.0 if kind == "osfp" else 30.0), (pcb, 16.0 if kind == "osfp" else 12.0)):
        I.drive_prop(g_, "location", 2, root, "p_open_%s" % mode, expr_scale=I.mm(lift))
    # plug B (closed): placed at x = 2R, facing the same direction (U-bend cable)
    gb = I.empty("%s_plug_b_group" % mode, sub, ga, loc_mm=(2 * R_BEND, 0, 0))
    K.build_plug(kind, mode, sub, gb, M, "%s_b" % mode, False, od, tab_mat=None, seed=5)
    D = K.DIMS[kind]
    y0 = D["Yf"] - K.BOOT_LEN
    path = K.u_cable(M, sub, ga, "%s_cable" % mode, y0, od, jm, R=R_BEND, L1=L1)
    zc = 6.5 if kind == "osfp" else 4.9
    path[:, 2] = zc * 0.001
    cab = I.tube_mesh("%s_cable" % mode, path[None], od / 2 * 0.001, sides=16, mat=M[jm], caps=True)
    I.place(cab, sub, ga)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * 1000
    lengths[mode] = round(float(seg), 1)
    C.hook("%s_cable_mid" % mode, sub, ga, loc=_v(R_BEND, y0 - L1 - R_BEND, zc))
    C.hook("%s_plug_b_axis" % mode, sub, gb, loc=_v(0, 0, zc))
# cross-sections: true size and x20 copies
cs = C.sub_collection(coll, "VARIANT_cross_sections")
ge = I.empty("cross_sections_root", cs, root, loc_mm=(0, 340, 0))
K.cross_section_twinax(M, cs, ge, (0, 0, 0), "xsec_twinax_1x", 1.0)
K.cross_section_aoc(M, cs, ge, (30, 0, 0), "xsec_aoc_1x", 1.0)
K.cross_section_twinax(M, cs, ge, (150, 0, 120), "xsec_twinax_20x", 20.0)
K.cross_section_aoc(M, cs, ge, (300, 0, 120), "xsec_aoc_20x", 20.0)
meta = dict(
    description="Three cable assemblies with U-bend runs, each ending in two plugs: DAC (OSFP, 8 twinax pairs 28 AWG + 4 low-speed wires), AEC (QSFP-DD, retimer paddle card, thinner cable), AOC (OSFP, 8-fibre ribbon, VCSEL/PD optical engine). Plug A of each is openable. Cross-sections of the twinax and AOC cable at true size and x20.",
    sources=[
        {"what": "OSFP MSA Rev 5.22 and QSFP-DD HW Rev 5.1 plug outlines and card edges (same dimensions as modules)", "url": "https://www.osfpmsa.org/specification.html", "accessed": "2026-10-02"},
        {"what": "AWG wire diameters: 26 AWG 0.405 mm, 28 AWG 0.321 mm, 30 AWG 0.255 mm (standard table, level A); 28 AWG used for the cross-section", "url": "n/a (standard AWG table)", "accessed": "2026-10-02"},
        {"what": "Fibre geometry: 125 um cladding, 250 um coated, 9 um core (G.652)", "url": "n/a (standard)", "accessed": "2026-10-02"},
    ],
    dimension_table=[
        dict(item="plug outline (OSFP / QSFP-DD)", value="22.58 x 13.0 / 18.35 x 8.5 (front block 19 x 13.5)", unit="mm", source="MSA specs, same as modules", accuracy="A"),
        dict(item="card-edge pads", value="30 per side pitch 0.6 / 38 per side 2 rows pitch 0.8", unit="count", source="MSA specs", accuracy="A"),
        dict(item="DAC cable OD", value=7.0, unit="mm", source="own geometric estimate from the cross-section (8 pairs 28 AWG + 4 wires + foil/braid + jacket); product data 6-9 mm", accuracy="C"),
        dict(item="AEC cable OD", value=5.6, unit="mm", source="estimate (30-32 AWG)", accuracy="C"),
        dict(item="AOC cable OD", value=3.0, unit="mm", source="task: 3.0 mm jacket class", accuracy="B"),
        dict(item="conductor 28 AWG / dielectric OD", value="0.321 / 0.62", unit="mm", source="AWG table / estimate", accuracy="B"),
        dict(item="U-bend radius / total length per assembly", value="%s mm / %s" % (K.__dict__.get("x", 120), lengths), unit="mm", source="model parameter (bend radius > 15 x OD)", accuracy="C"),
        dict(item="boot length", value=K.BOOT_LEN, unit="mm", source="estimate", accuracy="C"),
    ],
    hooks={"HOOK_<type>_a_plug_axis / HOOK_<type>_b_plug_axis": "insertion axis of each plug (local +Y)", "HOOK_<type>_a_open": "on plug A top shell", "HOOK_<type>_cable_mid": "mid-point of the U bend (types dac, aec, aoc)"},
    custom_properties={"p_open_dac": "0..1", "p_open_aec": "0..1", "p_open_aoc": "0..1"},
    cross_section_notes="x1 at (0,340) mm; x20 copies at (150,340,120) and (300,340,120) mm under scaled empties; layers: jacket, overall foil/braid, 8 pair foils, 16 dielectric-coated conductors, 8 drain wires, 4 low-speed wires (twinax); jacket, aramid yarn, ribbon matrix, 8 fibres + cores, 2 strength rods (AOC).",
    origin="ROOT at the origin; each assembly is a child empty at x = 0 / 420 / 840 mm; plug A forward-stop plane at its empty; cable is a fixed U bend in the XY plane (not a rig).",
    simplifications=["cable shape is baked (no slack/length control); only the U bend", "no braid weave; solder joints are pads only", "DSP-less AOC engine is generic", "cable jackets have no printed text", "AEC retimer text generic, no logos"],
    intended_usage="S6 AOC whip (cable only, scaled), S1/S2 DAC/AEC context; cross-sections for explanatory cutaways.",
)
meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
meta["triangles_unique_meshes"] = I.unique_tris(coll)
blend = os.path.join(OUT, ASSET + ".blend")
C.finish(ASSET, blend, coll, meta, preview_dir=None)
prev = os.path.join(OUT, "previews")
I.preview_views(coll, prev, ASSET, [("overview_top", (400, -150, 0), (0, -0.05, 1), 1250, 40), ("overview_three_quarter", (400, -150, 0), (0.5, -1.0, 0.8), 1100, 40),
                                    ("dac_plug_closeup", (0, -35, 6), (0.7, -0.9, 0.6), 190, 50), ("aec_plug_closeup", (420, -35, 4), (0.7, -0.9, 0.6), 160, 50),
                                    ("xsec_twinax_20x", (150, 340.5, 120), (0, -1, 0), 330, 50), ("xsec_aoc_20x", (300, 340.5, 120), (0, -1, 0), 150, 50)])
for md in ("dac", "aec", "aoc"):
    root["p_open_%s" % md] = 1.0
root.update_tag()
bpy.context.view_layer.update()
I.preview_views(coll, prev, ASSET, [("aec_opened_closeup", (420, -35, 14), (0.4, -0.8, 0.9), 170, 50), ("aoc_opened_closeup", (840, -35, 18), (0.8, -0.8, 0.3), 220, 50)])
