"""Build passives_library: SMD passives, magnetics, QFN power parts, connectors, test points and hardware, each its own object.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_passives_library.py -- assets/components/packaging
Every part is a separate mesh object named passives_library_<name>, origin at the footprint center, board-contact plane at z = 0.
Parts are laid out on a display grid (rows per category); to use one, duplicate the object (linked data) and move it. Mesh-only parts: use
the instancing sources in the hgx_baseboard / xpu assets as an example of Geometry Nodes scatter (NG_pk_scatter).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import parts  # noqa: E402
from pk import MB, MM, PI  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
AID = "passives_library"
P = AID + "_"
mat = pk.mat
C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="B")
dims = pk.Dims()

REG = []   # (category, name, builder, source, level, text)
CS = parts.CHIP_SIZES
for sz in ("0201", "0402", "0603", "0805", "1206"):
    REG.append(("mlcc", "mlcc_%s" % sz, (lambda s=sz: parts.mlcc(s)), "EIA %s chip: L x W x T = %.2f x %.2f x %.2f mm (nominal, Murata/Samsung MLCC datasheets)" % ((sz,) + CS[sz]), "A", None))
for sz in ("0402", "0603", "0805", "1206"):
    REG.append(("resistor", "res_%s" % sz, (lambda s=sz: parts.chip_resistor(s)), "EIA %s thick-film chip resistor, T %.2f mm (Yageo RC series nominal)" % (sz, parts.RES_T[sz]), "A", None))
REG.append(("inductor", "ferrite_bead_0603", parts.ferrite_bead, "EIA 0603 ferrite bead 1.6 x 0.8 x 0.8", "A", None))
REG.append(("inductor", "chip_inductor_0603", parts.chip_inductor_0603, "EIA 0603 chip inductor 1.6 x 0.8 x 0.8", "A", None))
REG.append(("inductor", "power_inductor_7x7", lambda: parts.power_inductor(7.0, 7.0, 3.5), "shielded molded power inductor 7.0 x 7.0 x 3.5 mm (class of Coilcraft XAL7030 / Wurth WE-MAPI 7x7; footprint 7-7.5 mm, height 3-4.8 mm)", "B", "1R0"))
REG.append(("inductor", "power_inductor_10x10", lambda: parts.power_inductor(10.0, 10.0, 5.0), "shielded molded power inductor 10.0 x 10.0 x 5.0 mm (class of XAL1050 / WE-MAPI 10x10; height 4-6 mm)", "B", "R22"))
for case in ("A", "B", "C", "D"):
    REG.append(("capacitor", "tantalum_case_%s" % case, (lambda c=case: parts.tantalum(c)), "EIA 535BAAC case %s molded tantalum" % case, "A", None))
REG.append(("capacitor", "polymer_cap_case_D", lambda: parts.tantalum("D", polymer=True), "polymer tantalum, case D 7343 (7.3 x 4.3 x 3.1 mm class)", "A", None))
REG.append(("capacitor", "alu_electrolytic_6p3x7p7", parts.alu_electrolytic, "SMD aluminum electrolytic 6.3 x 7.7 mm (V-chip)", "A", None))
REG.append(("magnetics", "toroid_12mm", parts.toroid, "ferrite toroid OD 12 / ID 6 / H 5 mm, 28 turns (generic; sizes C)", "C", None))
REG.append(("magnetics", "flyback_transformer", parts.flyback_transformer, "small SMD flyback transformer with 2 x 5 pins, bobbin 13 x 11 mm (generic; C)", "C", None))
REG.append(("power_ic", "drmos_5x6", parts.drmos_5x6, "DrMOS / power stage PQFN 5 x 6 x 0.8 mm class with 3 large pads (generic; B)", "B", "DR56"))
REG.append(("power_ic", "dcdc_controller_qfn_4x4", parts.dcdc_controller_4x4, "DC-DC controller QFN 4 x 4 mm, 24 leads at 0.5 mm pitch, 2.7 mm exposed pad", "B", "DC44"))
REG.append(("connector", "b2b_receptacle_2x25", parts.b2b_receptacle, "board-to-board receptacle, 50 pos, 0.5 mm pitch (generic 0.5 mm class; height 1.5, width 4.0)", "B", None))
REG.append(("connector", "b2b_plug_2x25", parts.b2b_plug, "board-to-board plug, 50 pos, 0.5 mm pitch", "B", None))
REG.append(("connector", "power_connector_2x3", parts.power_connector_2x3, "2 x 3 power header, 4.2 mm pitch (Mini-Fit class), 11 mm tall", "B", None))
REG.append(("testpoint", "test_point_pad_1p0", parts.test_point_pad, "SMD test pad dia 1.0 mm with 1.6 mm mask opening", "B", None))
REG.append(("testpoint", "test_point_loop", parts.test_point_loop, "through-hole loop test point (5 mm loop, 0.4 mm wire)", "B", None))
REG.append(("hardware", "screw_m3x6", parts.screw_m3, "M3 x 6 pan-head cross-recess screw, head 5.6 x 2.4 mm (ISO 7045 class); origin at the head seating plane", "A", None))
REG.append(("hardware", "standoff_m3_hex_8", parts.standoff_m3_hex, "M3 F-F hex standoff, 5.5 mm across flats, 8 mm long", "A", None))
REG.append(("hardware", "washer_m3", parts.washer_m3, "M3 washer, ID 3.2 mm / OD 7 mm / 0.5 mm", "A", None))
REG.append(("indicator", "led_0603_green", lambda: parts.led_0603("green"), "0603 LED (1.6 x 0.8 mm), emissive lens", "B", None))
REG.append(("indicator", "led_0603_red", lambda: parts.led_0603("red"), "0603 LED, red lens", "B", None))

cats = []
for r in REG:
    if r[0] not in cats:
        cats.append(r[0])
y = 0.0
rows = {}
built = {}
for cat in cats:
    items = [r for r in REG if r[0] == cat]
    x = 0.0
    rowh = 0.0
    sub = C.sub_collection(coll, "CAT_" + cat)
    for (c_, name, fn, srcnote, lvl, txt) in items:
        mb, mnames = fn()
        o = mb.to_obj(P + name, [mat(m) if isinstance(m, str) else m for m in mnames], sub, root)
        bbx = [v for v in o.bound_box]
        w_ = (max(v[0] for v in bbx) - min(v[0] for v in bbx)) / MM
        d_ = (max(v[1] for v in bbx) - min(v[1] for v in bbx)) / MM
        h_ = (max(v[2] for v in bbx) - min(v[2] for v in bbx)) / MM
        cx = x + w_ / 2 + 0.5
        o.location = (cx * MM, (y - d_ / 2) * MM, 0)
        o["pk_footprint_size_mm"] = [round(w_, 3), round(d_, 3), round(h_, 3)]
        if txt:
            t = pk.text(P + name + "_mark", txt, min(w_, d_) * 0.22, (cx, y - d_ / 2, h_), mat("steel"), sub, root, extrude_mm=0.01)
        x = cx + w_ / 2 + 1.5 + (2.0 if w_ > 4 else 0.0)
        rowh = max(rowh, d_)
        dims.add(name, [round(w_, 3), round(d_, 3), round(h_, 3)], "mm (L x W x H bbox)", srcnote, lvl)
        C.hook("origin_" + name, sub, o, loc=(0, 0, 0))
    y -= rowh + 4.0
bb, tris = pk.eval_stats(coll)
meta = {
    "title": "SMD passives, magnetics, power ICs, connectors, test points and hardware library",
    "accuracy_level": "A for EIA chip sizes and standard hardware; B/C for magnetics and connectors (generic)",
    "origin": "root at the origin; parts are on a display grid in the file (rows per category at -Y); each object's origin is its footprint center with the board-contact plane at z = 0",
    "sources": [{"what": "EIA chip sizes, tantalum cases (EIA 535BAAC), ISO 7045 pan head screw dimensions, typical datasheet nominals", "note": "recalled standard values, not re-fetched this session; verify against datasheets for any close-up"}],
    "usage": "duplicate with linked mesh data (Alt+D) or scatter with Geometry Nodes; mats are MAT_packaging_*; marking text objects named *_mark",
    "simplifications": ["no solder fillets on chip parts (pads are not modeled; place on pcb_generator pads)", "connector pin counts and magnetics are generic"],
    "scene_usage": "S4 board (hgx_baseboard builds its populations from these builders); S2 tray boards",
}
blend = os.path.join(OUT, AID + ".blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
lo, hi = bb
W_ = hi[0] - lo[0]
H_ = hi[1] - lo[1]
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
views = [dict(name="top", loc=(cx, cy - 0.01, max(W_, H_ * 1.33) * 0.95), tgt=(cx, cy, 0), lens=50),
         dict(name="three_quarter", loc=(cx + W_ * 0.5, cy - H_ * 1.1, W_ * 0.7), tgt=(cx, cy, 0), lens=50, floor=True),
         dict(name="front", loc=(cx, cy - H_ * 2.0 - 40, 25), tgt=(cx, cy, 2), lens=60, floor=True)]
# close-up rows
rowy = {}
for o in coll.all_objects:
    if o.name.startswith(P) and "pk_footprint_size_mm" in o.keys():
        rowy[o.name] = (o.location.x / MM, o.location.y / MM)


def near(name):
    return rowy[P + name]


pngs = []
sets = [("closeup_mlcc", ["mlcc_0201", "mlcc_1206"], 18), ("closeup_inductors_caps", ["power_inductor_7x7", "tantalum_case_D"], 30),
        ("closeup_magnetics", ["toroid_12mm", "flyback_transformer"], 34), ("closeup_ics_connectors", ["drmos_5x6", "b2b_receptacle_2x25"], 30),
        ("closeup_hardware", ["screw_m3x6", "standoff_m3_hex_8"], 24)]
cv = list(views)
for nm, ns, dist in sets:
    ax = sum(near(n)[0] for n in ns) / len(ns)
    ay = sum(near(n)[1] for n in ns) / len(ns)
    cv.append(dict(name=nm, loc=(ax, ay - dist, dist * 0.7), tgt=(ax, ay, 1.5), lens=50, floor=True))
pngs = pk.render_views(coll, pv, AID, cv)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm": bb, "size_mm": [round(hi[k] - lo[k], 3) for k in range(3)], "triangles_including_instances": tris,
             "dimension_table": dims.rows, "hooks": [h["name"] for h in hooks][:3] + ["HOOK_origin_<part> (one per part)"], "material_slots": mnames, "custom_properties": props,
             "previews": [os.path.relpath(p, OUT) for p in pngs]})
pk.write_json(blend, meta)
print("DONE", AID, tris, len(REG), meta["size_mm"])
