"""Build narrowcom_retimer_chip: parody retimer BGA (black mold, gold laser marking, valley mark).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_narrowcom_retimer_chip.py -- assets/components/packaging
z = 0 at the ball-tip plane (= top of the PCB pads when mounted); chip centered in X/Y; A1 corner at (-X, +Y); front = -Y.
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
AID = "narrowcom_retimer_chip"
P = AID + "_"
mat = pk.mat
C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="C")
root["p_glow"] = 0.0
src = C.sub_collection(coll, "SOURCES")
cmount = C.sub_collection(coll, "VARIANT_mounted")
chs = C.sub_collection(coll, "VARIANT_heat_spreader")
for c_ in (cmount, chs):
    c_.hide_render = True
    c_.hide_viewport = True
dims = pk.Dims()

BODY = 15.0
PITCH, BALL_D = 0.8, 0.45
N = 17
CV = 5
HB = 0.68 * BALL_D            # collapsed standoff
SUB_T = 0.50
MOLD_T = 0.90
Z0 = HB                       # substrate underside
Z1 = Z0 + SUB_T               # substrate top
ZM = Z1 + MOLD_T              # mold top
PCB_T = 1.6
PADH = 0.035


def obj(name, mb, mats, c=coll, parent=root, loc=(0, 0, 0)):
    return mb.to_obj(P + name, [mat(m) for m in mats], c, parent, loc)


# ---- package substrate
x = BODY / 2
sub_pts = [(-x + 1.2, x), (x - 0.2, x), (x, x - 0.2), (x, -x + 0.2), (x - 0.2, -x), (-x + 0.2, -x), (-x, -x + 0.2), (-x, x - 1.2)]
mb = MB()
mb.prism(sub_pts, Z0, Z1, mi=(0, 1, 2))
obj("substrate", mb, ["substrate", "fr4", "mask_green"])
# ---- mold compound body (matte black, chamfered top edge)
mb = MB()
mb.box((0, 0, Z1 + MOLD_T / 2), (BODY - 0.1, BODY - 0.1, MOLD_T), mi=0, bevel=0.18, seg=2)
mold = obj("mold", mb, ["mold_black"])
# ---- balls
cells = []
for j in range(N):
    for i in range(N):
        ci, cj = i - (N - 1) / 2, j - (N - 1) / 2
        if abs(ci) <= (CV - 1) / 2 and abs(cj) <= (CV - 1) / 2:
            continue
        if i in (0, N - 1) and j in (0, N - 1):
            continue
        cells.append((ci * PITCH, cj * PITCH, 0.0))
mb = MB()
mb.lathe([(0.26 * BALL_D, 0.0), (0.485 * BALL_D, 0.28 * HB), (BALL_D / 2, 0.5 * HB), (0.44 * BALL_D, 0.78 * HB), (0.34 * BALL_D, HB)], seg=14, mi=0)
bsrc = pk.source(mb.to_obj(P + "src_ball", [mat("solder")], src, root))
pk.scatter(P + "balls", cells, bsrc, coll, root)
# fiducial + A1 on underside
mb = MB()
mb.prism([(-x + 0.4, x - 0.4), (-x + 1.8, x - 0.4), (-x + 0.4, x - 1.8)], Z0 - 0.02, Z0 - 0.001, mi=0)
obj("pin1_underside", mb, ["silkscreen"])

# ---- marking on the mold top: valley mark, wordmark, part-number rows, pin-1 dimple
mark_mat = mat("laser_gold")
vm = pk.valley_mark_mesh(2.4, 0.24, 0.0, 0.006, mi=0)
vmo = vm.to_obj(P + "valley_mark", [mark_mat], coll, root, loc=(-5.0, 3.75, ZM))
pk.text(P + "wordmark", "NARROWCOM", 1.25, (2.4, 3.6, ZM), mark_mat, coll, root, extrude_mm=0.006)
rows = ["NCX-RT0800-A1", "8CH 106G PAM4 RETIMER", "TW 2640  LOT 000001", "NARROWCOM  R1  0001"]
for k, t in enumerate(rows):
    pk.text(P + "row_%d" % k, t, 0.95 if k else 1.0, (0.0, 1.0 - k * 1.55, ZM), mark_mat, coll, root, extrude_mm=0.005)
pk.text(P + "row_qr", "[ 00 ]", 0.8, (0.0, -5.6, ZM), mark_mat, coll, root, extrude_mm=0.005)
# pin-1 dimple: ring + dark dish
mb = MB()
mb.lathe([(0.0, 0.0), (0.62, 0.0), (0.62, 0.012), (0.0, 0.012)], c=(-x + 1.7, x - 1.7, ZM), seg=24, mi=0, smooth=False)
mb.lathe([(0.0, 0.012), (0.34, 0.012), (0.34, 0.018), (0.0, 0.018)], c=(-x + 1.7, x - 1.7, ZM), seg=24, mi=1, smooth=False)
obj("pin1_dimple", mb, ["mold_black", "plastic_black"])

# ---- glow flash helpers: glow plane (transparent unless p_glow > 0) and hooks
gm = bpy.data.materials.new("MAT_packaging_chip_glow")
gm.use_nodes = True
nt = gm.node_tree
for n_ in list(nt.nodes):
    nt.nodes.remove(n_)
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1.0, 0.45, 0.10, 1)
em.inputs["Strength"].default_value = 3.0
tr = nt.nodes.new("ShaderNodeBsdfTransparent")
mix = nt.nodes.new("ShaderNodeMixShader")
outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tr.outputs[0], mix.inputs[1])
nt.links.new(em.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], outn.inputs[0])
fc = nt.driver_add('nodes["Mix Shader"].inputs[0].default_value')
fc.driver.type = "SCRIPTED"
fc.driver.expression = "min(max(v, 0.0), 1.0)"
vv = fc.driver.variables.new()
vv.name = "v"
vv.type = "SINGLE_PROP"
vv.targets[0].id = root
vv.targets[0].data_path = '["p_glow"]'
try:
    gm.surface_render_method = "BLENDED"
except Exception:
    pass
gm.use_backface_culling = False
mb = MB()
mb.rrect((0, 0, 0), BODY + 3.0, BODY + 3.0, 0.02, 2.0, seg=4, mi=0)
g = mb.to_obj(P + "glow", [gm], coll, root, loc=(0, 0, -0.03))
g["note"] = "emissive orange plate under the chip; transparent at p_glow = 0, glowing at p_glow = 1 (driven by the root custom property p_glow)"
C.hook("glow_flash", coll, root, loc=(0, 0, 0.2 * MM))
C.hook("chip_top_center", coll, root, loc=(0, 0, ZM * MM))
C.hook("pin1", coll, root, loc=((-x + 1.7) * MM, (x - 1.7) * MM, ZM * MM))

# ---- VARIANT_mounted: blue PCB patch, pads, vias, underfill fillet, passives, silkscreen
PW, PD = 40.0, 32.0
zt_pcb = -PADH
mb = MB()
mb.box((0, 0, zt_pcb - PCB_T / 2), (PW, PD, PCB_T), mi=(0, 1, 2) if False else 0, bevel=0.05, seg=1)
obj("pcb_patch", mb, ["mask_blue"], c=cmount)
mb = MB()
r = 0.26
for (px, py, _) in cells:
    mb.lathe([(0.0, zt_pcb), (r, zt_pcb), (r, 0.0), (0.0, 0.0)], c=(px, py, 0), seg=14, mi=0, smooth=False)
obj("pads", mb, ["gold_enig"], c=cmount)
mb = MB()
mb.loft_rects((BODY + 1.3, BODY + 1.3, zt_pcb), (BODY, BODY, Z0 + 0.25), mi=0)
obj("underfill_fillet", mb, ["underfill"], c=cmount)
# vias and test pads
vsrc_mb = MB()
vsrc_mb.lathe([(0.15, 0.0), (0.30, 0.0), (0.30, 0.02), (0.15, 0.02)], seg=12, mi=0, smooth=False)
vsrc = pk.source(vsrc_mb.to_obj(P + "src_via_ring", [mat("gold_enig")], src, root))
vp = []
for k in range(14):
    vp.append((-6.5 + k * 1.0, -9.6, zt_pcb))
    vp.append((-6.5 + k * 1.0, 9.6, zt_pcb))
pk.scatter(P + "pcb_vias", vp, vsrc, cmount, root)
# passives (0402 caps) around the chip with solder fillets (small tin wedges)
mbc, cm = parts.mlcc("0402")
cs = pk.source(mbc.to_obj(P + "src_cap0402", [mat(m) for m in cm], src, root))
cp, crz = [], []
for k in range(8):
    cp.append((-11.5, -6 + k * 1.7, zt_pcb))
    crz.append(PI / 2)
    cp.append((11.5, -6 + k * 1.7, zt_pcb))
    crz.append(PI / 2)
pk.scatter(P + "pcb_caps", cp, cs, cmount, root, rz=crz)
mbp = MB()
for (px, py, _) in cp:
    for s_ in (-1, 1):
        mbp.box((px, py + s_ * 0.55, zt_pcb + 0.025), (0.5, 0.45, 0.05), mi=0)
obj("pcb_cap_pads", mbp, ["gold_enig"], c=cmount)
pk.text(P + "silk_u", "U7", 1.6, (-12.5, 12.0, zt_pcb), mat("silkscreen"), cmount, root, extrude_mm=0.02)
pk.text(P + "silk_u", "NARROWCOM RT-EVB", 1.2, (0, -14.2, zt_pcb), mat("silkscreen"), cmount, root, extrude_mm=0.02)
mb = MB()
mb.ribbon([(-x - 0.7, x + 0.7), (x + 0.7, x + 0.7), (x + 0.7, -x - 0.7), (-x - 0.7, -x - 0.7)], 0.15, zt_pcb, zt_pcb + 0.02, mi=0, closed=True)
obj("silk_outline", mb, ["silkscreen"], c=cmount)

# ---- VARIANT_heat_spreader
HSP = 13.2
mb = MB()
mb.box((0, 0, ZM + 0.1 + 1.0 / 2), (HSP, HSP, 1.0), mi=0, bevel=0.2, seg=2)
obj("heat_spreader", mb, ["nickel"], c=chs)
mb = MB()
mb.box((0, 0, ZM + 0.05), (HSP - 0.4, HSP - 0.4, 0.1), mi=0)
obj("spreader_tim", mb, ["thermal_pad"], c=chs)
C.hook("spreader_top", chs, root, loc=(0, 0, (ZM + 1.1) * MM))

# ---- dimension table
dims.add("package body", "15.0 x 15.0", "mm", "plausible 8-channel retimer BGA body (project brief suggests 15 x 15 mm); real parts range roughly 12-17 mm; no vendor part was copied", "C")
dims.add("ball pitch / ball diameter", "0.8 / 0.45", "mm", "JEDEC fine-pitch BGA 0.8 mm; typical 0.45 mm ball (see bga_lga_family sources)", "B")
dims.add("ball array", "17 x 17 minus 5 x 5 center minus 4 corners", "cells", "illustrative void pattern", "C")
dims.add("ball count", len(cells), "count", "counted from the array", "A")
dims.add("collapsed standoff", round(HB, 3), "mm", "0.68 x ball diameter", "C")
dims.add("substrate / mold thickness", "%.2f / %.2f" % (SUB_T, MOLD_T), "mm", "typical overmolded BGA; total height %.2f mm" % (ZM), "C")
dims.add("marking layout", "valley mark + NARROWCOM wordmark + 4 rows", "", "parody mark drawn from scratch: valley (arch with the tip flipped upside down) over a wave line; part number NCX-RT0800-A1 is invented", "A")
dims.add("PCB patch (mounted variant)", "40 x 32 x 1.6", "mm", "illustrative evaluation-board patch, blue solder mask (as in the Broadcom photo reference, material only)", "C")
dims.add("heat spreader (variant)", "13.2 x 13.2 x 1.0", "mm", "illustrative nickel-plated copper cap with 0.1 mm TIM", "C")

meta = {
    "title": "NARROWCOM retimer chip (parody)", "accuracy_level": "C (plausible generic) ; marking A (our own artwork)",
    "origin": "z = 0 at the ball-tip plane; X/Y centered; mounted variant PCB top at z = -0.035 mm; A1 corner at (-X, +Y)",
    "sources": [{"what": "material reference only (matte black mold, shallow gold laser lettering, rows of marking text, blue PCB): Wentao's chip photo (not reused, no logo traced)"},
                {"what": "ball pitch/diameter", "file": "assets/components/packaging/bga_lga_family.json"}],
    "variants": {"BASE": "bare chip on its balls (default)", "VARIANT_mounted": "blue PCB patch with pads, vias, 0402 caps, underfill edge fillet, silkscreen", "VARIANT_heat_spreader": "nickel cap with TIM on top of the mold (hides the marking)"},
    "glow": "object narrowcom_retimer_chip_glow + custom property p_glow (0..1) via driver; hooks HOOK_glow_flash (chip center at the pad plane), HOOK_chip_top_center",
    "instancing": "balls, vias, caps via Geometry Nodes (NG_pk_scatter)",
    "simplifications": ["no solder fillets on passives (balls and underfill only)", "traces not modeled on the patch (see pcb_generator)", "dimple is a ring plus dark disc, not a real depression"],
    "scene_usage": "S2 retimer rows (instance many; glow flash with p_glow); S4 board retimers",
}
blend = os.path.join(OUT, AID + ".blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
bb, tris = pk.eval_stats(coll)
pngs = []
R = lambda views, pre="": pngs.extend(pk.render_views(coll, pv, AID, [dict(v, name=pre + v["name"]) for v in views]))
R([dict(name="front", loc=(0, -42, 9), tgt=(0, 0, 0.9), lens=50, floor=True),
   dict(name="three_quarter", loc=(24, -28, 22), tgt=(0, 0, 0.8), lens=50, floor=True),
   dict(name="top", loc=(0, -0.1, 44), tgt=(0, 0, 1), lens=50),
   dict(name="closeup_marking", loc=(-2, -9, 11), tgt=(-1, 2, 1.3), lens=70, floor=True),
   dict(name="underside", loc=(0, -0.1, -42), tgt=(0, 0, 0), lens=50),
   dict(name="closeup_balls", loc=(-5.5, -9.0, -4.5), tgt=(-5.5, -4.0, 0.2), lens=70)])
cmount.hide_render = cmount.hide_viewport = False
bpy.context.view_layer.update()
R([dict(name="three_quarter", loc=(48, -50, 38), tgt=(0, 0, 0), lens=50, floor=True),
   dict(name="closeup_fillet", loc=(8, -16, 3.2), tgt=(6.5, -7.5, 0.6), lens=80, floor=True)], "mounted_")
chs.hide_render = chs.hide_viewport = False
bpy.context.view_layer.update()
R([dict(name="three_quarter", loc=(48, -50, 38), tgt=(0, 0, 0), lens=50, floor=True)], "heat_spreader_")
root["p_glow"] = 1.0
bpy.context.view_layer.update()
R([dict(name="three_quarter", loc=(48, -50, 38), tgt=(0, 0, 0), lens=50, floor=True)], "glow_")
root["p_glow"] = 0.0
cmount.hide_render = cmount.hide_viewport = True
chs.hide_render = chs.hide_viewport = True
C.save(blend)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)],
             "triangles_including_instances_default_variant": tris, "dimension_table": dims.rows, "hooks": hooks,
             "material_slots": mnames, "custom_properties": props, "previews": [os.path.relpath(p, OUT) for p in pngs]})
pk.write_json(blend, meta)
print("DONE", AID, tris, len(cells), meta["size_mm"])
