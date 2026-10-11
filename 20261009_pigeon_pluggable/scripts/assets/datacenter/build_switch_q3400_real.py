"""switch_q3400_real: true-scale Quantum-X800 Q3400-RA style 4U switch (438 W x 177.8 H x 850 D mm, 72 OSFP cages).

Reference/scale asset for the bird rack. Sources: research/racks_refs.md section 3. No logo; parody badge "NVYDIA QX-3400".
Frame: origin = bottom centre of the faceplate front plane... (see meta.frame). FRONT = +X, width along Y, Z up.
Run: scripts/bslot.sh -b --python scripts/assets/datacenter/build_switch_q3400_real.py -- assets/components/datacenter
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import rk_lib as K  # noqa: E402
import rk_q3400 as Q  # noqa: E402

OUT, KV = K.parse_args()
AID = "switch_q3400_real"
PREV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(OUT)), "..", "..", "outputs", "assets", "datacenter"))
K.reset()
L = K.palette()
coll, root = K.new_asset(AID, accuracy="B")
parts, lay, labels = Q.chassis(Q.REAL, L)
K.instantiate(parts, coll, root, AID)
bpy.context.view_layer.update()
Q.label_texts(Q.REAL, L, labels, coll, root, AID, "NVYDIA\nQX-3400")   # parody badge (owner rule)
K.hook("front_center", coll, root, (0.002, 0, Q.REAL["H"] / 2000))
bpy.context.view_layer.update()

os.makedirs(OUT, exist_ok=True)
bb = K.bbox(coll)
meta = dict(asset_id=AID, date="2026-10-10", units="m",
            frame="origin = faceplate front plane, centre of width, chassis bottom; FRONT = +X, width along Y, Z up; body extends to x = -0.85",
            size_m=[round(bb[1][i] - bb[0][i], 4) for i in range(3)], triangles=K.count_tris(coll),
            cages=len(lay["cells"]), layout="3 block rows x 3 column groups x (2 x 4) cages; label strip between cage pairs",
            measured=dict(cage_pitch_mm=lay["ow"], cage_outline_h_mm=lay["oh"], group_gap_mm=round(lay["gap"], 2),
                          block_gap_mm=round(lay["vgap"], 2), margin_mm=round(lay["marg"], 2)),
            sources="research/racks_refs.md section 3 (NVIDIA datasheet 3231555, XDR switch UM specs + front/rear figures)",
            accuracy=dict(envelope="A", front_layout="B (figure measured, +-1 mm)", rear="B", internals="not modelled"),
            hooks=K.hooks_of(coll))
K.save_blend(os.path.join(OUT, AID + ".blend"))
K.export_glb(coll, os.path.join(OUT, AID + ".glb"))
K.write_json(os.path.join(OUT, AID + ".json"), meta)
print("ASSET", AID, "tris", meta["triangles"], "size_m", meta["size_m"])

if int(KV.get("previews", 1)):
    os.makedirs(PREV, exist_ok=True)
    K.preview_setup(res=(1296, 520), samples=16, world=0.15, exposure=-0.2)
    K.light("key", "AREA", (1.5, -0.6, 0.8), 16, 1.0, (0, 0, 0.09))
    K.light("fill", "AREA", (1.4, 0.8, 0.3), 6, 1.0, (0, 0, 0.09))
    K.light("rear", "AREA", (-1.8, 0.4, 0.6), 20, 1.0, (-0.85, 0, 0.09))
    V = lambda n: os.path.join(PREV, "%s_%s.png" % (AID, n))
    K.render_view(V("front"), (1.0, 0, 0.0889), (0, 0, 0.0889), ortho=0.438)
    K.render_view(V("rear"), (-2.0, 0, 0.0889), (-0.85, 0, 0.0889), ortho=0.438)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1200, 800
    K.render_view(V("three_quarter"), (0.55, -0.42, 0.32), (-0.2, 0, 0.06), lens=40)
