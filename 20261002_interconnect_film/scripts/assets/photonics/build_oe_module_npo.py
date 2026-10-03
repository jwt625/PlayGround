"""oe_module_npo: near-package-optics module: HDI board carrier with LGA underside, PIC+EIC+16-fibre FAU under a heat-spreader lid,
ribbon routed on the board to an MT-16 board-edge optical connector (housing with latch) at the +X end.
Origin = centre of the underside; +X = connector end. Variants: VARIANT_closed, VARIANT_open.
Run: Blender -b --python build_oe_module_npo.py -- <out_dir>
"""
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402
import dies as D  # noqa: E402
import oe as O  # noqa: E402

C = P.C
ASSET = "oe_module_npo"
MM, UM = P.MM, P.UM

B_X, B_Y, B_T = 44.0 * MM, 30.0 * MM, 1.2 * MM
PIC_C = (-8.0 * MM, 0.0)
LID_X0, LID_X1, LID_Y = -15.0 * MM, 4.0 * MM, 8.0 * MM
LID_Z0, LID_Z1 = 2.5 * MM, 3.3 * MM
WALL = 0.6 * MM
CONN_X0, CONN_X1 = 14.0 * MM, 26.0 * MM


def lid_geo(top):
    g = P.Geo()
    g.box(LID_X0, -LID_Y, LID_Z0, LID_X1, LID_Y, LID_Z1)
    g.box(LID_X0, -LID_Y, top, LID_X0 + WALL, LID_Y, LID_Z0)
    g.box(LID_X0 + WALL, -LID_Y, top, LID_X1, -LID_Y + WALL, LID_Z0)
    g.box(LID_X0 + WALL, LID_Y - WALL, top, LID_X1, LID_Y, LID_Z0)
    n = 3.8 * MM
    g.box(LID_X1 - WALL, -LID_Y + WALL, top, LID_X1, -n, LID_Z0)
    g.box(LID_X1 - WALL, n, top, LID_X1, LID_Y - WALL, LID_Z0)
    return g


def connector(coll, root, name, ztop):
    """Board-edge MT-16 plug housing; ferrule mating face at x = CONN_X1; returns ferrule centre z and rear x."""
    zc = 3.0 * MM
    outer_h, outer_w = 7.0 * MM, 14.0 * MM
    inner_h, inner_w = 3.4 * MM, 7.4 * MM
    x0, x1 = CONN_X0, CONN_X1
    zb = zc - outer_h / 2
    g = P.Geo()
    g.box(x0, -outer_w / 2, zb, x1, outer_w / 2, zc - inner_h / 2)  # floor
    g.box(x0, -outer_w / 2, zc + inner_h / 2, x1, outer_w / 2, zb + outer_h)  # roof
    g.box(x0, -outer_w / 2, zc - inner_h / 2, x1, -inner_w / 2, zc + inner_h / 2)
    g.box(x0, inner_w / 2, zc - inner_h / 2, x1, outer_w / 2, zc + inner_h / 2)
    g.build(name + "_connector_housing", P.mat("plastic_aqua"), coll, root, bevel=(0.3 * MM, 2))
    # latch tab on the roof
    P.Geo().box(x0 + 2.0 * MM, -2.0 * MM, zb + outer_h, x1 - 1.0 * MM, 2.0 * MM, zb + outer_h + 0.8 * MM).build(name + "_connector_latch", P.mat("plastic_aqua"), coll, root, bevel=(0.15 * MM, 2))
    # mounting block under the housing
    P.Geo().box(x0, -outer_w / 2, ztop, x0 + 6.0 * MM, outer_w / 2, zb).build(name + "_connector_base", P.mat("anodized"), coll, root)
    O_mt = P.build_mt_ferrule(coll, root, name + "_mt", 16, 250 * UM, loc=(x1 - 1.0 * MM, 0.0, zc), with_pins=True, pin_len=1.5 * MM)
    return zc, x0


def build_variant(coll, root, name, with_lid, detail):
    car = O.carrier_substrate(coll, root, name, B_X, B_Y, B_T, 0.0, lga_pitch=1.0 * MM)
    ztop = car["top"]
    # board surface: HDI via-in-pad dots and decoupling
    pts = [(x * MM, y * MM) for x in (-18.5, -17.0) for y in (-6, -4.5, -3, 3, 4.5, 6)] + [(x * MM, y * MM) for x in (6.0, 8.0, 10.0) for y in (-6.0, 6.0)]
    O.mlcc_row(coll, root, name, pts, ztop)
    eng = O.build_engine(coll, root, name, ztop, pic_center=PIC_C, detail=detail, eic_bond="hybrid")
    zc, xrear = connector(coll, root, name, ztop)
    start = eng["fau_exit"]
    pg = O.build_pigtail(coll, root, name, start, 16, [(start[0] + 6 * MM, 0.0, start[2] + 0.15 * MM), (start[0] + 10 * MM, 0.0, start[2] + 0.8 * MM), (xrear - 1.0 * MM + 8.0 * MM - 8.0 * MM, 0.0, zc)], mt=False)
    if with_lid:
        P.Geo().box(0, 0, 0, 0, 0, 0)  # placeholder keeps API symmetry
        lid = lid_geo(ztop).build(name + "_lid_body", P.mat("oe_body"), coll, root, bevel=(0.15 * MM, 2))
        t = C.text_mesh(name + "_marking", "NARROWCOM NPO-16", 1.2 * MM, loc=((LID_X0 + LID_X1) / 2, 0.0, LID_Z1 + 0.005 * MM), extrude=0.01 * MM, mat=P.mat("etch"))
        C.add(t, coll, root)
        P.Geo().box(PIC_C[0] + D.EIC_SITE[0] - D.EIC_X / 2 - 2.0 * MM, -D.EIC_Y / 2 - 0.5 * MM, eng["eic_back"], PIC_C[0] + D.EIC_SITE[0] + D.EIC_X / 2 + 2.0 * MM,
                    D.EIC_Y / 2 + 0.5 * MM, LID_Z0).build(name + "_thermal_pad", P.mat("heatsink_grey"), coll, root)
    return car, eng, zc


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="C")
    vc = C.sub_collection(coll, "VARIANT_closed")
    vo = C.sub_collection(coll, "VARIANT_open")
    car, eng, zc = build_variant(vc, root, ASSET, True, "low")
    car2, eng2, _ = build_variant(vo, root, ASSET + "_open", False, "high")
    for o in vo.all_objects:
        if o.parent == root:
            o.location.y += 0.07
    C.hook("lid_top", vc, root, loc=((LID_X0 + LID_X1) / 2, 0.0, LID_Z1))
    C.hook("connector_face", vc, root, loc=(CONN_X1, 0.0, zc))
    C.hook("fiber_exit", vc, root, loc=(eng["fau_exit"][0], 0.0, eng["fau_exit"][2]))
    C.hook("underside_center", vc, root, loc=(0, 0, 0))
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("HDI module board", "44 x 30 x 1.2", "mm", "plausible size; no public NPO engine dimensions found (Cheng 2025 shows NPO as module on HDI interposer next to the ASIC package)", "C", "range 30-60 x 20-40 mm"),
        P.dim("LGA array", "43 x 28 pads, 0.6 mm dia, 1.0 mm pitch", "mm", "estimate", "C"),
        P.dim("engine", "same PIC / EIC / 16-fibre FAU as oe_module_cpo", "-", "see oe_module_cpo", "C"),
        P.dim("lid", "19 x 16 x 0.8 plate, top at 3.3 mm", "mm", "estimate", "C"),
        P.dim("optical connector", "MT-16 ferrule 6.4 x 2.5 mm in a 14 x 7 x 12 mm plug housing, face at x = 26 mm", "mm", "ferrule IEC 61754-5 (A); housing generic (C)", "B"),
        P.dim("ribbon on board", "16 x 250 um coated fibres, ~25 mm", "mm", "estimate", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "NPO: engine near the ASIC on its own HDI board, optical connector at the board edge (fibre exits via the connector, not a pigtail). Open variant (no lid, full-detail PIC) offset +70 mm in y.",
        [dict(src="blog 2025/20251219_CPO/pluggable-npo-cpo-comparison.webp (Cheng et al., Opt. Express 33, 24190, 2025, Fig. 2)", accessed="2026-10-02", used_for="NPO topology: engine on HDI interposer, fibre to connector"),
         dict(src="20260320_OFC W1D.7 (16-channel NPO engine)", accessed="2026-10-02", used_for="NPO engine channel count context"),
         dict(src="blog ranovus-optical-engines-on-asic.webp; Intel OCI photo", accessed="2026-10-02", used_for="engine anatomy")],
        dims, ["HOOK_lid_top", "HOOK_connector_face", "HOOK_fiber_exit", "HOOK_underside_center"], P.used_material_names(), [],
        "S3 (NPO module next to the ASIC, vendors argue over connector/LGA/BGA), S4 contrast.",
        ["Connector housing generic", "Board routing/vias not modelled", "No ASIC package/socket"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["triangles_evaluated_incl_instances"] = P.eval_tris(coll)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    prev = os.path.join(out_dir, "previews")
    shots = [
        dict(name="front", target=(0.0, 0, 0.002), dist=0.11, az=0, el=10),
        dict(name="three_quarter", target=(0.0, 0, 0.002), dist=0.1, az=35, el=30),
        dict(name="top", target=(0.0, 0, 0.0), dist=0.1, az=0, el=89.5),
        dict(name="closeup_connector", target=(0.022, 0, 0.003), dist=0.03, az=62, el=20),
        dict(name="open_closeup_engine", target=(-0.006, 0.07, 0.002), dist=0.03, az=30, el=40),
        dict(name="underside_lga", target=(0, 0, 0.0), dist=0.07, az=0, el=-80),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots[:5], floor=-0.0001)
    outs += P.render_shots([coll], prev, ASSET, shots[5:], floor=None)
    print(outs)


if __name__ == "__main__":
    main()
