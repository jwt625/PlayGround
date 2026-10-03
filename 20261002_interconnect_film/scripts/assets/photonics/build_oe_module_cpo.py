"""oe_module_cpo: co-packaged optical engine module (Ranovus Odin 8P / Intel OCI class): organic carrier with LGA underside,
PIC + flip-chip EIC, 16-fibre glass-V-groove FAU, heat-spreader lid with flat top (fried-egg platform), 16-fibre ribbon pigtail to MT-16.
Origin = centre of the underside (z=0 at LGA pad bottoms), +X = fibre exit, +Z up. Variants: VARIANT_closed, VARIANT_open (lid removed).
Run: Blender -b --python build_oe_module_cpo.py -- <out_dir>
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
ASSET = "oe_module_cpo"
MM, UM = P.MM, P.UM

CAR_X, CAR_Y, CAR_T = 24.0 * MM, 20.0 * MM, 1.0 * MM
LID_X0, LID_X1, LID_Y = -10.0 * MM, 8.5 * MM, 8.0 * MM
LID_Z0, LID_Z1 = 2.5 * MM, 3.3 * MM
WALL = 0.6 * MM


def lid_geo(carrier_top):
    g = P.Geo()
    g.box(LID_X0, -LID_Y, LID_Z0, LID_X1, LID_Y, LID_Z1)
    zb = carrier_top
    g.box(LID_X0, -LID_Y, zb, LID_X0 + WALL, LID_Y, LID_Z0)
    g.box(LID_X0 + WALL, -LID_Y, zb, LID_X1, -LID_Y + WALL, LID_Z0)
    g.box(LID_X0 + WALL, LID_Y - WALL, zb, LID_X1, LID_Y, LID_Z0)
    notch = 3.8 * MM
    g.box(LID_X1 - WALL, -LID_Y + WALL, zb, LID_X1, -notch, LID_Z0)
    g.box(LID_X1 - WALL, notch, zb, LID_X1, LID_Y - WALL, LID_Z0)
    return g


def build_variant(coll, root, name, with_lid, detail):
    car = O.carrier_substrate(coll, root, name, CAR_X, CAR_Y, CAR_T, 0.0)
    ztop = car["top"]
    # decoupling capacitors under the lid
    pts = [(-9.0 * MM, y * MM) for y in (-6, -4.5, -3, 3, 4.5, 6)] + [(-7.6 * MM, y * MM) for y in (-6, -4.5, 4.5, 6)] + [(5.0 * MM, y * MM) for y in (-7, 7)]
    O.mlcc_row(coll, root, name, pts, ztop)
    eng = O.build_engine(coll, root, name, ztop, detail=detail, eic_bond="hybrid")
    pg = O.build_pigtail(coll, root, name, eng["fau_exit"], 16,
                         [(eng["fau_exit"][0] + 8 * MM, 0.0, eng["fiber_axis_z"] - 0.1 * MM), (eng["fau_exit"][0] + 22 * MM, 3.0 * MM, 1.7 * MM),
                          (eng["fau_exit"][0] + 36 * MM, 3.0 * MM, 1.2 * MM), (eng["fau_exit"][0] + 44 * MM, 3.0 * MM, 1.2 * MM)])
    if with_lid:
        lg = lid_geo(ztop)
        lid = lg.build(name + "_lid_body", P.mat("oe_body"), coll, root, bevel=(0.15 * MM, 2))
        # thermal pedestal under the lid
        # engraved marking on the flat top
        t = C.text_mesh(name + "_marking", "NARROWCOM OE-16", 1.3 * MM, loc=((LID_X0 + LID_X1) / 2, 0.0, LID_Z1 + 0.005 * MM), extrude=0.01 * MM, mat=P.mat("etch"))
        C.add(t, coll, root)
        P.Geo().box(D.EIC_SITE[0] - D.EIC_X / 2 - 2.0 * MM, -D.EIC_Y / 2 - 0.5 * MM, eng["eic_back"], D.EIC_SITE[0] + D.EIC_X / 2 + 2.0 * MM,
                    D.EIC_Y / 2 + 0.5 * MM, LID_Z0).build(name + "_thermal_pad", P.mat("heatsink_grey"), coll, root, loc=(-2.0 * MM, 0, 0))
    return car, eng, pg


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="C")
    vc = C.sub_collection(coll, "VARIANT_closed")
    vo = C.sub_collection(coll, "VARIANT_open")
    car, eng, pg = build_variant(vc, root, ASSET, True, "low")
    # open variant offset in +y for side-by-side layout
    car2, eng2, pg2 = build_variant(vo, root, ASSET + "_open", False, "high")
    for o in vo.all_objects:
        if o.parent == root:
            o.location.y += 0.06
    C.hook("lid_top", vc, root, loc=((LID_X0 + LID_X1) / 2, 0.0, LID_Z1))
    C.hook("fiber_exit", vc, root, loc=(LID_X1, 0.0, eng["fiber_axis_z"]))
    C.hook("mt_face", vc, root, loc=pg["mt_face"])
    C.hook("underside_center", vc, root, loc=(0, 0, 0))
    root["p_lid_top_z_mm"] = LID_Z1 / MM
    root["p_open_variant_y_offset_mm"] = 60.0
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("carrier substrate", "24 x 20 x 1.0", "mm", "estimate from Ranovus/MediaTek photo (OE cover about 1/5 of the ASIC lid width); Intel OCI shows PIC+EIC on a PCB-like carrier. No public dimensions", "C", "range 18-30 x 15-24 mm"),
        P.dim("LGA pads", "22 x 18 (396), 0.6 mm dia, 1.0 mm pitch", "mm", "estimate", "C"),
        P.dim("PIC", "7.0 x 9.0 x 0.775", "mm", "see pic_die", "C"),
        P.dim("EIC", "2.4 x 4.2 x 0.30, flip-chip on PIC, 17 um stand-off", "mm", "see eic_die", "C"),
        P.dim("FAU", "16 fibres, 250 um pitch, glass base 0.8 + lid 0.5, 5.75 x 5.0 mm", "mm", "Ranovus Odin 8P 16-fibre FVGA at 250 um pitch (HC34 2022); sizes estimated", "B"),
        P.dim("lid body", "18.5 x 16 x 0.8 plate on 0.6 mm walls, notch 7.6 mm for fibres", "mm", "estimate from Ranovus photo (flat metal cover with fibre notch)", "C"),
        P.dim("overall height (lid top)", 3.3, "mm", "derived from stack: carrier 1.03 + PIC 0.775 + EIC 0.3 + pad + plate", "C", "range 2.5-4.5"),
        P.dim("pigtail", "16 x 250 um coated fibres, 44 mm modelled, MT-16 ferrule 6.4 x 2.5 x 8.0 mm", "mm", "IEC 61754-5 MT outline", "A (ferrule) / C (length)"),
        P.dim("flat top for fried egg", "18.5 x 16 mm at z = 3.3 mm, flatness not modelled", "mm", "storyboard S4", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "Closed module at the origin; open variant (no lid, full-detail PIC) offset +60 mm in y. The lid is one object with the single body material MAT_photonics_oe_body (duplicate the material per instance for heat colours). The marking text uses MAT_photonics_etch.",
        [dict(src="blog ranovus-optical-engines-on-asic.webp", accessed="2026-10-02", used_for="module arrangement, flat metal cover with fibre notch, fibres exiting to the side"),
         dict(src="blog intel-oci-eic-pic-fiber.webp; Intel Hot Chips 2024", accessed="2026-10-02", used_for="EIC on PIC with FAU in V-grooves"),
         dict(src="HC34 2022 Ranovus", accessed="2026-10-02", used_for="16-fibre FVGA, 250 um pitch")],
        dims, ["HOOK_lid_top", "HOOK_fiber_exit", "HOOK_mt_face", "HOOK_underside_center"], P.used_material_names(),
        ["p_lid_top_z_mm = 3.3", "p_open_variant_y_offset_mm = 60"],
        "S3 (NPO/CPO contrast), S4 (16 OEs around the XPU, eggs on lid_top, heating colours), S5 (die on engine).",
        ["Carrier internals (vias, routing) not modelled", "No thermal-interface fillets", "Pigtail is 44 mm long and floats at fixed height (no routing to a connector bulkhead)"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["triangles_evaluated_incl_instances"] = P.eval_tris(coll)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    prev = os.path.join(out_dir, "previews")
    cx = (LID_X0 + LID_X1) / 2
    shots = [
        dict(name="front", target=(0.012, 0, 0.0015), dist=0.12, az=0, el=10),
        dict(name="three_quarter", target=(0.012, 0, 0.0015), dist=0.12, az=35, el=30),
        dict(name="top", target=(0.012, 0.0, 0.0), dist=0.11, az=0, el=89.5),
        dict(name="closeup_lid", target=(cx, 0, 0.003), dist=0.045, az=20, el=35),
        dict(name="open_three_quarter", target=(0.0, 0.06, 0.0015), dist=0.05, az=30, el=40),
        dict(name="open_closeup_fau_pic_eic", target=(-0.002, 0.06, 0.002), dist=0.022, az=25, el=38),
        dict(name="closeup_mt_ferrule", target=(0.0 + pg["mt_face"][0], pg["mt_face"][1], pg["mt_face"][2]), dist=0.028, az=40, el=25),
        dict(name="underside_lga", target=(0, 0, 0.0), dist=0.04, az=0, el=-80),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots[:7], floor=-0.0001)
    outs += P.render_shots([coll], prev, ASSET, shots[7:], floor=None)
    print(outs)


if __name__ == "__main__":
    main()
