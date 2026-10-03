"""pic_eic_stack: EIC hybrid-bonded face-to-face onto a thinned PIC with TSVs, PIC on C4 bumps on an organic substrate patch
(Marvell/TSMC COUPE and NVIDIA ISSCC 2026 style). Variants: assembled, exploded (EIC lifted and flipped to show both pad faces),
cutaway (y>=0 half, TSV row shown), detail inset (x500, real proportions of the interface region).
Run: Blender -b --python build_pic_eic_stack.py -- <out_dir>
"""
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402
import dies as D  # noqa: E402

C = P.C
ASSET = "pic_eic_stack"
MM, UM, PI = P.MM, P.UM, math.pi

SUB_X, SUB_Y, SUB_T = 12.0 * MM, 14.0 * MM, 0.8 * MM
PIC_T_THIN = 0.10 * MM
C4_PITCH = 130 * UM
C4_NX, C4_NY = 53, 67
C4_H = 70 * UM
BOND_GAP = 1 * UM


def c4_points(z):
    return [((i - (C4_NX - 1) / 2) * C4_PITCH, (j - (C4_NY - 1) / 2) * C4_PITCH, z) for i in range(C4_NX) for j in range(C4_NY)]


def build_stack(coll, root, name, dz_c4=0.0, dz_pic=0.0, dz_eic=0.0, eic_flip=False, realize_bumps=False, ymin=None, tsv=False):
    sub_top = SUB_T
    g = P.Geo().box(-SUB_X / 2, -SUB_Y / 2, 0, SUB_X / 2, SUB_Y / 2, sub_top * 0.97)
    g.build(name + "_organic_substrate_core", P.mat("substrate_edge"), coll, root, bevel=(0.06 * MM, 2))
    P.Geo().box(-SUB_X / 2 + 0.02 * MM, -SUB_Y / 2 + 0.02 * MM, sub_top * 0.97, SUB_X / 2 - 0.02 * MM, SUB_Y / 2 - 0.02 * MM, sub_top).build(
        name + "_solder_mask", P.mat("organic_substrate"), coll, root)
    zc4 = sub_top + dz_c4
    pts = c4_points(zc4)
    if realize_bumps:
        g = P.Geo()
        gp = P.Geo()
        for (x, y, z) in pts:
            if ymin is not None and y < ymin - 1e-9:
                continue
            g.sphere(x, y, z + 0.035 * MM, 40 * UM, nseg=10, nring=5, sz=0.9)
            gp.cyl(x, y, z, z + 0.012 * MM, 48 * UM, n=10)
        g.build(name + "_c4_bumps", P.mat("solder"), coll, root, smooth=True)
        gp.build(name + "_ubm_pads", P.mat("copper"), coll, root)
    else:
        proto = P.Geo().sphere(0, 0, 0.035 * MM, 40 * UM, nseg=8, nring=4, sz=0.9).build(name + "_c4_proto", P.mat("solder"), coll, root, loc=(0, 0, zc4), smooth=True)
        P.instancer(name + "_c4_bumps", proto, pts, coll, root)
        pp = P.Geo().cyl(0, 0, 0, 0.012 * MM, 48 * UM, n=8).build(name + "_ubm_proto", P.mat("copper"), coll, root, loc=(0, 0, zc4))
        P.instancer(name + "_ubm_pads", pp, pts, coll, root)
    z_pic0 = sub_top + C4_H + dz_pic
    pic = D.build_pic(coll, root, name + "_pic", loc=(0, 0, z_pic0), detail="low", with_bumps=True, bond_pads="hybrid", thickness=PIC_T_THIN)
    pic_top = z_pic0 + PIC_T_THIN
    if tsv:
        gt = P.Geo()
        for i in range(C4_NX):
            x = (i - (C4_NX - 1) / 2) * C4_PITCH
            gt.cyl(x, 0.0, z_pic0, pic_top - D.BEOL_T, 15 * UM, n=10)
        gt.build(name + "_tsv_row_y0", P.mat("copper"), coll, root)
    eic_root_z = pic_top + BOND_GAP + D.EIC_T + dz_eic
    e = D.build_eic(coll, root, name + "_eic", loc=(D.EIC_SITE[0], D.EIC_SITE[1], eic_root_z), bond="hybrid", detail="low")
    if eic_flip is False:
        e["root"].rotation_euler = (PI, 0, 0)
    else:
        e["root"].location = (D.EIC_SITE[0], D.EIC_SITE[1] + 0.0, pic_top + BOND_GAP + dz_eic)  # face up (pads visible)
    return dict(pic_top=pic_top, eic_back=pic_top + BOND_GAP + D.EIC_T, pic=pic, eic=e)


def inset(coll, root, name, scale, loc):
    """Interface cross-section at real proportions under a scaled empty. Layers (um, z up): substrate, C4, UBM, thinned PIC with TSV,
    BOX + device + BEOL, hybrid bond, EIC BEOL, EIC Si. Window 120 um wide in x and 60 um in y; section at y=0."""
    sr = bpy.data.objects.new(name + "_scale_root", None)
    sr.empty_display_type = "ARROWS"
    sr.empty_display_size = 0.01
    coll.objects.link(sr)
    sr.parent = root
    sr.scale = (scale, scale, scale)
    sr.location = loc
    W, Dp = 120 * UM, 60 * UM
    hx, hy = W / 2, Dp / 2

    def box(nm, z0, z1, matname, bevel=None):
        P.Geo().box(-hx, -hy, z0 * UM, hx, hy, z1 * UM).build(name + "_" + nm, P.mat(matname), coll, sr, bevel=bevel)

    box("substrate", -30, 0, "organic_substrate")
    P.Geo().cyl(0, 0, 0, 5 * UM, 48 * UM, n=24).build(name + "_ubm_pad", P.mat("copper"), coll, sr)
    P.Geo().sphere(0, 0, 40 * UM, 40 * UM, nseg=24, nring=12, sz=0.9).build(name + "_c4_bump", P.mat("solder"), coll, sr, smooth=True)
    z_si0, z_si1 = 70.0, 170.0
    box("pic_si_thinned", z_si0, z_si1, "silicon")
    g = P.Geo()
    g.cyl(0, 0, z_si0 * UM, z_si1 * UM, 5 * UM, n=16)  # TSV 10 um dia
    g.build(name + "_tsv", P.mat("copper"), coll, sr)
    box("pic_box", z_si1, z_si1 + 2.0, "box_oxide")
    box("pic_beol_oxide", z_si1 + 2.0, z_si1 + 10.0, "cladding")
    P.Geo().box(-hx, -0.25 * UM, (z_si1 + 2.0) * UM, hx, 0.25 * UM, (z_si1 + 2.22) * UM).build(name + "_pic_waveguide", P.mat("waveguide_si"), coll, sr)
    P.Geo().box(-hx, -2 * UM, (z_si1 + 5.0) * UM, hx, 2 * UM, (z_si1 + 5.5) * UM).build(name + "_pic_metal1", P.mat("metal1"), coll, sr)
    z_b = z_si1 + 10.0
    gp1, gp2 = P.Geo(), P.Geo()
    for i in range(-6, 7):
        for j in range(-3, 4):
            gp1.cyl(i * 9 * UM, j * 9 * UM, (z_b - 1.0) * UM, z_b * UM, 2.25 * UM, n=12)
            gp2.cyl(i * 9 * UM, j * 9 * UM, z_b * UM, (z_b + 1.0) * UM, 2.25 * UM, n=12)
    gp1.build(name + "_pic_bond_pads", P.mat("copper"), coll, sr)
    gp2.build(name + "_eic_bond_pads", P.mat("copper"), coll, sr)
    box("hybrid_bond_oxide_interface", z_b - 0.0, z_b + 0.0001, "glass_edge")
    box("eic_beol_oxide", z_b + 0.0, z_b + 7.0, "cladding")
    P.Geo().box(-hx, -2 * UM, (z_b + 3.0) * UM, hx, 2 * UM, (z_b + 3.5) * UM).build(name + "_eic_metal_top", P.mat("metal2"), coll, sr)
    box("eic_si", z_b + 7.0, z_b + 60.0, "silicon")
    return sr


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="B")
    va = C.sub_collection(coll, "VARIANT_assembled")
    vx = C.sub_collection(coll, "VARIANT_exploded")
    vc = C.sub_collection(coll, "VARIANT_cutaway")
    vi = C.sub_collection(coll, "VARIANT_detail_inset_x500")
    s1 = build_stack(va, root, ASSET)
    # exploded variant displaced in +y
    s2 = build_stack(vx, root, ASSET + "_exploded", dz_c4=1.5 * MM, dz_pic=3.5 * MM, dz_eic=7.0 * MM, eic_flip=True)
    for o in vx.all_objects:
        if o.parent == root:
            o.location.y += 0.03
            o.location.z += 0.0
    s3 = build_stack(vc, root, ASSET + "_cutaway", realize_bumps=True, ymin=0.0, tsv=True)
    for o in vc.all_objects:
        if o.parent == root:
            o.location.y -= 0.03
    P.refresh()
    P.cut_collection(vc, 1, -0.03, keep_positive=True)
    sr = inset(vi, root, ASSET + "_inset", 500.0, (0.06, 0.0, 0.0))
    P.refresh()
    P.cut_collection(vi, 1, 0.0, keep_positive=True)
    top = s1["eic_back"]
    C.hook("eic_back_center", va, root, loc=(D.EIC_SITE[0], D.EIC_SITE[1], top))
    C.hook("bond_interface_center", va, root, loc=(D.EIC_SITE[0], D.EIC_SITE[1], s1["pic_top"] + BOND_GAP / 2))
    C.hook("substrate_underside_center", va, root, loc=(0, 0, 0))
    C.hook("fiber_grating_coupler_site", va, root, loc=(D.PIC_X / 2 - 0.9 * MM, 0.0, s1["pic_top"]))
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("EIC / PIC", "2.4 x 4.2 x 0.30 / 7.0 x 9.0 x 0.10 (thinned, TSV)", "mm", "pic_die/eic_die sizes; thinned PIC thickness estimated (COUPE shows TSV through PIC)", "C"),
        P.dim("EIC-PIC bond", "Cu-Cu hybrid bond, face-to-face, 9 um pitch (texture), 1 um gap shown", "um", "TSMC SoIC-X pitch roadmap 9 um (2023) / 6 um (2025); Marvell COUPE slide: EIC-PIC hybrid bonding; NVIDIA OFC M4B.2 Cu-Cu hybrid bond", "B"),
        P.dim("TSV through PIC", "10 um dia (inset), drawn 30 um dia in the mm-scale cutaway", "um", "Marvell COUPE slide (TSV through PIC); diameter estimate", "C"),
        P.dim("C4 bumps to organic substrate", "80 um dia, 70 um collapsed height, 130 um pitch, 53 x 67", "um", "Marvell COUPE slide (C4 to organic substrate); pitch/diameter estimated (typical C4 100-180 um)", "C"),
        P.dim("organic substrate patch", "12 x 14 x 0.8", "mm", "estimate", "C"),
        P.dim("interface inset", "x500, window 120 x 60 um: substrate, C4, UBM, thinned Si + 10 um TSV, BOX 2 um, device layer 0.22 um, BEOL oxide, hybrid pads 4.5 um at 9 um, EIC BEOL 7 um", "um", "layer ordering from Marvell slide / AIM process figure; thicknesses typical", "C"),
        P.dim("vertical coupling", "not modelled; HOOK_fiber_grating_coupler_site marks the PIC side", "-", "COUPE: vertical grating couplers + Si lens; NVIDIA: GC 1.3 dB, 20 nm bandwidth (M4B.2)", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "Four variants laid out side by side: assembled (origin), exploded (+30 mm y), cutaway (-30 mm y, section plane y=-30 mm world, i.e. through the centre row of C4 bumps/TSVs), detail inset x500 at x=0.06 m (scale root empty carries x500; mesh transforms identity).",
        [dict(src="blog 2026/OFC2026/IMG_4055.JPG (Marvell COUPE cross-section)", accessed="2026-10-02", used_for="layer order: Support Si, EIC, hybrid bonding, PIC, TSV, C4, organic substrate"),
         dict(src="20260320_OFC M4B.2 (NVIDIA)", accessed="2026-10-02", used_for="7 nm EIC on 65 nm PIC, Cu-Cu hybrid bonding, FAU, organic substrate"),
         dict(src="TSMC SoIC-X pitch roadmap (Tom's Hardware / TrendForce)", accessed="2026-10-02", used_for="9 um hybrid-bond pitch")],
        dims, [h.name for h in coll.all_objects if h.name.startswith("HOOK_")], P.used_material_names(), [],
        "S5 (EIC/PIC bonding step and engine anatomy), explainer shots of the 3D-stack.",
        ["Hybrid-bond pads in the mm-scale variants are a texture", "TSV row only in the cutaway", "No FAU on top (vertical coupling not modelled)", "No underfill/lid"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["triangles_evaluated_incl_instances"] = P.eval_tris(coll)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    prev = os.path.join(out_dir, "previews")
    shots = [
        dict(name="front", target=(0, 0, 0.0012), dist=0.05, az=0, el=8),
        dict(name="three_quarter", target=(0, 0, 0.0012), dist=0.045, az=35, el=30),
        dict(name="top", target=(0, 0, 0.001), dist=0.020, az=0, el=89.5),
    ]
    outs = P.render_shots([va], prev, ASSET, shots, floor=-0.0001)
    outs += P.render_shots([vx], prev, ASSET, [dict(name="exploded_three_quarter", target=(0, 0.03, 0.0045), dist=0.055, az=30, el=22),
                                                dict(name="exploded_front", target=(0, 0.03, 0.0045), dist=0.03, az=0, el=5)], floor=-0.0001)
    outs += P.render_shots([vc], prev, ASSET, [dict(name="cutaway_three_quarter", target=(0, -0.03, 0.0008), dist=0.022, az=-25, el=28),
                                                dict(name="cutaway_closeup_layers", target=(0.0, -0.03, 0.0011), dist=0.0055, az=0, el=8)], floor=-0.0001)
    outs += P.render_shots([vi], prev, ASSET, [dict(name="inset_x500", target=(0.06, 0.0, 0.07), dist=0.30, az=-10, el=8),
                                                dict(name="inset_x500_bond_interface", target=(0.06, 0.0, 0.089), dist=0.04, az=-10, el=8)], floor=-0.03)
    print(outs)


if __name__ == "__main__":
    main()
