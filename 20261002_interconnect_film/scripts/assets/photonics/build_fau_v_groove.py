"""fau_v_groove: glass V-groove fiber array units (8/12/16 fibers; 250 um and 127 um pitch), 12-fiber MT ferrule with guide
pins, and the fibre-to-chip edge-coupling interface (x200 scaled, plain and sectioned).
Frame of a FAU: chip-facing end face at x = 0, block extends to -X, fibres exit to -X (callers rotate). z=0 = underside of glass base.
Run: Blender -b --python build_fau_v_groove.py -- <out_dir>
"""
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
ASSET = "fau_v_groove"
MM, UM = P.MM, P.UM


def edge_interface(coll, root, name, scale):
    """Single-channel fibre/V-groove/PIC-edge interface at real size under a scaled empty (scale x)."""
    sroot = bpy.data.objects.new(name + "_scale_root", None)
    sroot.empty_display_type = "ARROWS"
    sroot.empty_display_size = 0.01
    coll.objects.link(sroot)
    sroot.parent = root
    sroot.scale = (scale, scale, scale)
    L = 0.35 * MM
    fau = P.build_fau(coll, sroot, name + "_fau", 1, 250 * UM, base_w=0.30 * MM, length=L, base_t=0.30 * MM, lid_t=0.20 * MM,
                      fiber_len_out=0.0, with_coating=False, epoxy=False)
    zc = fau["z_axis"]
    # index-matching epoxy gap and PIC edge
    gap = 10 * UM
    P.Geo().box(0.0, -0.15 * MM, zc - 62.5 * UM, gap, 0.15 * MM, zc + 62.5 * UM).build(name + "_index_match_epoxy", P.mat("epoxy"), coll, sroot)
    xe = gap
    sub_t = 0.25 * MM
    box_t, wg_t = 2.0 * UM, 0.22 * UM
    z_box_top = zc - wg_t / 2
    z_box_bot = z_box_top - box_t
    trench = 150 * UM
    # silicon substrate with trench (recess) at the edge
    P.Geo().box(xe + trench, -0.15 * MM, z_box_bot - sub_t, xe + 0.6 * MM, 0.15 * MM, z_box_bot).build(name + "_pic_substrate_si", P.mat("silicon"), coll, sroot)
    P.Geo().box(xe, -0.15 * MM, z_box_bot, xe + 0.6 * MM, 0.15 * MM, z_box_top).build(name + "_pic_box_oxide", P.mat("box_oxide"), coll, sroot)
    P.Geo().box(xe, -0.15 * MM, z_box_top, xe + 0.6 * MM, 0.15 * MM, zc + 10 * UM).build(name + "_pic_cladding", P.mat("glass_edge"), coll, sroot)
    # inverse-taper waveguide (tip 0.18 um -> 0.5 um over 200 um), 220 nm thick
    g = P.Geo()
    n = 20
    for k in range(n):
        xa, xb = 200 * UM * k / n, 200 * UM * (k + 1) / n
        w = 0.18 * UM + (0.5 * UM - 0.18 * UM) * (k + 0.5) / n
        g.box(xe + xa, -w / 2, z_box_top, xe + xb, w / 2, z_box_top + wg_t)
    g.box(xe + 200 * UM, -0.25 * UM, z_box_top, xe + 0.6 * MM, 0.25 * UM, z_box_top + wg_t)
    g.build(name + "_pic_edge_coupler_taper", P.mat("waveguide_si"), coll, sroot)
    return sroot, fau


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="B")
    specs = [(8, 250), (12, 250), (16, 250), (8, 127), (12, 127), (16, 127)]
    ypos = {}
    for k, (n, p) in enumerate(specs):
        vc = C.sub_collection(coll, "VARIANT_%df_%d" % (n, p))
        y = k * 8 * MM
        tag = "%df_%d" % (n, p)
        info = P.build_fau(vc, root, "fau_%s" % tag, n, p * UM, loc=(0, y, 0), length=5 * MM, fiber_len_out=2.5 * MM)
        C.hook("chip_face_%s" % tag, vc, root, loc=(0, y, info["z_axis"]))
        C.hook("fiber_exit_%s" % tag, vc, root, loc=(-5 * MM - 1.0 * MM - 2.5 * MM, y, info["z_axis"]))
        ypos[tag] = y
    vm = C.sub_collection(coll, "VARIANT_mt_ferrule_12f")
    mt = P.build_mt_ferrule(vm, root, "mt12", 12, loc=(0, -12 * MM, 0), with_pins=True)
    # short ribbon at the rear
    path = P.catmull([(-8 * MM, -12 * MM, 0), (-14 * MM, -12 * MM, 0), (-24 * MM, -12 * MM, 0)], 4)
    P.ribbon_fibers(path, 12, 250 * UM, 125 * UM, vm, root, "mt12_ribbon")
    C.hook("mt_mating_face", vm, root, loc=(0, -12 * MM, 0))
    C.hook("mt_fiber_exit", vm, root, loc=(-24 * MM, -12 * MM, 0))
    SC = 200.0
    ve = C.sub_collection(coll, "VARIANT_edge_coupling_interface_x200")
    vs = C.sub_collection(coll, "VARIANT_edge_coupling_section_x200")
    r1, f1 = edge_interface(ve, root, "ecif", SC)
    r1.location = (0.15, 0.10, 0.0)
    r2, f2 = edge_interface(vs, root, "ecsec", SC)
    r2.location = (0.15, 0.30, 0.0)
    P.refresh()
    P.cut_collection(vs, 1, 0.0, keep_positive=True)
    C.hook("ecif_interface_center", ve, root, loc=(0.15, 0.10, f1["z_axis"] * SC))
    root["p_note"] = "variants laid out along +y (6 FAUs 8 mm apart), MT ferrule at y=-12 mm, x200 interface variants at x=0.15 m"
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("fibre cladding diameter", 125, "um", "SMF-28 / G.652 standard", "A"),
        P.dim("fibre pitch variants", "250 and 127", "um", "brief; Ranovus Odin 8P V-groove pitch 250 um (HC34 2022)", "A"),
        P.dim("V-groove", "included angle 70.5 deg (KOH <100> Si style), apex depth 90 um -> fibre axis 18.25 um above the top surface", "um", "geometric consequence of 125 um fibre; groove depth chosen", "B"),
        P.dim("glass base / lid / length", "0.85 / 0.5 / 5.0", "mm", "typical FAU class; estimates (vendor FAUs vary 0.5-1.5 mm thick)", "C"),
        P.dim("block width", "fibre span + 2.0 mm (8f250: 3.75, 12f250: 4.75, 16f250: 5.75; 127 um: 2.9, 3.4, 3.9)", "mm", "estimate", "C"),
        P.dim("MT ferrule body", "6.4 x 2.5 x 8.0", "mm", "IEC 61754-5 / US Conec (width x height; length 8.0 mm typical)", "A"),
        P.dim("MT guide pins", "0.7 dia, 4.6 pitch, 6 mm protrusion modelled", "mm", "IEC 61754-5 (0.699 mm option)", "A"),
        P.dim("MT fibre rows", "12 x 250 um pitch", "um", "MT-12", "A"),
        P.dim("edge-coupling gap", 10, "um", "index-matching epoxy gap, illustrative (real 5-20 um)", "C"),
        P.dim("edge coupler inverse taper", "tip 0.18 um -> 0.5 um over 200 um, 220 nm thick", "um", "generic Si inverse taper; AIM cross-section shows low-loss edge coupler in trench", "C"),
        P.dim("edge trench", "Si recessed 150 um behind the oxide membrane", "um", "AIM Photonics process cross-section figure (edge coupler cavity); depth illustrative", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "Family: 6 FAU variants (VARIANT_<n>f_<pitch>), MT ferrule, edge-coupling interface (plain and y>=0 section, both scaled x200 via parent empty). Hide variants you do not need.",
        [dict(src="US Conec / IEC 61754-5 (search results)", accessed="2026-10-02", used_for="MT ferrule dims"),
         dict(src="HC34 Ranovus 2022", accessed="2026-10-02", used_for="250 um V-groove pitch, 16-fibre FVGA"),
         dict(src="blog intel-oci-eic-pic-fiber.webp", accessed="2026-10-02", used_for="FAU as glass V-groove block with fibre ribbon"),
         dict(src="blog aim-photonics-process-cross-section.webp", accessed="2026-10-02", used_for="edge coupler cavity")],
        dims, [h.name for h in coll.all_objects if h.name.startswith("HOOK_")], P.used_material_names(), [],
        "S3/S4/S5 (OE module fibre attach, FAU attach station), edge-coupling explainer shots.",
        ["Fibre end faces cleaved flat (no 8 deg polish)", "127 um variants: bare fibre stubs instead of fan-out", "Epoxy bead is a rounded box", "Fibre core drawn as emissive cylinder"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    prev = os.path.join(out_dir, "previews")
    y16 = ypos["16f_250"]
    y16b = ypos["16f_127"]
    shots = [
        dict(name="overview_top", target=(-0.004, 0.014, 0), dist=0.085, az=0, el=89.5),
        dict(name="overview_three_quarter", target=(-0.004, 0.020, 0.001), dist=0.065, az=35, el=35),
        dict(name="front", target=(-0.004, 0.02, 0.001), dist=0.07, az=0, el=8),
        dict(name="closeup_16f_250_end", target=(0, y16, 0.001), dist=0.012, az=-55, el=25),
        dict(name="closeup_16f_127_grooves", target=(-0.001, y16b, 0.0009), dist=0.0045, az=-65, el=22),
        dict(name="closeup_mt_ferrule", target=(0.0, -12 * MM, 0), dist=0.022, az=-40, el=25),
        dict(name="closeup_mt_end_face", target=(0.0, -12 * MM, 0), dist=0.0095, az=72, el=8),
        dict(name="edge_coupling_interface", target=(0.16, 0.10, 0.063), dist=0.22, az=-28, el=22),
        dict(name="edge_coupling_section", target=(0.16, 0.30, 0.063), dist=0.22, az=0, el=6),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots, floor=-0.002)
    print(outs)


if __name__ == "__main__":
    main()
