"""pic_die: complete silicon-photonics die 7 x 9 mm: 16 edge couplers + 4 laser-input couplers on the +X edge, 2 x 64 microring
modulators/drop rings with Ge PDs, metal fan-out, 60 x 105 microbump pad field (40 um) for the EIC, periphery bond pads, seal ring,
fiducials, grating-coupler test structures, probe pads.
Run: Blender -b --python build_pic_die.py -- <out_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402
import dies as D  # noqa: E402

C = P.C
ASSET = "pic_die"
MM = P.MM


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="C")
    info = D.build_pic(coll, root, ASSET, detail="high", with_bumps=True, bond_pads="microbump")
    top = info["top"]
    C.hook("edge_coupler_center", coll, root, loc=(D.PIC_X / 2, 0, top - 0.01 * MM))
    C.hook("laser_input_center", coll, root, loc=(D.PIC_X / 2, -2.875 * MM, top - 0.01 * MM))
    C.hook("eic_site_center", coll, root, loc=(D.EIC_SITE[0], D.EIC_SITE[1], top))
    C.hook("die_top_center", coll, root, loc=(0, 0, top))
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("die size (x depth, y along the edge-coupler edge)", "7.0 x 9.0", "mm", "coordinator brief (about 7 x 9 mm); cross-check: Intel OCI photo with 16 fibers at 250 um pitch gives PIC about 8 x 8 mm (+-1) incl. PIC strip", "C", "range 7-11 mm"),
        P.dim("die thickness", 0.775, "mm", "300 mm wafer standard thickness (unthinned)", "B"),
        P.dim("BEOL film thickness (shown)", 10, "um", "estimate", "C"),
        P.dim("edge couplers", "16 data (8 TX + 8 RX) + 4 laser, 250 um pitch", "-", "Intel OCI 8 fiber pairs; Ranovus Odin 8P 16-fiber FVGA at 250 um pitch (HC34 2022)", "B"),
        P.dim("microring radius (true), drawn rib width", "7.5 um, 1.0 um drawn (0.5 um true)", "um", "see microring_modulator_cell", "B"),
        P.dim("rings", "128 (8 TX + 8 RX channels x 8 rings, alternating sides, 40 um pitch)", "-", "Intel OCI 8 wavelengths per fiber with ring mux/DMUX (HC2024)", "B"),
        P.dim("waveguide drawn width", 2.0, "um", "true 0.5 um; widened 4x for macro legibility", "C"),
        P.dim("EIC site / bump field", "2.4 x 4.2 mm, 60 x 105 pads, pitch 40 um", "mm", "EIC size from OCI photo ratio (EIC about 13% of PIC area); pitch estimate (microbump class 25-55 um)", "C"),
        P.dim("periphery pads", "100 um squares, 150 um pitch", "um", "estimate", "C"),
        P.dim("seal ring", "25 um wide, 60 um inside the edge", "um", "estimate (true 5-10 um class)", "C"),
        P.dim("fiducials", "200 x 20 um crosses", "um", "OCI photo shows cross fiducials ~100-200 um", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "Die photo-style PIC. Origin: die centre in xy, bottom at z=0, edge couplers on +X edge, top surface at z=0.775 mm. Pad field instanced with Geometry Nodes (6300 copper pads).",
        [dict(src="jwt625.github.io/assets/images/2025/20251219_CPO/intel-oci-eic-pic-fiber.webp, intel-oci-chip-pencil.webp, intel-photonic-ic-layout.webp", accessed="2026-10-02", used_for="layout style, EIC/PIC ratio, pad frames"),
         dict(src="Intel Hot Chips 2024 OCI (hc2024.hotchips.org)", accessed="2026-10-02", used_for="channel counts, ring modulators, V-groove attach"),
         dict(src="Ranovus HC34 2022", accessed="2026-10-02", used_for="250 um pitch, 16-fiber FVGA")],
        dims, ["HOOK_edge_coupler_center", "HOOK_laser_input_center", "HOOK_eic_site_center", "HOOK_die_top_center"], P.used_material_names(), [],
        "S5 (die yanked back through the line), S4 context, wafer-level test shots; macro render of a die photo.",
        ["Waveguides 4x and ring ribs 2x wider than true", "Laser manifold/splitter tree schematic", "Metal fan-out representative, not a real netlist", "No edge-coupler trench depth modelling"],
    )
    P.refresh()
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["triangles_evaluated_incl_instances"] = P.eval_tris(coll)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    z = top
    prev = os.path.join(out_dir, "previews")
    shots = [
        dict(name="front", target=(0, 0, 0.0004), dist=0.020, az=0, el=12),
        dict(name="three_quarter", target=(0, 0, 0.0004), dist=0.020, az=35, el=35),
        dict(name="top", target=(0, 0, z), dist=0.0125, az=0, el=89.5),
        dict(name="closeup_edge_couplers", target=(3.3 * MM, 0.9 * MM, z), dist=0.0035, az=70, el=40),
        dict(name="closeup_ring_bank", target=(1.8 * MM, 1.0 * MM, z), dist=0.0016, az=0, el=70),
        dict(name="closeup_bump_pads", target=(-0.9 * MM, 1.6 * MM, z), dist=0.0009, az=10, el=50),
        dict(name="closeup_corner_seal_fiducial", target=(-3.0 * MM, -3.9 * MM, z), dist=0.0016, az=20, el=45),
        dict(name="closeup_gc_test", target=(-2.2 * MM, -3.2 * MM, z), dist=0.0015, az=0, el=60),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots, floor=-0.0001)
    print(outs)


if __name__ == "__main__":
    main()
