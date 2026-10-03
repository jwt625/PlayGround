"""eic_die: electronic IC die 2.4 x 4.2 x 0.3 mm (driver/TIA/SerDes macro floorplan), active face up (+z), 60 x 105 Cu-pillar
solder-cap microbumps at 40 um pitch matching pic_die; second variant with 9 um hybrid-bond Cu pad face (procedural).
Run: Blender -b --python build_eic_die.py -- <out_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402
import dies as D  # noqa: E402

C = P.C
ASSET = "eic_die"
MM = P.MM


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="C")
    v1 = C.sub_collection(coll, "VARIANT_microbump")
    v2 = C.sub_collection(coll, "VARIANT_hybrid_bond")
    a = D.build_eic(v1, root, ASSET + "_ub", loc=(0, 0, 0), bond="microbump")
    b = D.build_eic(v2, root, ASSET + "_hb", loc=(3.6 * MM, 0, 0), bond="hybrid")
    C.hook("bond_face_center", v1, root, loc=(0, 0, D.EIC_T))
    C.hook("bond_face_center_hybrid", v2, root, loc=(3.6 * MM, 0, D.EIC_T))
    C.hook("flip_note_rotate_180_about_x_to_bond_face_down", coll, root, loc=(0, 0, D.EIC_T / 2))
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("die size", "2.4 x 4.2", "mm", "estimate from Intel OCI photo (EIC about 2.1 x 4 mm if 16 fibers at 250 um); NVIDIA 1.33 Tb/s/mm^2 areal density consistent with an interface-limited footprint (OFC 2026 M4B.2)", "C"),
        P.dim("thickness", 0.30, "mm", "typical thinned EIC; estimate", "C"),
        P.dim("microbump pitch / array", "40 um, 60 x 105 = 6300", "um", "estimate; pad site identical to pic_die; microbump class 25-55 um", "C"),
        P.dim("bump geometry", "Cu pillar 20 um dia x 12 um + SnAg dome 5 um", "um", "generic microbump", "C"),
        P.dim("hybrid-bond pad pitch", 9, "um", "TSMC SoIC-X roadmap 9 um (2023) -> 6 um (2025) (Tom's Hardware / TrendForce)", "B"),
        P.dim("floorplan", "16 lanes of driver / TIA / SerDes slices, PLL, control", "-", "schematic, matches NVIDIA 8+1 lane link concept (OFC M4B.2); not a real layout", "C"),
        P.dim("seal ring", "20 um wide, 40 um inside the edge", "um", "estimate", "C"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "Active face up (+z). To bond face down onto pic_die, rotate the root children 180 deg about X; pad pattern is symmetric. VARIANT_hybrid_bond uses a procedural 9 um pad texture (MAT_photonics_hybrid_bond_face), VARIANT_microbump uses GN-instanced pillars and solder caps.",
        [dict(src="blog intel-oci-eic-pic-fiber.webp, intel-oci-chip-pencil.webp", accessed="2026-10-02", used_for="EIC/PIC size ratio"),
         dict(src="OFC 2026 M4B.2 (NVIDIA)", accessed="2026-10-02", used_for="7 nm EIC / 65 nm PIC, Cu-Cu hybrid bond, lane architecture"),
         dict(src="TSMC SoIC-X pitch roadmap (Tom's Hardware search result)", accessed="2026-10-02", used_for="9 um hybrid-bond pitch")],
        dims, ["HOOK_bond_face_center", "HOOK_bond_face_center_hybrid"], P.used_material_names(), [],
        "S5 (EIC/PIC bonding step), pic_eic_stack, OE modules.", ["Floorplan schematic", "Hybrid-bond pads are a texture, not geometry"],
    )
    P.refresh()
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["triangles_evaluated_incl_instances"] = P.eval_tris(coll)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    z = D.EIC_T
    prev = os.path.join(out_dir, "previews")
    shots = [
        dict(name="front", target=(1.8 * MM, 0, 0.15 * MM), dist=0.016, az=0, el=12),
        dict(name="three_quarter", target=(1.8 * MM, 0, 0.15 * MM), dist=0.016, az=35, el=35),
        dict(name="top", target=(1.8 * MM, 0, z), dist=0.0105, az=0, el=89.5),
        dict(name="closeup_microbumps", target=(-0.9 * MM, -1.8 * MM, z), dist=0.0004, az=10, el=40),
        dict(name="closeup_macros_corner", target=(-0.8 * MM, -1.7 * MM, z), dist=0.0018, az=0, el=65),
        dict(name="closeup_hybrid_pad_texture", target=(3.6 * MM, 0, z), dist=0.00012, az=0, el=70),
    ]
    outs = P.render_shots([coll], prev, ASSET, shots, floor=-0.0001)
    print(outs)


if __name__ == "__main__":
    main()
