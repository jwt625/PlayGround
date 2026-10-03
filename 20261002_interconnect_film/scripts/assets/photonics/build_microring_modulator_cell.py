"""microring_modulator_cell: one silicon microring modulator at true proportions, with layer stack, heater, contacts,
vias, M1/M2 and pads. Variants: real size, scaled x10000 (1 um = 1 cm), and a y=0 cross-section cutaway (scaled).

Run: Blender -b --python build_microring_modulator_cell.py -- <out_dir>   (out_dir = assets/components/photonics)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
PI = math.pi
ASSET = "microring_modulator_cell"
SCALE_UP = 10000.0  # 1 um -> 1 cm

# ---- geometry in micrometres (sources in the metadata table)
R = 7.5  # ring centre-line radius (A: OFC 2026 Th2A.14 5 um; AMF 12 um; chosen mid-range 7.5)
W = 0.5  # rib / bus width
H = 0.22  # device layer
SLAB = 0.09  # slab thickness
GAP = 0.18  # bus-ring edge gap (A: AMF 5x200G MRM paper, PMC10831040)
BOX = 2.0
SUB = 10.0  # displayed substrate thickness (truncated; wafer is 775 um)
Z_HEAT0, Z_HEAT1 = 0.85, 0.95
Z_M1_0, Z_M1_1 = 1.4, 1.9
Z_M2_0, Z_M2_1 = 2.6, 3.3
Z_PAD0, Z_PAD1 = 3.9, 4.4
Z_CLAD = 3.9
PAD = 35.0
PAD_PITCH = 45.0
PAD_Y = -60.0
BUS_Y = R + W / 2 + GAP + W / 2
BUS_L = 65.0


def deg(a):
    return math.radians(a)


def ang_ranges(a0, a1):
    """Normalise an angular range (rad, a1 may exceed 2pi) to list of ranges inside [0, 2pi]."""
    out = []
    if a1 - a0 >= 2 * PI - 1e-9:
        return [(0.0, 2 * PI)]
    a0m = a0 % (2 * PI)
    span = a1 - a0
    if a0m + span <= 2 * PI + 1e-12:
        return [(a0m, a0m + span)]
    return [(a0m, 2 * PI), (0.0, a0m + span - 2 * PI)]


def clip_ranges(ranges, lo, hi):
    out = []
    for (a, b) in ranges:
        s, e = max(a, lo), min(b, hi)
        if e - s > 1e-9:
            out.append((s, e))
    return out


def build_cell(coll, root, U, tag, half=False):
    """Build all parts (coordinates in um * U). half=True keeps only y >= 0 angular halves for ring parts."""
    ob = {}

    def add(name, geo, mname, smooth=False, bevel=None, loc=(0, 0, 0)):
        geo.scale(U)
        o = geo.build("%s_%s" % (ASSET, name) + tag, P.mat(mname), coll, root, smooth=smooth, loc=loc, bevel=bevel)
        ob[name] = o
        return o

    def annuli(g, r0, r1, z0, z1, a0=0.0, a1=2 * PI, n=96):
        rng = ang_ranges(a0, a1)
        if half:
            rng = clip_ranges(rng, 0.0, PI)
        for (s, e) in rng:
            g.annulus(0, 0, r0, r1, z0, z1, n=n, a0=s, a1=e)
        return g

    ext_x, ext_y0, ext_y1 = 100.0, -85.0, 28.0
    # substrate, BOX
    add("substrate_si", P.Geo().box(-ext_x, ext_y0, -BOX - SUB, ext_x, ext_y1, -BOX), "silicon")
    add("box_oxide", P.Geo().box(-ext_x, ext_y0, -BOX, ext_x, ext_y1, 0.0), "box_oxide")
    # ---- device layer, implant-coloured
    sector = deg(65.0)  # slab removed within +-65 deg of the bus direction (+y) so the bus can pass
    oa0, oa1 = PI / 2 + sector, PI / 2 + 2 * PI - sector
    g = P.Geo()
    annuli(g, R - W / 2, R, 0, H)  # rib, inner (p) half
    annuli(g, R - 1.35, R - W / 2, 0, SLAB)  # p slab
    add("si_p", g, "p_implant")
    g = P.Geo()
    annuli(g, R, R + W / 2, 0, H)  # rib, outer (n) half
    annuli(g, R + W / 2, R + 1.6, 0, SLAB, oa0, oa1, n=96)  # n slab (away from the bus)
    add("si_n", g, "n_implant")
    g = P.Geo()
    annuli(g, 0, R - 1.35, 0, SLAB)  # p+ hub
    add("si_pplus", g, "pplus_implant")
    g = P.Geo()
    annuli(g, R + 1.6, R + 4.5, 0, SLAB, oa0, oa1, n=96)  # n+ contact slab
    add("si_nplus", g, "nplus_implant")
    # bus waveguide (undoped), passes the ring at the gap
    g = P.Geo().strip([(-BUS_L, BUS_Y), (BUS_L, BUS_Y)], W, 0, H)
    if half:
        g = P.Geo().box(-BUS_L, BUS_Y - W / 2, 0, BUS_L, BUS_Y + W / 2, H)
    add("bus_waveguide", g, "waveguide_si")
    # ---- contacts / vias (tungsten), M1, heater
    g = P.Geo()
    for k in range(10):
        a = 2 * PI * k / 10
        if half and math.sin(a) < -1e-9:
            continue
        g.cyl(3.5 * math.cos(a), 3.5 * math.sin(a), SLAB, Z_M1_0, 0.2, n=8)
    for k in range(16):
        a = deg(195 + 150 * k / 15)
        if half and math.sin(a) < 0:
            continue
        g.cyl(10.4 * math.cos(a), 10.4 * math.sin(a), SLAB, Z_M1_0, 0.2, n=8)
    add("vias_contact", g, "tungsten_via")
    g = P.Geo()
    for sx in (-1, 1):  # heater end vias
        x, y = sx * R * math.cos(deg(50)), R * math.sin(deg(50))
        g.cyl(x, y, Z_HEAT1, Z_M1_0, 0.35, n=8)
    if not half:
        add("vias_heater", g, "tungsten_via")
    g = P.Geo()
    hx = R * math.cos(deg(50))
    hy = R * math.sin(deg(50))
    if not half:
        g.annulus(0, 0, 0, 4.6, Z_M1_0, Z_M1_1, n=48)
        for a0, a1 in [(deg(195), deg(345))]:
            g.annulus(0, 0, 9.9, 11.1, Z_M1_0, Z_M1_1, n=64, a0=a0, a1=a1)
        for sx in (-1, 1):
            g.cyl(sx * hx, hy, Z_M1_0, Z_M1_1, 1.0, n=16)
    else:
        g.annulus(0, 0, 0, 4.6, Z_M1_0, Z_M1_1, n=48, a0=0, a1=PI)
    add("metal1", g, "metal1")
    # heater (TiN) arc: 280 deg, gap at north (over the bus coupling region)
    g = P.Geo()
    rng = ang_ranges(deg(130), deg(410))
    if half:
        rng = clip_ranges(rng, 0.0, PI)
    for (s, e) in rng:
        g.annulus(0, 0, R - 0.8, R + 0.8, Z_HEAT0, Z_HEAT1, n=96, a0=s, a1=e)
    add("heater_tin", g, "tin_heater")
    if not half:
        # via1 (M1 -> M2) and M2 straps
        g = P.Geo()
        g.cyl(0, 0, Z_M1_1, Z_M2_0, 0.5, n=10)
        for sx in (-1, 1):
            g.cyl(sx * hx, hy, Z_M1_1, Z_M2_0, 0.5, n=10)
        gx, gy = 10.5 * math.cos(deg(285)), 10.5 * math.sin(deg(285))
        g.cyl(gx, gy, Z_M1_1, Z_M2_0, 0.5, n=10)
        add("vias_m1_m2", g, "tungsten_via")
        g = P.Geo()
        w2 = 2.0
        g.strip([(0, 0), (0, -14), (-22.5, -45), (-22.5, PAD_Y)], w2, Z_M2_0, Z_M2_1)
        g.strip([(-hx, hy), (-67.5, hy), (-67.5, PAD_Y)], w2, Z_M2_0, Z_M2_1)
        g.strip([(hx, hy), (67.5, hy), (67.5, PAD_Y)], w2, Z_M2_0, Z_M2_1)
        g.strip([(gx, gy), (22.5, -40), (22.5, PAD_Y)], w2, Z_M2_0, Z_M2_1)
        add("metal2", g, "metal2")
        g = P.Geo()
        for x in (-67.5, -22.5, 22.5, 67.5):
            g.cyl(x, PAD_Y, Z_M2_1, Z_PAD0, 1.2, n=12)
        add("vias_m2_pad", g, "tungsten_via")
        g = P.Geo()
        for x in (-67.5, -22.5, 22.5, 67.5):
            g.box_c(x, PAD_Y, (Z_PAD0 + Z_PAD1) / 2, PAD, PAD, Z_PAD1 - Z_PAD0)
        add("pads", g, "gold_pad", bevel=(0.5 * U, 1))
    # cladding (transparent SiO2) up to the passivation
    add("cladding_sio2", P.Geo().box(-ext_x, ext_y0, 0.002, ext_x, ext_y1, Z_CLAD), "cladding")
    return ob


def main():
    out_dir = C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics"
    out_dir = os.path.abspath(out_dir)
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="A")
    root["p_heat"] = 0.0
    root["p_ring_scale_note"] = "scaled variant is x10000 (1 um = 1 cm)"
    vr = C.sub_collection(coll, "VARIANT_real_size")
    vs = C.sub_collection(coll, "VARIANT_scaled_x10000")
    vc = C.sub_collection(coll, "VARIANT_cross_section_x10000")
    build_cell(vr, root, 1e-6, "")
    ob_s = build_cell(vs, root, 1e-6 * SCALE_UP, "_x10000")
    build_cell(vc, root, 1e-6 * SCALE_UP, "_xsec", half=True)
    # keep only y >= 0 half of everything and cap; the substrate/BOX/cladding boxes are cut by plane y = 0
    P.cut_collection(vc, 1, 0.0, keep_positive=True)
    # hooks (scaled variant positions)
    u = 1e-6 * SCALE_UP
    C.hook("ring_center", vs, root, loc=(0, 0, 0.11 * u))
    C.hook("bus_in", vs, root, loc=(-BUS_L * u, BUS_Y * u, 0.11 * u))
    C.hook("bus_out", vs, root, loc=(BUS_L * u, BUS_Y * u, 0.11 * u))
    C.hook("pad_array_center", vs, root, loc=(0, PAD_Y * u, Z_PAD1 * u))
    h = C.hook("ring_center_real", vr, root, loc=(0, 0, 0.11e-6))
    # drivers: p_heat drives heater + ring emission (both variants share the materials)
    heat = P.mat("tin_heater")
    ringm = P.mat("p_implant")
    for m, strength in ((heat, 6.0), (ringm, 1.5)):
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.35, 0.05, 1)
        fc = m.node_tree.driver_add('nodes["Principled BSDF"].inputs["Emission Strength"].default_value')
        d = fc.driver
        d.type = "SCRIPTED"
        var = d.variables.new()
        var.name = "heat"
        var.type = "SINGLE_PROP"
        var.targets[0].id = root
        var.targets[0].data_path = '["p_heat"]'
        d.expression = "heat*%s" % strength

    sx = 0.2 * 2 * 100.0 * u
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("ring centre-line radius", R, "um", "Th2A.14 (OFC 2026) 5 um ring; AMF 5x200G MRM paper 12 um (PMC10831040); 7.5 um chosen inside 5-10 um brief", "B", "range 5-12 um across public MRMs"),
        P.dim("rib / bus waveguide width", W, "um", "PMC10831040 (AMF, 500 nm wide, 220 nm thick, 90 nm slab)", "A"),
        P.dim("device layer (rib) thickness", H, "um", "220 nm SOI standard; PMC10831040", "A"),
        P.dim("slab thickness", SLAB, "um", "PMC10831040 (90 nm slab)", "A"),
        P.dim("bus-ring coupling gap (edge to edge)", GAP, "um", "PMC10831040 (180 nm)", "A", "brief range 100-300 nm"),
        P.dim("BOX thickness", BOX, "um", "brief; standard 220 nm SOI with 2 um BOX; not stated in PMC10831040", "B"),
        P.dim("displayed substrate thickness", SUB, "um", "truncated for display; real wafer 775 um (300 mm SOI)", "C"),
        P.dim("junction", "radial lateral p inside / n outside, p+ hub, n+ outer arc; 4 implant colours", "-", "PMC10831040 describes 4 implant layers (Z-shape); radial layout is a simplification", "C"),
        P.dim("implant dopings (reference)", "n,p 3e18; n+,p+ 6e18 cm^-3", "cm^-3", "PMC10831040 (not modelled, colours only)", "B"),
        P.dim("TiN heater", "1.6 um wide arc 280 deg (open over the bus), z 0.85-0.95 um above BOX top (R=7.5)", "um", "geometry estimated; typical heater 0.5-1 um above waveguide; ~500 ohm in PMC10831040", "C"),
        P.dim("M1 / M2 / pad heights", "M1 z 1.4-1.9, M2 z 2.6-3.3, pad z 3.9-4.4 (um above device-layer bottom)", "um", "estimated from generic 3-metal BEOL; AIM cross-section figure shows M1/M2/ML stack", "C"),
        P.dim("pad size / pitch", "35 x 35 / 45", "um", "estimated (microbump/probe pad class)", "C"),
        P.dim("cell footprint (real)", "200 x 113 x ~0.0164 (incl. 10 um substrate)", "um", "derived", "C"),
        P.dim("scale-up factor", SCALE_UP, "x", "1 um = 1 cm; same proportions as the real-size variant", "A"),
    ]
    hooks = ["HOOK_ring_center", "HOOK_bus_in", "HOOK_bus_out", "HOOK_pad_array_center", "HOOK_ring_center_real"]
    mats = P.used_material_names()
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "One silicon MRM cell with layer stack. Collections: VARIANT_real_size (1 um = 1e-6 m), VARIANT_scaled_x10000 (1 um = 1 cm), VARIANT_cross_section_x10000 (cut at y=0, keep y>=0). Hide two of the three. Cladding is a transparent SiO2 slab (MAT_photonics_cladding). Ring bus at +y; pads at -y.",
        [
            dict(src="https://pmc.ncbi.nlm.nih.gov/articles/PMC10831040", accessed="2026-10-02", used_for="rib/slab/gap, junction implant layout, TiN heater, AMF foundry"),
            dict(src="20260320_OFC/extracted_text/Th2A.14-057bbb3c-dd1a-4dd6-b068378f8d72ef2e_so4440191.txt", accessed="2026-10-02", used_for="5 um radius ring, FSR 19.5 nm"),
            dict(src="jwt625.github.io/assets/images/2025/20251219_CPO/aim-photonics-process-cross-section.webp", accessed="2026-10-02", used_for="layer ordering: Si device, M1, M2, ML in oxide; edge-coupler cavity"),
            dict(src="HC34 Ranovus 2022 (hc34.hotchips.org ... HC2022.Ranovus.ChristophSchulien.v13.pdf)", accessed="2026-10-02", used_for="ring modulator footprint typically < 50x50 um^2 (GF 45SPCLO)"),
        ],
        dims, hooks, mats,
        ["p_heat (0..1): drives emission of MAT_photonics_tin_heater (heater glow) and MAT_photonics_p_implant (ring tint) via drivers"],
        "S4 PIC close-up context and ring explanation; use VARIANT_scaled_x10000 for macro shots.",
        ["Radial junction instead of foundry Z-shape junction", "Substrate truncated to 10 um", "Metal routing and via counts schematic", "Pad openings in passivation not modelled"],
    )
    # JSON bbox should refer to the real-size variant
    scn = bpy_scene()
    for c in (vs, vc):
        c.hide_render = True
        c.hide_viewport = True
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    for c in (vs, vc):
        c.hide_render = False
        c.hide_viewport = False
    # rewrite JSON with variant sizes
    import json
    jp = os.path.splitext(blend)[0] + ".json"
    with open(jp) as f:
        j = json.load(f)
    j["size_mm_real_variant"] = [round(1e3 * (x), 6) for x in real_size(vr)]
    j["triangles_real_variant"] = C.count_tris(vr)
    j["variants"] = {"VARIANT_real_size": "1:1", "VARIANT_scaled_x10000": "x10000", "VARIANT_cross_section_x10000": "x10000, y>=0 half"}
    with open(jp, "w") as f:
        json.dump(j, f, indent=2, sort_keys=True)
    prev = os.path.join(out_dir, "previews")
    ctr = (0, -0.3, 0)
    shots = [
        dict(name="front", target=(0, -0.3, -0.04), dist=3.2, az=0, el=8, lens=50),
        dict(name="three_quarter", target=(0, -0.25, 0), dist=3.2, az=35, el=32, lens=50),
        dict(name="top", target=(0, -0.28, 0), dist=3.0, az=0, el=89.5, lens=50),
        dict(name="closeup_ring", target=(0, 0.0, 0.0), dist=0.42, az=20, el=40, lens=50),
        dict(name="closeup_gap_bus", target=(0.0, 0.080, 0.0), dist=0.14, az=12, el=38, lens=50),
        dict(name="closeup_stack_section", target=(0.0, 0.0, -0.05), dist=0.55, az=0, el=6, lens=50),
    ]
    vr.hide_render = True
    outs = P.render_shots([vs], prev, ASSET, shots[:5], floor=-0.125)
    for c in (vs,):
        c.hide_render = True
    outs += P.render_shots([vc], prev, ASSET, [dict(name="cutaway_section", target=(0.0, 0.0, -0.04), dist=0.45, az=0, el=8, lens=50),
                                                   dict(name="cutaway_closeup_layers", target=(0.04, 0.0, -0.005), dist=0.11, az=0, el=4, lens=50),
                                                   dict(name="cutaway_closeup_rib_section", target=(0.075, 0.0, 0.0), dist=0.035, az=0, el=3, lens=50),
                                                   dict(name="cutaway_section_three_quarter", target=(0.0, 0.05, -0.02), dist=0.5, az=-35, el=28, lens=50)], floor=-0.125)
    print(outs)


def bpy_scene():
    import bpy
    return bpy.context.scene


def real_size(vr):
    import bpy
    from mathutils import Vector
    pts = []
    for o in vr.all_objects:
        if o.type == "MESH":
            for v in o.bound_box:
                pts.append(o.matrix_world @ Vector(v))
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


if __name__ == "__main__":
    main()
