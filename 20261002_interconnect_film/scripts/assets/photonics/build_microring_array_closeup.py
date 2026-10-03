"""microring_array_closeup: PIC section with 3 bus waveguides and 24 microrings (8 per bus, alternating sides), each ring
its own object + material MAT_photonics_ring_00..23, substrate split into 28 vertical slabs (MAT_photonics_slab_00..27),
heaters, M1 straps, pads, grating couplers at the bus ends, die-seal ring. Real-size and x15000 variants.

Run: Blender -b --python build_microring_array_closeup.py -- <out_dir>
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
PI = math.pi
ASSET = "microring_array_closeup"
SCALE_UP = 15000.0  # 1 um -> 15 mm

R, W, H, SLAB, GAP = 7.5, 0.5, 0.22, 0.09, 0.18
BOX = 2.0
SUBT = 200.0  # displayed substrate thickness (section; real 775 um)
CHIP_X = 500.0
CHIP_Y = 510.0
BUS_Y = [-150.0, 0.0, 150.0]
N_RING_PER_BUS = 8
PITCH = 40.0
X0 = -140.0
NSLAB = 28
Z_HEAT0, Z_HEAT1 = 0.85, 0.95
Z_M1_0, Z_M1_1 = 1.4, 1.9
Z_PAD0, Z_PAD1 = 2.3, 2.8
Z_CLAD = 2.3
RING_OFF = R + W / 2 + GAP + W / 2  # bus centre to ring centre


def deg(a):
    return math.radians(a)


def ring_parts(sy):
    """Ring local geometry (um). sy=+1: ring north of its bus (bus to the south)."""
    gapc = deg(270) if sy > 0 else deg(90)
    gring = P.Geo()
    gring.annulus(0, 0, R - W / 2, R + W / 2, 0, H, n=96)
    gring.annulus(0, 0, R - 2.0, R - W / 2, 0, SLAB, n=96)
    # outer slab leaves the bus side open (+-65 deg around the bus direction)
    a0 = gapc + deg(65)
    a1 = gapc + 2 * PI - deg(65)
    gring.annulus(0, 0, R + W / 2, R + 2.0, 0, SLAB, n=96, a0=a0, a1=a1)
    gh = P.Geo()
    gh.annulus(0, 0, R - 0.8, R + 0.8, Z_HEAT0, Z_HEAT1, n=96, a0=gapc + deg(40), a1=gapc + 2 * PI - deg(40))
    # heater end positions
    e1 = (R * math.cos(gapc + deg(40)), R * math.sin(gapc + deg(40)))
    e2 = (R * math.cos(gapc - deg(40)), R * math.sin(gapc - deg(40)))
    ends = sorted([e1, e2], key=lambda p: p[0])
    gm = P.Geo()
    gv = P.Geo()
    pads = P.Geo()
    for (ex, ey), px in zip(ends, (-18.0, 18.0)):
        gv.cyl(ex, ey, Z_HEAT1, Z_M1_0, 0.35, n=8)
        gm.cyl(ex, ey, Z_M1_0, Z_M1_1, 1.0, n=12)
        sgn = 1 if px > 0 else -1
        gm.strip([(ex, ey), (sgn * 15.5, ey), (sgn * 15.5, sy * 30.0), (px, sy * 40.0)], 1.6, Z_M1_0, Z_M1_1)
        gv.cyl(px, sy * 40.0, Z_M1_1, Z_PAD0, 0.8, n=10)
        pads.box_c(px, sy * 40.0, (Z_PAD0 + Z_PAD1) / 2, 30.0, 30.0, Z_PAD1 - Z_PAD0)
    return gring, gh, gm, gv, pads


def grating_coupler(xc, yc, direction):
    """Grating coupler: taper (60 um) + 12 um wide grating of 32 teeth, period 0.63 um. direction=+1: light enters from +x end."""
    g = P.Geo()
    L_t, W_g = 60.0, 12.0
    # taper polygon as strip with varying width -> use a few boxes (stepped wedge approximated by quads)
    pts = []
    nseg = 60
    for k in range(nseg):
        x_a = L_t * k / nseg
        x_b = L_t * (k + 1) / nseg
        w_a = W + (W_g - W) * k / nseg
        w_b = W + (W_g - W) * (k + 1) / nseg
        wm = (w_a + w_b) / 2
        g.box(x_a, -wm / 2, 0, x_b, wm / 2, H)
    per, nt = 0.63, 32
    gx0 = L_t
    gg = P.Geo()
    for t in range(nt):
        gg.box(gx0 + t * per + per * 0.5, -W_g / 2, 0, gx0 + t * per + per, W_g / 2, H)
    g.box(gx0, -W_g / 2, 0, gx0 + nt * per, W_g / 2, 0.15)  # shallow-etched base (70 nm etch of 220)
    g.merge(gg)
    # orient: taper joins the bus at x=xc. direction=+1 -> grating extends toward +x? we mirror so GC sits outside the bus
    out = P.Geo()
    for (x, y, z) in g.v:
        out.v.append((xc + direction * x, yc + y, z))
    out.f = [tuple(f) if direction > 0 else tuple(reversed(f)) for f in g.f]
    return out


def build(coll, root, U, tag):
    def build_geo(name, geo, mname_or_mat, loc=(0, 0, 0), smooth=False, bevel=None):
        geo.scale(U)
        m = mname_or_mat if not isinstance(mname_or_mat, str) else P.mat(mname_or_mat)
        return geo.build("%s_%s%s" % (ASSET, name, tag), m, coll, root, loc=tuple(c * U for c in loc), smooth=smooth, bevel=bevel)

    objs = {"rings": [], "slabs": []}
    # substrate slabs (28 vertical slabs along x)
    sw = CHIP_X / NSLAB
    for k in range(NSLAB):
        xc = -CHIP_X / 2 + (k + 0.5) * sw
        g = P.Geo().box(-sw / 2, -CHIP_Y / 2, -SUBT / 2, sw / 2, CHIP_Y / 2, SUBT / 2)
        m = slab_mats[k]
        o = build_geo("slab_%02d" % k, g, m, loc=(xc, 0, -BOX - SUBT / 2))
        objs["slabs"].append(o)
    build_geo("box_oxide", P.Geo().box(-CHIP_X / 2, -CHIP_Y / 2, -BOX + 0.002, CHIP_X / 2, CHIP_Y / 2, 0.0), "box_oxide")
    build_geo("cladding_sio2", P.Geo().box(-CHIP_X / 2, -CHIP_Y / 2, 0.003, CHIP_X / 2, CHIP_Y / 2, Z_CLAD), "cladding")
    # buses, grating couplers, rings, heaters
    gbus = P.Geo()
    ggc = P.Geo()
    gheat = P.Geo()
    gm1 = P.Geo()
    gvia = P.Geo()
    gpad = P.Geo()
    xend = CHIP_X / 2 - 40.0 - 60.0 - 32 * 0.63 * 1.0  # bus ends where the taper starts (left/right)
    xbus = 150.0
    ring_i = 0
    for b, by in enumerate(BUS_Y):
        gbus.box(-xbus, by - W / 2, 0, xbus, by + W / 2, H)
        ggc.merge(grating_coupler(-xbus, by, -1))
        ggc.merge(grating_coupler(xbus, by, +1))
        for k in range(N_RING_PER_BUS):
            sy = 1 if k % 2 == 0 else -1
            rx = X0 + k * PITCH
            ry = by + sy * RING_OFF
            gring, gh, gm, gv, pads = ring_parts(sy)
            idx = b * N_RING_PER_BUS + k
            o = build_geo("ring_%02d" % idx, gring, ring_mats[idx], loc=(rx, ry, 0))
            objs["rings"].append(o)
            gheat.merge(gh, rx, ry, 0)
            gm1.merge(gm, rx, ry, 0)
            gvia.merge(gv, rx, ry, 0)
            gpad.merge(pads, rx, ry, 0)
    build_geo("bus_waveguides", gbus, "waveguide_si")
    build_geo("grating_couplers", ggc, "waveguide_si")
    build_geo("heaters_tin", gheat, "tin_heater")
    build_geo("metal1_straps", gm1, "metal1")
    build_geo("vias", gvia, "tungsten_via")
    build_geo("pads", gpad, "gold_pad")
    # die-seal ring (metal frame, 5 um wide, 12 um inside the edge)
    gs = P.Geo()
    inset = 12.0
    hx, hy = CHIP_X / 2 - inset, CHIP_Y / 2 - inset
    gs.strip([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)], 5.0, Z_M1_0, Z_M1_1, closed=True)
    build_geo("die_seal_ring", gs, "copper")
    return objs


slab_mats = []
ring_mats = []


def main():
    out_dir = C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics"
    out_dir = os.path.abspath(out_dir)
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="B")
    root["p_wave_pos"] = -0.3
    root["p_wave_width"] = 0.22
    root["p_wave_gain"] = 8.0
    root["p_slab_gain_ratio"] = 0.25
    # materials with per-slab / per-ring drivers
    for i in range(24):
        m = P.mat("ring_%02d" % i, (0.10, 0.45, 0.75), 0.7, 0.25)
        ring_mats.append(m)
    for i in range(NSLAB):
        m = P.mat("slab_%02d" % i, (0.10, 0.12, 0.16), 0.8, 0.35)
        slab_mats.append(m)

    def drive(m, xn, ratio_expr, col):
        bs = m.node_tree.nodes["Principled BSDF"]
        bs.inputs["Emission Color"].default_value = (*col, 1)
        fc = m.node_tree.driver_add('nodes["Principled BSDF"].inputs["Emission Strength"].default_value')
        d = fc.driver
        d.type = "SCRIPTED"
        for nm, prop in (("pos", "p_wave_pos"), ("wid", "p_wave_width"), ("gain", "p_wave_gain"), ("ratio", "p_slab_gain_ratio")):
            v = d.variables.new()
            v.name = nm
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["%s"]' % prop
        d.expression = "%s*gain*max(0,1-abs(pos-%.5f)/wid)" % (ratio_expr, xn)

    for b in range(3):
        for k in range(N_RING_PER_BUS):
            idx = b * N_RING_PER_BUS + k
            xn = (X0 + k * PITCH + CHIP_X / 2) / CHIP_X
            drive(ring_mats[idx], xn, "1", (1.0, 0.30, 0.04))
    for k in range(NSLAB):
        drive(slab_mats[k], (k + 0.5) / NSLAB, "ratio", (1.0, 0.45, 0.12))
    vr = C.sub_collection(coll, "VARIANT_real_size")
    vs = C.sub_collection(coll, "VARIANT_scaled_x15000")
    build(vr, root, 1e-6, "")
    build(vs, root, 1e-6 * SCALE_UP, "_x15000")
    u = 1e-6 * SCALE_UP
    C.hook("array_center", vs, root, loc=(0, 0, 0))
    C.hook("wave_start", vs, root, loc=(-CHIP_X / 2 * u, 0, 0))
    C.hook("wave_end", vs, root, loc=(CHIP_X / 2 * u, 0, 0))
    C.hook("array_center_real", vr, root, loc=(0, 0, 0))
    root["p_note"] = "p_wave_pos 0..1 sweeps a heat wave across rings and slabs left to right (driver-based); or animate MAT_photonics_ring_NN / slab_NN yourself"

    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("ring radius", R, "um", "Th2A.14 5 um; AMF 12 um; chosen 7.5 (brief 5-10 um)", "B"),
        P.dim("rib width x height / slab", "0.5 x 0.22 / 0.09", "um", "PMC10831040 (AMF)", "A"),
        P.dim("bus-ring gap (edge to edge)", GAP, "um", "PMC10831040", "A"),
        P.dim("ring pitch along bus (alternating sides)", PITCH, "um", "estimated layout; Ranovus: RRM footprint < 50x50 um^2 each (HC34 2022)", "C", "same-side pitch 80 um"),
        P.dim("bus spacing", 150.0, "um", "estimated to fit pad rows", "C"),
        P.dim("heater pads", "30 x 30 um, 2 per ring, +-18 um from ring centre, 40 um from bus line", "um", "estimated", "C"),
        P.dim("chip section", "%d x %d" % (CHIP_X, CHIP_Y), "um", "chosen section of a PIC; substrate shown %d um thick (real 775 um)" % SUBT, "C"),
        P.dim("grating coupler", "taper 60 um, 12 um wide, 32 teeth, period 0.63 um, 70 nm etch", "um", "typical 1550 nm 220 nm SOI GC (period ~0.6-0.65 um); not from a specific PDK", "C"),
        P.dim("die-seal ring", "5 um wide frame, 12 um inside edge", "um", "generic; real seal rings 5-10 um class", "C"),
        P.dim("scale-up factor", SCALE_UP, "x", "1 um = 15 mm, chip 7.5 m x 7.65 m; ring diameter 0.225 m; real proportions", "A", "rings are 3% of chip width at true proportions; scale individual ring objects (origin at ring centre) to exaggerate"),
        P.dim("slab count / ring count", "28 / 24", "-", "storyboard S4 requirement", "A"),
    ]
    mats = P.used_material_names()
    hooks = ["HOOK_array_center", "HOOK_wave_start", "HOOK_wave_end", "HOOK_array_center_real"]
    props = [
        "p_wave_pos (-0.3..1.3): heat-wave centre, 0 = left chip edge, 1 = right chip edge (drives MAT_photonics_ring_NN and slab_NN emission)",
        "p_wave_width (0.22), p_wave_gain (8.0), p_slab_gain_ratio (0.25)",
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET,
        "24 microrings (ring_00..23: bus 0 = y -150 um row, then 0, then +150) and 28 substrate slabs along x. Ring object origin = ring centre; slab origin = slab centre. Variants: real size and x15000. Use the scaled variant for macro shots. Light/dark materials driven by root custom properties.",
        [
            dict(src="https://pmc.ncbi.nlm.nih.gov/articles/PMC10831040", accessed="2026-10-02", used_for="ring geometry"),
            dict(src="20260320_OFC Th2A.14", accessed="2026-10-02", used_for="5 um ring"),
            dict(src="HC34 2022 Ranovus Odin 8P (Schulien)", accessed="2026-10-02", used_for="ring footprint < 50x50 um^2"),
            dict(src="jwt625.github.io 2025/20251219_CPO/intel-photonic-ic-layout.webp, intel-100g-modulator-teardown.webp", accessed="2026-10-02", used_for="visual layout of pad/strap/ring fields"),
        ],
        dims, hooks, mats, props,
        "S4 PIC close-up (heat wave over rings and substrate).",
        ["Only M1 + pad layer (no M2)", "Substrate section 200 um thick", "Bus ends at grating couplers; no splitters", "Modulator p-n junction not coloured (heater-tuned rings)"],
    )
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    jp = os.path.splitext(blend)[0] + ".json"
    j = json.load(open(jp))
    j["size_mm_real_variant"] = [CHIP_X * 1e-3, CHIP_Y * 1e-3, (SUBT + BOX + Z_PAD1) * 1e-3]
    j["triangles_real_variant"] = C.count_tris(vr)
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)

    # previews: set a mid-chip heat wave only in memory
    root["p_wave_pos"] = 0.4
    bpy_update()
    prev = os.path.join(out_dir, "previews")
    cs = 7.5
    shots = [
        dict(name="front", target=(0, 0, -1.4), dist=12.5, az=0, el=10, lens=50),
        dict(name="three_quarter", target=(0, 0, -0.8), dist=12.5, az=32, el=32, lens=50),
        dict(name="top", target=(0, 0, 0), dist=11.5, az=0, el=89.5, lens=50),
        dict(name="oblique_wave", target=(0.5, -0.3, 0), dist=3.6, az=-20, el=28, lens=50),
        dict(name="closeup_ring_pads", target=(-2.1, 0.12, 0), dist=1.1, az=15, el=45, lens=50),
        dict(name="closeup_grating_coupler", target=(-3.15, 0, 0), dist=0.75, az=-25, el=40, lens=50),
        dict(name="closeup_substrate_slabs", target=(-1.5, -3.8, -0.8), dist=4.0, az=-15, el=15, lens=50),
    ]
    vr.hide_render = True
    outs = P.render_shots([vs], prev, ASSET, shots, floor=-(BOX + SUBT) * u)
    # also a cold (no wave) reference
    root["p_wave_pos"] = -5.0
    bpy_update()
    outs += P.render_shots([vs], prev, ASSET, [dict(name="cold_top", target=(0, 0, 0), dist=11.5, az=0, el=89.5, lens=50)], floor=-(BOX + SUBT) * u)
    print(outs)


def bpy_update():
    import bpy
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()


if __name__ == "__main__":
    main()
