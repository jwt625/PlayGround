"""grating_coupler_closeup (stretch): vertical grating coupler of a 220 nm SOI PIC with a fibre tilted 10 deg above it and an
emissive mode cone; adiabatic taper from a 0.5 um waveguide. Real-size and x5000 variants (1 um = 5 mm).
Run: Blender -b --python build_grating_coupler_closeup.py -- <out_dir>
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
ASSET = "grating_coupler_closeup"
SCALE = 5000.0
PER, NT, WG = 0.63, 32, 12.0  # period um, teeth, grating width um
TAPER = 150.0


def build(coll, root, U, tag):
    cfib = C.sub_collection(coll, "LAYER_fiber" + tag)
    ccp = C.sub_collection(coll, "LAYER_chip" + tag)

    def b(name, g, m, smooth=False, bevel=None):
        g.scale(U)
        tgt = cfib if name.startswith(("fiber", "mode")) else ccp
        return g.build("%s_%s%s" % (ASSET, name, tag), P.mat(m) if isinstance(m, str) else m, tgt, root, smooth=smooth, bevel=bevel)
    build.layers = (cfib, ccp)

    x0, x1, y = -TAPER - 25.0, 45.0, 22.0
    b("substrate_si", P.Geo().box(x0, -y, -2.0 - 40.0, x1, y, -2.0), "silicon")
    b("box_oxide", P.Geo().box(x0, -y, -2.0, x1, y, 0.0), "box_oxide")
    b("cladding_sio2", P.Geo().box(x0, -y, 0.003, x1, y, 3.0), "cladding")
    g = P.Geo()
    g.box(x0, -0.25, 0, -TAPER, 0.25, 0.22)  # access waveguide
    n = 60
    for k in range(n):
        xa, xb = -TAPER + TAPER * k / n, -TAPER + TAPER * (k + 1) / n
        w = 0.5 + (WG - 0.5) * (k + 0.5) / n
        g.box(xa, -w / 2, 0, xb, w / 2, 0.22)
    # grating slab (150 nm remaining) and 70 nm ridges: period 0.63 um, 50 % duty, 32 teeth
    g.box(0, -WG / 2, 0, NT * PER, WG / 2, 0.15)
    b("waveguide_taper_slab", g, "waveguide_si")
    gr = P.Geo()
    for t in range(NT):
        gr.box(t * PER, -WG / 2, 0.15, t * PER + PER * 0.5, WG / 2, 0.22)
    b("grating_teeth", gr, "pic_waveguide")
    # fibre tilted 10 deg from vertical, end 10 um above the cladding, centred on the grating
    tilt = math.radians(10)
    cx = NT * PER / 2
    p0 = (cx, 0, 13.0)
    L = 130.0
    p1 = (cx + L * math.sin(tilt), 0, 13.0 + L * math.cos(tilt))
    b("fiber_glass", P.Geo().tube([p0, p1], 62.5, n=24), "fiber_glass", smooth=True)
    b("fiber_core", P.Geo().tube([(cx - 0.02, 0, 13.0 - 0.05), p1], 4.1, n=12), "fiber_core", smooth=True)
    beam = P.mat("beam_cone", (0.4, 0.9, 1.0), 0.0, 0.3, 0.12, emit=(0.3, 0.8, 1.0), emit_s=2.0)
    b("mode_cone", P.Geo().tube([(cx, 0, 13.0), (cx, 0, 0.22)], 5.2, n=20), beam, smooth=True)


def main():
    out_dir = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/photonics")
    C.reset()
    coll, root = C.new_asset(ASSET, accuracy="C")
    vr = C.sub_collection(coll, "VARIANT_real_size")
    vs = C.sub_collection(coll, "VARIANT_scaled_x5000")
    build(vr, root, 1e-6, "")
    build(vs, root, 1e-6 * SCALE, "_x5000")
    vs_chip = build.layers[1]
    u = 1e-6 * SCALE
    C.hook("grating_center", vs, root, loc=(NT * PER / 2 * u, 0, 0.22 * u))
    C.hook("fiber_end", vs, root, loc=(NT * PER / 2 * u, 0, 13.0 * u))
    P.refresh()
    blend = os.path.join(out_dir, ASSET + ".blend")
    dims = [
        P.dim("platform", "220 nm SOI, 2 um BOX, 70 nm shallow etch (150 nm slab)", "um", "standard SOI; AMF 220 nm Si (PMC10831040); etch depth typical", "B"),
        P.dim("grating period / teeth / width", "0.63 um (1550 nm design) / 32 / 12 um", "um", "typical literature range 0.58-0.65 um for 1550 nm at ~10 deg; NVIDIA uses O-band GC (period ~0.53 um by scaling); not from one PDK", "C"),
        P.dim("taper", "150 um adiabatic, 0.5 to 12 um", "um", "typical 100-300 um", "C"),
        P.dim("fibre tilt", "10 deg from the chip normal, 10 um above cladding", "deg", "typical 8-12 deg; NVIDIA OFC M4B.2 reports vertical GCs, 1.3 dB loss, 20 nm 1-dB bandwidth", "B"),
        P.dim("mode field dia (cone)", 10.4, "um", "SMF-28 MFD ~9.2 um at 1310, 10.4 um at 1550", "B"),
        P.dim("scale-up factor", SCALE, "x", "1 um = 5 mm", "A"),
    ]
    meta = P.meta_base(
        ASSET, "ASSET_" + ASSET, "Grating coupler close-up; scaled variant is the one to use in scenes (x5000). Edge coupler close-up: see fau_v_groove (edge_coupling_interface).",
        [dict(src="20260320_OFC M4B.2 (NVIDIA vertical GC)", accessed="2026-10-02", used_for="vertical coupling context"),
         dict(src="PMC10831040 (AMF)", accessed="2026-10-02", used_for="220 nm Si platform")],
        dims, ["HOOK_grating_center", "HOOK_fiber_end"], P.used_material_names(), [], "Coupling explainer shots (stretch item).",
        ["Period/etch generic", "No apodization", "Beam cone is illustrative"])
    meta = C.finish(ASSET, blend, coll, meta, preview_dir=None)
    prev = os.path.join(out_dir, "previews")
    cxm = NT * PER / 2 * u
    shots = [
        dict(name="front", target=(-0.2, 0, 0.8), dist=4.5, az=0, el=8),
        dict(name="three_quarter", target=(-0.1, 0, 0.3), dist=2.2, az=35, el=30),
        dict(name="top", target=(-0.4, 0, 0.0), dist=2.2, az=0, el=89.5),
        dict(name="closeup_fiber_over_grating", target=(cxm, 0, 0.05), dist=0.9, az=25, el=25),
    ]
    vr.hide_render = True
    outs = P.render_shots([vs], prev, ASSET, shots, floor=-0.21)
    outs += P.render_shots([vs_chip], prev, ASSET, [dict(name="closeup_teeth", target=(cxm, 0, 0.0), dist=0.28, az=-35, el=42), dict(name="closeup_taper_to_grating", target=(-0.3, 0, 0.0), dist=0.7, az=0, el=55)], floor=-0.21)
    print(outs)


if __name__ == "__main__":
    main()
