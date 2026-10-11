"""RX optics FX layer (workstream X, DevLog-009 Sec. 8): fixture cleanup, optical arrival, electrical signal glow.
Blender 4.2 module.

    import sys; sys.path.insert(0, "scripts/assets/optics_rx"); import rx_fx
    rx_fx.rx_clean_fixtures(root)                    # ghost-glass bench / shelf / turning block, lit Si plate edges
    fx = rx_fx.build_rx_fx(root)                     # root = RX_OPTICS_ROOT (optics_rx.blend, or after the GLB import)
    rx_fx.rx_light_arrive(fx, f_on=10, f_front_end=60)               # 8 cyan beams converge onto the PD apertures, PAM4
    rx_fx.rx_signal_glow(fx, f0=50, f1=150, rate=40.0, seed=7)       # amber pulses PD -> G-S-S-G bonds -> TIA -> CPW -> flex
    track = rx_fx.rx_front_track(fx, 10, 60, f_sig0=50, f_sig_end=62) # [(frame, world xyz)] for the camera

Geometry is built in the subassembly frame (mm x unit, default 0.001 m) as children of `parent`, so it follows any parent
scale (the macro set uses x1000). Thin emission ribbons / tubes only; keyed node values + a frame driver, no per-frame
Python. Colours: optical = RX cyan (scripts/vfx/materials.py RX_CYAN), electrical = warm amber (DevLog-018 V3, same as T).
Distances (`fx_dist`) are physical path lengths in mm. The FX mesh / material code follows T's tx_fx.py (copied, not
imported, so the two layers stay independent).

The optical paths are re-traced here with the same constants as build_optics_rx.py (v3) and asserted to land on the PD
apertures (< 1 um); bond and pad positions come from textures/rx_layout.json (the file the builder used).
"""
import json
import math
import os

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ADIR = os.path.join(PROJ, "assets", "components", "optics_rx")

OPT_COL = (0.1, 0.85, 1.0)         # RX cyan (scripts/vfx/materials.py RX_CYAN)
ELEC_COL = (1.0, 0.60, 0.13)       # warm amber (same as tx_fx ELEC_COL)

# ------------------------------------------------------------------ constants mirrored from build_optics_rx.py (v3)
N_FS, N_SI = 1.4469, 3.504
COL_PITCH, ROW_PITCH = 1.00, 0.75
LENS_T, LENS_CA = 0.50, 0.70
PLATE_TILT, PLATE_SHIFT, PLATE_W = 30.0, 0.25, 4.73
X_T = -3.90
X_F = X_T + 0.25
Z_BB = -0.70
Z_PD_TOP = -1.20
PD_PITCH, PD_D = 0.50, 0.30
Z_CAR = -1.45
ZT1 = Z_CAR + 0.003
PD_X1 = X_T + 0.12
PD_X0 = PD_X1 - PD_D
TIA_X1 = PD_X0 - 0.08
FLEX_X0, FLEX_X1 = -6.85, -6.55
ROWS = {"top": +ROW_PITCH / 2, "bot": -ROW_PITCH / 2}
COLS = [(i - 1.5) * COL_PITCH for i in range(4)]
PD_Y = [(k - 3.5) * PD_PITCH for k in range(8)]
LANE_OF = {("top", i): 2 * i for i in range(4)}
LANE_OF.update({("bot", i): 2 * i + 1 for i in range(4)})
_th = math.radians(PLATE_TILT)
PLATE_T = PLATE_SHIFT / (math.sin(_th) * (1 - math.cos(_th) / math.sqrt(N_SI ** 2 - math.sin(_th) ** 2)))
PLATE_XEXT = PLATE_W * math.sin(_th) + PLATE_T * math.cos(_th)
PLATE_XC = -LENS_T - 0.12 - PLATE_XEXT / 2
X_START = 3.0                      # beams start 3 mm in front of the eye
W_LENS = 0.30                      # beam radius at the lens (mm), as in rx_optics_refs.md
SURF = 0.004                       # ribbon lift above die / trace surfaces (mm)


def _parts():
    return json.load(open(os.path.join(ADIR, "optics_rx_parts.json")))


def _layout():
    return json.load(open(os.path.join(ADIR, "textures", "rx_layout.json")))


# ------------------------------------------------------------------ helpers (mm)
def resample(pts, step=0.02):
    out = [Vector(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = Vector(a), Vector(b)
        n = max(1, int(math.ceil((b - a).length / step)))
        for i in range(1, n + 1):
            out.append(a.lerp(b, i / n))
    return out


def cumdist(pts, d0=0.0):
    d = [d0]
    for a, b in zip(pts[:-1], pts[1:]):
        d.append(d[-1] + (b - a).length)
    return d


def refract(d, n, n1, n2):
    cosi = -d.dot(n)
    eta = n1 / n2
    k = 1 - eta * eta * (1 - cosi * cosi)
    return (d * eta + n * (eta * cosi - math.sqrt(k))).normalized()


def reflect(d, n):
    return (d - n * (2 * d.dot(n))).normalized()


def hit(p, d, q, n):
    return p + d * ((q - p).dot(n) / d.dot(n))


def bond_curve(p0, p1, loop=0.07, n=22):
    """Same Bezier as build_optics_rx.bond_curve (ball end p0 lifted 10 um, stitch end p1 lifted 6 um by the caller)."""
    p0, p1 = Vector(p0), Vector(p1)
    top = max(p0.z, p1.z) + loop
    c1 = Vector((p0.x, p0.y, top + 0.03))
    c2 = p0.lerp(p1, 0.70)
    c2.z = top - 0.005
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(p0 * (1 - t) ** 3 + c1 * 3 * (1 - t) ** 2 * t + c2 * 3 * (1 - t) * t * t + p1 * t ** 3)
    return out


def tia_xy(y, s):
    return (TIA_X1 - s, y)


def pd_xy(y, s):
    return (PD_X1 - s, y)


# ------------------------------------------------------------------ optical trace (exact, same surfaces as the build)
def trace(row, col, R, dy=0.0, dz=0.0):
    sag = R - math.sqrt(R * R - (LENS_CA / 2) ** 2)
    yc, zc = COLS[col], ROWS[row]
    y, z = yc + dy, zc + dz
    xs = sag - (R - math.sqrt(R * R - dy * dy - dz * dz))
    centre = Vector((sag - R, yc, zc))
    pts = [Vector((X_START, y, z))]
    p = Vector((xs, y, z))
    pts.append(p)
    d = refract(Vector((-1.0, 0.0, 0.0)), (p - centre).normalized(), 1.0, N_FS)
    p = hit(p, d, Vector((-LENS_T, 0, 0)), Vector((1, 0, 0)))
    pts.append(p)
    d = refract(d, Vector((1, 0, 0)), N_FS, 1.0)
    phi = math.radians(PLATE_TILT if row == "top" else -PLATE_TILT)
    c, nrm = Vector((PLATE_XC, 0, 0)), Vector((math.cos(phi), math.sin(phi), 0.0))
    p = hit(p, d, c + nrm * (PLATE_T / 2), nrm)
    pts.append(p)
    d = refract(d, nrm, 1.0, N_SI)
    p = hit(p, d, c - nrm * (PLATE_T / 2), nrm)
    pts.append(p)
    d = refract(d, nrm, N_SI, 1.0)
    p = hit(p, d, Vector((X_F, 0, 0)), Vector((1, 0, 0)))
    pts.append(p)
    d = refract(d, Vector((1, 0, 0)), 1.0, N_FS)
    fn = Vector((1, 0, -1)).normalized()
    p = hit(p, d, Vector((X_T, 0, zc)), fn)
    pts.append(p)
    d = reflect(d, fn)
    p = hit(p, d, Vector((0, 0, Z_BB)), Vector((0, 0, 1)))
    pts.append(p)
    d = refract(d, Vector((0, 0, 1)), N_FS, 1.0)
    p = hit(p, d, Vector((0, 0, Z_PD_TOP)), Vector((0, 0, 1)))
    pts.append(p)
    return pts


def optical_paths():
    Rs = _parts()["optics"]["lens_R_mm"]
    paths = []
    for row in ("top", "bot"):
        for col in range(4):
            k = LANE_OF[(row, col)]
            pts = trace(row, col, Rs[row])
            land = pts[-1]
            assert abs(land.y - PD_Y[k]) < 1e-3 and abs(land.x - X_T) < 1e-3, "rx_fx trace miss lane %d" % (k + 1)
            rp = resample(pts, 0.03)
            dd = cumdist(rp)
            # beam radius: collimated W_LENS outside / at the lens, converging linearly (by path length) to the focus
            d_lens = cumdist(pts)[1]
            d_end = dd[-1]
            rad = [W_LENS if d <= d_lens else max(0.006, W_LENS * (d_end - d) / (d_end - d_lens)) for d in dd]
            paths.append(dict(pts=rp, dist=dd, rad=rad, lane=k, land=land, d_land=d_end))
    return paths


# ------------------------------------------------------------------ electrical paths (per lane, P and N lines)
def electrical_paths():
    lay = _layout()
    tp = {p["name"]: p for p in lay["tia"]["pads"]}
    pp = {(p["lane"], p["kind"]): p for p in lay["pd"]["pads"]}
    ap = {a["lane"]: a for a in lay["pd"]["apertures"]}
    coils = {c["name"]: c for c in lay["tia"]["inductors"]}
    ZP = Z_PD_TOP
    RF_X0 = TIA_X1 - lay["tia"]["die"]["D"] - 0.09
    paths, coil_hits = [], []
    for L in range(1, 9):
        yc = PD_Y[L - 1]
        for sgn, pk, ti, to, ci in ((-1, "pd_p", "L%d_INP", "L%d_OUTP", 1), (1, "pd_n", "L%d_INN", "L%d_OUTN", 2)):
            segs = []
            a = ap[L]
            p_pad = pp[(L, pk)]
            x0, y0 = pd_xy(a["y"], a["s"])
            x1, y1 = pd_xy(p_pad["y"], p_pad["s"])
            segs.append(("pd", resample([(x0, y0, ZP + SURF), (x1, y1, ZP + SURF)], 0.01), 0.010))
            tin = tp[ti % L]
            xt, yt = tia_xy(tin["y"], tin["s"])
            segs.append(("bond", resample(bond_curve((x1, y1, ZP + 0.010), (xt, yt, ZP + 0.006), loop=0.065), 0.01), 0.0))
            route = [(0.05, 0.065), (0.05, 0.150), (0.100, 0.215), (0.06, 0.272), (0.04, 0.34), (0.04, 0.405),
                     (0.12, 0.45), (0.12, 0.50), (0.12, 0.60), (0.06, 0.68), (0.12, 0.78), (0.12, 0.85), (0.12, 0.94),
                     (0.09, 1.00), (0.075, 1.065), (0.075, 1.12), (0.05, 1.20), (0.05, 1.235)]
            die = [(*tia_xy(yc + sgn * u, s), ZP + SURF) for (u, s) in route]
            segs.append(("die", resample(die, 0.01), 0.010))
            tout = tp[to % L]
            xo, yo = tia_xy(tout["y"], tout["s"])
            yc_cpw = yc + sgn * 0.05
            segs.append(("bond", resample(bond_curve((xo, yo, ZP + 0.010), (RF_X0 - 0.05, yc_cpw, ZT1 + 0.006), loop=0.06),
                                          0.01), 0.0))
            cpw = [(RF_X0 - 0.05, yc_cpw, ZT1 + SURF), (FLEX_X1 + 0.02, yc_cpw, ZT1 + SURF),
                   ((FLEX_X0 + FLEX_X1) / 2, yc_cpw, ZT1 + 0.004 + SURF)]
            segs.append(("cpw", resample(cpw, 0.02), 0.045))
            d0 = 0.0
            for kind, pts, w in segs:
                dd = cumdist(pts, d0)
                if kind == "die":
                    # coil flashes: whole spiral lit at the distance where the route passes its centre
                    for cname in ("L%d_inp" % L if sgn < 0 else "L%d_inn" % L, "L%d_pk1%s" % (L, "a" if sgn < 0 else "b"),
                                  "L%d_pk2%s" % (L, "a" if sgn < 0 else "b"), "L%d_out%s" % (L, "a" if sgn < 0 else "b")):
                        c = coils[cname]
                        cx, cy = tia_xy(c["y"], c["s"])
                        j = min(range(len(pts)), key=lambda i: (pts[i].x - cx) ** 2 + (pts[i].y - cy) ** 2)
                        coil_hits.append(dict(pts=[Vector((*tia_xy(a_, b_), ZP + SURF)) for (a_, b_) in c["pts"]],
                                              d=dd[j], w=c["w"] * 1.6, lane=L - 1, pol=0 if sgn < 0 else 1))
                paths.append(dict(kind=kind, pts=pts, dist=dd, width=w, lane=L - 1, pol=0 if sgn < 0 else 1))
                d0 = dd[-1]
    dmax = max(max(p["dist"]) for p in paths)
    return paths, coil_hits, dmax


# ------------------------------------------------------------------ meshes with per-vertex attributes (after tx_fx.FXMesh)
class FXMesh:
    def __init__(self):
        self.v, self.f, self.attr = [], [], {"fx_dist": [], "fx_lane": [], "fx_mod": []}

    def _push(self, p, d, lane, mod):
        self.v.append(tuple(p))
        self.attr["fx_dist"].append(d)
        self.attr["fx_lane"].append(float(lane))
        self.attr["fx_mod"].append(float(mod))

    def ribbon(self, pts, dist, width, lane, mod, normal=(0, 0, 1)):
        nrm = Vector(normal)
        o = len(self.v)
        n = len(pts)
        for i, p in enumerate(pts):
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t = t.normalized() if t.length > 1e-12 else Vector((1, 0, 0))
            side = nrm.cross(t).normalized() * (width / 2)
            self._push(p - side, dist[i], lane, mod)
            self._push(p + side, dist[i], lane, mod)
        for i in range(n - 1):
            a = o + 2 * i
            self.f.append((a, a + 2, a + 3, a + 1))

    def tube(self, pts, dist, r, lane, mod, sides=8):
        n = len(pts)
        rr = r if isinstance(r, (list, tuple)) else [r] * n
        o = len(self.v)
        prev = None
        for i, p in enumerate(pts):
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t = t.normalized() if t.length > 1e-12 else Vector((1, 0, 0))
            if prev is None:
                ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
                side = t.cross(ref).normalized()
            else:
                side = (prev - t * prev.dot(t)).normalized()
            prev = side
            up = t.cross(side).normalized()
            for kk in range(sides):
                a = 2 * math.pi * kk / sides
                self._push(p + (side * math.cos(a) + up * math.sin(a)) * rr[i], dist[i], lane, mod)
        for i in range(n - 1):
            for kk in range(sides):
                j = (kk + 1) % sides
                self.f.append((o + i * sides + kk, o + i * sides + j, o + (i + 1) * sides + j, o + (i + 1) * sides + kk))

    def disc(self, c, r, d, lane, mod, seg=20):
        o = len(self.v)
        self._push(c, d, lane, mod)
        for kk in range(seg):
            a = 2 * math.pi * kk / seg
            self._push(c + Vector((r * math.cos(a), r * math.sin(a), 0)), d, lane, mod)
        for kk in range(seg):
            self.f.append((o, o + 1 + kk, o + 1 + (kk + 1) % seg))

    def to_object(self, name, material, parent, unit, coll):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(x * unit, y * unit, z * unit) for (x, y, z) in self.v], [], self.f)
        for k, vals in self.attr.items():
            at = me.attributes.new(k, "FLOAT", "POINT")
            at.data.foreach_set("value", vals)
        me.materials.append(material)
        me.validate()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        ob.parent = parent
        ob["fx"] = 1
        ob["explode"] = [0.0, 0.0, 0.0]
        return ob


# ------------------------------------------------------------------ materials (emission x transparency, keyed values)
def _n(nt, typ, loc, **kw):
    n = nt.nodes.new(typ)
    n.location = loc
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def _math(nt, op, a, b=None, loc=(0, 0), clamp=False):
    m = _n(nt, "ShaderNodeMath", loc, operation=op)
    m.use_clamp = clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            m.inputs[i].default_value = x
        else:
            nt.links.new(x, m.inputs[i])
    return m.outputs[0]


def _value(nt, name, val, loc):
    v = _n(nt, "ShaderNodeValue", loc, name=name, label=name)
    v.outputs[0].default_value = val
    return v


def _time_driver(node):
    fc = node.outputs[0].driver_add("default_value")
    drv = fc.driver
    drv.type = "SCRIPTED"
    var = drv.variables.new()
    var.name = "fps"
    var.type = "SINGLE_PROP"
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = bpy.context.scene
    var.targets[0].data_path = "render.fps"
    drv.expression = "frame / fps"


def fx_material(name, color, strength, electrical=False, alpha_gain=1.5):
    """Keyed values: front_mm, master, mod_on (optical), sym_per_mm, rate, seed; time from a frame driver."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    nt = m.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = _n(nt, "ShaderNodeOutputMaterial", (1400, 0))
    at_d = _n(nt, "ShaderNodeAttribute", (-1200, 200), attribute_name="fx_dist", attribute_type="GEOMETRY").outputs["Fac"]
    at_l = _n(nt, "ShaderNodeAttribute", (-1200, 0), attribute_name="fx_lane", attribute_type="GEOMETRY").outputs["Fac"]
    at_m = _n(nt, "ShaderNodeAttribute", (-1200, -200), attribute_name="fx_mod", attribute_type="GEOMETRY").outputs["Fac"]
    front = _value(nt, "front_mm", -1.0, (-1200, 400)).outputs[0]
    master = _value(nt, "master", 0.0, (-1200, 600)).outputs[0]
    k = _value(nt, "sym_per_mm", 3.0 if electrical else 2.0, (-1200, -400)).outputs[0]
    rate = _value(nt, "rate", 40.0 if electrical else 30.0, (-1200, -600)).outputs[0]
    seed = _value(nt, "seed", 7.0, (-1200, -800)).outputs[0]
    tnode = _value(nt, "time_s", 0.0, (-1200, -1000))
    _time_driver(tnode)
    t = tnode.outputs[0]
    lag = _math(nt, "SUBTRACT", front, at_d, (-900, 400))
    reveal = _math(nt, "DIVIDE", lag, 0.25, (-700, 400), clamp=True)
    head = _math(nt, "MULTIPLY", _math(nt, "EXPONENT", _math(nt, "MULTIPLY", _math(nt, "POWER", _math(nt, "DIVIDE", lag, 0.35, (-700, 600)), 2.0, (-550, 600)), -1.0, (-400, 600)), None, (-250, 600)), 2.5, (-100, 600))
    head = _math(nt, "MULTIPLY", head, _math(nt, "GREATER_THAN", lag, -0.6, (-250, 750)), (0, 650))
    sym = _math(nt, "FLOOR", _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", at_d, k, (-900, -400)), _math(nt, "MULTIPLY", t, rate, (-900, -600)), (-700, -500)), None, (-550, -500))
    comb = _n(nt, "ShaderNodeCombineXYZ", (-400, -500))
    nt.links.new(sym, comb.inputs[0])
    nt.links.new(_math(nt, "MULTIPLY", at_l, 7.31, (-550, -700)), comb.inputs[1])
    nt.links.new(seed, comb.inputs[2])
    wn = _n(nt, "ShaderNodeTexWhiteNoise", (-250, -500), noise_dimensions="3D")
    nt.links.new(comb.outputs[0], wn.inputs["Vector"])
    lvl = _math(nt, "DIVIDE", _math(nt, "FLOOR", _math(nt, "MULTIPLY", wn.outputs["Value"], 3.999, (-100, -500)), None, (50, -500)), 3.0, (200, -500))
    lo = 0.15 if electrical else 0.06
    pam = _math(nt, "ADD", _math(nt, "MULTIPLY", lvl, 1.0 - lo, (350, -500)), lo, (500, -500))
    if electrical:
        # fx_mod code (electrical): polarity bit (0 = P line, 1 = N line, complementary PAM4 level) + 2 x coil flag
        pol = _math(nt, "FLOORED_MODULO", at_m, 2.0, (-100, -700))
        dim = _math(nt, "GREATER_THAN", at_m, 1.5, (-100, -850))
        lvl = _math(nt, "ADD", lvl, _math(nt, "MULTIPLY", pol, _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", lvl, 2.0, (200, -650)), (300, -650)), (400, -650)), (500, -650))
        pam = _math(nt, "ADD", _math(nt, "MULTIPLY", lvl, 1.0 - lo, (600, -650)), lo, (700, -650))
        frac = _math(nt, "FRACT", _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", at_d, k, (-900, -300)), _math(nt, "MULTIPLY", t, rate, (-900, -800)), (-700, -300)), None, (-550, -300))
        dash = _math(nt, "LESS_THAN", frac, 0.55, (-400, -300))
        sig = _math(nt, "ADD", _math(nt, "MULTIPLY", dash, pam, (650, -400)), 0.12, (800, -400))
        inten = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "MULTIPLY", reveal, sig, (950, -200)), head, (1050, 0)), master, (1150, 0))
        inten = _math(nt, "MULTIPLY", inten, _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", dim, 0.55, (950, -900)), (1050, -900)), (1150, -100))
    else:
        mod_on = _value(nt, "mod_on", 0.0, (-1200, 800)).outputs[0]
        gate = _math(nt, "MULTIPLY", at_m, mod_on, (650, -300))
        modf = _math(nt, "ADD", _math(nt, "MULTIPLY", gate, _math(nt, "SUBTRACT", pam, 1.0, (650, -500)), (800, -400)), 1.0, (950, -400))
        inten = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "MULTIPLY", reveal, modf, (950, -200)), head, (1050, 0)), master, (1150, 0))
    em = _n(nt, "ShaderNodeEmission", (1150, 200))
    em.inputs["Color"].default_value = (*color, 1)
    nt.links.new(_math(nt, "MULTIPLY", inten, strength, (1150, 350)), em.inputs["Strength"])
    tr = _n(nt, "ShaderNodeBsdfTransparent", (1150, -150))
    mix = _n(nt, "ShaderNodeMixShader", (1300, 0))
    nt.links.new(_math(nt, "MINIMUM", _math(nt, "MULTIPLY", inten, alpha_gain, (1150, 450)), 1.0, (1250, 450)), mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*color, 1)
    return m


# ------------------------------------------------------------------ fixture cleanup (scene-safe: works on GLB imports)
FIXTURES = ("rx_optical_bench", "rx_interleaver_shelf_posts", "rx_turning_block_stepped")


def _ghost_glass(name, tint=(0.55, 0.75, 0.95), alpha=0.05, edge=0.8):
    """Nearly clear glass for EEVEE: low alpha body + facing-ratio edge glow so the outline still reads."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    if hasattr(m, "use_transparent_shadow"):
        m.use_transparent_shadow = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tint, 1)
    b.inputs["Roughness"].default_value = 0.05
    b.inputs["Alpha"].default_value = alpha
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.15
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = edge
    nt.links.new(lw.outputs["Facing"], mul.inputs[0])
    b.inputs["Emission Color"].default_value = (*tint, 1)
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    m["fx_ghost"] = 1
    return m


def rx_clean_fixtures(parent, si_plates=True, lenslets=False, facets=True):
    """Make the structural glass (bench, plate shelf + posts, turning-block body) nearly clear with lit edges, keep the
    Au facets; optionally Si plates translucent violet-grey with lit edges and lenslets clear. Geometry and positions are
    untouched (mechanics unchanged). Works on optics_rx.blend and on a GLB import (name prefixes, .001 suffixes ok).
    Returns the list of touched objects (the scene can also hide them: obj.hide_render)."""
    kids = [o for o in parent.children_recursive if o.type == "MESH"]
    g_body = _ghost_glass("fx_rx_ghost_glass")
    g_si = _ghost_glass("fx_rx_ghost_si", tint=(0.62, 0.58, 0.95), alpha=0.28, edge=2.0)
    g_lens = _ghost_glass("fx_rx_ghost_lens", tint=(0.6, 0.95, 1.0), alpha=0.18, edge=1.6)
    g_au = _ghost_glass("fx_rx_ghost_au_facet", tint=(1.0, 0.75, 0.35), alpha=0.16, edge=1.2)   # mirrors read dark in EEVEE
    touched = []
    for o in kids:
        base = o.name.split(".")[0]
        if base in FIXTURES:
            for slot in o.material_slots:
                if slot.material and slot.material.name.startswith("gold_mirror"):
                    if facets:
                        slot.material = g_au
                elif slot.material:
                    slot.material = g_body
            touched.append(o)
        elif si_plates and base.startswith("rx_si_interleaver_plate"):
            for slot in o.material_slots:
                slot.material = g_si
            touched.append(o)
        elif lenslets and base.startswith("rx_lenslet_array"):
            for slot in o.material_slots:
                slot.material = g_lens
            touched.append(o)
        o.visible_shadow = o.visible_shadow and base not in FIXTURES
    return touched


# ------------------------------------------------------------------ public API
def build_rx_fx(parent, unit=0.001, collection_name="rx_fx"):
    coll = bpy.data.collections.get(collection_name) or bpy.data.collections.new(collection_name)
    if collection_name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    opaths = optical_paths()
    epaths, coil_hits, d_el = electrical_paths()
    d_opt = max(p["d_land"] for p in opaths)
    m_beam = fx_material("fx_rx_optical_beam", OPT_COL, 0.9, alpha_gain=0.10)       # faint converging cone (unmodulated)
    m_core = fx_material("fx_rx_optical_core", OPT_COL, 10.0)                       # thin bright core + PD spots
    m_el = fx_material("fx_rx_electrical", ELEC_COL, 2.6, electrical=True)      # v4: less bloom so the bits read
    mb, mc = FXMesh(), FXMesh()
    for p in opaths:
        mb.tube(p["pts"], p["dist"], p["rad"], p["lane"], 0, sides=12)
        mc.tube(p["pts"], p["dist"], 0.008, p["lane"], 1, sides=6)
        c = Vector((p["land"].x, p["land"].y, Z_PD_TOP + 0.003))
        mc.disc(c, 0.022, p["d_land"] - 0.05, p["lane"], 1)                    # aperture spot
        mc.disc(c + Vector((0, 0, 0.001)), 0.045, p["d_land"] + 0.10, p["lane"], 1)   # halo (lights a moment later)
    me = FXMesh()
    for p in epaths:
        if p["kind"] == "bond":
            me.tube(p["pts"], p["dist"], 0.0145, p["lane"], p["pol"], sides=6)     # just outside the 12.5 um wire
        else:
            me.ribbon(p["pts"], p["dist"], 0.016 if p["kind"] == "die" else p["width"], p["lane"], p["pol"])
    for ch in coil_hits:
        me.ribbon(ch["pts"], [ch["d"]] * len(ch["pts"]), ch["w"], ch["lane"], ch["pol"] + 2)
    objs = dict(beam=mb.to_object("fx_rx_optical_beams", m_beam, parent, unit, coll),
                core=mc.to_object("fx_rx_optical_cores_pd_spots", m_core, parent, unit, coll),
                elec=me.to_object("fx_rx_electrical_signal", m_el, parent, unit, coll))
    # front tracks (mm): optical = mean of the 8 chief rays at each distance (lane lengths differ by row: use the
    # fraction of each lane's own length); electrical = mean of all P/N paths at each distance
    otrack = []
    for i in range(41):
        f = i / 40
        pts = []
        for p in opaths:
            dd = f * p["d_land"]
            j = min(range(len(p["dist"])), key=lambda q: abs(p["dist"][q] - dd))
            pts.append(p["pts"][j])
        otrack.append((f * d_opt, tuple(sum(pts, Vector()) / len(pts))))
    allp = [(q, d) for p in epaths for q, d in zip(p["pts"], p["dist"])]
    etrack = []
    for i in range(31):
        dd = d_el * i / 30
        sel = [q for q, d in allp if abs(d - dd) < 0.12]
        if sel:
            etrack.append((dd, tuple(sum(sel, Vector()) / len(sel))))
    return dict(objs=objs, m_beam=m_beam, m_core=m_core, m_el=m_el, d_opt=d_opt, d_el=d_el,
                d_lens=cumdist(trace("top", 0, _parts()["optics"]["lens_R_mm"]["top"]))[1],
                d_land={p["lane"] + 1: round(p["d_land"], 4) for p in opaths},
                otrack_mm=otrack, etrack_mm=etrack, parent=parent, unit=unit)


def _key(mat, node_name, frame, value, interp="LINEAR"):
    n = mat.node_tree.nodes[node_name]
    n.outputs[0].default_value = value
    n.outputs[0].keyframe_insert("default_value", frame=frame)
    fc = mat.node_tree.animation_data.action.fcurves.find('nodes["%s"].outputs[0].default_value' % node_name)
    for kp in fc.keyframe_points:
        if abs(kp.co[0] - frame) < 1e-6:
            kp.interpolation = interp


def rx_light_arrive(fx, f_on, f_front_end, f_off=None, rate=30.0, seed=7):
    """8 modulated (PAM4) cyan beams enter the eye at f_on; the front travels from 3 mm in front of the eye to the PD
    apertures (linear in path length) by f_front_end; aperture spots and halos light when the front lands. The light is
    already modulated on arrival (it comes from the far-end TX), so blinking is on from f_on. Optional 6-frame fade-out."""
    for m in (fx["m_beam"], fx["m_core"]):
        m.node_tree.nodes["rate"].outputs[0].default_value = rate
        m.node_tree.nodes["seed"].outputs[0].default_value = float(seed)
        _key(m, "master", f_on - 1, 0.0, "CONSTANT")
        _key(m, "master", f_on, 1.0)
        _key(m, "mod_on", f_on - 1, 1.0, "CONSTANT")
        _key(m, "front_mm", f_on - 1, -1.0, "CONSTANT")
        _key(m, "front_mm", f_on, 0.0)
        _key(m, "front_mm", f_front_end, fx["d_opt"] + 0.4)
        if f_off is not None:
            _key(m, "master", f_off, 1.0)
            _key(m, "master", f_off + 6, 0.0)
    f_lens = f_on + (f_front_end - f_on) * fx["d_lens"] / fx["d_opt"]
    return dict(f_on=f_on, f_front_end=f_front_end, f_lens=round(f_lens, 2), d_opt_mm=round(fx["d_opt"], 3))


def rx_land_frames(fx, f_on, f_front_end):
    """Frames at which the light front lands on the PDs: (first lanes = bottom row, last lanes = top row)."""
    lo, hi = min(fx["d_land"].values()), max(fx["d_land"].values())
    return (f_on + (f_front_end - f_on) * lo / fx["d_opt"], f_on + (f_front_end - f_on) * hi / fx["d_opt"])


def rx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6, intensity=1.0):
    """Electrical 'changing bits' (same look as tx_signal_glow: amber, PAM4 symbols travelling at rate / sym_per_mm mm/s,
    55 % duty dashes, 4 random levels per symbol from (symbol, lane, seed)). Arrival front from the PD apertures through
    the differential G-S-S-G bonds (P and N lines carry complementary levels), across each TIA channel (spirals flash
    dimmed as symbols pass), out the GSSG bonds and along the CPW to the flex pads over arrive_frames from f0; fades out
    over `fade` frames ending at f1. intensity scales the whole layer (keyed on 'master'). Start it when the light lands:
    f0 = rx_land_frames(fx, f_on, f_front_end)[0]."""
    m = fx["m_el"]
    nt = m.node_tree.nodes
    nt["rate"].outputs[0].default_value = rate
    nt["seed"].outputs[0].default_value = float(seed)
    nt["sym_per_mm"].outputs[0].default_value = sym_per_mm
    f0 = int(round(f0))
    _key(m, "master", f0 - 1, 0.0, "CONSTANT")
    _key(m, "master", f0, intensity)
    _key(m, "master", f1 - fade, intensity)
    _key(m, "master", f1, 0.0)
    _key(m, "front_mm", f0 - 1, -1.0, "CONSTANT")
    _key(m, "front_mm", f0, 0.0)
    _key(m, "front_mm", f0 + arrive_frames, fx["d_el"] + 0.5)
    return dict(f0=f0, f1=f1, rate=rate, seed=seed, speed_mm_per_s=rate / sym_per_mm, d_el_mm=round(fx["d_el"], 3),
                intensity=intensity)


def rx_front_track(fx, f_on, f_front_end, f_sig0=None, f_sig_end=None, world=True):
    """[(frame, position)] of the optical front (eye -> PD) and, if f_sig0/f_sig_end are given, then of the electrical
    front (PD -> TIA -> CPW -> flex). World metres via the parent matrix (world=True) or subassembly mm."""
    M = fx["parent"].matrix_world
    out = []

    def put(fr, p):
        pv = Vector(p)
        out.append((fr, tuple(M @ (pv * fx["unit"])) if world else tuple(pv)))

    for d, p in fx["otrack_mm"]:
        put(f_on + (f_front_end - f_on) * d / fx["d_opt"], p)
    if f_sig0 is not None and f_sig_end is not None:
        for d, p in fx["etrack_mm"]:
            put(f_sig0 + (f_sig_end - f_sig0) * d / fx["d_el"], p)
    return out
