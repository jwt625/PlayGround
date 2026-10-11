"""TX optics FX layer (workstream T, DevLog-008 Sec. 8): optical light-up and electrical signal glow. Blender 4.2 module.

Use inside Blender (scene build or the test script):
    import sys; sys.path.insert(0, "scripts/assets/optics_tx"); import tx_fx
    fx = tx_fx.build_tx_fx(parent)            # parent = the subassembly root (OPTICS_TX_ROOT after a GLB import, or in optics_tx.blend)
    tx_fx.tx_light_up(fx, f_on=10, f_front_end=80)                 # DFB on, light front through lenses, PIC, MZMs, outputs
    tx_fx.tx_signal_glow(fx, f0=40, f1=150, rate=40.0, seed=7)    # RF pulses: flex contacts -> carrier -> GSSG bonds -> TW electrodes
    track = tx_fx.tx_front_track(fx, f_on=10, f_front_end=80)     # [(frame, world position)] of the optical front, for the camera

Geometry is built in the subassembly frame (mm * unit, default 0.001 m) as children of `parent`, so it follows any parent
scale (the macro set uses x1000). All FX objects are thin emission ribbons / tubes (cheap in EEVEE); everything is driven by
keyed node values and a frame driver, no per-frame Python. Colours: optical = TX red (scripts/vfx/materials.py TX_RED),
electrical = warm amber (DevLog-018 V3). Distances are physical path lengths in mm along the light / RF path.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tx_geom as g  # noqa: E402

OPT_COL = (1.0, 0.08, 0.05)        # TX red (scripts/vfx/materials.py TX_RED)
ELEC_COL = (1.0, 0.60, 0.13)       # warm amber, distinct from the optical red
SURF = 0.004                       # ribbon lift above the PIC / trace surface (mm)

# carrier / bond constants mirrored from build_optics_tx.py (v1.1)
T_TR = 0.018
X_CTOP = g.X_CTOP
CY0 = g.Y_BACK - 1.60
FLEX_RF_Y = (CY0, CY0 + 0.45)
RF_CPAD_Y = g.Y_BACK - 0.20
PT = 0.004


# ------------------------------------------------------------------ path helpers (mm)
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


def bond_curve(p0, p1, h=0.13, n=14, up=(1, 0, 0)):
    """Same loop shape as build_optics_tx.bond()."""
    p0, p1, u = Vector(p0), Vector(p1), Vector(up)
    pts = []
    for i in range(n + 1):
        t = i / n
        lift = h * math.sin(math.pi * t / 1.6) ** 0.55 if t < 0.8 else h * (1 - (t - 0.8) / 0.2) ** 1.3
        pts.append(p0.lerp(p1, t) + u * lift)
    return pts


def pic_xyz(u, v, lift=SURF):
    return Vector((g.X_W + lift, g.Y_EDGE - v, u))


# ------------------------------------------------------------------ optical path network with distances
def optical_paths():
    """Returns list of dict(pts=[Vector mm], dist=[mm], lane=int or -1, mod=0/1, kind) and the max distance."""
    T = g.TRAIN
    paths = []
    # free-space laser trains: DFB facet -> lens1 -> isolator -> lens2 -> laser SSC (distance 0 at the facet)
    for gi, zl in enumerate(g.LASER_U):
        pts = resample([(g.X_W, T["dfb_facet"], zl), (g.X_W, g.Y_EDGE, zl)], 0.05)
        paths.append(dict(pts=pts, dist=cumdist(pts), lane=-1 - gi, mod=0, kind="laser_train"))
    L_train = T["dfb_facet"] - g.Y_EDGE
    # on-chip polylines: BFS from the laser SSCs over the waveguide set (+ SSC segments)
    wg, mmi, ssc = g.waveguides()
    polys = [[(u, v) for (u, v) in p] for p in wg]
    for (u, v0, v1) in ssc:
        polys.append([(u, v0), (u, v1)])
    polys = [resample([(u, v, 0.0) for (u, v) in p], 0.02) for p in polys]   # (u, v, 0) vectors
    reached = {}
    for i, p in enumerate(polys):
        for end in (0, -1):
            if abs(p[end].y) < 1e-6 and any(abs(p[end].x - ul) < 1e-6 for ul in g.LASER_U):
                pp = p if end == 0 else p[::-1]
                reached[i] = (pp, cumdist(pp, L_train))
    changed = True
    TOL = 0.09
    while changed:
        changed = False
        pool = [(q, d) for (pp, dd) in reached.values() for q, d in zip(pp, dd)]
        for i, p in enumerate(polys):
            if i in reached:
                continue
            best = None
            for end in (0, -1):
                for q, d in pool:
                    e = (p[end] - q).length
                    if e < TOL and (best is None or e < best[0]):
                        best = (e, end, d)
            if best:
                e, end, d = best
                pp = p if end == 0 else p[::-1]
                reached[i] = (pp, cumdist(pp, d + e))
                changed = True
    d_mzm = []
    # modulated flag: polylines that start at / after the MZM output MMI (v <= 3.51) and are not the laser input segments;
    # lane id (for per-lane PAM4 randomness): outputs by the SSC position, MZM region by the lane position, tree = -1
    for i, (pp, dd) in reached.items():
        v0 = pp[0].y
        mod = 1 if (v0 <= g.V_MMI_OUT[1] + 0.01 and dd[0] > L_train + 1.0) else 0
        if v0 < 0.3 and mod:
            lane = min(range(8), key=lambda k: abs(g.OUT_U[k] - pp[0].x))
        elif v0 < 6.6 and dd[0] > L_train + 1.0:
            lane = min(range(8), key=lambda k: abs(g.LANE_U[k] - pp[0].x))
        else:
            lane = -1
        paths.append(dict(pts=[pic_xyz(q.x, q.y) for q in pp], dist=dd, lane=lane, mod=mod, kind="pic"))
        if mod and v0 > 3.0:
            d_mzm.append(dd[0])
    # MMI glow pads at the distance of the nearest reached point
    pool = [(q, d) for (pp, dd) in reached.values() for q, d in zip(pp, dd)]
    mmi_pads = []
    for (u0, v0, u1, v1) in mmi:
        c = Vector(((u0 + u1) / 2, (v0 + v1) / 2, 0))
        d = min(pool, key=lambda qd: (qd[0] - c).length)[1]
        mmi_pads.append(((u0, v0, u1, v1), d))
    # output free-space paths: SSC facet -> collimator -> staircase fold -> rhomboid -> eye plate -> 1 mm out
    out_d = {}
    for (pp, dd) in reached.values():
        if abs(pp[-1].y) < 1e-6:
            k = min(range(8), key=lambda k_: abs(g.OUT_U[k_] - pp[-1].x))
            if abs(g.OUT_U[k] - pp[-1].x) < 1e-3:
                out_d[k] = dd[-1]
    c_r = g.RH["x0"] + g.RH["s"] / 2
    for d in g.lane_map():
        k = d["lane"]
        z, yc, zo, dz = d["z_in"], d["y_out"], d["z_out"], d["dz"]
        pts = [(g.X_W, g.Y_EDGE, z), (g.X_W, yc, z)]
        if abs(dz) > 1e-9:
            pts += [(c_r, yc, z), (c_r, yc, zo)]
        pts += [(-g.EYE["t"], yc, zo), (0.0, yc, zo), (1.0, yc, zo)]
        pts = resample(pts, 0.05)
        paths.append(dict(pts=pts, dist=cumdist(pts, out_d.get(k, 0.0)), lane=k, mod=1, kind="output"))
    dmax = max(max(p["dist"]) for p in paths)
    d_eye = min(d for p in paths if p["kind"] == "output" for q, d in zip(p["pts"], p["dist"]) if q.x >= -g.EYE["t"])
    return paths, mmi_pads, dmax, out_d, (min(d_mzm) if d_mzm else dmax), d_eye


def electrical_paths():
    """Per lane, both signal lines: flex contact -> carrier trace -> GSSG bond -> PIC pad -> feed -> TW electrode -> termination."""
    paths = []
    yp = g.Y_EDGE - (g.V_RF_PAD[0] + g.V_RF_PAD[1]) / 2
    for k, uk in enumerate(g.LANE_U):
        for j, sg in ((3 * k + 1, -1), (3 * k + 2, 1)):
            zj = g.RF_PAD_U[j]
            mid = (g.RF_PAD_U[3 * k + 1] + g.RF_PAD_U[3 * k + 2]) / 2
            zc = mid + sg * 0.075
            xs = X_CTOP + T_TR + SURF
            trace = [((FLEX_RF_Y[0] + FLEX_RF_Y[1]) / 2, zj), (FLEX_RF_Y[1], zj), (FLEX_RF_Y[1] + 0.25, zc),
                     (RF_CPAD_Y - 0.30, zc), (RF_CPAD_Y - 0.08, zj), (RF_CPAD_Y + 0.02, zj)]
            tr = resample([(xs, y, z) for (y, z) in trace], 0.03)
            d_tr = cumdist(tr)
            bd = bond_curve((g.X_W + PT, yp + 0.02, zj), (X_CTOP + T_TR, RF_CPAD_Y + 0.02, zj))[::-1]
            bd = resample(bd, 0.02)
            d_bd = cumdist(bd, d_tr[-1])
            ue = uk + sg * 0.04
            on = [(zj, (g.V_RF_PAD[0] + g.V_RF_PAD[1]) / 2), (zj, g.V_RF_PAD[0] - 0.12), (ue, g.V_EL[1]), (ue, g.V_EL[0]),
                  (ue, (g.V_TERM[0] + g.V_TERM[1]) / 2), (uk + sg * 0.0825, (g.V_TERM[0] + g.V_TERM[1]) / 2)]
            pc = resample([pic_xyz(u, v, 0.006) for (u, v) in on], 0.03)
            d_pc = cumdist(pc, d_bd[-1])
            paths.append(dict(pts=tr, dist=d_tr, lane=k, kind="trace", width=0.07))
            paths.append(dict(pts=bd, dist=d_bd, lane=k, kind="bond", width=0.0))
            paths.append(dict(pts=pc, dist=d_pc, lane=k, kind="electrode", width=0.04))
    dmax = max(max(p["dist"]) for p in paths)
    return paths, dmax


# ------------------------------------------------------------------ mesh builders with per-vertex attributes
class FXMesh:
    def __init__(self):
        self.v, self.f, self.attr = [], [], {"fx_dist": [], "fx_lane": [], "fx_mod": []}

    def _push(self, p, d, lane, mod):
        self.v.append(tuple(p))
        self.attr["fx_dist"].append(d)
        self.attr["fx_lane"].append(float(lane))
        self.attr["fx_mod"].append(float(mod))

    def ribbon(self, pts, dist, width, lane, mod, normal=(1, 0, 0)):
        nrm = Vector(normal)
        o = len(self.v)
        n = len(pts)
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
            t = t.normalized() if t.length > 1e-12 else Vector((0, 1, 0))
            side = nrm.cross(t).normalized() * (width / 2)
            self._push(p - side, dist[i], lane, mod)
            self._push(p + side, dist[i], lane, mod)
        for i in range(n - 1):
            a = o + 2 * i
            self.f.append((a, a + 2, a + 3, a + 1))

    def tube(self, pts, dist, r, lane, mod, sides=6):
        n = len(pts)
        o = len(self.v)
        prev = None
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
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
                self._push(p + (side * math.cos(a) + up * math.sin(a)) * r, dist[i], lane, mod)
        for i in range(n - 1):
            for kk in range(sides):
                j = (kk + 1) % sides
                self.f.append((o + i * sides + kk, o + i * sides + j, o + (i + 1) * sides + j, o + (i + 1) * sides + kk))

    def quad(self, corners, d, lane, mod):
        o = len(self.v)
        for c in corners:
            self._push(c, d, lane, mod)
        self.f.append((o, o + 1, o + 2, o + 3))

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


# ------------------------------------------------------------------ materials (emission + transparency, keyed values)
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


def fx_material(name, color, strength, electrical=False):
    """Emission x mask. Keyed values: 'front_mm' (reveal front), 'master' (0-1), 'mod_on' (0-1, optical only),
    'sym_per_mm', 'seed', 'rate' (symbols per second, travelling). Time from a frame driver."""
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
    k = _value(nt, "sym_per_mm", 3.0 if electrical else 4.0, (-1200, -400)).outputs[0]
    rate = _value(nt, "rate", 40.0 if electrical else 60.0, (-1200, -600)).outputs[0]
    seed = _value(nt, "seed", 7.0, (-1200, -800)).outputs[0]
    tnode = _value(nt, "time_s", 0.0, (-1200, -1000))
    _time_driver(tnode)
    t = tnode.outputs[0]
    # reveal: soft front + bright head
    lag = _math(nt, "SUBTRACT", front, at_d, (-900, 400))
    reveal = _math(nt, "DIVIDE", lag, 0.25, (-700, 400), clamp=True)
    head = _math(nt, "MULTIPLY", _math(nt, "EXPONENT", _math(nt, "MULTIPLY", _math(nt, "POWER", _math(nt, "DIVIDE", lag, 0.35, (-700, 600)), 2.0, (-550, 600)), -1.0, (-400, 600)), None, (-250, 600)), 2.5, (-100, 600))
    head = _math(nt, "MULTIPLY", head, _math(nt, "GREATER_THAN", lag, -0.6, (-250, 750)), (0, 650))
    # PAM4 symbols travelling along the path: index = floor(d * k - t * rate); level from 3D white noise (sym, lane, seed)
    sym = _math(nt, "FLOOR", _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", at_d, k, (-900, -400)), _math(nt, "MULTIPLY", t, rate, (-900, -600)), (-700, -500)), None, (-550, -500))
    comb = _n(nt, "ShaderNodeCombineXYZ", (-400, -500))
    nt.links.new(sym, comb.inputs[0])
    nt.links.new(_math(nt, "ADD", _math(nt, "MULTIPLY", at_l, 7.31, (-550, -700)), 0.0, (-450, -700)), comb.inputs[1])
    nt.links.new(seed, comb.inputs[2])
    wn = _n(nt, "ShaderNodeTexWhiteNoise", (-250, -500), noise_dimensions="3D")
    nt.links.new(comb.outputs[0], wn.inputs["Vector"])
    lvl = _math(nt, "DIVIDE", _math(nt, "FLOOR", _math(nt, "MULTIPLY", wn.outputs["Value"], 3.999, (-100, -500)), None, (50, -500)), 3.0, (200, -500))
    lo = 0.15 if electrical else 0.04                      # optical: deeper PAM4 contrast so the modulation reads
    pam = _math(nt, "ADD", _math(nt, "MULTIPLY", lvl, 1.0 - lo, (350, -500)), lo, (500, -500))
    if electrical:
        # dashed pulses: duty 0.55 inside each symbol, PAM4 amplitude, dim base trace glow
        frac = _math(nt, "FRACT", _math(nt, "SUBTRACT", _math(nt, "MULTIPLY", at_d, k, (-900, -300)), _math(nt, "MULTIPLY", t, rate, (-900, -800)), (-700, -300)), None, (-550, -300))
        dash = _math(nt, "LESS_THAN", frac, 0.55, (-400, -300))
        sig = _math(nt, "ADD", _math(nt, "MULTIPLY", dash, pam, (650, -400)), 0.12, (800, -400))
        inten = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "MULTIPLY", reveal, sig, (950, -200)), head, (1050, 0)), master, (1150, 0))
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
    nt.links.new(_math(nt, "MINIMUM", _math(nt, "MULTIPLY", inten, 1.5, (1150, 450)), 1.0, (1250, 450)), mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*color, 1)
    return m


def dfb_glow_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*OPT_COL, 1)
    b.inputs["Emission Color"].default_value = (*OPT_COL, 1)
    b.inputs["Emission Strength"].default_value = 0.0
    return m


# ------------------------------------------------------------------ public API
def build_tx_fx(parent, unit=0.001, collection_name="tx_fx"):
    """Create the FX objects under `parent`. Returns a dict handle used by the keying functions."""
    coll = bpy.data.collections.get(collection_name) or bpy.data.collections.new(collection_name)
    if collection_name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    opaths, mmi_pads, d_opt, out_d, d_mzm_out, d_eye = optical_paths()
    epaths, d_el = electrical_paths()
    m_opt = fx_material("fx_tx_optical", OPT_COL, 9.0)
    m_el = fx_material("fx_tx_electrical", ELEC_COL, 3.5, electrical=True)
    mo_pic, mo_free = FXMesh(), FXMesh()
    for p in opaths:
        if p["kind"] == "pic":
            mo_pic.ribbon(p["pts"], p["dist"], 0.012, p["lane"], p["mod"])
        else:
            mo_free.tube(p["pts"], p["dist"], 0.012 if p["kind"] == "output" else 0.02, p["lane"], p["mod"], sides=6)
    for (u0, v0, u1, v1), d in mmi_pads:
        mo_pic.quad([pic_xyz(u0, v0), pic_xyz(u1, v0), pic_xyz(u1, v1), pic_xyz(u0, v1)], d, -1, 0)
    me = FXMesh()
    for p in epaths:
        if p["kind"] == "bond":
            me.tube(p["pts"], p["dist"], 0.016, p["lane"], 0, sides=6)
        else:
            me.ribbon(p["pts"], p["dist"], p["width"], p["lane"], 0)
    objs = dict(
        opt_pic=mo_pic.to_object("fx_tx_optical_pic", m_opt, parent, unit, coll),
        opt_free=mo_free.to_object("fx_tx_optical_free_space", m_opt, parent, unit, coll),
        elec=me.to_object("fx_tx_electrical_rf", m_el, parent, unit, coll),
    )
    m_dfb = dfb_glow_material("fx_tx_dfb_on")
    glow = FXMesh()
    for zl in g.LASER_U:
        c = Vector((g.X_W - 0.004, g.TRAIN["dfb_facet"] - 0.01, zl))
        glow.tube([c + Vector((0, 0.0, -0.001)), c + Vector((0, -0.03, 0))], [0, 0], 0.025, -1, 0, sides=10)
    objs["dfb"] = glow.to_object("fx_tx_dfb_on", m_dfb, parent, unit, coll)
    # front track (mm, subassembly frame): mean position of path points within +/-0.3 mm of each distance, then a
    # +/-3-sample moving average (the raw mean jumps between branches); Z centred (the layout is mirror-symmetric in Z)
    allp = [(q, d) for p in opaths for q, d in zip(p["pts"], p["dist"])]
    raw = []
    for i in range(53):
        dd = d_opt * i / 52
        sel = [q for q, d in allp if abs(d - dd) < 0.3]
        if sel:
            raw.append((dd, sum(sel, Vector()) / len(sel)))
    track = []
    for i, (dd, c) in enumerate(raw):
        win = [cc for _, cc in raw[max(0, i - 3):i + 4]]
        m_ = sum(win, Vector()) / len(win)
        track.append((dd, (m_.x, m_.y, 0.0)))
    return dict(objs=objs, m_opt=m_opt, m_el=m_el, m_dfb=m_dfb, d_opt=d_opt, d_el=d_el, d_mzm_out=d_mzm_out, d_eye=d_eye,
                d_ssc_out=max(out_d.values()) if out_d else d_opt, d_laser_ssc=g.TRAIN["dfb_facet"] - g.Y_EDGE,
                track_mm=track, parent=parent, unit=unit)


def _key(mat, node_name, frame, value, interp="LINEAR"):
    n = mat.node_tree.nodes[node_name]
    n.outputs[0].default_value = value
    n.outputs[0].keyframe_insert("default_value", frame=frame)
    fc = mat.node_tree.animation_data.action.fcurves.find('nodes["%s"].outputs[0].default_value' % node_name)
    for kp in fc.keyframe_points:
        if abs(kp.co[0] - frame) < 1e-6:
            kp.interpolation = interp


def tx_light_up(fx, f_on, f_front_end, f_mod_on=None, dfb_ramp=4, f_off=None):
    """DFB facets on at f_on (ramp dfb_ramp frames); light front travels 0 -> full path (laser trains, PIC, outputs, eye)
    linearly between f_on and f_front_end; the MZM outputs start PAM4 blinking at f_mod_on (default: when the front
    passes the MZM output MMIs). Optional f_off switches everything off (6-frame fade)."""
    m, md = fx["m_opt"], fx["m_dfb"]
    _key(m, "master", f_on - 1, 0.0, "CONSTANT")
    _key(m, "master", f_on, 1.0)
    _key(m, "front_mm", f_on - 1, -1.0, "CONSTANT")
    _key(m, "front_mm", f_on, 0.0)
    _key(m, "front_mm", f_front_end, fx["d_opt"] + 0.5)
    if f_mod_on is None:
        f_mod_on = f_on + (f_front_end - f_on) * fx["d_mzm_out"] / fx["d_opt"]
    _key(m, "mod_on", f_on, 0.0, "CONSTANT")
    _key(m, "mod_on", int(round(f_mod_on)), 1.0)
    b = md.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    for fr, v in ((f_on - 1, 0.0), (f_on + dfb_ramp, 40.0)):
        b.default_value = v
        b.keyframe_insert("default_value", frame=fr)
    # base-asset emitters (also present after a GLB import): DFB facet glow on with the DFB, eye lenslets when the front
    # reaches the eye plate
    f_eye = f_on + (f_front_end - f_on) * fx["d_eye"] / fx["d_opt"]
    for mname, fr_on, ramp in (("tx_dfb_facet_glow", f_on, dfb_ramp), ("tx_eye_lens_red", f_eye, 4)):
        for mb in [x for x in bpy.data.materials if x.name == mname or x.name.startswith(mname + ".")]:
            bs = mb.node_tree.nodes.get("Principled BSDF") if mb.use_nodes else None
            if bs is None:
                continue
            inp = bs.inputs["Emission Strength"]
            full = mb.get("fx_full_emission", inp.default_value)
            mb["fx_full_emission"] = full
            for fr, v in ((fr_on - 1, 0.0), (fr_on + ramp, full)):
                inp.default_value = v
                inp.keyframe_insert("default_value", frame=fr)
    if f_off is not None:
        _key(m, "master", f_off, 1.0)
        _key(m, "master", f_off + 6, 0.0)
        for fr, v in ((f_off, 40.0), (f_off + 6, 0.0)):
            b.default_value = v
            b.keyframe_insert("default_value", frame=fr)
    return dict(f_on=f_on, f_front_end=f_front_end, f_mod_on=f_mod_on, f_eye=f_eye)


def tx_signal_glow(fx, f0, f1, rate=40.0, seed=7, sym_per_mm=3.0, arrive_frames=12, fade=6):
    """RF signal glow: arrival front from the flex contacts to the terminations over arrive_frames starting at f0, then
    travelling PAM4 pulses (rate symbols/s, sym_per_mm spatial density -> speed rate/sym_per_mm mm/s in subassembly mm);
    fades out over `fade` frames ending at f1. Per-lane randomness from (lane, seed)."""
    m = fx["m_el"]
    nt = m.node_tree.nodes
    nt["rate"].outputs[0].default_value = rate
    nt["seed"].outputs[0].default_value = float(seed)
    nt["sym_per_mm"].outputs[0].default_value = sym_per_mm
    _key(m, "master", f0 - 1, 0.0, "CONSTANT")
    _key(m, "master", f0, 1.0)
    _key(m, "master", f1 - fade, 1.0)
    _key(m, "master", f1, 0.0)
    _key(m, "front_mm", f0 - 1, -1.0, "CONSTANT")
    _key(m, "front_mm", f0, 0.0)
    _key(m, "front_mm", f0 + arrive_frames, fx["d_el"] + 0.5)
    return dict(f0=f0, f1=f1, rate=rate, seed=seed, speed_mm_per_s=rate / sym_per_mm)


def tx_front_track(fx, f_on, f_front_end, world=True):
    """[(frame, position)] of the optical front; position in world metres (world=True, via the parent matrix) or
    subassembly mm. Use it to aim / pan the camera along the light (lasers -> PIC back edge -> outputs -> eye)."""
    out = []
    M = fx["parent"].matrix_world
    for d, p in fx["track_mm"]:
        fr = f_on + (f_front_end - f_on) * d / fx["d_opt"]
        pv = Vector(p)
        out.append((fr, tuple(M @ (pv * fx["unit"])) if world else tuple(pv)))
    return out
