"""Build the TX optics + PIC subassembly (workstream T, DevLog-008). Blender 4.2, headless.

Run from the project folder (always through the slot wrapper, one Blender process at a time):
  scripts/bslot.sh -b --factory-startup --python scripts/assets/optics_tx/build_optics_tx.py -- assets/components/optics_tx

Needs the PIC textures first (make_pic_texture.py). Writes optics_tx.blend, optics_tx.glb, optics_tx_parts.json.
Frame (DevLog-005 Sec. 3): origin = centre of the 2x4 eye-plate outer surface (dome bases), +X = light out of the eye,
+Z up; Blender meters, true scale. Geometry numbers live in tx_geom.py (mm); grades in research/tx_optics_refs.md.
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tx_geom as g  # noqa: E402

MM = 0.001
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["assets/components/optics_tx"]
OUT_DIR = os.path.abspath(ARGS[0])
ASSET = "optics_tx"
TEX = os.path.join(OUT_DIR, "textures")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.save_version = 0
SCN = bpy.context.scene
SCN.unit_settings.system = "METRIC"
SCN.unit_settings.scale_length = 1.0


# ================================================================== mesh builder (mm, subassembly frame)
class MB:
    def __init__(self):
        self.v, self.f, self.uv = [], [], {}

    def add(self, verts, faces):
        o = len(self.v)
        self.v += [tuple(float(c) for c in p) for p in verts]
        self.f += [[i + o for i in fc] for fc in faces]
        return self

    def face(self, pts, outward):
        """Add one polygon, wound so its normal points along `outward`."""
        a, b, c = Vector(pts[0]), Vector(pts[1]), Vector(pts[2])
        n = (b - a).cross(c - a)
        if n.dot(Vector(outward)) < 0:
            pts = pts[::-1]
        o = len(self.v)
        self.v += [tuple(p) for p in pts]
        self.f.append(list(range(o, o + len(pts))))
        return self

    def box(self, x0, x1, y0, y1, z0, z1):
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        z0, z1 = sorted((z0, z1))
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return self.add(v, f)

    def prism(self, prof, a0, a1, plane):
        """Extrude polygon prof [(p, q)] between a0..a1; plane 'xy' (along Z), 'xz' (along Y), 'yz' (along X)."""
        def P(p, q, a):
            return {"xy": (p, q, a), "xz": (p, a, q), "yz": (a, p, q)}[plane]
        n = len(prof)
        v = [P(p, q, a0) for p, q in prof] + [P(p, q, a1) for p, q in prof]
        f = [list(range(n))[::-1], list(range(n, 2 * n))]
        for i in range(n):
            j = (i + 1) % n
            f.append([i, j, n + j, n + i])
        return self.add(v, f)

    def cyl(self, c, r, h, axis="x", segs=20):
        cx, cy, cz = c
        v = []
        for k in range(segs):
            a = 2 * math.pi * k / segs
            ca, sa = math.cos(a) * r, math.sin(a) * r
            for s in (-0.5, 0.5):
                if axis == "x":
                    v.append((cx + s * h, cy + ca, cz + sa))
                elif axis == "y":
                    v.append((cx + ca, cy + s * h, cz + sa))
                else:
                    v.append((cx + ca, cy + sa, cz + s * h))
        f = []
        for k in range(segs):
            j = (k + 1) % segs
            f.append([2 * k, 2 * j, 2 * j + 1, 2 * k + 1])
        f.append([2 * k for k in range(segs)][::-1])
        f.append([2 * k + 1 for k in range(segs)])
        return self.add(v, f)

    def ellipsoid(self, c, rx, ry, rz, segs=12, rings=6):
        v, f = [], []
        for i in range(rings + 1):
            th = math.pi * i / rings
            for k in range(segs):
                ph = 2 * math.pi * k / segs
                v.append((c[0] + rx * math.sin(th) * math.cos(ph), c[1] + ry * math.sin(th) * math.sin(ph), c[2] + rz * math.cos(th)))
        for i in range(rings):
            for k in range(segs):
                j = (k + 1) % segs
                f.append([i * segs + k, i * segs + j, (i + 1) * segs + j, (i + 1) * segs + k])
        return self.add(v, f)

    def cap(self, c, axis, R, a, segs=28, rings=6):
        """Spherical cap (lens surface), base disk of radius a centred at c, apex along +/-axis ('x','y','-y',...).
        Closed by the base disk."""
        sgn = -1 if axis.startswith("-") else 1
        ax = axis[-1]
        sag = R - math.sqrt(R * R - a * a)
        v = []
        for i in range(rings + 1):
            rho = a * (1 - i / rings)
            h = math.sqrt(R * R - rho * rho) - (R - sag)
            for k in range(segs):
                ph = 2 * math.pi * k / segs
                p, q = rho * math.cos(ph), rho * math.sin(ph)
                d = sgn * h
                if ax == "x":
                    v.append((c[0] + d, c[1] + p, c[2] + q))
                elif ax == "y":
                    v.append((c[0] + p, c[1] + d, c[2] + q))
                else:
                    v.append((c[0] + p, c[1] + q, c[2] + d))
        f = []
        for i in range(rings):
            for k in range(segs):
                j = (k + 1) % segs
                f.append([i * segs + k, i * segs + j, (i + 1) * segs + j, (i + 1) * segs + k])
        f.append([k for k in range(segs)][::-1])
        return self.add(v, f)

    def tube(self, pts, r, sides=6, radii=None):
        pts = [Vector(p) for p in pts]
        n = len(pts)
        if n < 2:
            return self
        v, f = [], []
        prev_side = None
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
            if t.length < 1e-12:
                t = Vector((1, 0, 0))
            t.normalize()
            if prev_side is None:
                ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
                side = t.cross(ref).normalized()
            else:
                side = (prev_side - t * prev_side.dot(t)).normalized()
            up = t.cross(side).normalized()
            prev_side = side
            rr = radii[i] if radii else r
            for k in range(sides):
                a = 2 * math.pi * k / sides
                v.append(tuple(p + (side * math.cos(a) + up * math.sin(a)) * rr))
        for i in range(n - 1):
            for k in range(sides):
                j = (k + 1) % sides
                f.append([i * sides + k, i * sides + j, (i + 1) * sides + j, (i + 1) * sides + k])
        f.append(list(range(sides))[::-1])
        f.append([(n - 1) * sides + k for k in range(sides)])
        return self.add(v, f)


# ================================================================== collections, materials
TOP = SCN.collection
COLL = {}
for cname in ("carrier", "pic", "wirebonds", "laser", "laser_optics", "output_optics", "eye", "beams_viz"):
    c = bpy.data.collections.new("tx_" + cname)
    TOP.children.link(c)
    COLL[cname] = c
ROOT = bpy.data.objects.new("OPTICS_TX_ROOT", None)
ROOT.empty_display_type = "PLAIN_AXES"
ROOT.empty_display_size = 0.003
TOP.objects.link(ROOT)
ROOT["asset_id"] = ASSET
ROOT["explode"] = [0.0, 0.0, 0.0]
ROOT["units"] = "meters, true scale; origin = centre of the 2x4 eye-plate outer surface (dome bases); +X = light out of the eye; +Z up"

MATS = {}


def mat(name, color, metal=0.0, rough=0.5, emit=0.0, ecol=None, alpha=1.0, coat=0.0, thin_film=0.0, ior=1.45):
    if name in MATS:
        return MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    if thin_film and "Thin Film Thickness" in b.inputs:
        b.inputs["Thin Film Thickness"].default_value = thin_film
        b.inputs["Thin Film IOR"].default_value = 1.9
    if emit:
        b.inputs["Emission Color"].default_value = (*(ecol or color), 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
        m.use_transparency_overlap = True
        m.use_backface_culling = False
    m.diffuse_color = (*color, alpha)
    MATS[name] = m
    return m


M = dict(
    carrier=mat("tx_carrier_laminate_black", (0.012, 0.014, 0.016), 0.0, 0.42, coat=0.15),
    carrier_core=mat("tx_carrier_core", (0.30, 0.28, 0.16), 0.0, 0.6),
    gold=mat("tx_gold", (1.0, 0.74, 0.32), 1.0, 0.32),
    gold_wire=mat("tx_gold_wire", (1.0, 0.78, 0.38), 1.0, 0.45),
    aln=mat("tx_aln_ceramic", (0.80, 0.80, 0.78), 0.0, 0.42),
    bench=mat("tx_aln_bench", (0.70, 0.71, 0.70), 0.0, 0.38),
    si_side=mat("tx_silicon_die_side", (0.10, 0.11, 0.14), 0.5, 0.25),
    inp=mat("tx_inp_die", (0.07, 0.065, 0.075), 0.4, 0.22),
    si_lens=mat("tx_si_lens_ar", (0.10, 0.10, 0.12), 0.35, 0.06, coat=0.6, thin_film=420.0),
    fused_silica=mat("tx_fused_silica", (0.78, 0.90, 0.98), 0.0, 0.03, alpha=0.28, coat=0.3, ior=1.447),
    bk7=mat("tx_bk7_glass", (0.78, 0.90, 0.98), 0.0, 0.03, alpha=0.28, coat=0.3, ior=1.504),
    eye_glass=mat("tx_eye_plate_glass", (0.95, 0.60, 0.55), 0.0, 0.03, alpha=0.18, coat=0.5, ior=1.504),
    eye_lens=mat("tx_eye_lens_red", (1.0, 0.10, 0.06), 0.0, 0.08, emit=1.2, alpha=0.75, coat=0.6, ior=1.504),
    polarizer=mat("tx_polarizer_glass", (0.05, 0.06, 0.07), 0.2, 0.12, alpha=0.85),
    garnet=mat("tx_garnet_faraday", (0.16, 0.12, 0.05), 0.2, 0.15, coat=0.4),
    epoxy=mat("tx_uv_epoxy", (0.95, 0.75, 0.30), 0.0, 0.2, alpha=0.45),
    pedestal=mat("tx_si_pedestal", (0.20, 0.21, 0.24), 0.6, 0.3),
    solder=mat("tx_solder_ausn", (0.80, 0.78, 0.70), 1.0, 0.3),
    facet_glow=mat("tx_dfb_facet_glow", (1.0, 0.08, 0.04), 0.0, 0.3, emit=12.0),
    beam_core=mat("tx_beam_core_red", (1.0, 0.08, 0.04), 0.0, 0.5, emit=8.0),
    beam_env=mat("tx_beam_envelope_red", (1.0, 0.12, 0.06), 0.0, 0.5, emit=1.6, alpha=0.10),
    via=mat("tx_via_ring", (0.85, 0.85, 0.82), 1.0, 0.25),
    quartz=mat("tx_quartz_waveplate", (0.85, 0.80, 0.95), 0.0, 0.05, alpha=0.5, coat=0.4, ior=1.54),
)

# PIC top-face material: layout albedo + packed metal/rough (G rough, B metal) for glTF
pm = bpy.data.materials.new("tx_pic_layout")
pm.use_nodes = True
nt = pm.node_tree
bsdf = nt.nodes["Principled BSDF"]
ti = nt.nodes.new("ShaderNodeTexImage")
ti.image = bpy.data.images.load(os.path.join(TEX, "pic_albedo.png"))
ti.interpolation = "Cubic"
tm = nt.nodes.new("ShaderNodeTexImage")
tm.image = bpy.data.images.load(os.path.join(TEX, "pic_mr.png"))
tm.image.colorspace_settings.name = "Non-Color"
sep = nt.nodes.new("ShaderNodeSeparateColor")
nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(tm.outputs["Color"], sep.inputs["Color"])
nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
M["pic_layout"] = pm

PARTS = []


def obj_from_mb(name, mb, material, cname, explode=(0, 0, 0), label=None, smooth=False, info=None, materials=None,
                face_mats=None, uvs=None, recalc=True):
    if not mb.v:
        raise ValueError("empty mesh " + name)
    xs, ys, zs = zip(*mb.v)
    org = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    verts = [((x - org[0]) * MM, (y - org[1]) * MM, (z - org[2]) * MM) for x, y, z in mb.v]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], mb.f)
    if recalc:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    COLL[cname].objects.link(ob)
    ob.parent = ROOT
    ob.location = (org[0] * MM, org[1] * MM, org[2] * MM)
    for m_ in (materials or [material]):
        me.materials.append(m_)
    if face_mats:
        for p, mi in zip(me.polygons, face_mats):
            p.material_index = mi
    if uvs:
        uvl = me.uv_layers.new(name="UVMap")
        for p in me.polygons:
            for li in p.loop_indices:
                vi = me.loops[li].vertex_index
                uvl.data[li].uv = uvs.get(vi, (0.0, 0.0))
    for p in me.polygons:
        p.use_smooth = smooth
    ob["explode"] = [e * MM for e in explode]
    ob["group"] = cname
    if label:
        ob["label"] = label
    rec = dict(name=name, group=cname, label=label or "", explode_mm=list(explode),
               bbox_mm=[round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4), round(max(zs) - min(zs), 4)],
               center_mm=[round(c, 4) for c in org], tris=sum(len(f) - 2 for f in mb.f))
    if info:
        rec.update(info)
        for k_, v_ in info.items():
            if isinstance(v_, (int, float, str)):
                ob[k_] = v_
    PARTS.append(rec)
    return ob


HOOKS = []


def hook(name, loc, direction, doc):
    h = bpy.data.objects.new(name, None)
    h.empty_display_type = "ARROWS"
    h.empty_display_size = 0.0015
    TOP.objects.link(h)
    h.parent = ROOT
    h.location = tuple(c * MM for c in loc)
    d = Vector(direction).normalized()
    h.rotation_mode = "QUATERNION"
    # local +X = hook direction (interface); local +Z kept as close to world +Z as possible (world +Y if d is vertical)
    h.rotation_quaternion = d.to_track_quat("X", "Y" if abs(d.z) > 0.9 else "Z")
    h["doc"] = doc
    h["direction"] = list(d)
    h["explode"] = [0.0, 0.0, 0.0]
    HOOKS.append(dict(name=name, location_mm=list(loc), direction=list(d), local_axis="+X", doc=doc))
    return h


def bond(mb, p0, p1, h=0.15, r=0.0125, n=14, up=(1, 0, 0)):
    """Ball-wedge wire bond from p0 (ball) to p1 (wedge); loop rises along `up` (PIC face normal)."""
    p0, p1, u = Vector(p0), Vector(p1), Vector(up)
    pts = []
    for i in range(n + 1):
        t = i / n
        base = p0.lerp(p1, t)
        if t < 0.8:
            lift = h * math.sin(math.pi * t / 1.6) ** 0.55
        else:
            lift = h * (1 - (t - 0.8) / 0.2) ** 1.3
        pts.append(base + u * lift)
    pts[0] = p0
    mb.tube(pts, r, sides=6)
    BONDS.append(pts)
    return pts


BONDS = []
EX = dict(
    carrier=(0, 0, 0), pic=(1.6, 0, 0), bonds=(0.8, 0, 0), bench=(0.8, 2.6, 0), laser_z=0.8,
    sub=(2.0, 6.2, 0), dfb=(2.8, 6.2, 0), lens1=(3.6, 5.4, 0), iso=(3.6, 4.6, 0), lens2=(3.6, 3.8, 0),
    coll=(3.0, 0.4, 0), lid=(3.4, -1.4, 0), pedestal=(2.4, 1.6, 0), rails=(2.4, 1.6, 0), stair=(4.4, 1.6, 0),
    window=(5.6, 1.6, 0), rhombi=(6.8, 1.6, 0), eye=(9.0, 1.6, 0),
)

# ================================================================== CARRIER (laminate) + traces, pads, vias
CY0, CY1 = g.Y_BACK - 1.60, g.BENCH["y1"] + 0.30
CZ = 4.60
CAV = dict(y0=g.Y_BACK - 0.06, y1=g.BENCH["y1"] + 0.06, z=3.46)          # D cavity for PIC + bench
mb = MB().box(g.X_CAR0, g.X_CAR1, CY0, CY1, -CZ, CZ)
mb.box(g.X_CAR1, g.X_CTOP, CY0, CAV["y0"], -CZ, CZ)
mb.box(g.X_CAR1, g.X_CTOP, CAV["y1"], CY1, -CZ, CZ)
mb.box(g.X_CAR1, g.X_CTOP, CAV["y0"], CAV["y1"], -CZ, -CAV["z"])
mb.box(g.X_CAR1, g.X_CTOP, CAV["y0"], CAV["y1"], CAV["z"], CZ)
BOXES = [("carrier_wall_back", (g.X_CAR1, g.X_CTOP, CY0, CAV["y0"], -CZ, CZ)),
         ("carrier_wall_front", (g.X_CAR1, g.X_CTOP, CAV["y1"], CY1, -CZ, CZ)),
         ("carrier_wall_-z", (g.X_CAR1, g.X_CTOP, CAV["y0"], CAV["y1"], -CZ, -CAV["z"])),
         ("carrier_wall_+z", (g.X_CAR1, g.X_CTOP, CAV["y0"], CAV["y1"], CAV["z"], CZ))]
obj_from_mb("tx_carrier", mb, M["carrier"], "carrier", EX["carrier"], label="carrier (laminate)",
            info=dict(grade="D", note="laminate with a cavity for the PIC and bench so bond pads are level (Cisco PIC photo shows a cavity)"))
ZC1 = g.X_CTOP                       # carrier top surface outside the cavity (subassembly X)
T_TR = 0.018                         # D trace thickness (ENIG over Cu)
tr = MB()
pads_c = MB()
vias = MB()
RF_CPAD_Y = g.Y_BACK - 0.20
FLEX_RF_Y = (CY0, CY0 + 0.45)
rf_nets = []
for j, zj in enumerate(g.RF_PAD_U):
    kind = g.RF_PAD_KIND[j]
    pads_c.box(ZC1, ZC1 + T_TR, RF_CPAD_Y - 0.10, RF_CPAD_Y + 0.10, zj - 0.08, zj + 0.08)
    w = 0.09 if kind == "S" else 0.16
    # differential S pairs pulled together (coupled lines) between the bond pad and the flex pad
    if kind == "S":
        lane = (j - 1) // 3
        mid = (g.RF_PAD_U[3 * lane + 1] + g.RF_PAD_U[3 * lane + 2]) / 2
        zc = mid + (-0.075 if j % 3 == 1 else 0.075)
    else:
        zc = zj
    path = [(RF_CPAD_Y - 0.08, zj), (RF_CPAD_Y - 0.30, zc), (FLEX_RF_Y[1] + 0.25, zc), (FLEX_RF_Y[1], zj)]
    for (ya, za), (yb, zb) in zip(path[:-1], path[1:]):
        dy, dz = yb - ya, zb - za
        L = math.hypot(dy, dz)
        ny, nz = -dz / L * w / 2, dy / L * w / 2
        tr.prism([(ya + ny, za + nz), (yb + ny, zb + nz), (yb - ny, zb - nz), (ya - ny, za - nz)], ZC1, ZC1 + T_TR, "yz")
    pads_c.box(ZC1, ZC1 + T_TR, FLEX_RF_Y[0] + 0.02, FLEX_RF_Y[1], zj - 0.08, zj + 0.08)
    lane = j // 3
    rf_nets.append(dict(pad=j, kind=kind, z_mm=round(zj, 4),
                        net=("GND" if kind == "G" else "TX%d%s" % ((j - 1) // 3 + 1, "p" if j % 3 == 1 else "n"))))
# DC: PIC side pads -> carrier pads at |Z| 3.85 -> bottom-edge DC field (-Z side direct, +Z side via inner layer)
asg = g.dc_assign()
DC_CPAD_Z = 3.80
FIELD_Z = (-CZ, -CZ + 0.40)
dc_nets = []
field = []
for s in (-1, 1):
    for d in asg[s]:
        y = g.Y_EDGE - g.DC_PAD_V[d["m"]]
        zc = s * DC_CPAD_Z
        pads_c.box(ZC1, ZC1 + T_TR, y - 0.065, y + 0.065, zc - 0.10, zc + 0.10)
        if s < 0:
            tr.box(ZC1, ZC1 + T_TR, y - 0.03, y + 0.03, FIELD_Z[1], zc)
            fy = y
        else:
            tr.box(ZC1, ZC1 + T_TR, y - 0.03, y + 0.03, zc + 0.10, zc + 0.32)
            vias.cyl((ZC1 + T_TR / 2, y, zc + 0.36), 0.05, T_TR * 1.4, axis="x", segs=12)
            fy = g.Y_EDGE - 5.70 - 0.17 * d["m"]
            vias.cyl((ZC1 + T_TR / 2, fy, FIELD_Z[1] + 0.12), 0.05, T_TR * 1.4, axis="x", segs=12)
        field.append(fy)
        pads_c.box(ZC1, ZC1 + T_TR, fy - 0.06, fy + 0.06, FIELD_Z[0] + 0.02, FIELD_Z[1])
        dc_nets.append(dict(net=d["net"], side="-Z" if s < 0 else "+Z", pic_pad_v_mm=round(g.DC_PAD_V[d["m"]], 3),
                            field_y_mm=round(fy, 3)))
# laser DC: bench edge pads -> carrier pads -> field (LD1 on -Z direct, LD2 on +Z via)
LD_NETS = []
for gi, zl in enumerate(g.LASER_U):
    s = 1 if zl > 0 else -1
    for ni, net in enumerate(("LD%d+" % (gi + 1), "LD%d-" % (gi + 1))):
        y = g.TRAIN["sub0"] + 0.35 + 0.55 * ni
        zc = s * DC_CPAD_Z
        pads_c.box(ZC1, ZC1 + T_TR, y - 0.10, y + 0.10, zc - 0.12, zc + 0.12)
        fy = g.Y_EDGE - 2.40 - 0.20 * (2 * gi + ni)
        if s < 0:
            tr.box(ZC1, ZC1 + T_TR, y - 0.04, y + 0.04, -4.10 - 0.1 * ni, zc)
            tr.box(ZC1, ZC1 + T_TR, fy - 0.04, y + 0.04, -4.10 - 0.1 * ni - 0.04, -4.10 - 0.1 * ni + 0.04)
            tr.box(ZC1, ZC1 + T_TR, fy - 0.04, fy + 0.04, FIELD_Z[1], -4.10 - 0.1 * ni)
        else:
            tr.box(ZC1, ZC1 + T_TR, y - 0.04, y + 0.04, zc + 0.12, zc + 0.34)
            vias.cyl((ZC1 + T_TR / 2, y, zc + 0.38), 0.05, T_TR * 1.4, axis="x", segs=12)
            vias.cyl((ZC1 + T_TR / 2, fy, FIELD_Z[1] + 0.12), 0.05, T_TR * 1.4, axis="x", segs=12)
        pads_c.box(ZC1, ZC1 + T_TR, fy - 0.07, fy + 0.07, FIELD_Z[0] + 0.02, FIELD_Z[1])
        field.append(fy)
        LD_NETS.append(dict(net=net, carrier_pad_y_mm=round(y, 3), field_y_mm=round(fy, 3)))
obj_from_mb("tx_carrier_traces", tr, M["gold"], "carrier", EX["carrier"], label="carrier routing (RF pairs + DC)",
            info=dict(grade="D"))
obj_from_mb("tx_carrier_pads", pads_c, M["gold"], "carrier", EX["carrier"], info=dict(grade="D"))
obj_from_mb("tx_carrier_vias", vias, M["via"], "carrier", EX["carrier"], info=dict(grade="D", note="inner-layer routing implied"))

# ================================================================== PIC die (textured top) + raised pads
x0, x1 = g.X_PIC0, g.X_W
y0, y1 = g.Y_BACK, g.Y_EDGE
z0, z1 = -g.HALF_W, g.HALF_W
mb = MB()
# top face (X = x1) with UVs: image x = u (= Z) left->right, image row 0 = optical edge (v = 0, Y = Y_EDGE)
mb.add([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], [[0, 1, 2, 3]])
uvs = {0: (0.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 0.0)}
mb.face([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0))
mb.face([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0))
mb.face([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0))
mb.face([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1))
mb.face([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1))
pic = obj_from_mb("tx_pic_siph_die", mb, None, "pic", EX["pic"], label="TX SiPh PIC (layout texture)",
                  materials=[M["pic_layout"], M["si_side"]], face_mats=[0, 1, 1, 1, 1, 1], uvs=uvs, recalc=False,
                  info=dict(grade="B (7.0 x 7.5, Carmel8) / D (0.75 thick)", size_mm="7.0 x 7.5 x 0.75",
                            texture="textures/pic_albedo.png 4200x4500 (600 px/mm)"))
# raised bond-pad metal (geometry for parallax; positions = texture)
_, _, rfpads = g.rf_metal()
dcr = g.dc_routing()
pp = MB()
PT = 0.004
for (u0, v0, u1, v1) in rfpads:
    pp.box(x1, x1 + PT, g.Y_EDGE - v1, g.Y_EDGE - v0, u0, u1)
for (u0, v0, u1, v1, net) in dcr["pads"]:
    pp.box(x1, x1 + PT, g.Y_EDGE - v1, g.Y_EDGE - v0, u0, u1)
obj_from_mb("tx_pic_bond_pads", pp, M["gold"], "pic", EX["pic"], info=dict(grade="D", note="25 RF (GSSG, shared G) + 28 DC"))
# die-attach epoxy line under the PIC
mb = MB().box(g.X_CAR1, g.X_PIC0, y0 + 0.05, y1 - 0.05, z0 + 0.05, z1 - 0.05)
obj_from_mb("tx_pic_die_attach", mb, M["solder"], "pic", EX["pic"], info=dict(grade="D"))
BOXES.append(("pic_die", (x0, x1, y0, y1, z0, z1)))

# ================================================================== LASER BENCH, CoS submounts, DFBs, laser bonds
B = g.BENCH
mb = MB().box(g.X_CAR1, g.X_BENCH1, B["y0"], B["y1"], -B["z"], B["z"])
BOXES.append(("bench", (g.X_CAR1, g.X_BENCH1, B["y0"], B["y1"], -B["z"], B["z"])))
obj_from_mb("tx_laser_bench_aln", mb, M["bench"], "laser", EX["bench"], label="laser/optics bench (AlN)",
            info=dict(grade="D", note="laser sub-block (PSM8 photo ~9.2 x 4.9 mm, C) reduced to the parts it carries"))
bp = MB()                                           # bench gold traces / pads
for zl in g.LASER_U:
    s = 1 if zl > 0 else -1
    for ni in range(2):
        y = g.TRAIN["sub0"] + 0.35 + 0.55 * ni
        bp.box(g.X_BENCH1, g.X_BENCH1 + T_TR, y - 0.08, y + 0.08, s * 2.98, s * 3.30)          # edge pad
        bp.box(g.X_BENCH1, g.X_BENCH1 + T_TR, y - 0.035, y + 0.035, s * (abs(zl) + g.SUB["W"] / 2 + 0.06), s * 3.0)
        bp.box(g.X_BENCH1, g.X_BENCH1 + T_TR, y - 0.09, y + 0.09, s * (abs(zl) + g.SUB["W"] / 2 + 0.04),
               s * (abs(zl) + g.SUB["W"] / 2 + 0.22))
    # bench fiducial crosses (D)
    for yy in (B["y0"] + 0.25, B["y1"] - 0.25):
        bp.box(g.X_BENCH1, g.X_BENCH1 + T_TR, yy - 0.10, yy + 0.10, s * 3.15 - 0.012, s * 3.15 + 0.012)
        bp.box(g.X_BENCH1, g.X_BENCH1 + T_TR, yy - 0.012, yy + 0.012, s * 3.15 - 0.10, s * 3.15 + 0.10)
obj_from_mb("tx_laser_bench_traces", bp, M["gold"], "laser", EX["bench"], info=dict(grade="D"))

T = g.TRAIN
L1, L2, I = g.LENS1, g.LENS2, g.ISO
s1 = L1["R"] - math.sqrt(L1["R"] ** 2 - (L1["CA"] / 2) ** 2)
s2 = L2["R"] - math.sqrt(L2["R"] ** 2 - (L2["CA"] / 2) ** 2)


def lex(key, s):
    """Per-laser explode: base offset + outward Z spread (s = +/-1)."""
    e = EX[key]
    return (e[0], e[1], s * EX["laser_z"])


for gi, zl in enumerate(g.LASER_U):
    s = 1 if zl > 0 else -1
    tag = "LD%d" % (gi + 1)
    sub, die, sp, glow, lb = MB(), MB(), MB(), MB(), MB()
    sub.box(g.X_BENCH1, g.X_SUB1, T["sub0"], T["sub1"], zl - g.SUB["W"] / 2, zl + g.SUB["W"] / 2)
    # submount metallization: die pad (LD-) wrapping to the outer LD- pad, LD+ pad on the outer side
    sp.box(g.X_SUB1, g.X_SUB1 + 0.006, T["dfb_facet"] - 0.01, T["dfb_back"] + 0.12, zl - 0.20, zl + 0.20)
    sp.box(g.X_SUB1, g.X_SUB1 + 0.006, T["sub0"] + 0.25, T["sub0"] + 0.62, zl + s * 0.22, zl + s * 0.38)
    sp.box(g.X_SUB1, g.X_SUB1 + 0.006, T["sub0"] + 0.78, T["sub1"] - 0.08, zl + s * 0.22, zl + s * 0.38)
    sp.box(g.X_SUB1, g.X_SUB1 + 0.006, T["dfb_back"] + 0.05, T["dfb_back"] + 0.12, zl, zl + s * 0.30)
    die.box(g.X_SUB1 + 0.006, g.X_W, T["dfb_facet"], T["dfb_back"], zl - g.DFB["W"] / 2, zl + g.DFB["W"] / 2)
    sp.box(g.X_W, g.X_W + 0.004, T["dfb_facet"] + 0.08, T["dfb_back"] - 0.05, zl - 0.012, zl + 0.012)        # ridge contact
    sp.box(g.X_W, g.X_W + 0.004, T["dfb_facet"] + 0.20, T["dfb_back"] - 0.30, zl - 0.11, zl + 0.11)          # p pad
    glow.ellipsoid((g.X_W - 0.004, T["dfb_facet"] - 0.004, zl), 0.006, 0.012, 0.018, segs=10, rings=5)
    # LD+ : 4 parallel bonds die p-pad -> submount LD+ pad (Cisco photo shows ~5 parallel bonds per pad, C)
    for q in range(4):
        yq = T["dfb_facet"] + 0.26 + q * 0.09
        lb.ellipsoid((g.X_W + 0.012, yq, zl + s * 0.05), 0.02, 0.02, 0.02, segs=8, rings=4)
        bond(lb, (g.X_W + 0.006, yq, zl + s * 0.05), (g.X_SUB1 + 0.008, yq - 0.02, zl + s * 0.30), h=0.12)
    # submount -> bench pads (2 bonds per net), bench edge -> carrier pads (2 bonds per net)
    for ni in range(2):
        y = T["sub0"] + 0.35 + 0.55 * ni
        zs = zl + s * 0.30
        for q in (-0.04, 0.04):
            bond(lb, (g.X_SUB1 + 0.008, y + q, zs), (g.X_BENCH1 + T_TR, y + q, s * (abs(zl) + g.SUB["W"] / 2 + 0.15)), h=0.10)
            bond(lb, (g.X_BENCH1 + T_TR, y + q, s * 3.18), (ZC1 + T_TR, y + q, s * DC_CPAD_Z), h=0.42)
    obj_from_mb("tx_dfb_submount_aln_" + tag, sub, M["aln"], "laser", lex("sub", s),
                label="AlN chip-on-submount" if gi == 0 else None, info=dict(grade="D", size_mm="1.4 x 0.8 x 0.25"))
    obj_from_mb("tx_dfb_submount_metal_" + tag, sp, M["gold"], "laser", lex("sub", s), info=dict(grade="D"))
    obj_from_mb("tx_dfb_laser_die_" + tag, die, M["inp"], "laser", lex("dfb", s),
                label="CW DFB (1311 nm)" if gi == 0 else None,
                info=dict(grade="B/D", size_mm="1.0 x 0.3 x 0.1", note="1 mm cavity (Casela class); p-side up"))
    obj_from_mb("tx_dfb_facet_glow_" + tag, glow, M["facet_glow"], "laser", lex("dfb", s), smooth=True, info=dict(grade="viz"))
    obj_from_mb("tx_laser_wirebonds_dc_" + tag, lb, M["gold_wire"], "wirebonds", lex("sub", s),
                label="laser bias bonds (LS)" if gi == 0 else None, info=dict(grade="C/D", wire="25 um Au"))
    # micro-optics: lens 1 (collimator, flat toward the DFB), isolator, lens 2 (focuser, convex toward the beam)
    l1, l2, iso_p, iso_g, ep = MB(), MB(), MB(), MB(), MB()
    l1.box(g.X_W - L1["W"] / 2, g.X_W + L1["W"] / 2, T["lens1_apex"] + s1, T["lens1_flat"], zl - L1["L"] / 2, zl + L1["L"] / 2)
    l1.cap((g.X_W, T["lens1_apex"] + s1, zl), "-y", L1["R"], L1["CA"] / 2)
    l2.box(g.X_W - L2["W"] / 2, g.X_W + L2["W"] / 2, T["lens2_flat"], T["lens2_apex"] - s2, zl - L2["L"] / 2, zl + L2["L"] / 2)
    l2.cap((g.X_W, T["lens2_apex"] - s2, zl), "y", L2["R"], L2["CA"] / 2)
    # light travels -Y: input polarizer (iso1 side) -> latching garnet (45 deg) -> output polarizer -> half-wave plate
    # (rotates back to TE for the SSC)
    hw = MB()
    iso_p.box(g.X_W - I["H"] / 2, g.X_W + I["H"] / 2, T["iso1"] - I["pol"], T["iso1"], zl - I["W"] / 2, zl + I["W"] / 2)
    iso_p.box(g.X_W - I["H"] / 2, g.X_W + I["H"] / 2, T["iso0"] + I["hwp"], T["iso0"] + I["hwp"] + I["pol"],
              zl - I["W"] / 2, zl + I["W"] / 2)
    hw.box(g.X_W - I["H"] / 2, g.X_W + I["H"] / 2, T["iso0"], T["iso0"] + I["hwp"], zl - I["W"] / 2, zl + I["W"] / 2)
    iso_g.box(g.X_W - I["H"] / 2 + 0.01, g.X_W + I["H"] / 2 - 0.01, T["iso0"] + I["hwp"] + I["pol"], T["iso1"] - I["pol"],
              zl - I["W"] / 2 + 0.01, zl + I["W"] / 2 - 0.01)
    for (ya, yb) in ((T["lens1_apex"], T["lens1_flat"]), (T["lens2_flat"], T["lens2_apex"]), (T["iso0"], T["iso1"])):
        ep.box(g.X_BENCH1, g.X_W - 0.30, ya + 0.02, yb - 0.02, zl - 0.45, zl + 0.45)
    obj_from_mb("tx_si_lens1_collimator_" + tag, l1, M["si_lens"], "laser_optics", lex("lens1", s),
                label="Si lens 1 (collimator)" if gi == 0 else None,
                info=dict(grade="A", part=L1["part"], size_mm="1.2 x 0.6 x 0.5", EFL_mm=0.34, WD_mm=0.194, CA_mm=0.5))
    obj_from_mb("tx_si_lens2_focuser_" + tag, l2, M["si_lens"], "laser_optics", lex("lens2", s),
                label="Si lens 2 (focuser)" if gi == 0 else None,
                info=dict(grade="B", part=L2["part"], size_mm="1.0 x 0.6 x 0.5", EFL_mm=0.45, WD_mm=0.30, CA_mm=0.5))
    obj_from_mb("tx_isolator_polarizers_" + tag, iso_p, M["polarizer"], "laser_optics", lex("iso", s),
                info=dict(grade="D", optional=1, note="polarizer / Faraday garnet / polarizer (US 9,787,054); none documented for DR8"))
    obj_from_mb("tx_isolator_garnet_" + tag, iso_g, M["garnet"], "laser_optics", lex("iso", s),
                label="micro isolator (optional)" if gi == 0 else None,
                info=dict(grade="C/D", optional=1, size_mm="0.6 x 0.6 x 0.55",
                          note="Cisco 400G DR4 photo shows isolators; latching garnet assumed (no magnet)"))
    obj_from_mb("tx_isolator_halfwave_plate_" + tag, hw, M["quartz"], "laser_optics", lex("iso", s),
                info=dict(grade="D", optional=1, note="half-wave plate rotates the 45 deg isolator output back to TE"))
    obj_from_mb("tx_laser_optics_epoxy_" + tag, ep, M["epoxy"], "laser_optics", lex("iso", s), info=dict(grade="D"))

# ================================================================== PIC wire bonds: HS differential GSSG + DC
hs, dcb, balls = MB(), MB(), MB()
yp = g.Y_EDGE - (g.V_RF_PAD[0] + g.V_RF_PAD[1]) / 2
for j, zj in enumerate(g.RF_PAD_U):
    offs = (-0.035, 0.035) if g.RF_PAD_KIND[j] == "G" else (0.0,)
    for o in offs:
        p0 = (g.X_W + PT, yp + 0.02, zj + o)
        p1 = (ZC1 + T_TR, RF_CPAD_Y + 0.02, zj + o)
        bond(hs, p0, p1, h=0.13)
        balls.ellipsoid((g.X_W + PT + 0.012, yp + 0.02, zj + o), 0.024, 0.024, 0.024, segs=8, rings=4)
for (u0, v0, u1, v1, net) in dcr["pads"]:
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    s = 1 if uc > 0 else -1
    p0 = (g.X_W + PT, g.Y_EDGE - vc, uc)
    p1 = (ZC1 + T_TR, g.Y_EDGE - vc, s * (DC_CPAD_Z - 0.02))
    bond(dcb, p0, p1, h=0.16)
    balls.ellipsoid((g.X_W + PT + 0.012, g.Y_EDGE - vc, uc), 0.024, 0.024, 0.024, segs=8, rings=4)
obj_from_mb("tx_wirebonds_hs_diff_gssg", hs, M["gold_wire"], "wirebonds", EX["bonds"],
            label="HS differential bonds (GSSG)", info=dict(grade="D", wire="25 um Au", count=34,
                                                            note="16 signal (8 pairs) + 9 G pads x 2 wires; ~0.4 mm span, pads level (carrier cavity)"))
obj_from_mb("tx_wirebonds_dc_ls", dcb, M["gold_wire"], "wirebonds", EX["bonds"], label="DC bonds (heaters, MPD)",
            info=dict(grade="D", wire="25 um Au", count=28))
obj_from_mb("tx_wirebond_balls", balls, M["gold_wire"], "wirebonds", EX["bonds"], smooth=True, info=dict(grade="D"))

# ================================================================== OUTPUT OPTICS
GD = g.gaussian_design()
C_ = g.COLL
# FAU-attach collimator block (fused silica) bonded to the edge facet + 1x8 lens array on its exit face
mb = MB().box(C_["x0"], C_["x1"], C_["y0"] + 0.01, C_["y0"] + C_["L"], -C_["z"], C_["z"])
obj_from_mb("tx_fau_collimator_block", mb, M["fused_silica"], "output_optics", EX["coll"],
            label="FAU-attach collimator block", info=dict(grade="D", size_mm="0.63 x 0.80 x 3.2", n=g.N_FS,
                                                           note="bonded to the PIC facet with index-matched UV epoxy"))
la = MB()
for z in [(i - 5.5) * g.EDGE_PITCH for i in range(12)]:
    la.cap((g.X_W, C_["y0"] + C_["L"], z), "y", GD["coll_lens_R_mm"], 0.11, segs=20, rings=4)
obj_from_mb("tx_fau_lens_array_1x8", la, M["fused_silica"], "output_optics", EX["coll"], smooth=True,
            label="1x12 collimating lens array (8 lanes + loopbacks)", info=dict(grade="D", pitch_mm=0.25, R_mm=round(GD["coll_lens_R_mm"], 4)))
mb = MB().box(g.X_W, g.X_W + 0.30, C_["y0"] - C_["lid_l"], C_["y0"], -C_["z"], C_["z"])
obj_from_mb("tx_pic_edge_lid", mb, M["fused_silica"], "output_optics", EX["lid"], label="PIC edge glass lid",
            info=dict(grade="B/D", note="silica cover lid at the edge coupler (US 9,787,054), enlarges the FAU bond face"))
fe = MB()
fe.box(C_["x0"], g.X_W + 0.30, C_["y0"], C_["y0"] + 0.01, -C_["z"], C_["z"])                 # facet bond line
fe.box(g.X_W + 0.30, g.X_W + 0.33, C_["y0"] - 0.25, C_["y0"] + 0.25, -C_["z"], C_["z"])       # bead over the joint
obj_from_mb("tx_fau_uv_epoxy_fillets", fe, M["epoxy"], "output_optics", EX["coll"], info=dict(grade="D"))

# staircase 45 deg TIR turning prism (fused silica): one facet per lane band, at the lane's target column
S = g.STAIR
LM = g.lane_map()
st = MB()
xb, xt, ys = S["xb"], S["xt"], S["y0"]
bands = []
for d in LM:
    zc = d["z_in"]
    bands.append((zc - 0.125, zc + 0.125, d["y_out"]))
for i, (za, zb, ya) in enumerate(bands):
    yf0, yf1 = ya + (xb - g.X_W), ya + (xt - g.X_W)       # facet ends at xb / xt (45 deg)
    zm = (za + zb) / 2
    st.face([(xb, ys, za), (xb, yf0, za), (xb, yf0, zb), (xb, ys, zb)], (-1, 0, 0))                  # bottom
    st.face([(xt, ys, za), (xt, yf1, za), (xt, yf1, zb), (xt, ys, zb)], (1, 0, 0))                   # top (exit)
    st.face([(xb, ys, za), (xt, ys, za), (xt, ys, zb), (xb, ys, zb)], (0, -1, 0))                    # entrance
    st.face([(xb, yf0, za), (xt, yf1, za), (xt, yf1, zb), (xb, yf0, zb)], (-1, 1, 0))                # TIR facet
    prof = [(xb, ys), (xb, yf0), (xt, yf1), (xt, ys)]
    if i == 0:
        st.face([(p, q, za) for p, q in prof], (0, 0, -1))
    if i == len(bands) - 1:
        st.face([(p, q, zb) for p, q in prof], (0, 0, 1))
    if i < len(bands) - 1:
        yb_next = bands[i + 1][2]
        if abs(yb_next - ya) > 1e-9:
            lo, hi = sorted((ya, yb_next))
            out_dir = (0, 0, 1) if ya > yb_next else (0, 0, -1)
            st.face([(xb, lo + (xb - g.X_W), zb), (xb, hi + (xb - g.X_W), zb), (xt, hi + (xt - g.X_W), zb),
                     (xt, lo + (xt - g.X_W), zb)], out_dir)
obj_from_mb("tx_staircase_tir_prism", st, M["bk7"], "output_optics", EX["stair"],
            label="45 deg staircase TIR prism (1x8 -> columns)", recalc=False,
            info=dict(grade="D", n=g.N_BK7, note="8 bands x 0.25 mm; facet per lane at its eye column; N-BK7 class, TIR 45 > 41.7 deg"))
# pedestal under the staircase (silicon spacer on the bench)
ymax = max(d["y_out"] for d in LM) + (xt - g.X_W)
mb = MB().box(g.X_BENCH1 + 0.01, xb, ys, ymax - (xt - xb) + 0.05, -1.0, 1.0)
obj_from_mb("tx_staircase_pedestal", mb, M["fused_silica"], "output_optics", EX["pedestal"], info=dict(grade="D"))
# window plate on the staircase top, carries the rhomboids
mb = MB().box(xt, xt + 0.05, ys, ymax + 0.12, -1.05, 1.05)
obj_from_mb("tx_rhomboid_window_plate", mb, M["fused_silica"], "output_optics", EX["window"], info=dict(grade="D"))
# row-shift rhomboid micro-prisms (N-BK7 class)
R_ = g.RH
rh = MB()
n_rh = 0
for d in LM:
    dz = d["dz"]
    if abs(dz) < 1e-9:
        continue
    n_rh += 1
    s_ = R_["s"]
    c = R_["x0"] + s_ / 2
    z = d["z_in"]
    sg = 1 if dz > 0 else -1
    A = abs(dz)
    prof = [(c - s_ / 2, z - sg * s_ / 2), (c + s_ / 2, z + sg * s_ / 2), (c + s_ / 2, z + sg * (A + s_ / 2)),
            (c - s_ / 2, z + sg * (A - s_ / 2))]
    rh.prism(prof, d["y_out"] - 0.11, d["y_out"] + 0.11, "xz")
obj_from_mb("tx_row_shift_rhomboids", rh, M["bk7"], "output_optics", EX["rhombi"], label="row-shift rhomboids x%d" % n_rh,
            info=dict(grade="D", n=g.N_BK7, count=n_rh, aperture_mm=g.RH["s"]))
# eye plate support posts (4 slim AlN posts at the plate corners, clear of the beams and the staircase)
mb = MB()
for (ya, yb) in ((-1.30, -1.14), (1.42, 1.58)):
    for sz in (-1, 1):
        mb.box(g.X_BENCH1 + 0.01, -g.EYE["t"], ya, yb, sz * 1.15, sz * 1.29)
obj_from_mb("tx_eye_plate_posts", mb, M["aln"], "output_optics", EX["rails"], info=dict(grade="D"))

# ================================================================== EYE PLATE: 2x4 Galilean expanders (concave in, convex out)
E = g.EYE
EH = 2.60
plate = MB().box(-E["t"], 0.0, -E["W"] / 2, E["W"] / 2, -EH / 2, EH / 2)
po = obj_from_mb("tx_eye_plate_2x4", plate, M["eye_glass"], "eye", EX["eye"], label="2x4 eye plate (Galilean x3.5)",
                 info=dict(grade="D", size_mm="1.2 x 3.2 x 2.6", n=g.N_BK7, M=E["M"],
                           R_in_mm=round(GD["eye_R_in_concave_mm"], 4), R_out_mm=round(GD["eye_R_out_convex_mm"], 4)))
# concave inner lenslets: boolean difference with spheres
Rin = GD["eye_R_in_concave_mm"]
sag_in = Rin - math.sqrt(Rin * Rin - E["a_in"] ** 2)
cutters = []
for c_y in g.COLS_Y:
    for r_z in g.ROWS_Z:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=Rin * MM,
                                             location=((-E["t"] - Rin + sag_in) * MM, c_y * MM, r_z * MM))
        cutters.append(bpy.context.active_object)
for cobj in cutters[1:]:
    cobj.select_set(True)
bpy.context.view_layer.objects.active = cutters[0]
for cobj in cutters:
    cobj.select_set(True)
bpy.ops.object.join()
cut = bpy.context.active_object
md = po.modifiers.new("concave_lenslets", "BOOLEAN")
md.operation = "DIFFERENCE"
md.solver = "EXACT"
md.object = cut
bpy.context.view_layer.objects.active = po
bpy.ops.object.modifier_apply(modifier=md.name)
bpy.data.objects.remove(cut, do_unlink=True)
for p in po.data.polygons:
    p.use_smooth = False
PARTS[-1]["tris"] = sum(len(p.vertices) - 2 for p in po.data.polygons)
dm = MB()
for c_y in g.COLS_Y:
    for r_z in g.ROWS_Z:
        dm.cap((0.0, c_y, r_z), "x", GD["eye_R_out_convex_mm"], E["a_out"], segs=32, rings=8)
obj_from_mb("tx_eye_lenslets_out_2x4", dm, M["eye_lens"], "eye", EX["eye"], smooth=True, label="2x4 output lenslets",
            info=dict(grade="D", pitch_mm=0.75, aperture_mm=2 * E["a_out"]))

# ================================================================== BEAMS (viz only; hidden in background shots)
core, env = MB(), MB()
lane_cores = []
lane_paths = []
for d in LM:
    z, yc, zo, dz = d["z_in"], d["y_out"], d["z_out"], d["dz"]
    c = R_["x0"] + R_["s"] / 2
    pts = [(g.X_W, g.Y_EDGE, z), (g.X_W, C_["y0"] + C_["L"], z), (g.X_W, yc, z)]
    if abs(dz) > 1e-9:
        pts += [(c, yc, z), (c, yc, zo)]
    pts += [(-E["t"], yc, zo), (0.0, yc, zo), (1.0, yc, zo)]
    lane_paths.append(dict(lane=d["name"], points_mm=[[round(a, 4) for a in p] for p in pts]))
    lc = MB().tube(pts, 0.006, sides=6)
    lane_cores.append((d, lc))
    # envelopes: SSC -> lens (diverging), collimated run, expansion in the eye plate, output beam
    ylens = C_["y0"] + C_["L"]
    env.tube([(g.X_W, g.Y_EDGE, z), (g.X_W, ylens, z)], 0, sides=10, radii=[0.004, GD["collimated_w_mm"]])
    run = pts[1:-2]
    env.tube(run, GD["collimated_w_mm"], sides=10)
    env.tube([(-E["t"], yc, zo), (0.0, yc, zo)], 0, sides=12, radii=[GD["w_at_eye_inner_mm"], GD["eye_out_w_mm"]])
    env.tube([(0.0, yc, zo), (1.0, yc, zo)], GD["eye_out_w_mm"], sides=12)
for zl in g.LASER_U:
    pts = [(g.X_W, T["dfb_facet"], zl), (g.X_W, T["lens1_apex"], zl), (g.X_W, T["lens2_apex"], zl), (g.X_W, g.Y_EDGE, zl),
           (g.X_W, g.Y_EDGE - 0.6, zl)]
    core.tube(pts[:4], 0.006, sides=6)
    wl = GD["laser_collimated_w_mm"]
    env.tube([(g.X_W, T["dfb_facet"], zl), (g.X_W, T["lens1_flat"] - 0.25, zl)], 0, sides=12, radii=[0.002, wl])
    env.tube([(g.X_W, T["lens1_flat"] - 0.25, zl), (g.X_W, T["lens2_flat"] + 0.25, zl)], wl, sides=12)
    env.tube([(g.X_W, T["lens2_flat"] + 0.25, zl), (g.X_W, g.Y_EDGE, zl)], 0, sides=12, radii=[wl, 0.002])
obj_from_mb("tx_beam_cores_laser", core, M["beam_core"], "beams_viz", (0, 0, 0), info=dict(grade="viz", viz_only=1))
for d, lc in lane_cores:
    t_ = d["lane"] / 7.0
    col = (1.0, 0.06 + 0.55 * t_, 0.04 + 0.06 * t_)
    mcore = mat("tx_beam_core_%s" % d["name"], col, 0.0, 0.5, emit=4.0)
    obj_from_mb("tx_beam_core_%s" % d["name"], lc, mcore, "beams_viz", (0, 0, 0),
                info=dict(grade="viz", viz_only=1, lane=d["name"], col=d["col"], row=d["row"]))
obj_from_mb("tx_beam_envelopes", env, M["beam_env"], "beams_viz", (0, 0, 0), smooth=True,
            info=dict(grade="viz", viz_only=1, note="1/e^2 radius from paraxial Gaussian design"))

# ================================================================== HOOKS
hook("HOOK_eye_aperture", (0.0, 0.0, 0.0), (1, 0, 0), "centre of the 2x4 eye-plate outer surface; +X = beam direction out")
hook("HOOK_pic_mount", (g.X_PIC0, (g.Y_EDGE + g.Y_BACK) / 2, 0.0), (-1, 0, 0), "PIC bottom-face centre (die attach to carrier)")
hook("HOOK_flex_rf", (ZC1, CY0, 0.0), (0, -1, 0), "centre of the 25-pad RF flex field at the carrier back edge (GSSG x8, 0.27 pitch)")
hook("HOOK_flex_dc", (ZC1, (min(field) + max(field)) / 2, -CZ), (0, 0, -1),
     "centre of the DC flex field on the carrier -Z edge (28 PIC DC + 4 laser nets)")
hook("HOOK_carrier_mount", (g.X_CAR0, (CY0 + CY1) / 2, 0.0), (-1, 0, 0), "carrier back-face centre (mount to the head frame)")

# ================================================================== HOOK ORIENTATION CHECK (local +X = documented direction)
bpy.context.view_layer.update()
for hk in HOOKS:
    ob_ = bpy.data.objects[hk["name"]]
    ax = (ob_.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
    err = (ax - Vector(hk["direction"])).length
    loc_ = Vector(ob_.matrix_world.translation) / MM
    perr = (loc_ - Vector(hk["location_mm"])).length
    if err > 1e-5 or perr > 1e-5:
        print("HOOK_AXIS_FAIL", hk["name"], tuple(ax), hk["direction"], perr)
        raise SystemExit(1)
print("HOOK_AXIS_OK", len(HOOKS))

# ================================================================== BOND CLEARANCE CHECK
WIRE_R = 0.0125
bad = []
for bi, pts in enumerate(BONDS):
    dense = []
    for a_, b_ in zip(pts[:-1], pts[1:]):
        for t_ in (0.0, 0.25, 0.5, 0.75):
            dense.append(a_.lerp(b_, t_))
    for q in dense:
        if (q - pts[0]).length < 0.04 or (q - pts[-1]).length < 0.04:
            continue
        for nm, (bx0, bx1, by0, by1, bz0, bz1) in BOXES:
            if (bx0 - WIRE_R < q.x < bx1 + WIRE_R and by0 - WIRE_R < q.y < by1 + WIRE_R and bz0 - WIRE_R < q.z < bz1 + WIRE_R):
                bad.append((bi, nm, tuple(round(c, 3) for c in q)))
                break
if bad:
    print("BOND_CLEARANCE_FAIL", len(bad), bad[:10])
    raise SystemExit(1)
print("BOND_CLEARANCE_OK bonds=%d boxes=%d" % (len(BONDS), len(BOXES)))

# ================================================================== SAVE / EXPORT / JSON
os.makedirs(OUT_DIR, exist_ok=True)
blend = os.path.join(OUT_DIR, ASSET + ".blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=blend)
glb = os.path.join(OUT_DIR, ASSET + ".glb")
bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", export_extras=True, export_apply=True, export_yup=True)

tris = sum(p["tris"] for p in PARTS)
xs = [p["center_mm"][0] for p in PARTS]
doc = dict(
    asset=ASSET, date="2026-10-10", workstream="T (DevLog-008)",
    frame="origin = centre of the 2x4 eye-plate outer surface (dome bases); +X = light out of the eye; +Z up; meters in "
          ".blend/.glb (glTF export Y-up), mm in this file. Hero mapping: +X -> pigeon -X (right eye outward), +Y -> pigeon -Y (beak).",
    scale_statement="True scale: every part is modeled at its real or estimated mm size (grades A-D below and in "
                    "research/tx_optics_refs.md); no part is scaled for visibility (exception: on the PIC texture, waveguide cores are drawn 3.5 um wide and SSC tips 2.4 um so they read at render resolution; real ~0.4 um / ~0.2 um). Overall carrier 9.2 (Z) x %.2f (Y) mm." % (CY1 - CY0),
    total_tris=tris, parts=PARTS, hooks=HOOKS, lane_map=LM, lane_beam_paths=lane_paths, rf_pads=rf_nets, dc_nets=dc_nets,
    laser_nets=LD_NETS, optics_design=GD, train_y_mm=g.TRAIN,
    stack_x_mm=dict(eye_outer=0.0, eye_inner=-g.EYE["t"], rhomboids=[g.RH["x0"], g.RH["x0"] + g.RH["s"]],
                    staircase=[g.STAIR["xb"], g.STAIR["xt"]], pic_top=g.X_W, pic_bottom=g.X_PIC0, carrier_top=g.X_CTOP, carrier_cavity_floor=g.X_CAR1,
                    carrier_bottom=g.X_CAR0, bench_top=g.X_BENCH1),
    beams_note="tx_beam_* objects (collection tx_beams_viz) are viz only: hide in background shots; any shot showing "
               "them carries '* beams invisible (1311 nm); rendered for viewing'.",
)
with open(os.path.join(OUT_DIR, ASSET + "_parts.json"), "w") as fh:
    json.dump(doc, fh, indent=1)
print("OPTICS_TX_BUILD_OK parts=%d tris=%d hooks=%d" % (len(PARTS), tris, len(HOOKS)))
