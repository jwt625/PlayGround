"""RX optics + TIA subassembly for the hero's left (RX) eye (workstream X, DevLog-009).  Blender 4.2, headless.

Run from the project folder (one Blender at a time, through the slot wrapper):
  scripts/bslot.sh -b --factory-startup --python scripts/assets/optics_rx/build_optics_rx.py

Frame (DevLog-005 Section 3): origin = centre of the 2x4 lenslet array outer (front) surface; +X = optical axis out of
the eye (light arrives travelling -X); +Z up; Y completes the right-handed frame.  Units: Blender metres, true scale
(all numbers below are in mm and scaled by 1e-3 when written).

Optical chain (outside -> in):
  2x4 lenslet array (fused silica, plano-convex, row-specific radii)  ->  two tilted Si plates (top row +30 deg, bottom row
  -30 deg about Z; thickness solved for a -/+0.25 mm Y shift) interleave the rows  ->  stepped fused-silica turning block
  (Au-coated 45 deg facets; top-row and bottom-row facets at the same X, alternating in Y) turns all 8 beams down (-Z)  ->
  1x8 top-illuminated PIN PD array at 0.50 mm pitch on an AlN submount.
Electrical: per lane a differential G-S-S-G input group: PD P -> TIA IN+, PD N -> TIA IN- (matched ball-wedge bonds) plus
two ground bonds TIA G -> PD-submount GND strip; PD bias via IN- from the TIA; TIA outputs GSSG -> AlN carrier
differential CPW -> flex pad row; DC/control from the TIA end pads via single-layer caps to carrier pads -> DC flex pads.
Sources and grades: research/rx_optics_refs.md, DevLog-009.
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ADIR = os.path.join(PROJ, "assets", "components", "optics_rx")
TDIR = os.path.join(ADIR, "textures")
S = 1e-3          # mm -> m
LAY = json.load(open(os.path.join(TDIR, "rx_layout.json")))

# ================================================================== design parameters (mm)
LAMBDA = 1.311e-3            # mm, O-band DR lane (IEEE 802.3 DR: 1304.5-1317.5 nm)
N_FS = 1.4469                # fused silica at 1310 nm (Malitson 1965 dispersion)                   B
N_SI = 3.504                 # silicon at 1310 nm, room temperature (Li 1980 / Frey 2006)           B
COL_PITCH, ROW_PITCH = 1.00, 0.75     # lenslet grid (design, D); rows at Z = +/-0.375
LENS_CA = 0.70               # lens clear aperture diameter (D)
LENS_T = 0.50                # lens substrate thickness (D)
LENS_Y, LENS_Z = 2.20, 0.85  # substrate half sizes (D)
PLATE_TILT = 30.0            # deg about Z (top +, bottom -)                                         D
PLATE_SHIFT = 0.25           # mm lateral shift = half the PD pitch                                 derived
PLATE_W, PLATE_H = 4.73, 0.80    # plate length along its face / height (covers |Y| <= 2.05)        D
PD_PITCH = 0.50              # A: 1x8 PIN array, 500 um pitch (Coherent INP05KK82D101 class)
PD_W, PD_D, PD_T = 4.00, 0.30, 0.15   # A (datasheet nominal 4000 x 300 x 150 um)
PD_AP = 0.020                # A: 20 um aperture
TIA_W, TIA_D, TIA_T = LAY["tia"]["die"]["W"], LAY["tia"]["die"]["D"], 0.20   # D estimate
X_T = -3.90                  # comb-mirror turning line = PD aperture line (design)
X_F = X_T + 0.25             # turning-block front (entry) face
Z_BB = -0.70                 # turning-block bottom (exit) face; 0.50 mm air gap above the PD
Z_CAR = -1.45                # carrier top
Z_PD_TOP = -1.20             # PD top = TIA top (co-planar for short bonds)
CAR_T = 0.38                 # AlN carrier thickness (D: common thin-film AlN 0.25-0.635)
WIRE_D = 0.025               # Au bond wire 25 um (1 mil)                                           D (typical)

# derived positions
PD_X1 = X_T + 0.12           # PD optical edge (aperture 120 um from it)
PD_X0 = PD_X1 - PD_D
TIA_X1 = PD_X0 - 0.08        # TIA input edge
TIA_X0 = TIA_X1 - TIA_D
CAR_X0, CAR_X1, CAR_Y = -6.95, 0.00, 3.20     # nothing but the lens domes sits ahead of the X = 0 eye plane
FLEX_X0, FLEX_X1 = -6.85, -6.55
ROWS = {"top": +ROW_PITCH / 2, "bot": -ROW_PITCH / 2}
COLS = [(i - 1.5) * COL_PITCH for i in range(4)]
PD_Y = [(k - 3.5) * PD_PITCH for k in range(8)]
LANE_OF = {("top", i): 2 * i for i in range(4)}
LANE_OF.update({("bot", i): 2 * i + 1 for i in range(4)})

th = math.radians(PLATE_TILT)
PLATE_T = PLATE_SHIFT / (math.sin(th) * (1 - math.cos(th) / math.sqrt(N_SI ** 2 - math.sin(th) ** 2)))
PLATE_XEXT = PLATE_W * math.sin(th) + PLATE_T * math.cos(th)
PLATE_XC = -LENS_T - 0.12 - PLATE_XEXT / 2

# ================================================================== scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
SCN = bpy.context.scene
SCN.unit_settings.system = "METRIC"
SCN.unit_settings.scale_length = 1.0
SCN.unit_settings.length_unit = "MILLIMETERS"

COLL = {}
for cname in ("rx_micro_optics", "rx_pd", "rx_tia", "rx_carrier", "rx_wirebonds", "rx_passives", "rx_hooks",
              "viz_beams"):
    c = bpy.data.collections.new(cname)
    SCN.collection.children.link(c)
    COLL[cname] = c
ROOT = bpy.data.objects.new("RX_OPTICS_ROOT", None)
SCN.collection.objects.link(ROOT)
ROOT.empty_display_size = 0.002
PARTS = []


# ================================================================== materials
def mat(name, color, metal=0.0, rough=0.5, trans=0.0, ior=1.45, emit=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    b.inputs["Transmission Weight"].default_value = trans
    if emit > 0:
        b.inputs["Emission Color"].default_value = (*color, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
    if trans > 0:
        m.use_raytrace_refraction = True
    return m


def tex_mat(name, base_png, height_png=None, rough_png=None, metal=0.0, bump=0.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(TDIR, base_png))
    t.interpolation = "Cubic"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = 0.35
    if rough_png:
        r = nt.nodes.new("ShaderNodeTexImage")
        r.image = bpy.data.images.load(os.path.join(TDIR, rough_png))
        r.image.colorspace_settings.name = "Non-Color"
        nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    if height_png:
        h = nt.nodes.new("ShaderNodeTexImage")
        h.image = bpy.data.images.load(os.path.join(TDIR, height_png))
        h.image.colorspace_settings.name = "Non-Color"
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.000002
        nt.links.new(h.outputs["Color"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


M = dict(
    fsilica=mat("fused_silica", (0.92, 0.97, 1.0), rough=0.02, trans=1.0, ior=N_FS),
    lens=mat("lenslet_fused_silica_ar", (0.80, 0.95, 1.0), rough=0.02, trans=1.0, ior=N_FS),
    si_plate=mat("silicon_plate_ar", (0.30, 0.30, 0.38), metal=0.75, rough=0.08),
    si=mat("silicon_bare", (0.22, 0.22, 0.25), metal=0.6, rough=0.25),
    gold=mat("gold_thin_film", (1.0, 0.72, 0.33), metal=1.0, rough=0.28),
    au_mirror=mat("gold_mirror", (1.0, 0.78, 0.40), metal=1.0, rough=0.05),
    cu_coil=mat("top_metal_cu", (0.95, 0.52, 0.30), metal=1.0, rough=0.3),
    wire=mat("gold_wire", (1.0, 0.76, 0.36), metal=1.0, rough=0.2),
    aln=mat("aln_ceramic", (0.80, 0.78, 0.74), rough=0.7),
    inp_side=mat("inp_side", (0.18, 0.15, 0.14), metal=0.3, rough=0.4),
    tia_side=mat("si_die_side", (0.10, 0.10, 0.12), metal=0.4, rough=0.35),
    shim=mat("cu_mo_shim", (0.75, 0.55, 0.42), metal=1.0, rough=0.35),
    epoxy=mat("uv_epoxy", (0.85, 0.80, 0.55), rough=0.3, trans=0.6, ior=1.5),
    slc_body=mat("slc_dielectric", (0.12, 0.11, 0.10), rough=0.5),
    mlcc_body=mat("mlcc_body", (0.55, 0.47, 0.36), rough=0.6),
    mlcc_term=mat("mlcc_tin", (0.80, 0.80, 0.82), metal=1.0, rough=0.3),
    flex_pad=mat("flex_pad_enig", (1.0, 0.80, 0.45), metal=1.0, rough=0.25),
    beam=mat("beam_viz_cyan", (0.10, 0.90, 1.0), emit=8.0),
    beam_m=mat("beam_viz_cyan_marginal", (0.05, 0.55, 0.75), emit=4.0),
    tia_top=tex_mat("tia_die_top", "tia_basecolor.png", "tia_height.png", "tia_rough.png", metal=0.25, bump=0.6),
    pd_top=tex_mat("pd_die_top", "pd_basecolor.png", "pd_height.png", None, metal=0.3, bump=0.5),
)


# ================================================================== geometry helpers (inputs in mm)
def link(obj, coll):
    COLL[coll].objects.link(obj)
    obj.parent = ROOT


def finish(obj, coll, explode, label=None, grade="", source="", dims=None, note=""):
    link(obj, coll)
    obj["explode"] = [e * S for e in explode]                         # Blender frame (Z up), metres
    obj["explode_gltf"] = [explode[0] * S, explode[2] * S, -explode[1] * S]   # glTF frame (Y up)
    if label:
        obj["part_label"] = label
    obj["grade"] = grade
    PARTS.append(dict(name=obj.name, label=label or obj.name, collection=coll, dims_mm=dims, grade=grade,
                      source=source, note=note, explode_m=[round(e * S, 6) for e in explode]))
    return obj


def mesh_obj(name, verts, faces, mats, face_mat=None, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c * S for c in v) for v in verts], [], faces)
    for m_ in mats:
        me.materials.append(m_)
    if face_mat:
        for p, mi in zip(me.polygons, face_mat):
            p.material_index = mi
    if uvs:
        uvl = me.uv_layers.new(name="UVMap")
        for p in me.polygons:
            for li in p.loop_indices:
                vi = me.loops[li].vertex_index
                uvl.data[li].uv = uvs[vi] if uvs[vi] is not None else (0.0, 0.0)
    me.validate()
    me.update()
    return bpy.data.objects.new(name, me)


BOX_F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def box_vf(x0, x1, y0, y1, z0, z1, off=0):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return v, [tuple(i + off for i in f) for f in BOX_F]


def box(name, x0, x1, y0, y1, z0, z1, m_, coll, explode, **kw):
    v, f = box_vf(x0, x1, y0, y1, z0, z1)
    o = mesh_obj(name, v, f, [m_])
    return finish(o, coll, explode, dims=[round(x1 - x0, 4), round(y1 - y0, 4), round(z1 - z0, 4)], **kw)


class MB:
    """mesh accumulator (mm)."""

    def __init__(self):
        self.v, self.f, self.fm, self.fs = [], [], [], []

    def add(self, verts, faces, mi=0, smooth=False):
        o = len(self.v)
        self.v += verts
        self.f += [tuple(i + o for i in fc) for fc in faces]
        self.fm += [mi] * len(faces)
        self.fs += [smooth] * len(faces)

    def box(self, x0, x1, y0, y1, z0, z1, mi=0):
        v, f = box_vf(x0, x1, y0, y1, z0, z1)
        self.add(v, f, mi)

    def prism_xz(self, pts, y0, y1, mi=0):
        """polygon in XZ (counter-clockwise seen from +Y) extruded over Y [y0, y1]."""
        n = len(pts)
        v = [(x, y0, z) for x, z in pts] + [(x, y1, z) for x, z in pts]
        f = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
        f += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        self.add(v, f, mi)

    def tube(self, pts, r, sides=8, mi=0):
        """swept circle along a polyline (parallel-transport frames)."""
        P = [Vector(p) for p in pts]
        n = len(P)
        T = []
        for i in range(n):
            a = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
            T.append(a.normalized())
        ref = Vector((0, 0, 1)) if abs(T[0].z) < 0.9 else Vector((1, 0, 0))
        N = T[0].cross(ref).normalized()
        verts = []
        for i in range(n):
            if i > 0:
                N = (N - T[i] * N.dot(T[i])).normalized()
            B = T[i].cross(N)
            for k in range(sides):
                a = 2 * math.pi * k / sides
                q = P[i] + (N * math.cos(a) + B * math.sin(a)) * r
                verts.append(tuple(q))
        faces = []
        for i in range(n - 1):
            for k in range(sides):
                a, b = i * sides + k, i * sides + (k + 1) % sides
                faces.append((a, b, b + sides, a + sides))
        faces.append(tuple(range(sides - 1, -1, -1)))
        faces.append(tuple(range((n - 1) * sides, n * sides)))
        self.add(verts, faces, mi, smooth=True)

    def sphere(self, c, r, sx=1.0, sy=1.0, sz=1.0, seg=10, rings=6, mi=0):
        verts, faces = [], []
        for j in range(rings + 1):
            ph = math.pi * j / rings
            for k in range(seg):
                a = 2 * math.pi * k / seg
                verts.append((c[0] + r * sx * math.sin(ph) * math.cos(a), c[1] + r * sy * math.sin(ph) * math.sin(a),
                              c[2] + r * sz * math.cos(ph)))
        for j in range(rings):
            for k in range(seg):
                a, b = j * seg + k, j * seg + (k + 1) % seg
                faces.append((a, a + seg, b + seg, b))
        self.add(verts, faces, mi, smooth=True)

    def cyl_z(self, x, y, r, z0, z1, seg=16, mi=0):
        v = [(x + r * math.cos(2 * math.pi * k / seg), y + r * math.sin(2 * math.pi * k / seg), z) for z in (z0, z1)
             for k in range(seg)]
        f = [tuple(range(seg - 1, -1, -1)), tuple(range(seg, 2 * seg))]
        f += [(k, (k + 1) % seg, seg + (k + 1) % seg, seg + k) for k in range(seg)]
        self.add(v, f, mi)

    def ribbon(self, pts, w, z0, z1, mi=0):
        """flat trace of width w along a 2D polyline (x, y), thickness z0..z1 (mitred corners)."""
        n = len(pts)
        L, R = [], []
        for i in range(n):
            p = Vector((pts[i][0], pts[i][1]))
            if i == 0:
                d = (Vector(pts[1]) - Vector(pts[0])).normalized()
                nrm = Vector((-d.y, d.x))
                off = nrm * (w / 2)
            elif i == n - 1:
                d = (Vector(pts[-1]) - Vector(pts[-2])).normalized()
                nrm = Vector((-d.y, d.x))
                off = nrm * (w / 2)
            else:
                d0 = (Vector(pts[i]) - Vector(pts[i - 1])).normalized()
                d1 = (Vector(pts[i + 1]) - Vector(pts[i])).normalized()
                n0, n1 = Vector((-d0.y, d0.x)), Vector((-d1.y, d1.x))
                m_ = (n0 + n1).normalized()
                off = m_ * (w / 2) / max(0.3, m_.dot(n0))
            L.append(p + off)
            R.append(p - off)
        verts = [(q.x, q.y, z0) for q in L] + [(q.x, q.y, z0) for q in R] + [(q.x, q.y, z1) for q in L] + \
                [(q.x, q.y, z1) for q in R]
        faces = []
        for i in range(n - 1):
            lb, rb, lt, rt = i, n + i, 2 * n + i, 3 * n + i
            faces += [(lb, lb + 1, rb + 1, rb)[::-1], (lt, rt, rt + 1, lt + 1)[::-1],
                      (lb, lt, lt + 1, lb + 1)[::-1], (rb, rb + 1, rt + 1, rt)[::-1]]
        faces += [(0, n, 3 * n, 2 * n), (n - 1, 3 * n - 1, 4 * n - 1, 2 * n - 1)]
        self.add(verts, faces, mi)

    def obj(self, name, mats):
        o = mesh_obj(name, self.v, self.f, mats, self.fm)
        if any(self.fs):
            for p_, sm in zip(o.data.polygons, self.fs):
                p_.use_smooth = sm
        return o


def hook(name, loc, rot_euler, doc, explode=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "ARROWS"
    e.empty_display_size = 0.0008
    e.location = Vector(loc) * S
    e.rotation_euler = rot_euler
    e["doc"] = doc
    e["explode"] = [c * S for c in explode]
    link(e, "rx_hooks")
    return e


# ================================================================== ray tracing (mm)
def refract(d, n, n1, n2):
    cosi = -d.dot(n)
    eta = n1 / n2
    k = 1 - eta * eta * (1 - cosi * cosi)
    assert k >= 0, "TIR"
    return (d * eta + n * (eta * cosi - math.sqrt(k))).normalized()


def reflect(d, n):
    return (d - n * (2 * d.dot(n))).normalized()


def hit(p, d, q, n):
    t = (q - p).dot(n) / d.dot(n)
    return p + d * t


def plate_frame(row):
    phi = math.radians(PLATE_TILT if row == "top" else -PLATE_TILT)
    zc = (0.05 + PLATE_H / 2) if row == "top" else -(0.05 + PLATE_H / 2)
    c = Vector((PLATE_XC, 0.0, zc))
    n = Vector((math.cos(phi), math.sin(phi), 0.0))
    return c, n, phi


def lens_radius(row):
    """paraxial: R = (n-1) * air-equivalent distance from the lens apex to the PD aperture plane (chief ray)."""
    zr = ROWS[row]
    p = Vector((0.0, 0.0, zr))
    d = Vector((-1.0, 0.0, 0.0))
    c, n, _ = plate_frame(row)
    L = LENS_T / N_FS                                   # substrate (apex sag added below)
    q0 = hit(p, d, Vector((-LENS_T, 0, 0)), Vector((1, 0, 0)))
    q1 = hit(q0, d, c + n * (PLATE_T / 2), n)
    L += (q1 - q0).length
    d1 = refract(d, n, 1.0, N_SI)
    q2 = hit(q1, d1, c - n * (PLATE_T / 2), n)
    L += (q2 - q1).length / N_SI
    d2 = refract(d1, n, N_SI, 1.0)
    q3a = hit(q2, d2, Vector((X_F, 0, 0)), Vector((1, 0, 0)))       # enter the glass turning block
    L += (q3a - q2).length
    d3 = refract(d2, Vector((1, 0, 0)), 1.0, N_FS)
    fn = Vector((1, 0, -1)).normalized()
    q3 = hit(q3a, d3, Vector((X_T, 0, zr)), fn)                     # Au-coated 45 deg facet
    L += (q3 - q3a).length / N_FS
    L += (q3.z - Z_BB) / N_FS + (Z_BB - Z_PD_TOP)
    sag = 0.0
    for _ in range(3):                                  # apex sits sag ahead of the front face: extra glass path
        R = (N_FS - 1) * (L + sag / N_FS)
        sag = R - math.sqrt(R * R - (LENS_CA / 2) ** 2)
    return R, L + sag / N_FS, sag


LENS = {row: lens_radius(row) for row in ROWS}


def trace_ray(row, col, dy, dz):
    """exact trace of an incoming ray parallel to -X hitting lenslet (row, col) at offset (dy, dz)."""
    R, L, sag = LENS[row]
    yc, zc = COLS[col], ROWS[row]
    pts = []
    y, z = yc + dy, zc + dz
    r2 = dy * dy + dz * dz
    xs = sag - (R - math.sqrt(R * R - r2))            # dome surface X (apex at X = sag, front face at X = 0)
    centre = Vector((sag - R, yc, zc))
    p = Vector((xs, y, z))
    pts.append(Vector((2.6, y, z)))
    pts.append(p)
    d = Vector((-1.0, 0.0, 0.0))
    n = (p - centre).normalized()
    d = refract(d, n, 1.0, N_FS)
    p = hit(p, d, Vector((-LENS_T, 0, 0)), Vector((1, 0, 0)))
    pts.append(p)
    d = refract(d, Vector((1, 0, 0)), N_FS, 1.0)
    c, nrm, _ = plate_frame(row)
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
    p = hit(p, d, Vector((X_T, 0, ROWS[row])), fn)
    pts.append(p)
    d = reflect(d, fn)
    p = hit(p, d, Vector((0, 0, Z_BB)), Vector((0, 0, 1)))
    pts.append(p)
    d = refract(d, Vector((0, 0, 1)), N_FS, 1.0)
    p = hit(p, d, Vector((0, 0, Z_PD_TOP)), Vector((0, 0, 1)))
    pts.append(p)
    return pts


LOG = []


def log(s):
    print(s)
    LOG.append(s)


log("plate t = %.4f mm (Si n=%.3f, tilt %.1f deg, shift %.3f), plate X extent %.3f, centre X %.3f"
    % (PLATE_T, N_SI, PLATE_TILT, PLATE_SHIFT, PLATE_XEXT, PLATE_XC))
TRACE = {}
for row in ROWS:
    R, L, sag = LENS[row]
    wf = LAMBDA * L / (math.pi * 0.30)
    log("lens row %s: R = %.4f mm, air-equivalent L = %.4f mm, sag = %.4f mm, focus waist (w0 0.30 mm) = %.2f um"
        % (row, R, L, sag, wf * 1000))
    for i in range(4):
        k = LANE_OF[(row, i)]
        chief = trace_ray(row, i, 0.0, 0.0)
        land = chief[-1]
        err = (land.y - PD_Y[k], land.x - X_T)
        spots = []
        for (dy, dz) in [(0.28, 0), (-0.28, 0), (0, 0.28), (0, -0.28)]:
            q = trace_ray(row, i, dy, dz)[-1]
            spots.append(((q.x - land.x) ** 2 + (q.y - land.y) ** 2) ** 0.5)
        TRACE[k] = dict(row=row, col=i, chief=[tuple(p) for p in chief])
        log("  lane %d (row %s col %d): lands Y %.4f (PD %.3f, err %.1f um), X err %.1f um, marginal spread %.2f um"
            % (k + 1, row, i, land.y, PD_Y[k], err[0] * 1000, err[1] * 1000, max(spots) * 1000))
        assert abs(err[0]) < 0.002 and abs(err[1]) < 0.002, "trace miss"

# ================================================================== MICRO-OPTICS
# lenslet array: substrate X [-LENS_T, 0] + 8 plano-convex domes on the front face
mb = MB()
mb.box(-LENS_T, 0.0, -LENS_Y, LENS_Y, -LENS_Z, LENS_Z)
for row, zc in ROWS.items():
    R, L, sag = LENS[row]
    for yc in COLS:
        nr, na = 14, 48
        verts = [(sag, yc, zc)]
        for j in range(1, nr + 1):
            rr = (LENS_CA / 2) * j / nr
            xs = sag - (R - math.sqrt(R * R - rr * rr))
            for k in range(na):
                a = 2 * math.pi * k / na
                verts.append((xs, yc + rr * math.cos(a), zc + rr * math.sin(a)))
        faces = [(0, 1 + (k + 1) % na, 1 + k) for k in range(na)]
        for j in range(nr - 1):
            for k in range(na):
                a, b = 1 + j * na + k, 1 + j * na + (k + 1) % na
                faces.append((a, b, b + na, a + na)[::-1])
        mb.add(verts, faces, smooth=True)
o = mb.obj("rx_lenslet_array_2x4", [M["lens"]])
finish(o, "rx_micro_optics", (8.0, 0, 0), label="2x4 lenslet array (fused silica)", grade="D",
       dims=[LENS_T, 2 * LENS_Y, 2 * LENS_Z],
       source="design; n(1310)=1.4469 Malitson; R per row from the exact chief-ray path (log)",
       note="R_top %.3f / R_bot %.3f mm, CA %.2f, col pitch %.2f, row pitch %.2f" %
            (LENS["top"][0], LENS["bot"][0], LENS_CA, COL_PITCH, ROW_PITCH))

# Si interleaver plates
for row in ("top", "bot"):
    c, n, phi = plate_frame(row)
    v, f = box_vf(-PLATE_T / 2, PLATE_T / 2, -PLATE_W / 2, PLATE_W / 2, -PLATE_H / 2, PLATE_H / 2)
    o = mesh_obj("rx_si_interleaver_plate_%s" % row, v, f, [M["si_plate"]])
    o.rotation_euler = (0, 0, phi)
    o.location = Vector(c) * S
    finish(o, "rx_micro_optics", (4.5, 0, 2.6 if row == "top" else 0.0),
           label="Si interleaver plate (%s row, %+.0f deg)" % (row, math.degrees(phi)), grade="D",
           dims=[round(PLATE_T, 4), PLATE_W, PLATE_H],
           source="design; lateral shift d = t sin(th)(1 - cos(th)/sqrt(n^2 - sin^2 th)); n_Si(1310)=3.504",
           note="AR-coated both faces (Si Fresnel loss ~30%/face uncoated)")

# glass shelf between the plates (dead zone between the two beam rows) + side posts + bench
SH_X0, SH_X1 = PLATE_XC - PLATE_XEXT / 2 + 0.05, PLATE_XC + PLATE_XEXT / 2 - 0.05
mb = MB()
mb.box(SH_X0, SH_X1, -2.55, 2.55, -0.05, 0.05)
for yy in (-2.55, 2.30):
    for xx in (SH_X1 - 0.35, SH_X0 + 0.05):
        mb.box(xx, xx + 0.30, yy, yy + 0.25, -0.85, -0.05)
o = mb.obj("rx_interleaver_shelf_posts", [M["fsilica"]])
finish(o, "rx_micro_optics", (4.5, 0, 1.3), label="glass shelf + posts", grade="D",
       dims=[round(SH_X1 - SH_X0, 3), 5.1, 0.10], source="design (shelf sits in the unlit gap between the rows)")
BENCH_X0 = PD_X1 + 0.10
box("rx_optical_bench", BENCH_X0, 0.0, -2.60, 2.60, Z_CAR, -0.85, M["fsilica"], "rx_micro_optics", (4.5, 0, -1.8),
    label="optical bench (glass)", grade="D", source="design")

# stepped glass turning block (fused silica, Au-coated 45 deg facets): top-row lanes (1,3,5,7) reflect on the upper
# facet level, bottom-row lanes (2,4,6,8) on the lower level, all facets on X = X_T; light enters the front face and
# leaves through the bottom face.  Built directly as one watertight solid (no boolean): Y bands with right-triangle
# cross-sections; side walls where the band height steps; T-vertices inserted so all edges are manifold.
bands = [(-2.60, -2.00, ROWS["bot"])] + [(PD_Y[k] - 0.25, PD_Y[k] + 0.25, ROWS["top"] if k % 2 == 0 else ROWS["bot"])
                                          for k in range(8)] + [(2.00, 2.60, ROWS["bot"])]
VI = {}
tv, tf, tm = [], [], []


def _v(x, y, z):
    key = (round(x, 6), round(y, 6), round(z, 6))
    if key not in VI:
        VI[key] = len(tv)
        tv.append(key)
    return VI[key]


def _f(pts, mi=0):
    tf.append(tuple(_v(*q) for q in pts))
    tm.append(mi)


zt_of = lambda zr: zr + (X_F - X_T)          # facet meets the entry face
xb_of = lambda zr: X_T + (Z_BB - zr)          # facet meets the exit face
for i, (y0, y1, zr) in enumerate(bands):
    zt, xb = zt_of(zr), xb_of(zr)
    L = bands[i - 1][2] if i > 0 else None
    R_ = bands[i + 1][2] if i < len(bands) - 1 else None
    ent = [(X_F, y0, Z_BB), (X_F, y1, Z_BB)]
    if R_ is not None and zt_of(R_) < zt - 1e-9:
        ent.append((X_F, y1, zt_of(R_)))
    ent += [(X_F, y1, zt), (X_F, y0, zt)]
    if L is not None and zt_of(L) < zt - 1e-9:
        ent.append((X_F, y0, zt_of(L)))
    _f(ent)                                                               # entry face
    bot = [(xb, y0, Z_BB)]
    if L is not None and xb_of(L) > xb + 1e-9:
        bot.append((xb_of(L), y0, Z_BB))
    bot += [(X_F, y0, Z_BB), (X_F, y1, Z_BB)]
    if R_ is not None and xb_of(R_) > xb + 1e-9:
        bot.append((xb_of(R_), y1, Z_BB))
    bot.append((xb, y1, Z_BB))
    _f(bot)                                                               # exit face
    _f([(X_F, y0, zt), (X_F, y1, zt), (xb, y1, Z_BB), (xb, y0, Z_BB)], mi=1)   # Au-coated facet
    if L is None:
        _f([(X_F, y0, Z_BB), (X_F, y0, zt), (xb, y0, Z_BB)])                 # end wall
    if R_ is None:
        _f([(X_F, y1, Z_BB), (X_F, y1, zt), (xb, y1, Z_BB)])
    elif abs(zt_of(R_) - zt) > 1e-9:                                         # step wall between bands
        hi, lo = (zr, R_) if zt > zt_of(R_) else (R_, zr)
        _f([(X_F, y1, zt_of(lo)), (X_F, y1, zt_of(hi)), (xb_of(hi), y1, Z_BB), (xb_of(lo), y1, Z_BB)])
o_ = mesh_obj("rx_turning_block_stepped", tv, tf, [M["fsilica"], M["au_mirror"]],
              face_mat=tm)
tb = o_
mbp = MB()                                     # two glass end posts (separate closed shells, touching the exit face)
for (y0, y1) in ((-2.60, -2.25), (2.25, 2.60)):
    mbp.box(X_T - 0.20, X_T + 0.15, y0, y1, Z_CAR, Z_BB)
me_ = tb.data
bm = bmesh.new()
bm.from_mesh(me_)
off = len(bm.verts)
for q in mbp.v:
    bm.verts.new(tuple(c * S for c in q))
bm.verts.ensure_lookup_table()
for fc in mbp.f:
    nf = bm.faces.new([bm.verts[off + i] for i in fc])
    nf.material_index = 0
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
nonman = sum(1 for e in bm.edges if not e.is_manifold)
bm.to_mesh(me_)
bm.free()
log("turning block: %d faces, %d non-manifold edges" % (len(me_.polygons), nonman))
assert nonman == 0, "turning block not watertight"
tb.name = "rx_turning_block_stepped"
tb.data.name = tb.name
finish(tb, "rx_micro_optics", (0, 0, 4.2), label="stepped glass turning block (Au-coated 45 deg facets)", grade="D",
       dims=[round(X_F - (X_T + Z_BB - ROWS["top"]), 3), 5.2, round(ROWS["top"] + 0.25 - Z_CAR, 3)],
       source="design; owner 400G/800G photos (glass turning block over the PD row); step-mirror principle",
       note="facets on X=%.2f; entry face X=%.2f; exit face Z=%.2f; coated facets (TIR alone would leak: "
            "critical angle 43.7 deg in fused silica)" % (X_T, X_F, Z_BB))

# ================================================================== PD array
SUB_X0 = TIA_X1 + 0.005                       # submount reaches to 5 um from the TIA shim
box("rx_pd_submount_aln", SUB_X0, PD_X1 + 0.06, -2.15, 2.15, Z_CAR, Z_PD_TOP - PD_T, M["aln"], "rx_pd",
    (0, 0, 0.8), label="PD submount (AlN)", grade="D", source="estimate")
# Au ground strip on the submount between the PD and the TIA (via-tied to the carrier GND): lands the flanking
# ground bonds of each differential input group
GS_X0, GS_X1 = SUB_X0, PD_X0 - 0.005
box("rx_pd_submount_gnd_strip", GS_X0, GS_X1, -2.10, 2.10, Z_PD_TOP - PD_T, Z_PD_TOP - PD_T + 0.003, M["gold"], "rx_pd",
    (0, 0, 0.8), label="submount GND strip", grade="D", source="design (ground return for the input G bonds)")
v, f = box_vf(PD_X0, PD_X1, -PD_W / 2, PD_W / 2, Z_PD_TOP - PD_T, Z_PD_TOP)
uvs = []
for (x, y, z) in v:
    s = PD_X1 - x
    uvs.append(((y + PD_W / 2) / PD_W, 1 - s / PD_D) if z == Z_PD_TOP else (0.002, 0.5))
o = mesh_obj("rx_pin_pd_array_1x8", v, f, [M["pd_top"], M["inp_side"]], face_mat=[1, 0, 1, 1, 1, 1], uvs=uvs)
finish(o, "rx_pd", (0, 0, 1.6), label="1x8 PIN PD array (InGaAs/InP)", grade="A",
       dims=[PD_D, PD_W, PD_T], source="Coherent INP05KK82D101 datasheet: 300 x 4000 x 150 um, 500 um pitch, "
                                        "20 um aperture, 65 um GSG pads (layout read from drawing, C)")

# ================================================================== TIA
box("rx_tia_shim", TIA_X0 - 0.02, TIA_X1, -TIA_W / 2 - 0.02, TIA_W / 2 + 0.02, Z_CAR, Z_PD_TOP - TIA_T, M["shim"],
    "rx_tia", (0, 0, 0.8), label="TIA shim (CuMo)", grade="D", source="design: brings the TIA top level with the PD")
v, f = box_vf(TIA_X0, TIA_X1, -TIA_W / 2, TIA_W / 2, Z_PD_TOP - TIA_T, Z_PD_TOP)
uvs = []
for (x, y, z) in v:
    s = TIA_X1 - x
    uvs.append(((y + TIA_W / 2) / TIA_W, 1 - s / TIA_D) if z == Z_PD_TOP else (0.0005, 0.9995))
o = mesh_obj("rx_octal_tia", v, f, [M["tia_top"], M["tia_side"]], face_mat=[1, 0, 1, 1, 1, 1], uvs=uvs)
finish(o, "rx_tia", (0, 0, 1.6), label="octal TIA (8 x 0.50 mm lanes)", grade="D",
       dims=[TIA_D, TIA_W, TIA_T], source="lane pitch 500 um (Semtech GN18xx class, A); die size not public -> estimate")


def tia_xy(y, s):
    return (TIA_X1 - s, y)


def pd_xy(y, s):
    return (PD_X1 - s, y)


mb = MB()
ZM0, ZM1 = Z_PD_TOP + 0.0002, Z_PD_TOP + 0.0032       # thick top metal 3 um (D)
for ind in LAY["tia"]["inductors"]:
    pts = [tia_xy(a, b) for (a, b) in ind["pts"]]
    mb.ribbon(pts, ind["w"], ZM0, ZM1)
o = mb.obj("rx_tia_spiral_inductors", [M["cu_coil"]])
finish(o, "rx_tia", (0, 0, 1.6), label="spiral peaking inductors (56)", grade="D",
       dims=None, source="layout estimate; SiGe/CMOS TIA micrographs (refs)", note="7 per lane: input, 2 shunt pairs, output pair")

# ================================================================== CARRIER + traces
box("rx_carrier_aln", CAR_X0, CAR_X1, -CAR_Y, CAR_Y, Z_CAR - CAR_T, Z_CAR, M["aln"], "rx_carrier", (0, 0, 0),
    label="AlN carrier (thin-film Au)", grade="D", source="estimate")
ZT0, ZT1 = Z_CAR, Z_CAR + 0.003
mb = MB()
# ground pad under the TIA shim
mb.box(TIA_X0 - 0.07, TIA_X1 + 0.02, -TIA_W / 2 - 0.07, TIA_W / 2 + 0.07, ZT0, ZT1 - 0.001)
RF_X0 = TIA_X0 - 0.09      # carrier RF bond pads start
cpw = {}
for k, yc in enumerate(PD_Y):
    for j, (dy, w, kind) in enumerate([(-0.19, 0.08, "G"), (-0.05, 0.045, "S"), (0.05, 0.045, "S"), (0.19, 0.08, "G")]):
        y = yc + dy
        mb.ribbon([(RF_X0, y), (FLEX_X1 + 0.02, y)], w, ZT0, ZT1)
        cpw[(k, j)] = (RF_X0 - 0.05, y)
# flex pad row (RF)
mb2 = MB()
for k, yc in enumerate(PD_Y):
    for dy, w in [(-0.19, 0.09), (-0.05, 0.055), (0.05, 0.055), (0.19, 0.09)]:
        mb2.box(FLEX_X0, FLEX_X1 + 0.02, yc + dy - w / 2, yc + dy + w / 2, ZT0, ZT1 + 0.004)
# DC traces and pads: -Y = power (VTIA, GND, VPD_BIAS), +Y = control (SCL, SDA, RSTn)
dc_pads = {p["name"]: p for p in LAY["tia"]["pads"] if p["kind"].startswith("tia_dc")}
FLEX_DC = {"lo": [("GND", -2.30), ("VPD_BIAS", -2.62), ("VTIA", -2.94)],
           "hi": [("TIA_RSTn", 2.30), ("TIA_SDA", 2.62), ("TIA_SCL", 2.94)]}
DCX = {}
for side, items in FLEX_DC.items():
    for nm, y in items:
        mb2.box(FLEX_X0, FLEX_X1 + 0.02, y - 0.10, y + 0.10, ZT0, ZT1 + 0.004)
        DCX[nm] = y
# carrier DC bond pads next to the TIA ends, ground strips with vias, and routed traces
gnd_strip = {}
for sgn, side in ((-1, "lo"), (1, "hi")):
    yy = sgn * (TIA_W / 2 + 0.33)
    mb.box(TIA_X0 + 0.02, TIA_X1 - 0.38, yy - 0.06, yy + 0.06, ZT0, ZT1)          # GND strip (via-stitched)
    gnd_strip[side] = yy
slc_sites, dc_land = [], {}
LAND_Y = {"VCC": -(TIA_W / 2 + 0.78), "VPD": -(TIA_W / 2 + 0.55), "SCL": TIA_W / 2 + 0.55, "SDA": TIA_W / 2 + 0.55,
          "RSTn": TIA_W / 2 + 0.55}
for p in LAY["tia"]["pads"]:
    if not p["kind"].startswith("tia_dc"):
        continue
    x, y = tia_xy(p["y"], p["s"])
    sgn = -1 if p["y"] < 0 else 1
    nm = p["kind"].replace("tia_dc_", "")
    if nm == "GND":
        dc_land[p["name"]] = (x, sgn * (TIA_W / 2 + 0.33), "GND")
        continue
    if nm in ("VCC", "VPD"):
        slc_sites.append((x, sgn * (TIA_W / 2 + 0.33), p["name"], nm))     # SLC sits on the via-stitched GND strip
    yl = LAND_Y[nm]
    mb.box(x - 0.06, x + 0.06, yl - 0.06, yl + 0.06, ZT0, ZT1)
    dc_land[p["name"]] = (x, yl, nm)


def land(nm):
    return [v for v in dc_land.values() if v[2] == nm]


# -Y power group (flex inside -> outside: GND, VPD_BIAS, VTIA); +Y control group (RSTn, SDA, SCL) - no crossings
vcc = sorted(land("VCC"), key=lambda v: -v[0])
yb = LAND_Y["VCC"]
mb.ribbon([(vcc[0][0], yb), (FLEX_X1 + 0.45, yb), (FLEX_X1 + 0.02, DCX["VTIA"])], 0.07, ZT0, ZT1)
xv, yv, _ = land("VPD")[0]
mb.ribbon([(xv, yv), (xv - 0.35, yv), (xv - 0.60, DCX["VPD_BIAS"]), (FLEX_X1 + 0.02, DCX["VPD_BIAS"])], 0.05, ZT0, ZT1)
mb.ribbon([(TIA_X0 + 0.02, gnd_strip["lo"]), (TIA_X0 - 0.10, gnd_strip["lo"]), (TIA_X0 - 0.30, DCX["GND"]),
           (FLEX_X1 + 0.02, DCX["GND"])], 0.10, ZT0, ZT1)
xr, yr, _ = land("RSTn")[0]
mb.ribbon([(xr, yr), (xr - 0.38, yr), (xr - 0.68, DCX["TIA_RSTn"]), (FLEX_X1 + 0.02, DCX["TIA_RSTn"])], 0.05, ZT0, ZT1)
xd, yd, _ = land("SDA")[0]
mb.ribbon([(xd, yd), (xd, 3.00), (xd - 0.66, 3.00), (xd - 0.96, DCX["TIA_SDA"]), (FLEX_X1 + 0.02, DCX["TIA_SDA"])],
          0.05, ZT0, ZT1)
xc_, yc_, _ = land("SCL")[0]
mb.ribbon([(xc_, yc_), (xc_, 3.12), (xc_ - 1.04, 3.12), (xc_ - 1.34, DCX["TIA_SCL"]), (FLEX_X1 + 0.02, DCX["TIA_SCL"])],
          0.05, ZT0, ZT1)
# TIA die-attach GND pad tied to both GND strips (bridges at the rear corners)
for sgn in (-1, 1):
    mb.box(TIA_X0 + 0.02, TIA_X0 + 0.12, min(sgn * (TIA_W / 2 + 0.05), sgn * (TIA_W / 2 + 0.30)),
           max(sgn * (TIA_W / 2 + 0.05), sgn * (TIA_W / 2 + 0.30)), ZT0, ZT1)
# 0201 bulk cap on VTIA: long axis along Y, terminal A on the VTIA bus, terminal B on a via-stitched GND land
MLCC_X = -5.75
mb.box(MLCC_X - 0.07, MLCC_X + 0.07, LAND_Y["VCC"] + 0.44, LAND_Y["VCC"] + 0.56, ZT0, ZT1)
# fiducials (Au crosses) near the front corners and beside the PD site (clear of all traces)
for (fx, fy) in [(CAR_X1 - 0.25, CAR_Y - 0.25), (CAR_X1 - 0.25, -CAR_Y + 0.25), (-3.60, CAR_Y - 0.25),
                 (-3.60, -CAR_Y + 0.25)]:
    mb.box(fx - 0.10, fx + 0.10, fy - 0.015, fy + 0.015, ZT0, ZT1)
    mb.box(fx - 0.015, fx + 0.015, fy - 0.10, fy + 0.10, ZT0, ZT1)
o = mb.obj("rx_carrier_traces_au", [M["gold"]])
finish(o, "rx_carrier", (0, 0, 0), label="carrier Au: GSSG CPW, DC traces, GND", grade="D", source="design")
o = mb2.obj("rx_flex_pads", [M["flex_pad"]])
finish(o, "rx_carrier", (0, 0, 0), label="flex pads (8 x GSSG + 6 DC)", grade="D", source="design")
# vias (dark filled dots) along the CPW grounds and GND strips
mb = MB()
for k, yc in enumerate(PD_Y):
    for dy in (-0.19, 0.19):
        for xx in (RF_X0 - 0.25, RF_X0 - 0.60):
            mb.cyl_z(xx, yc + dy, 0.02, ZT1, ZT1 + 0.0005, seg=10)
for side, yy in gnd_strip.items():
    for i in range(6):
        mb.cyl_z(TIA_X1 - 0.45 - i * 0.15, yy, 0.02, ZT1, ZT1 + 0.0005, seg=10)
    mb.cyl_z(TIA_X0 + 0.07, (1 if side == "hi" else -1) * (TIA_W / 2 + 0.18), 0.02, ZT1, ZT1 + 0.0005, seg=10)
mb.cyl_z(MLCC_X, LAND_Y["VCC"] + 0.50, 0.025, ZT1, ZT1 + 0.0005, seg=10)
o = mb.obj("rx_carrier_vias", [M["tia_side"]])
finish(o, "rx_carrier", (0, 0, 0), label="ground vias", grade="D", source="design")

# ================================================================== passives
mb = MB()
SLC = 0.25
slc_top = {}
for (x, ys, pname, nm) in slc_sites:
    mb.box(x - SLC / 2, x + SLC / 2, ys - SLC / 2, ys + SLC / 2, ZT1, ZT1 + 0.10, mi=0)
    mb.box(x - SLC / 2 + 0.02, x + SLC / 2 - 0.02, ys - SLC / 2 + 0.02, ys + SLC / 2 - 0.02, ZT1 + 0.10, ZT1 + 0.103,
           mi=1)
    slc_top[pname] = (x, ys, ZT1 + 0.103)
# 0201 bulk cap on VTIA (VPD is decoupled by its SLC): terminal A on the bus, terminal B on the GND land
y0 = LAND_Y["VCC"]
mb.box(MLCC_X - 0.15, MLCC_X + 0.15, y0 - 0.05, y0 + 0.05, ZT1, ZT1 + 0.30, mi=2)
mb.box(MLCC_X - 0.15, MLCC_X + 0.15, y0 + 0.05, y0 + 0.45, ZT1, ZT1 + 0.30, mi=3)
mb.box(MLCC_X - 0.15, MLCC_X + 0.15, y0 + 0.45, y0 + 0.55, ZT1, ZT1 + 0.30, mi=2)
o = mb.obj("rx_decoupling_caps", [M["slc_body"], M["gold"], M["mlcc_term"], M["mlcc_body"]])
finish(o, "rx_passives", (0, 0, 0.4), label="decoupling: SLC 0.25 sq + 0201 MLCC", grade="D",
       source="SLC size typical (D); 0201 = 0.6 x 0.3 x 0.3 EIA (A)", note="%d SLC, 1 MLCC (VTIA)" % len(slc_sites))

# ================================================================== wire bonds (25 um Au, ball-wedge)
WR = WIRE_D / 2


def bond_curve(p0, p1, loop=0.07, n=22):
    p0, p1 = Vector(p0), Vector(p1)
    top = max(p0.z, p1.z) + loop
    c1 = Vector((p0.x, p0.y, top + 0.03))
    c2 = p0.lerp(p1, 0.70)
    c2.z = top - 0.005
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t * t
        d = t ** 3
        out.append(tuple(p0 * a + c1 * b + c2 * c + p1 * d))
    return out


def add_bond(mb, p0, p1, loop=0.07):
    """ball on p0 (first bond), stitch on p1."""
    ball = (p0[0], p0[1], p0[2] + 0.010)
    mb.sphere((p0[0], p0[1], p0[2] + 0.008), 0.028, sz=0.45, seg=10, rings=5)
    mb.tube(bond_curve(ball, (p1[0], p1[1], p1[2] + 0.006), loop), WR, sides=8)
    along_x = abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1])        # wedge (stitch) elongated along the wire
    mb.sphere((p1[0], p1[1], p1[2] + 0.004), 0.020, sx=1.7 if along_x else 0.9, sy=0.9 if along_x else 1.7, sz=0.22,
              seg=10, rings=4)


pd_pads = {(p["lane"], p["kind"]): p for p in LAY["pd"]["pads"]}
tia_pads = {p["name"]: p for p in LAY["tia"]["pads"]}
ZP = Z_PD_TOP
ZGS = Z_PD_TOP - PD_T + 0.003                  # submount ground-strip top
mb = MB()
for L in range(1, 9):
    for pd_kind, tia_key in (("pd_p", "L%d_INP"), ("pd_n", "L%d_INN")):     # matched-length differential pair
        pp = pd_pads[(L, pd_kind)]
        tp = tia_pads[tia_key % L]
        x0, y0 = pd_xy(pp["y"], pp["s"])
        x1, y1 = tia_xy(tp["y"], tp["s"])
        add_bond(mb, (x0, y0, ZP), (x1, y1, ZP), loop=0.065)
    for g_key in ("L%d_GIa", "L%d_GIb"):                                     # flanking ground bonds
        tp = tia_pads[g_key % L]
        x1, y1 = tia_xy(tp["y"], tp["s"])
        add_bond(mb, (x1, y1, ZP), ((GS_X0 + GS_X1) / 2, y1, ZGS), loop=0.05)
o = mb.obj("rx_wirebonds_pd_to_tia_diff", [M["wire"]])
finish(o, "rx_wirebonds", (0, 0, 1.6), label="differential bonds PD -> TIA (G-S-S-G, 4 per lane)", grade="D",
       source="owner request (differential); pad pitch class from the PD datasheet (S1); owner 100G photo (short PD-TIA bonds)",
       note="32 wires: P->IN+, N->IN- (0.10 mm pair pitch, equal 0.21 mm spans) + 2 ground bonds TIA -> submount GND strip")
mb = MB()
for L in range(1, 9):
    for j, key in enumerate(("L%d_GNDa", "L%d_OUTP", "L%d_OUTN", "L%d_GNDb")):
        tp = tia_pads[key % L]
        x0, y0 = tia_xy(tp["y"], tp["s"])
        xc, yc = cpw[(L - 1, j)]
        add_bond(mb, (x0, y0, ZP), (xc, yc, ZT1), loop=0.06)
o = mb.obj("rx_wirebonds_tia_out_gssg", [M["wire"]])
finish(o, "rx_wirebonds", (0, 0, 0.8), label="GSSG output bonds TIA -> CPW", grade="D", source="design",
       note="32 wires (OUTP/OUTN paired, G either side)")
mb = MB()
for p in LAY["tia"]["pads"]:
    if not p["kind"].startswith("tia_dc"):
        continue
    x0, y0 = tia_xy(p["y"], p["s"])
    sgn = -1 if p["y"] < 0 else 1
    x, yl, nm = dc_land[p["name"]]
    if p["name"] in slc_top:
        sx, sy, sz = slc_top[p["name"]]
        add_bond(mb, (x0, y0, ZP), (sx, sy - sgn * 0.05, sz), loop=0.05)
        add_bond(mb, (sx, sy + sgn * 0.05, sz), (x, yl, ZT1), loop=0.05)
    else:
        add_bond(mb, (x0, y0, ZP), (x, yl, ZT1), loop=0.06)
o = mb.obj("rx_wirebonds_dc_ctrl", [M["wire"]])
finish(o, "rx_wirebonds", (0, 0, 0.8), label="DC / control bonds (VCC, VPD, GND, SCL, SDA, RSTn)", grade="D",
       source="design")

# ================================================================== beams (viz only; not exported to GLB)
mb_c, mb_m = MB(), MB()
for k, tr in TRACE.items():
    mb_c.tube([tuple(p) for p in tr["chief"]], 0.006, sides=6)
    if k in (0, 1, 6, 7):
        for (dy, dz) in [(0.28, 0), (-0.28, 0), (0, 0.28), (0, -0.28)]:
            pts = trace_ray(tr["row"], tr["col"], dy, dz)
            mb_m.tube([tuple(p) for p in pts], 0.003, sides=5)
for nm, mbx, mm in (("viz_beam_chief_rays", mb_c, M["beam"]), ("viz_beam_marginal_rays", mb_m, M["beam_m"])):
    o = mbx.obj(nm, [mm])
    link(o, "viz_beams")
    o["explode"] = [0.0, 0.0, 0.0]

# ================================================================== hooks
hook("HOOK_eye_aperture", (0, 0, 0), (0, 0, 0),
     "centre of the lenslet array front face; local +X = optical axis out of the eye (light arrives along -X)")
hook("HOOK_pd_mount", (X_T, 0, Z_PD_TOP), (0, 0, 0),
     "centre of the PD aperture row (top of the PD die); beams arrive along -Z; row along Y, lane k at Y=(k-4.5)*0.5 mm",
     explode=(0, 0, 1.6))
hook("HOOK_flex_rf", ((FLEX_X0 + FLEX_X1) / 2, 0, ZT1), (0, 0, math.pi),
     "centre of the RF flex pad row (8 x G-S-S-G, 0.50 mm lane pitch); local +X = flex exit direction (global -X)")
hook("HOOK_flex_dc", ((FLEX_X0 + FLEX_X1) / 2, -2.62, ZT1), (0, 0, math.pi),
     "centre of the power DC pad group (-Y): VTIA, GND, VPD_BIAS; local +X = flex exit direction")
hook("HOOK_flex_ctrl", ((FLEX_X0 + FLEX_X1) / 2, 2.62, ZT1), (0, 0, math.pi),
     "centre of the control pad group (+Y): TIA_SCL, TIA_SDA, TIA_RSTn; local +X = flex exit direction")
hook("HOOK_carrier_mount", ((CAR_X0 + CAR_X1) / 2, 0, Z_CAR - CAR_T), (0, 0, 0),
     "carrier bottom-face centre; mount plane normal = -Z")

# ================================================================== save / export / parts JSON
os.makedirs(ADIR, exist_ok=True)
for img in bpy.data.images:
    if img.source == "FILE":
        img.pack()
blend = os.path.join(ADIR, "optics_rx.blend")
bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backups
bpy.ops.wm.save_as_mainfile(filepath=blend)
# GLB: everything except the viz beams
bpy.ops.object.select_all(action="DESELECT")
for o in bpy.data.objects:
    if o.users_collection and o.users_collection[0].name != "viz_beams":
        o.select_set(True)
ROOT.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(ADIR, "optics_rx.glb"), export_format="GLB", use_selection=True,
                          export_extras=True, export_yup=True, export_apply=True)
tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
bb = [Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))]
for o in bpy.data.objects:
    if o.type == "MESH" and o.users_collection[0].name != "viz_beams":
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            bb[0] = Vector(map(min, bb[0], w))
            bb[1] = Vector(map(max, bb[1], w))
hooks = {o.name: dict(loc_m=[round(c, 7) for c in o.location], rot_euler=[round(c, 5) for c in o.rotation_euler],
                      doc=o["doc"]) for o in bpy.data.objects if o.name.startswith("HOOK_")}
out = dict(
    asset="optics_rx", date="2026-10-10", units="m (true scale)", frame=__doc__.split("Frame")[1].split("\n\n")[0],
    bbox_m=[list(map(lambda c: round(c, 7), bb[0])), list(map(lambda c: round(c, 7), bb[1]))],
    faces_incl_viz_beams=tris, explode_frames="explode = Blender Z-up metres; explode_gltf = glTF Y-up metres",
    hooks=hooks, parts=PARTS,
    optics=dict(n_fused_silica=N_FS, n_si=N_SI, plate_t_mm=round(PLATE_T, 5), plate_tilt_deg=PLATE_TILT,
                lens_R_mm={r: round(v[0], 4) for r, v in LENS.items()},
                air_equiv_path_mm={r: round(v[1], 4) for r, v in LENS.items()},
                lane_map={"lane %d" % (k + 1): "%s row, column %d" % (TRACE[k]["row"], TRACE[k]["col"] + 1)
                          for k in range(8)}),
    log=LOG)
with open(os.path.join(ADIR, "optics_rx_parts.json"), "w") as f:
    json.dump(out, f, indent=1)
log("BBOX mm: %s -> %s" % (tuple(round(c * 1000, 3) for c in bb[0]), tuple(round(c * 1000, 3) for c in bb[1])))
log("faces %d, parts %d, saved %s" % (tris, len(PARTS), os.path.relpath(blend, PROJ)))
