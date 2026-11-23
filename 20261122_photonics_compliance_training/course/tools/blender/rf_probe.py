"""Microwave GSG wafer probe, probe arm, manual XYZ positioner, fiber stage and platen (generic parts,
no vendor marks). Units mm. Dimensions and their sources: research/ref_probes/README.md.

Probe (generic 67 GHz class, after the published P-style body drawing of a coaxial probe): black anodized
body 35.3 mm from flange back to tip, 9.3 mm tall, 30.5 mm wide three-hole flange (4.7 mm holes at
10.16 mm pitch) flush with the top and open underneath for the arm plate, top view tapering to 12 mm
at the front face, 1.85 mm female jack (M7 x 0.75 thread) on a 45 degree face pointing up and back,
1.2 mm coax running 8.1 mm from the front face to a conical tip with three spring contacts at
150 um pitch (GSG), landing at about 22 degrees (contacts about 35 degrees).

Probe local frame: tip (signal contact) at the origin on the pad surface, +x toward the probe body
(the approach direction reversed), contacts along y, z up.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import common as C

IN = 25.4
L_TOT = 1.39 * IN            # flange back to tip
L_COAX = 0.32 * IN           # body front face to tip (x of the front face)
H_BOT = 2.0                  # body underside above the tip plane
H_BODY = 0.365 * IN          # body height incl. flange
FLANGE_T = 0.11 * IN
W_BACK, W_FRONT = 1.2 * IN, 0.47 * IN
PITCH = 0.150
COAX_R = 0.60


# ------------------------------------------------------------------ helpers
def boolean(ob, cutter, op="DIFFERENCE"):
    md = ob.modifiers.new("b", "BOOLEAN")
    md.object = cutter
    md.operation = op
    md.solver = "EXACT"
    cutter.hide_render = True
    cutter.hide_viewport = True
    return ob


def apply_mods(ob):
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    for md in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=md.name)


def lathe(name, prof, mat=None, seg=48, smooth=True, cap0=True, cap1=True):
    """Solid of revolution about local +z from a (r, z) profile (z increasing)."""
    bm = bmesh.new()
    rings = []
    for r, z in prof:
        if r < 1e-6:
            rings.append([bm.verts.new((0, 0, z))])
            continue
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z))
                      for k in range(seg)])
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for k in range(seg):
                bm.faces.new((a[0], b[(k + 1) % seg], b[k]))
        elif len(b) == 1:
            for k in range(seg):
                bm.faces.new((a[k], a[(k + 1) % seg], b[0]))
        else:
            for k in range(seg):
                bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    if cap0 and len(rings[0]) > 1:
        bm.faces.new(list(reversed(rings[0])))
    if cap1 and len(rings[-1]) > 1:
        bm.faces.new(rings[-1])
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    if mat is not None:
        C.set_mat(ob, mat)
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = True
        C.smooth(ob, 40)
    return ob


def axis_matrix(origin, axis):
    """Matrix that maps local +z to `axis` at `origin`."""
    q = Vector(axis).normalized().to_track_quat("Z", "Y")
    return Matrix.Translation(Vector(origin)) @ q.to_matrix().to_4x4()


def prism_side(name, prof, y0, y1, mat=None):
    """Prism from an (x, z) profile polygon (counterclockwise seen from -y), extruded from y0 to y1."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, y0, z)) for x, z in prof]
    f = bm.faces.new(vs)
    bm.normal_update()
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv, vec=(0, y1 - y0, 0))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    if mat is not None:
        C.set_mat(ob, mat)
    return ob


def fillet_path(points, radius, step=0.5):
    """Dense polyline through `points` with each corner replaced by a circular arc of `radius` (one value, or
    one per corner; shrunk where the adjacent segments are too short), so a Bezier tube through it follows
    the path without overshoot and its smallest bend radius is known."""
    pts = [Vector(p) for p in points]
    radii = list(radius) if isinstance(radius, (list, tuple)) else [radius] * (len(pts) - 2)
    out = [pts[0]]

    def line_to(q):
        a = out[-1]
        L = (q - a).length
        k = max(1, int(L / step))
        out.extend(a + (q - a) * (i / k) for i in range(1, k + 1))

    for i in range(1, len(pts) - 1):
        c = pts[i]
        a = (pts[i - 1] - c).normalized()
        b = (pts[i + 1] - c).normalized()
        cosang = max(-1.0, min(1.0, a.dot(b)))
        th = math.acos(cosang)                 # interior angle at the corner
        if th > math.pi - 1e-3:                # straight through
            line_to(c)
            continue
        t = radii[i - 1] / math.tan(th / 2)
        f0 = 0.95 if i == 1 else 0.45                 # an end segment can give its whole length to one arc
        f1 = 0.95 if i == len(pts) - 2 else 0.45
        t = min(t, f0 * (pts[i - 1] - c).length, f1 * (pts[i + 1] - c).length)
        r = t * math.tan(th / 2)
        t1, t2 = c + a * t, c + b * t
        w = (a + b).normalized()
        o = c + w * (r / math.sin(th / 2))
        line_to(t1)
        u0, u1 = (t1 - o), (t2 - o)
        n = max(4, int(r * (math.pi - th) / step))
        for k in range(1, n + 1):
            out.append(o + _slerp(u0, u1, k / n) * r)
    line_to(pts[-1])
    return out


def _slerp(u0, u1, f):
    a, b = u0.normalized(), u1.normalized()
    om = math.acos(max(-1.0, min(1.0, a.dot(b))))
    if om < 1e-6:
        return a
    return (a * math.sin((1 - f) * om) + b * math.sin(f * om)) / math.sin(om)


def knurl_mat(name, color=(0.75, 0.75, 0.76), scale=6.0, rough=0.3):
    """Metal with a straight knurl (bump stripes along the axis), object coordinates, z = axis."""
    m = C.principled(name, color=color, metallic=1.0, rough=rough)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    at = nt.nodes.new("ShaderNodeMath")
    at.operation = "ARCTAN2"
    nt.links.new(sep.outputs["Y"], at.inputs[0])
    nt.links.new(sep.outputs["X"], at.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = scale * 6.0
    nt.links.new(at.outputs[0], mul.inputs[0])
    sn = nt.nodes.new("ShaderNodeMath")
    sn.operation = "SINE"
    nt.links.new(mul.outputs[0], sn.inputs[0])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = 0.8
    b.inputs["Distance"].default_value = 0.08
    nt.links.new(sn.outputs[0], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
    return m


def materials():
    m = {}
    m["anod"] = C.principled("probe_anod", color=(0.03, 0.03, 0.033), rough=0.34, rough_var=0.06,
                             rough_scale=0.8, bump=0.15, bump_scale=60.0, bump_dist=0.003)
    m["anod_pos"] = C.principled("pos_anod", color=(0.025, 0.025, 0.027), rough=0.42, rough_var=0.06,
                                 rough_scale=0.3, bump=0.15, bump_scale=30.0, bump_dist=0.004)
    m["alu"] = C.principled("fiber_arm_alu", color=(0.78, 0.78, 0.79), metallic=1.0, rough=0.35, rough_var=0.06,
                            rough_scale=1.0, bump=0.2, bump_scale=40.0, bump_dist=0.003)
    m["ss"] = C.principled("stainless", color=(0.72, 0.71, 0.69), metallic=1.0, rough=0.18, rough_var=0.04,
                           rough_scale=4.0)
    m["steel"] = C.principled("cap_screw", color=(0.10, 0.10, 0.11), metallic=1.0, rough=0.35)
    m["base"] = C.principled("mag_base", color=(0.62, 0.62, 0.63), metallic=1.0, rough=0.30, rough_var=0.05,
                             rough_scale=0.5)
    m["coax"] = C.principled("probe_coax", color=(0.08, 0.075, 0.07), metallic=1.0, rough=0.30)
    m["tipau"] = C.principled("tip_gold", color=(1.0, 0.76, 0.40), metallic=1.0, rough=0.2)
    m["becu"] = C.principled("contact_becu", color=(0.92, 0.72, 0.50), metallic=1.0, rough=0.18)
    m["plate"] = C.principled("label_plate", color=(0.012, 0.012, 0.013), rough=0.15)
    m["ptfe"] = C.principled("ptfe", color=(0.85, 0.85, 0.83), rough=0.5)
    m["hole"] = C.principled("hole_dark", color=(0.0, 0.0, 0.0), rough=0.9)
    m["knurl"] = knurl_mat("knurl_ss")
    m["knurl_dark"] = knurl_mat("knurl_dark", color=(0.06, 0.06, 0.065), scale=5.0, rough=0.4)
    m["cable"] = C.principled("cable_jacket", color=(0.10, 0.19, 0.42), rough=0.55, bump=0.25, bump_scale=3.0,
                              bump_dist=0.05)
    m["boot"] = C.principled("cable_boot", color=(0.03, 0.03, 0.035), rough=0.55)
    m["micro"] = C.principled("micrometer_chrome", color=(0.80, 0.80, 0.80), metallic=1.0, rough=0.12)
    m["platen"] = C.principled("platen_ni", color=(0.55, 0.55, 0.54), metallic=1.0, rough=0.33, rough_var=0.05,
                               rough_scale=0.02)
    return m


def cap_screw(name, loc, d=4.0, m=None, rot=None):
    """Socket head cap screw head (M4: 7 mm x 4 mm) with a hex socket, axis +z at loc (head bottom)."""
    dk, k = 1.75 * d, d
    h = C.cylinder(name, dk / 2, k, (0, 0, k / 2), m["steel"], verts=32, bevel=0.25)
    sock = C.cylinder(name + "_hex", 0.75 * d / 2 / math.cos(math.pi / 6), k, (0, 0, k), None, verts=6)
    boolean(h, sock)
    apply_mods(h)
    bpy.data.objects.remove(sock)
    h.matrix_world = (rot if rot is not None else Matrix.Translation(Vector(loc))) @ h.matrix_world
    return h


# ------------------------------------------------------------------ probe
def _body_profile():
    """Side profile (x, z) of the body, from the published P-style drawing (inches from the back)."""
    def P(b, h):
        return (L_TOT - b * IN, H_BOT + h * IN)
    return [P(0, 0.255), P(0.304, 0.255), P(0.304, 0.0), P(0.40, 0.0), P(1.07, 0.0), P(1.07, 0.365),
            P(0.961, 0.365), P(0.726, 0.13), P(0.40, 0.13), P(0.40, 0.365), P(0, 0.365)]


def gsg_probe(m, name, landing=True):
    """Build the probe in its local frame under an empty and return (root, connector_axis_end, conn_dir).
    connector_axis_end: point on the connector axis at the jack mating face (local)."""
    root = C.empty(name)
    parts = []
    # --- body: side-profile prism cut by the top-view taper
    prof = _body_profile()
    body = prism_side(name + "_body", prof, -W_BACK / 2, W_BACK / 2, m["anod"])
    x_taper0 = L_TOT - 0.40 * IN
    x_front = L_COAX
    # taper cutters (one per side): vertical prisms removing the wedges outside the trapezoid
    k = (W_BACK / 2 - W_FRONT / 2) / (x_taper0 - x_front)   # half-width slope of the taper
    for sy in (-1, 1):
        poly = [(x_taper0 + 0.001, sy * W_BACK / 2), (x_front - 1.0, sy * (W_FRONT / 2 - k * 1.0)),
                (x_front - 1.0, sy * (W_BACK / 2 + 2)), (x_taper0 + 0.001, sy * (W_BACK / 2 + 2))]
        cut = C.extrude_polys(name + f"_taper{sy}", [poly], -5.0, 30.0)
        boolean(body, cut)
    # mounting holes (4.7 mm) through the flange, 3.9 mm from the back, 10.16 mm pitch
    hx = L_TOT - 0.155 * IN
    for k in (-1, 0, 1):
        hc = C.cylinder(name + f"_hole{k}", 0.185 * IN / 2, 10.0, (hx, k * 0.4 * IN, H_BOT + 0.31 * IN), verts=32)
        boolean(body, hc)
    # side set-screw hole near the front (seen on the product photos)
    for sy in (-1, 1):
        ss = C.cylinder(name + f"_set{sy}", 0.9, 4.0, (x_front + 2.2, sy * W_FRONT / 2, H_BOT + 0.17 * IN), verts=20,
                        rot=(math.radians(90), 0, 0))
        boolean(body, ss)
    apply_mods(body)
    for o in [o for o in bpy.data.objects if o.name.startswith(name + "_taper") or o.name.startswith(name + "_hole")
              or o.name.startswith(name + "_set")]:
        bpy.data.objects.remove(o)
    md = body.modifiers.new("bev", "BEVEL")
    md.width = 0.15
    md.segments = 2
    md.limit_method = "ANGLE"
    md.angle_limit = math.radians(50)
    parts.append(body)
    # --- connector on the 45 degree face: axis normal to the face, pointing up and back (+x, +z)
    b0, h0, b1, h1 = 0.726, 0.13, 0.961, 0.365
    fb, fh = (b0 + b1) / 2, (h0 + h1) / 2
    face_c = Vector((L_TOT - fb * IN, 0, H_BOT + fh * IN))
    cdir = Vector((1, 0, 1)).normalized()
    # 1.85 mm female jack: 6.2 mm neck, M7 x 0.75 thread (circular grooves), mating face with the
    # outer conductor bore and the dielectric-free socket
    prof = [(3.1, -0.5), (3.1, 0.6)]
    z = 0.6
    pitch = 0.75
    while z < 0.6 + 5.6 - 1e-6:
        prof += [(3.5, z + 0.12), (3.5, z + pitch / 2 - 0.06), (3.12, z + pitch - 0.08)]
        z += pitch
    zt = z
    prof += [(3.3, zt + 0.05), (1.6, zt + 0.1), (0.925, zt + 0.1), (0.925, zt - 2.0), (0.0, zt - 2.0)]
    jack = lathe(name + "_jack", prof, m["ss"], seg=48, cap0=True, cap1=False)
    jack.matrix_world = axis_matrix(face_c, cdir)
    parts.append(jack)
    pin = lathe(name + "_socket", [(0.0, zt - 2.0), (0.40, zt - 2.0), (0.40, zt - 0.25), (0.30, zt - 0.15),
                                   (0.0, zt - 0.15)], m["tipau"], seg=24)
    pin.matrix_world = axis_matrix(face_c, cdir)
    parts.append(pin)
    mate = face_c + cdir * zt
    # --- coax from the front face down to the tip, the cone and the contacts
    a_end = math.radians(22.0)
    a_con = math.radians(35.0)
    L_con = 0.40
    E = Vector((L_con * math.cos(a_con), 0, L_con * math.sin(a_con)))     # cone end face centre
    ax = Vector((math.cos(a_end), 0, math.sin(a_end)))                     # coax axis at the end (toward body)
    B = E + ax * 0.9                                                       # cone base (full coax diameter)
    F = Vector((x_front + 1.5, 0, H_BOT + COAX_R + 0.35))                  # inside the body
    F0 = Vector((x_front, 0, H_BOT + COAX_R + 0.35))
    pts = [B, B + ax * 1.6, F0 + Vector((-2.2, 0, -0.30)), F0, F]
    coax = C.curve_tube(name + "_coax", pts, COAX_R, m["coax"], resolution=24, bevel_res=6)
    sp = coax.data.splines[0]
    for bp in sp.bezier_points:
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    parts.append(coax)
    cone = lathe(name + "_cone", [(0.0, 0.0), (0.20, 0.0), (COAX_R + 0.006, 0.9), (COAX_R - 0.03, 0.93),
                                  (COAX_R - 0.03, 1.15), (0.0, 1.15)],
                 m["tipau"], seg=40)
    cone.matrix_world = axis_matrix(E, ax)
    parts.append(cone)
    # three contacts (BeCu): roots on the cone end face, contacts on the pads at 150 um pitch
    for k in (-1, 0, 1):
        w0, w1 = (0.075, 0.035) if k else (0.055, 0.028)
        t = 0.022
        r0 = E + Vector((0, k * 0.11, 0)) + ax * 0.02
        tip = Vector((0, k * PITCH, t / 2 + 0.0005))
        verts, faces = [], []
        n = 10
        for i in range(n + 1):
            u = i / n
            # slight downward curvature of the spring contact
            p = r0 + (tip - r0) * u + Vector((0, 0, 0.03 * math.sin(math.pi * u)))
            w = w0 + (w1 - w0) * u
            verts += [(p.x, p.y - w / 2, p.z - t / 2), (p.x, p.y + w / 2, p.z - t / 2),
                      (p.x, p.y + w / 2, p.z + t / 2), (p.x, p.y - w / 2, p.z + t / 2)]
        for i in range(n):
            o = 4 * i
            for j in range(4):
                faces.append((o + j, o + 4 + j, o + 4 + (j + 1) % 4, o + (j + 1) % 4))
        faces.append((0, 1, 2, 3))
        faces.append((4 * n + 3, 4 * n + 2, 4 * n + 1, 4 * n))
        c = C.mesh_obj(name + f"_contact{k}", verts, faces, m["becu"])
        parts.append(c)
    C.parent(parts, root)
    return root, mate, cdir


def cable_plug(m, name, mate, cdir, route, parent=None):
    """Mated 1.85 mm cable plug (8 mm hex coupling nut, knurled grip, boot) on the jack, then a
    phase-stable test cable along `route` (list of points after the boot, world or parent frame)."""
    parts = []
    mtx = axis_matrix(mate, cdir)
    # coupling nut covers the thread down to 1.2 mm above the neck
    nut = C.cylinder(name + "_nut", 8.0 / 2 / math.cos(math.pi / 6), 6.4, (0, 0, -3.2 + 1.6), m["ss"], verts=6,
                     bevel=0.25)
    nut.matrix_world = mtx @ nut.matrix_world
    grip = lathe(name + "_grip", [(0.0, 1.6), (4.3, 1.6), (4.3, 11.0), (3.9, 11.6), (0.0, 11.6)], m["knurl"], seg=64)
    grip.matrix_world = mtx
    boot = lathe(name + "_boot", [(0.0, 11.6), (3.6, 11.6), (3.4, 18.0), (2.9, 28.0), (2.75, 31.0), (0.0, 31.0)],
                 m["boot"], seg=40)
    boot.matrix_world = mtx
    parts += [nut, grip, boot]
    p0 = Vector(mate) + Vector(cdir) * 30.0
    # leaves the boot on the connector axis, bends up (30 mm radius) to run straight up in front of
    # the arm's drop plate, then follows `route` (first point: height of the vertical run) with 35 mm bends
    q = p0 + Vector(cdir) * 14.0
    pts = [p0, q, Vector((q.x, q.y, route[0][2]))] + [Vector(p) for p in route[1:]]
    path = fillet_path(pts, [30.0] + [40.0] * (len(pts) - 3), step=1.0)
    cab = C.curve_tube(name + "_cable", path, 2.75, m["cable"], resolution=4, bevel_res=6)
    parts.append(cab)
    if parent is not None:
        C.parent(parts, parent)
    return parts


# ------------------------------------------------------------------ arm and positioner
def probe_arm(m, name, x_pos, z_top_pos):
    """L-shaped arm in the probe frame: 16 x 6 mm bar under the probe flange (top at the flange
    underside), from the flange back to the drop plate at x_pos, drop plate 22 x 8 mm up to z_top_pos,
    three M4 cap screws through the flange into the bar."""
    z_fl = H_BOT + 0.255 * IN
    x0 = L_TOT - 0.304 * IN + 0.3
    parts = []
    bar = C.box(name + "_bar", (x_pos - x0, 22.0, 6.0), ((x0 + x_pos) / 2, 0, z_fl - 3.0), m["anod_pos"],
                bevel=0.4)
    parts.append(bar)
    drop = C.box(name + "_drop", (8.0, 26.0, z_top_pos - (z_fl - 6.0)),
                 (x_pos + 4.0, 0, (z_top_pos + z_fl - 6.0) / 2), m["anod_pos"], bevel=0.5)
    parts.append(drop)
    hx = L_TOT - 0.155 * IN
    for k in (-1, 0, 1):
        parts.append(cap_screw(name + f"_scr{k}", (hx, k * 0.4 * IN, H_BOT + H_BODY), 4.0, m))
    # screws holding the drop plate to the Z carriage
    for zz in (z_top_pos - 8.0, z_top_pos - 22.0):
        for yy in (-7.0, 7.0):
            s = cap_screw(name + f"_dps{zz:.0f}{yy:.0f}", (0, 0, 0), 4.0, m,
                          rot=Matrix.Translation(Vector((x_pos, yy, zz))) @ Matrix.Rotation(-math.pi / 2, 4, "Y"))
            parts.append(s)
    return parts


def micrometer(m, name, mtx, length=28.0):
    """Micrometer head along local +z from its mounting collar: sleeve, thimble, ratchet."""
    parts = []
    sl = lathe(name + "_sleeve", [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (4.8, 5.0), (4.8, length * 0.55),
                                  (0.0, length * 0.55)], m["micro"], seg=40)
    th = lathe(name + "_thimble", [(0.0, length * 0.5), (6.6, length * 0.5), (7.4, length * 0.58),
                                   (7.4, length * 0.95), (4.0, length * 0.98), (4.0, length * 1.15),
                                   (0.0, length * 1.15)], m["knurl"], seg=56)
    for o in (sl, th):
        o.matrix_world = mtx
        parts.append(o)
    return parts


def positioner(m, name, front, facing, z_platen, micrometers=2, h_body=88.8):
    """Manual XYZ positioner on the platen. `front`: (x, y) of the front-face centre; `facing`: unit
    (x, y) pointing from the positioner toward the probe. Returns (parts, carriage_front_x_local).
    Proportions after a published 89 x 60 mm, 88.8 mm tall RF positioner with magnetic base."""
    root = C.empty(name, loc=(front[0], front[1], z_platen),
                   rot=(0, 0, math.atan2(-facing[1], -facing[0])))
    # local frame: +x away from the probe (positioner extends to +x), front face at x = 0
    P = []
    base = C.box(name + "_base", (89.0, 60.0, 8.0), (44.5, 0, 4.0), m["base"], bevel=0.6)
    P.append(base)
    y1 = C.box(name + "_ystage", (80.0, 60.0, 16.0), (44.0, 0, 8.0 + 8.0), m["anod_pos"], bevel=0.5)
    x1 = C.box(name + "_xstage", (76.0, 58.0, 16.0), (44.0, 0, 24.0 + 8.0 + 0.4), m["anod_pos"], bevel=0.5)
    top = C.box(name + "_top", (64.0, 58.0, 12.0), (50.0, 0, 40.4 + 6.0 + 0.4), m["anod_pos"], bevel=0.8)
    P += [y1, x1, top]
    # cap screws on the top plate and the stage sides, blank label plate (no marks)
    for xx in (22.0, 78.0):
        for yy in (-24.0, 24.0):
            P.append(cap_screw(name + f"_ts{xx:.0f}{yy:.0f}", (xx, yy, 52.8), 3.0, m))
    P.append(C.box(name + "_plate", (30.0, 34.0, 0.4), (52.0, 0, 52.9), m["plate"], bevel=0.8))
    for sy in (-1, 1):
        for zz in (16.0, 32.4):
            for xx in (14.0, 74.0):
                P.append(cap_screw(name + f"_ss{sy}{zz:.0f}{xx:.0f}", (0, 0, 0), 3.0, m,
                                   rot=Matrix.Translation(Vector((xx, sy * 30.0 if zz < 20 else sy * 29.0, zz))) @
                                   Matrix.Rotation(-sy * math.pi / 2, 4, "X")))
    # roller-way seams
    for zz in (24.2, 40.6):
        P.append(C.box(name + f"_seam{zz:.0f}", (77.0, 60.6, 0.5), (44.0, 0, zz), m["hole"]))
    # Z column at the front, Z carriage plate facing the probe
    col = C.box(name + "_zcol", (16.0, 60.0, h_body - 8.0), (8.0 + 6.0, 0, 8.0 + (h_body - 8.0) / 2), m["anod_pos"],
                bevel=0.6)
    car = C.box(name + "_zcar", (6.0, 40.0, 52.0), (3.0, 0, 30.0 + 26.0), m["anod_pos"], bevel=0.4)
    P += [col, car]
    # tapped-hole rows on the column sides and on the carriage
    for sy in (-1, 1):
        for i in range(7):
            P.append(C.cylinder(name + f"_th{sy}{i}", 1.1, 0.4, (14.0, sy * 30.05, 16.0 + i * 10.0), m["hole"],
                                verts=12, rot=(math.radians(90), 0, 0)))
    # Z knob on a bracket on top of the column
    P.append(C.box(name + "_zbr", (14.0, 18.0, 8.0), (14.0, 0, h_body + 4.0), m["anod_pos"], bevel=0.4))
    P.append(C.cylinder(name + "_zstem", 2.0, 8.0, (14.0, 0, h_body + 12.0), m["ss"], verts=20))
    kn = lathe(name + "_zknob", [(0.0, 0.0), (12.5, 0.0), (12.5, 7.0), (11.5, 8.0), (0.0, 8.0)], m["knurl"], seg=72)
    kn.matrix_world = Matrix.Translation(Vector((14.0, 0, h_body + 15.0)))
    P.append(kn)
    P.append(C.cylinder(name + "_zcap", 11.3, 0.3, (14.0, 0, h_body + 15.0 + 8.05), m["plate"], verts=72))
    # X micrometer on the side, Y micrometer at the back
    mm = []
    mm += micrometer(m, name + "_mx", Matrix.Translation(Vector((40.0, 30.0, 32.0))) @
                     Matrix.Rotation(-math.pi / 2, 4, "X"))
    mm += micrometer(m, name + "_my", Matrix.Translation(Vector((89.0, -8.0, 16.0))) @
                     Matrix.Rotation(math.pi / 2, 4, "Y"))
    if micrometers >= 3:
        mm += micrometer(m, name + "_mz", Matrix.Translation(Vector((24.0, 0, h_body + 1.0))), length=24.0)
        for o in [o for o in P if o.name.startswith((name + "_zknob", name + "_zstem", name + "_zcap"))]:
            P.remove(o)
            bpy.data.objects.remove(o)
    for blk, (x, y, z) in ((name + "_mxc", (40.0, 29.0, 32.0)), (name + "_myc", (88.0, -8.0, 16.0))):
        P.append(C.box(blk, (12.0, 4.0, 12.0) if "mx" in blk else (4.0, 12.0, 12.0), (x, y, z), m["anod_pos"]))
    P += mm
    C.parent(P, root)
    return root


def platen(m, z_top, opening=(133.0, 133.0), size=(420.0, 360.0), t=14.0, y_off=0.0):
    """Nickel-plated platen plate with a rounded-rectangle opening over the chuck."""
    blk = C.box("platen", (size[0], size[1], t), (0, y_off, z_top - t / 2), m["platen"])
    ow, oh = opening
    r = 14.0
    pts = []
    for cx, cy, a0 in ((ow / 2 - r, oh / 2 - r, 0), (-ow / 2 + r, oh / 2 - r, 90), (-ow / 2 + r, -oh / 2 + r, 180),
                       (ow / 2 - r, -oh / 2 + r, 270)):
        for i in range(9):
            a = math.radians(a0 + 90 * i / 8)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    cut = C.extrude_polys("platen_open", [pts], z_top - t - 5, t + 10)
    boolean(blk, cut)
    md = blk.modifiers.new("bev", "BEVEL")
    md.width = 0.8
    md.segments = 2
    md.limit_method = "ANGLE"
    return blk
