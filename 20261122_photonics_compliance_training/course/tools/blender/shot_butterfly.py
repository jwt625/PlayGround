"""Shot 5: 14-pin butterfly laser diode in a finned TEC mount on an optical breadboard (M6 holes,
25 mm pitch), 900 um pigtail through an inline fiber isolator (arrow = forward direction) to an
FC/APC connector (green boot); an FC/PC patch cord (blue boot) lies next to it.

view=main     : bench view.             iso_reverse=1 : isolator spliced in backwards.
view=ferrules : macro side view of the two ferrule tips (8 degree APC polish vs PC dome).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import common as C  # noqa: E402

A = C.args(out="/tmp/bfly.jpg", samples=128, res=1920, noise=0.015, view="main", iso_reverse=0,
           look="AgX - Medium High Contrast", check_only=0)
rnd = random.Random(2)


# ------------------------------------------------------------------ materials
def mats():
    m = {}
    m["anod"] = C.principled("anod", color=(0.014, 0.014, 0.016), rough=0.42, rough_var=0.06, rough_scale=0.2,
                             bump=0.3, bump_scale=40.0, bump_dist=0.003)
    m["kovar_au"] = C.principled("pkg_gold", color=(0.98, 0.76, 0.42), metallic=1.0, rough=0.22, rough_var=0.06,
                                 rough_scale=0.5)
    m["lid"] = C.principled("pkg_lid", color=(0.80, 0.78, 0.74), metallic=1.0, rough=0.28, rough_var=0.08,
                            rough_scale=0.4)
    m["pin"] = C.principled("pin_au", color=(1.0, 0.76, 0.38), metallic=1.0, rough=0.15)
    m["ss"] = C.principled("steel", color=(0.62, 0.62, 0.63), metallic=1.0, rough=0.25, rough_var=0.06,
                           rough_scale=1.0)
    m["nickel"] = C.principled("nickel", color=(0.70, 0.68, 0.64), metallic=1.0, rough=0.2)
    m["rubber"] = C.principled("rubber", color=(0.02, 0.02, 0.02), rough=0.65)
    m["yellow"] = C.principled("jacket_yellow", color=(0.80, 0.52, 0.03), rough=0.4, coat=0.2)
    m["green"] = C.principled("apc_green", color=(0.05, 0.42, 0.10), rough=0.35, coat=0.2)
    m["blue"] = C.principled("pc_blue", color=(0.03, 0.14, 0.60), rough=0.35, coat=0.2)
    m["zirc"] = C.principled("zirconia", color=(0.88, 0.87, 0.84), rough=0.2, ior=2.1)
    m["engr"] = C.principled("mark", color=(0.15, 0.15, 0.15), rough=0.6)
    m["white_mark"] = C.principled("wmark", color=(0.85, 0.85, 0.85), rough=0.6)
    m["ink"] = C.principled("ink", color=(0.01, 0.01, 0.01), rough=0.5)
    m["hexsock"] = C.principled("hexsock", color=(0.01, 0.01, 0.01), rough=0.7)
    m["boot_slot"] = C.principled("boot_slot", color=(0.06, 0.06, 0.06), rough=0.6)
    m["bronze"] = C.principled("bronze", color=(0.80, 0.50, 0.30), metallic=1.0, rough=0.3)
    m["nickel_satin"] = C.principled("nickel_satin", color=(0.72, 0.70, 0.66), metallic=1.0, rough=0.22)
    m["thread"] = C.principled("thread_dark", color=(0.06, 0.06, 0.06), metallic=1.0, rough=0.4)
    m["grey_plastic"] = C.principled("grey_plastic", color=(0.18, 0.18, 0.19), rough=0.55)
    m["label_white"] = C.principled("label_white", color=(0.85, 0.85, 0.83), rough=0.45)
    m["sleeve"] = C.principled("splice_sleeve", color=(0.95, 0.95, 0.95), rough=0.1, ior=1.45, transmission=0.9)
    m["socket"] = C.principled("socket", color=(0.03, 0.03, 0.03), rough=0.5)
    return m


def breadboard_material():
    """Black anodized breadboard with M6 tapped holes on a 25 mm grid (shader holes + chamfer bump)."""
    mt = C.principled("breadboard", color=(0.016, 0.016, 0.018), rough=0.5, rough_var=0.08, rough_scale=0.02)
    nt = mt.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])

    def cell(sock):
        d = nt.nodes.new("ShaderNodeMath")
        d.operation = "MULTIPLY_ADD"  # x / 25 + 0.5 -> hole centres on multiples of 25 mm
        d.inputs[1].default_value = 1.0 / 25.0
        d.inputs[2].default_value = 0.5
        nt.links.new(sock, d.inputs[0])
        fr = nt.nodes.new("ShaderNodeMath")
        fr.operation = "FRACT"
        nt.links.new(d.outputs[0], fr.inputs[0])
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        nt.links.new(fr.outputs[0], sub.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 25.0
        nt.links.new(sub.outputs[0], mul.inputs[0])
        return mul.outputs[0]

    cx, cy = cell(sep.outputs["X"]), cell(sep.outputs["Y"])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(cx, cmb.inputs[0])
    nt.links.new(cy, cmb.inputs[1])
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(cmb.outputs[0], ln.inputs[0])
    # height: 0 inside the hole (r < 2.5), ramp across the chamfer to 1 at r = 3.4
    h = nt.nodes.new("ShaderNodeMapRange")
    h.inputs["From Min"].default_value = 2.5
    h.inputs["From Max"].default_value = 3.4
    nt.links.new(ln.outputs["Value"], h.inputs["Value"])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = 1.0
    b.inputs["Distance"].default_value = 0.6
    nt.links.new(h.outputs["Result"], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
    hole = nt.nodes.new("ShaderNodeMapRange")
    hole.inputs["From Min"].default_value = 2.45
    hole.inputs["From Max"].default_value = 2.6
    nt.links.new(ln.outputs["Value"], hole.inputs["Value"])
    # tapped hole interior: M6 thread crests (1 mm pitch, seen in perspective as close rings) on
    # a dark bore that gets darker toward the centre (depth)
    thr = nt.nodes.new("ShaderNodeMath")
    thr.operation = "MULTIPLY"
    thr.inputs[1].default_value = 2 * math.pi / 0.30
    nt.links.new(ln.outputs["Value"], thr.inputs[0])
    sn = nt.nodes.new("ShaderNodeMath")
    sn.operation = "SINE"
    nt.links.new(thr.outputs[0], sn.inputs[0])
    sn2 = nt.nodes.new("ShaderNodeMath")
    sn2.operation = "MULTIPLY_ADD"
    sn2.inputs[1].default_value = 0.5
    sn2.inputs[2].default_value = 0.5
    nt.links.new(sn.outputs[0], sn2.inputs[0])
    depth = nt.nodes.new("ShaderNodeMapRange")
    depth.inputs["From Min"].default_value = 0.6
    depth.inputs["From Max"].default_value = 2.5
    depth.inputs["To Min"].default_value = 0.0
    depth.inputs["To Max"].default_value = 0.10
    nt.links.new(ln.outputs["Value"], depth.inputs["Value"])
    inner = nt.nodes.new("ShaderNodeMath")
    inner.operation = "MULTIPLY"
    nt.links.new(sn2.outputs[0], inner.inputs[0])
    nt.links.new(depth.outputs["Result"], inner.inputs[1])
    inner_rgb = nt.nodes.new("ShaderNodeCombineColor")
    for i in range(3):
        nt.links.new(inner.outputs[0], inner_rgb.inputs[i])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(inner_rgb.outputs[0], mix.inputs["A"])
    mix.inputs["B"].default_value = (0.016, 0.016, 0.018, 1)
    nt.links.new(hole.outputs["Result"], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    # bore and threads are bare machined aluminium: metallic inside the hole
    met = nt.nodes.new("ShaderNodeMapRange")
    met.inputs["From Min"].default_value = 2.45
    met.inputs["From Max"].default_value = 2.6
    met.inputs["To Min"].default_value = 0.8
    met.inputs["To Max"].default_value = 0.0
    nt.links.new(ln.outputs["Value"], met.inputs["Value"])
    nt.links.new(met.outputs["Result"], p.inputs["Metallic"])
    return mt


# ------------------------------------------------------------------ parts
def shcs(name, loc, d_head, h_head, m, washer=0.0):
    """Socket head cap screw head (+ optional washer) seated at loc (bottom of head)."""
    parts = []
    z = loc[2]
    if washer:
        parts.append(C.cylinder(name + "_w", washer / 2, d_head * 0.16, (loc[0], loc[1], z + d_head * 0.08),
                                m["ss"], verts=48, bevel=0.1))
        z += d_head * 0.16
    parts.append(C.cylinder(name + "_h", d_head / 2, h_head, (loc[0], loc[1], z + h_head / 2), m["ss"], verts=48,
                            bevel=d_head * 0.04))
    parts.append(C.cylinder(name + "_hex", d_head * 0.25, h_head * 0.6, (loc[0], loc[1], z + h_head - h_head * 0.3 + 0.01),
                            m["hexsock"], verts=6))
    return parts


def butterfly(m, origin, rot_z):
    """14-pin butterfly package (body 20.8 x 12.7 x 7.6 mm on a 30 mm flange). Local frame: fiber
    exits along -x from the snout; 7 straight pins per long side (2.54 mm pitch) at z = 2."""
    root = C.empty("bfly", loc=origin, rot=(0, 0, math.radians(rot_z)))
    parts = []
    parts.append(C.box("flange", (30.0, 12.7, 1.0), (0, 0, 0.5), m["kovar_au"], bevel=0.15))
    parts.append(C.box("body", (20.8, 12.7, 7.6), (0, 0, 1.0 + 3.8), m["kovar_au"], bevel=0.25))
    parts.append(C.box("lid", (20.0, 11.9, 0.3), (0, 0, 8.75), m["lid"], bevel=0.1))
    # four M2 screws through the flange tabs into the mount
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts += shcs(f"fl{sx}{sy}", (sx * 13.0, sy * 4.2, 1.0), 3.8, 2.0, m)
    # straight pins out of the long side walls
    for sy in (-1, 1):
        for i in range(7):
            x = (i - 3) * 2.54
            parts.append(C.box(f"pin{sy}{i}", (0.5, 7.0, 0.25), (x, sy * (6.35 + 3.5), 2.0), m["pin"]))
    # fiber snout (on axis) and strain-relief boot
    parts.append(C.cylinder("snout", 1.75, 6.0, (-10.4 - 3.0, 0, 5.0), m["kovar_au"], verts=48,
                            rot=(0, math.radians(90), 0), bevel=0.1))
    # boot: 1.6 mm radius at the snout, tapering to 1.0 mm at the free end where the 900 um tube exits
    parts.append(C.cylinder("boot", 1.0, 18.0, (-16.4 - 9.0, 0, 5.0), m["rubber"], verts=48,
                            rot=(0, math.radians(90), 0), r2=1.6))
    parts.append(C.text_mesh("lid_t1", "DFB  1550 nm", 1.5, (0, 1.6, 8.9), h=0.01, mat=m["white_mark"]))
    parts.append(C.text_mesh("lid_t2", "SN 24-0817", 1.1, (0, -1.4, 8.9), h=0.01, mat=m["white_mark"]))
    C.parent(parts, root)
    return root


def dsub(m, name, face_x, y, z, sgn, cable_nodes):
    """D-sub 9 receptacle on an end face (normal +-x) with a mated plug, backshell and cable; the cable
    leaves the backshell on axis and follows cable_nodes [(point, tangent or None)]."""
    parts = []
    parts.append(C.box(name + "_shell", (1.0, 18.0, 10.0), (face_x + sgn * 0.5, y, z), m["nickel"], bevel=0.4))
    parts.append(C.box(name + "_plug", (8.0, 16.5, 8.5), (face_x + sgn * 5.0, y, z), m["nickel"], bevel=0.6))
    parts.append(C.box(name + "_back", (16.0, 31.0, 12.0), (face_x + sgn * 17.0, y, z), m["grey_plastic"],
                       bevel=1.5))
    for sy in (-1, 1):
        parts.append(C.cylinder(name + f"_jk{sy}", 1.6, 6.0, (face_x + sgn * 11.0, y + sy * 12.5, z), m["nickel"],
                                verts=24, rot=(0, math.radians(90), 0), bevel=0.2))
    s0 = Vector((face_x + sgn * 25.0, y, z))
    parts.append(route(name + "_cable", [(s0, (sgn, 0, 0))] + cable_nodes, 2.6, m["grey_plastic"],
                       touch=(name + "_back",), start=(s0, (sgn, 0, 0))))
    return parts


def mount(m, origin):
    """Butterfly laser mount: base plate with four counterbored M6 clearance holes on the 25 mm
    grid (+-50, +-25 mm from the centre), finned heat-sink body with a TEC under the package seat,
    pin sockets with clamp bars on both long sides, D-sub connectors (laser, TEC) with cables."""
    ox, oy, oz = origin
    parts = []
    base = C.box("mount_base", (112.0, 62.0, 8.0), (ox, oy, 4.0), m["anod"], bevel=0.5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cb = C.cylinder("cb", 5.6, 13.0, (ox + sx * 50, oy + sy * 25, 8.0), None, verts=48)
            th = C.cylinder("th", 3.3, 30.0, (ox + sx * 50, oy + sy * 25, 0.0), None, verts=32)
            for c in (cb, th):
                md = base.modifiers.new("b", "BOOLEAN")
                md.object = c
                md.solver = "EXACT"
                c.hide_render = True
                c.hide_viewport = True
            # M6 socket head cap screw seated in the counterbore (head 10 x 6 mm), into the grid hole
            parts += shcs("m6", (ox + sx * 50, oy + sy * 25, 8.0 - 6.5), 10.0, 6.0, m)
    base.modifiers.move(0, len(base.modifiers) - 1)  # bevel after the booleans
    # finned body (heat sink) 44 x 30 x 16 on the base, fins along the long sides
    parts.append(C.box("mount_body", (44.0, 30.0, 16.0), (ox, oy, 8.0 + 8.0), m["anod"], bevel=0.4))
    for sy in (-1, 1):
        for i in range(6):
            parts.append(C.box(f"fin{sy}{i}", (44.0, 1.4, 14.0), (ox, oy + sy * (15.0 + 1.2 + i * 2.6), 8.0 + 7.0),
                               m["anod"], bevel=0.15))
    # package seat plate (copper, gold flashed) on top of the TEC
    parts.append(C.box("seat", (32.0, 16.0, 1.5), (ox, oy, 24.0 + 0.75), m["kovar_au"], bevel=0.2))
    # pin sockets (black glass-filled plastic) under the pins, clamp bars over them, 2 x M2 each
    pin_z = 25.5 + 2.0
    for sy in (-1, 1):
        yc = oy + sy * 10.5
        parts.append(C.box(f"sock{sy}", (24.0, 4.5, pin_z - 0.125 - 24.0), (ox, yc, (24.0 + pin_z - 0.125) / 2),
                           m["socket"], bevel=0.2))
        parts.append(C.box(f"clamp{sy}", (21.0, 3.6, 1.4), (ox, yc, pin_z + 0.125 + 0.7), m["anod"], bevel=0.25))
        for sx in (-1, 1):
            parts += shcs(f"cl{sy}{sx}", (ox + sx * 9.0, yc, pin_z + 0.125 + 1.4), 3.8, 1.6, m)
    # end-face labels and D-sub connectors: laser diode (+x end), TEC (-x end, under the snout)
    parts.append(C.text_mesh("lbl_ld", "LD", 2.2, (ox + 22.05, oy - 10.5, 21.5), h=0.01, mat=m["white_mark"],
                             rot=(math.radians(90), 0, math.radians(90))))
    parts.append(C.text_mesh("lbl_tec", "TEC", 2.2, (ox - 22.05, oy + 10.0, 21.5), h=0.01, mat=m["white_mark"],
                             rot=(math.radians(90), 0, math.radians(-90))))
    # cables (5.2 mm OD) clear the base plate (they used to cut through it) and lie on the table
    c = 2.6
    parts += dsub(m, "db_ld", ox + 22.0, oy, 15.0, 1, [
        ((ox + 63, oy + 6, 12.5), None), ((ox + 73, oy + 25, 6.0), None), ((ox + 77, oy + 50, c), None),
        ((ox + 79, oy + 105, c), None), ((ox + 80, oy + 199, c), None)])
    # TEC cable: turns toward the back (R ~ 28 mm) while dropping past the plate's left edge, away from the
    # fiber pigtail that comes down on the front side of the same end
    parts += dsub(m, "db_tec", ox - 22.0, oy, 15.0, -1, [
        ((ox - 61, oy + 3.75, 12.2), None), ((ox - 71.25, oy + 14, 5.4), None), ((ox - 75, oy + 28, c), (0, 1, 0)),
        ((ox - 73, oy + 65, c), None), ((ox - 65, oy + 125, c), None), ((ox - 40, oy + 199, c), None)])
    return parts


FC_LEN = 52.0  # tip to boot end (mm): 24 mm plug + 28 mm boot


def fc_rest(tip_xy, yaw_xy):
    """Pose of an FC connector lying on the table: it rests on the rear edge of the knurled nut
    (r = 5.0 mm, 24 mm behind the tip) and on the boot end (r = 1.35 mm, 52 mm behind the tip), so
    the axis rises toward the tip at atan(3.65 / 28) = 7.4 deg. Returns (tip, direction)."""
    a = math.atan2(5.0 - 1.35, 52.0 - 24.0)
    h = Vector((yaw_xy[0], yaw_xy[1], 0.0)).normalized()
    d = h * math.cos(a) + Vector((0, 0, math.sin(a)))
    return Vector((tip_xy[0], tip_xy[1], 5.0 * math.cos(a) + 24.0 * math.sin(a))), d


def _knurl_sleeve(name, r_out, r_root, z0, z1, mat, teeth=48):
    """Coupling-nut knurl: straight axial grooves, as on FC nuts (star profile extruded along z)."""
    bm = bmesh.new()
    prof = []
    for i in range(teeth * 2):
        a = 2 * math.pi * i / (teeth * 2)
        r = r_out if i % 2 == 0 else r_root
        prof.append((r * math.cos(a), r * math.sin(a)))
    v0 = [bm.verts.new((x, y, z0)) for x, y in prof]
    v1 = [bm.verts.new((x, y, z1)) for x, y in prof]
    n = len(prof)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((v0[i], v0[j], v1[j], v1[i]))
    bm.faces.new(list(reversed(v0)))
    bm.faces.new(v1)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    C.set_mat(ob, mat)
    return ob


def fc_connector(m, name, tip, direction, boot_mat, apc=True, ferrule_roll_deg=0.0, cap=None):
    """FC connector (IEC 61754-13 style), local z = axis, ferrule tip at z = 0. Layers, front to back,
    sized from vendor drawings and photos (see media_plan_render.md): zirconia ferrule (2.5 mm)
    protruding ~4.3 mm from the plug-body barrel (6 mm OD) that carries the anti-rotation key (a
    ~2 mm wide radial tab at the barrel root, in the APC angle plane) and a split bronze retaining
    ring; the free-rotating coupling nut (internal M8 x 0.75 thread; smooth front section, axially
    knurled rear section, ~10 mm OD); strain-relief boot tapering toward the cable; optional cap."""
    d = Vector(direction).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    root = C.empty(name, loc=tip)
    root.matrix_world = Matrix.Translation(Vector(tip)) @ rot
    parts = []
    fr = ferrule_mesh(name + "_ferrule", apc, m["zirc"], L=8.0)
    fr.rotation_euler = (0, 0, math.radians(ferrule_roll_deg))
    parts.append(fr)
    zb0, zb1 = -4.3, -9.2                      # plug-body barrel
    parts.append(C.cylinder(name + "_barrel", 3.0, zb0 - zb1, (0, 0, (zb0 + zb1) / 2), m["nickel"], verts=64,
                            bevel=0.15))
    parts.append(C.cylinder(name + "_bore", 1.32, 0.3, (0, 0, zb0 + 0.12), m["hexsock"], verts=48))
    # key: radial tab on the barrel root, toward local +y (the key plane = APC angle plane)
    parts.append(C.box(name + "_key", (2.0, 1.0, 2.2), (0, 3.4, zb1 + 1.1), m["nickel"], bevel=0.08))
    # split bronze retaining ring at the barrel root, gap at the side
    ring = C.cylinder(name + "_ring", 3.25, 0.35, (0, 0, zb1 + 0.25), m["bronze"], verts=64)
    hole = C.cylinder(name + "_ringh", 3.0, 1.0, (0, 0, zb1 + 0.25), None, verts=64)
    gap = C.box(name + "_ringg", (0.6, 2.0, 1.0), (3.1, -0.4, zb1 + 0.25), None)
    for c in (hole, gap):
        md = ring.modifiers.new("b", "BOOLEAN")
        md.object = c
        md.solver = "EXACT"
        c.hide_render = True
        c.hide_viewport = True
        c.parent = root
    parts.append(ring)
    # coupling nut: smooth front section with the M8 opening, knurled rear section
    zn0, zn1, zn2 = -9.0, -13.6, -24.0
    nut = C.cylinder(name + "_nutf", 4.8, zn0 - zn1, (0, 0, (zn0 + zn1) / 2), m["nickel"], verts=96, bevel=0.25)
    opening = C.cylinder(name + "_nuto", 3.65, 3.0, (0, 0, zn0), None, verts=64)
    md = nut.modifiers.new("b", "BOOLEAN")
    md.object = opening
    md.solver = "EXACT"
    nut.modifiers.move(len(nut.modifiers) - 1, 0)
    opening.hide_render = True
    opening.hide_viewport = True
    opening.parent = root
    parts.append(nut)
    parts.append(C.cylinder(name + "_thread", 3.66, 1.4, (0, 0, zn0 - 0.75), m["thread"], verts=64))
    parts.append(_knurl_sleeve(name + "_knurl", 5.0, 4.75, zn2, zn1, m["nickel_satin"]))
    # crimp sleeve behind the nut, then a short strain-relief boot that starts at about the crimp
    # diameter and tapers to the jacket, with flex slots over its rear two thirds
    parts.append(C.cylinder(name + "_crimp", 2.6, 2.0, (0, 0, zn2 - 1.0), m["nickel"], verts=48, bevel=0.2))
    zb_0, L_boot, rb0, rb1 = zn2 - 1.5, 26.5, 3.0, 1.35
    parts.append(C.cylinder(name + "_boot", rb0, L_boot, (0, 0, zb_0 - L_boot / 2), boot_mat, verts=64, r2=rb1,
                            rot=(math.pi, 0, 0), bevel=0.3))
    for i in range(8):
        z = zb_0 - 9.0 - i * 2.1
        rr = rb0 + (rb1 - rb0) * ((zb_0 - z) / L_boot)
        parts.append(C.cylinder(name + f"_slot{i}", rr + 0.01, 0.5, (0, 0, z), m["boot_slot"], verts=48))
    if cap is not None:  # PVC dust cap over the ferrule, seated on the barrel front
        parts.append(C.cylinder(name + "_cap", 2.4, 6.0, (0, 0, zb0 + 3.0 + 0.05), cap, verts=48, bevel=0.5))
        parts.append(C.cylinder(name + "_caplip", 2.9, 0.8, (0, 0, zb0 + 0.45), cap, verts=48, bevel=0.2))
    C.parent(parts, root)
    bpy.context.view_layer.update()
    for p in parts:
        p.matrix_parent_inverse = Matrix.Identity(4)
    back = Vector(tip) - d * FC_LEN
    return root, back


def ferrule_mesh(name, apc, mat, r=1.25, L=10.0):
    """2.5 mm zirconia ferrule, tip at z = 0. APC: end face polished at 8 deg (plus slight dome);
    PC: spherical dome (radius ~ 15 mm). 125 um fiber end visible as a darker disc."""
    bm = bmesh.new()
    seg, rings = 96, 18
    t8 = math.tan(math.radians(8.0))
    verts_rings = []
    # end face as a grid of rings
    for k in range(rings + 1):
        rr = r * 0.80 * k / rings
        ring = []
        for j in range(seg):
            a = 2 * math.pi * j / seg
            x, y = rr * math.cos(a), rr * math.sin(a)
            if apc:
                z = -y * t8 - (x * x + y * y) / (2 * 12.0)  # angle in the key plane
            else:
                z = -(x * x + y * y) / (2 * 15.0)
            ring.append(bm.verts.new((x, y, z)))
        verts_rings.append(ring)
    # chamfer to full radius
    ch = []
    for j in range(seg):
        a = 2 * math.pi * j / seg
        x, y = r * math.cos(a), r * math.sin(a)
        z0 = verts_rings[-1][j].co.z
        ch.append(bm.verts.new((x, y, z0 - 0.35)))
    tail = []
    for j in range(seg):
        a = 2 * math.pi * j / seg
        tail.append(bm.verts.new((r * math.cos(a), r * math.sin(a), -L)))
    center = verts_rings[0]
    for k in range(rings):
        a_, b_ = verts_rings[k], verts_rings[k + 1]
        for j in range(seg):
            jj = (j + 1) % seg
            if k == 0:
                continue
            bm.faces.new((a_[j], b_[j], b_[jj], a_[jj]))
    cpt = bm.verts.new((0, 0, 0))
    for j in range(seg):
        jj = (j + 1) % seg
        bm.faces.new((cpt, verts_rings[1][j], verts_rings[1][jj]))
    for a_, b_ in ((verts_rings[-1], ch), (ch, tail)):
        for j in range(seg):
            jj = (j + 1) % seg
            bm.faces.new((a_[j], b_[j], b_[jj], a_[jj]))
    bm.faces.new(list(reversed(tail)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    for p in ob.data.polygons:
        p.use_smooth = True
    C.set_mat(ob, mat)
    # fiber end (125 um) as a darker glass disc on the end face
    fib = C.principled(name + "_glass", color=(0.25, 0.27, 0.28), rough=0.05, ior=1.46)
    disc = C.cylinder(name + "_fib", 0.0625, 0.004, (0, 0, 0.002), fib, verts=32)
    disc.rotation_euler = (-math.radians(8.0) if apc else 0, 0, 0)
    disc.parent = ob
    return ob


def lay_cable(name, pts, radius, mat):
    return C.curve_tube(name, [Vector(p) for p in pts], radius, mat, resolution=32, bevel_res=4)


TABLE_XY = ((-300.0, 300.0), (-225.0, 225.0))  # breadboard footprint, top face at z = 0
ROUTES = []  # fibers and cables registered for check_routes()


def route(name, nodes, radius, mat, touch=(), start=None, end=None, min_bend=20.0, hf=0.36):
    """Fiber or cable as a cubic Bezier through nodes [(point, tangent or None)] with explicit handles
    (tangent given, else the bisector of the neighbour chords; 0.36 x chord per side), so the shape is
    not left to AUTO handles: nodes on the table (z = radius) get a horizontal tangent and two such
    nodes in a row give a span that stays exactly on the table. touch: objects this line enters or
    starts from (boots, splice sleeves), skipped in the solid check; start/end: (point, axis) the line
    must leave from / arrive at, tangent to the component axis."""
    P = [Vector(p) for p, _ in nodes]
    T = []
    for i, (_, t) in enumerate(nodes):
        if t is not None:
            v = Vector(t)
        elif i == 0:
            v = P[1] - P[0]
        elif i == len(P) - 1:
            v = P[-1] - P[-2]
        else:
            v = (P[i] - P[i - 1]).normalized() + (P[i + 1] - P[i]).normalized()
        if t is None and abs(P[i].z - radius) < 1e-6:
            v.z = 0.0
        T.append(v.normalized())
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = 32
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(P) - 1)
    for i, bp in enumerate(sp.bezier_points):
        bp.handle_left_type = bp.handle_right_type = "FREE"
        bp.co = P[i]
        hl = hf * (P[i] - P[i - 1]).length if i > 0 else 1.0
        hr = hf * (P[i + 1] - P[i]).length if i < len(P) - 1 else 1.0
        bp.handle_left = P[i] - T[i] * hl
        bp.handle_right = P[i] + T[i] * hr
    ob = C.link(bpy.data.objects.new(name, cu))
    cu.materials.append(mat)
    ROUTES.append(dict(name=name, r=radius, touch=tuple(touch), start=start, end=end, min_bend=min_bend))
    return ob


def _left(u):
    return Vector((-u.y, u.x))


def _hd(deg):
    return Vector((math.cos(math.radians(deg)), math.sin(math.radians(deg))))


def csc(p1, h1, rho1, p2, h2, rho2):
    """Plan-view path from pose (p1, heading h1 deg) to pose (p2, h2): arc of signed radius rho1
    (+ left / counterclockwise, - right), common tangent line, arc rho2 (Dubins CSC with unequal radii).
    Returns primitives ("arc", centre, rho, start heading, sweep deg) / ("line", start, dir, length)."""
    p1, p2 = Vector(p1).to_2d(), Vector(p2).to_2d()
    u1, u2 = _hd(h1), _hd(h2)
    c1, c2 = p1 + rho1 * _left(u1), p2 + rho2 * _left(u2)
    D, dl = c2 - c1, rho2 - rho1
    L = math.sqrt(D.length_squared - dl * dl)
    u = Vector((L * D.x + dl * D.y, -dl * D.x + L * D.y)) / (L * L + dl * dl)
    hu = math.degrees(math.atan2(u.y, u.x))

    def sweep(a, b, rho):
        return (b - a) % 360.0 if rho > 0 else -((a - b) % 360.0)
    t1 = c1 - rho1 * _left(u)
    return [("arc", c1, rho1, h1, sweep(h1, hu, rho1)), ("line", t1, u, L), ("arc", c2, rho2, hu, sweep(hu, h2, rho2))]


def line(p, h, length):
    return [("line", Vector(p).to_2d(), _hd(h), length)]


def _prim_len(pr):
    return pr[3] if pr[0] == "line" else abs(pr[2]) * abs(math.radians(pr[4]))


def plan_eval(prims, s):
    """Plan position and unit heading at horizontal arc length s."""
    for pr in prims:
        n = _prim_len(pr)
        if s <= n + 1e-9 or pr is prims[-1]:
            if pr[0] == "line":
                return pr[1] + pr[2] * s, pr[2].copy()
            _, c, rho, h0, sw = pr
            u = _hd(h0 + math.copysign(math.degrees(s / abs(rho)), sw))
            return c - rho * _left(u), u
        s -= n
    raise ValueError


def zprofile(keys):
    """Height and slope dz/ds from segments (s0, s1, z0, z1, kind) held flat in between. kind "S": two
    parabolas, zero slope at both ends (curvature 4 dz / L^2); "P0": one parabola, zero slope at s0;
    "P1": one parabola, zero slope at s1 (end slope 2 dz / L, used to meet a tilted boot)."""
    def f(s):
        z = keys[0][2]
        for s0, s1, z0, z1, kind in keys:
            if s < s0:
                return z, 0.0
            if s > s1:
                z = z1
                continue
            L, h, u = s1 - s0, z1 - z0, (s - s0) / (s1 - s0)
            if kind == "S":
                return (z0 + 2 * h * u * u, 4 * h * u / L) if u < 0.5 else (z1 - 2 * h * (1 - u) ** 2, 4 * h * (1 - u) / L)
            if kind == "P0":
                return z0 + h * u * u, 2 * h * u / L
            return z1 - h * (1 - u) ** 2, 2 * h * (1 - u) / L
        return z, 0.0
    return f


def path_nodes(prims, zf, step=3.0):
    """Dense Bezier nodes (point, exact tangent) along plan primitives with the height profile zf."""
    total = sum(_prim_len(p) for p in prims)
    n = max(2, int(math.ceil(total / step)))
    out = []
    for i in range(n + 1):
        s = total * i / n
        p, u = plan_eval(prims, s)
        z, dz = zf(s)
        out.append((Vector((p.x, p.y, z)), Vector((u.x, u.y, dz))))
    return out, total


def _centerline(ob, res=256):
    """Tessellated centerline of a curve object (what Cycles renders, at a finer resolution)."""
    cu = ob.data
    bd, ru = cu.bevel_depth, cu.resolution_u
    cu.bevel_depth, cu.resolution_u = 0.0, res
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    pts = [ob.matrix_world @ v.co for v in me.vertices]
    ev.to_mesh_clear()
    cu.bevel_depth, cu.resolution_u = bd, ru
    return pts


def _resample(pts, step):
    """Polyline resampled at uniform arc-length spacing (last point kept)."""
    out, acc, nxt = [pts[0].copy()], 0.0, step
    for a, b in zip(pts[:-1], pts[1:]):
        L = (b - a).length
        while L > 0 and acc + L >= nxt:
            out.append(a.lerp(b, (nxt - acc) / L))
            nxt += step
        acc += L
    if (out[-1] - pts[-1]).length > 1e-6:
        out.append(pts[-1].copy())
    return out


def check_routes(routes, exclude=("ceiling", "dust"), tol=0.02, table_xy=TABLE_XY, floor="breadboard"):
    """Sample every registered fiber / cable densely and print violations of: (a) table top: centre at
    least one radius above z = 0 over the breadboard; (b) solids: signed distance from the centre to
    every other rendered solid (mesh, cable, other fiber) at least one radius; (c) continuity: one open
    spline, no sample gaps, ends on their attachment points and tangent to the component axis; bend
    radius >= the route minimum. Also lists solids that sink into the table top. Returns the count."""
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update()
    lines = {r["name"]: _centerline(bpy.data.objects[r["name"]]) for r in routes}
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    solids = {}
    bad = 0
    for ob in bpy.data.objects:
        if ob.type not in ("MESH", "CURVE") or ob.hide_render or ob.name.split(".")[0] in exclude:
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        if me is None or not me.polygons:
            ev.to_mesh_clear()
            continue
        vs = [ob.matrix_world @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        ev.to_mesh_clear()
        lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
        hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
        solids[ob.name] = (BVHTree.FromPolygons(vs, polys), lo, hi)
        if ob.type == "MESH" and lo.z < -tol and ob.name != floor:
            print(f"CHECK solid  {ob.name:22s} sinks {-lo.z:.2f} mm into the table top")
            bad += 1
    for r in routes:
        nm, rad = r["name"], r["r"]
        raw = lines[nm]
        sp = bpy.data.objects[nm].data.splines
        if len(sp) != 1 or sp[0].use_cyclic_u:
            print(f"CHECK {nm}: {len(sp)} splines (expected one open spline)")
            bad += 1
        gap = max((b - a).length for a, b in zip(raw[:-1], raw[1:]))
        pts = _resample(raw, 0.25)
        length = sum((b - a).length for a, b in zip(raw[:-1], raw[1:]))
        # (a) table top
        tmin, tat = 1e9, None
        for p in pts:
            if table_xy[0][0] <= p.x <= table_xy[0][1] and table_xy[1][0] <= p.y <= table_xy[1][1]:
                if p.z - rad < tmin:
                    tmin, tat = p.z - rad, p
        if tmin < -tol:
            print(f"CHECK {nm}: below table top by {-tmin:.2f} mm at ({tat.x:.1f}, {tat.y:.1f})")
            bad += 1
        # (b) solids
        worst = {}
        for p in pts:
            for sn, (bvh, lo, hi) in solids.items():
                if sn == nm or sn.split(".")[0] in r["touch"]:
                    continue
                m = rad + 0.1
                if not (lo.x - m <= p.x <= hi.x + m and lo.y - m <= p.y <= hi.y + m and lo.z - m <= p.z <= hi.z + m):
                    continue
                inside = lo.x <= p.x <= hi.x and lo.y <= p.y <= hi.y and lo.z <= p.z <= hi.z
                hit = bvh.find_nearest(p) if inside else bvh.find_nearest(p, m)
                if hit[0] is None:
                    continue
                loc, nrm = hit[0], hit[1]
                d = (p - loc).length
                sd = d if (p - loc).dot(nrm) >= 0 else -d
                c = sd - rad
                if c < worst.get(sn, (1e9, None))[0]:
                    worst[sn] = (c, p)
        for sn, (c, p) in sorted(worst.items(), key=lambda kv: kv[1][0]):
            if c < -tol:
                print(f"CHECK {nm}: inside {sn} by {-c:.2f} mm at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
                bad += 1
        # (c) continuity, attachment, bend radius
        if gap > 2.0:
            print(f"CHECK {nm}: gap of {gap:.2f} mm between centerline samples")
            bad += 1
        for key, p0, p1 in (("start", raw[0], pts[min(8, len(pts) - 1)]),
                            ("end", raw[-1], pts[max(0, len(pts) - 9)])):
            if r[key] is None:
                continue
            att, ax = Vector(r[key][0]), Vector(r[key][1]).normalized()
            tang = (p1 - p0).normalized() if key == "start" else (p0 - p1).normalized()
            off = (p0 - att).length
            ang = math.degrees(tang.angle(ax))
            if off > 0.05 or ang > 3.0:
                print(f"CHECK {nm}: {key} {off:.2f} mm from its attachment, {ang:.1f} deg off the axis")
                bad += 1
        q = _resample(raw, 0.5)
        k = 4
        rmin, rat = 1e9, None
        for i in range(k, len(q) - k):
            a, b, c3 = q[i - k], q[i], q[i + k]
            ar = (b - a).cross(c3 - a).length / 2
            if ar < 1e-9:
                continue
            R = (b - a).length * (c3 - b).length * (c3 - a).length / (4 * ar)
            if R < rmin:
                rmin, rat = R, b
        if rmin < r["min_bend"]:
            print(f"CHECK {nm}: bend radius {rmin:.1f} mm < {r['min_bend']:.0f} mm at ({rat.x:.1f}, {rat.y:.1f}, {rat.z:.1f})")
            bad += 1
        cl = min([c for c, _ in worst.values()] + [99.0])
        print(f"CHECK {nm:8s} r={rad:.2f} len={length:6.1f} mm  table clr={tmin:+.2f}  solid clr={cl:+.2f}  "
              f"min bend R={rmin:6.1f} mm  max step={gap:.2f} mm")
    print(f"CHECK violations: {bad}")
    return bad


def isolator(m, center, direction, reverse):
    d = Vector(direction).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    root = C.empty("iso")
    root.matrix_world = Matrix.Translation(Vector(center)) @ rot
    parts = [C.cylinder("iso_body", 2.75, 35.0, (0, 0, 0), m["ss"], verts=64, bevel=0.3)]
    for s in (1, -1):
        parts.append(C.cylinder(f"iso_boot{s}", 1.2, 8.0, (0, 0, s * 21.5), m["rubber"], verts=32,
                                r2=0.7, rot=(0 if s > 0 else math.pi, 0, 0)))  # same tapered boot on both ends
    # printed label band (white) with a long direction arrow (about 60 % of the body length) and text;
    # the reversed variant uses the same decal turned end for end
    parts.append(C.cylinder("iso_label", 2.78, 26.0, (0, 0, 0), m["label_white"], verts=64))
    sgn = -1 if reverse else 1
    L = 21.0
    arrow = [(-0.35, -L / 2 * sgn), (0.35, -L / 2 * sgn), (0.35, (L / 2 - 3.2) * sgn), (1.25, (L / 2 - 3.2) * sgn),
             (0, L / 2 * sgn), (-1.25, (L / 2 - 3.2) * sgn), (-0.35, (L / 2 - 3.2) * sgn)]
    if reverse:
        arrow = list(reversed(arrow))
    ar = C.extrude_polys("iso_arrow", [[(x, y) for x, y in arrow]], 0.0, 0.03, m["ink"])
    ar.rotation_euler = (math.radians(90), 0, 0)
    ar.location = (0, 2.80, 0)
    parts.append(ar)
    t = C.text_mesh("iso_t", "ISO 1550 nm  SM", 1.1, (0, 0, 0), h=0.03, mat=m["ink"])
    t.rotation_euler = (math.radians(-90), math.radians(-90), 0)
    t.location = (-1.6 if not reverse else 1.6, 2.80, 0.0)
    if reverse:
        t.rotation_euler = (math.radians(-90), math.radians(90), 0)
    parts.append(t)
    C.parent(parts, root)
    bpy.context.view_layer.update()
    for p in parts:
        p.matrix_parent_inverse = Matrix.Identity(4)
    return root


# ------------------------------------------------------------------ scenes
def scene_main():
    m = mats()
    C.world_hdri("hospital_room", strength=0.6, rot_z_deg=110, bg_color=(0.03, 0.03, 0.033))
    bb = C.box("breadboard", (600, 450, 12.7), (0, 0, -6.35), breadboard_material())
    # mount on the 25 mm grid (screws into the holes at x = -25 / 75, y = 0 / 50), package on its seat
    mount(m, (25.0, 25.0, 0.0))
    butterfly(m, (25.0, 25.0, 25.5), 0.0)
    # pigtail (900 um tube): leaves the boot on axis, arcs down past the left edge of the base plate
    # (clear of the TEC D-sub and its cable), lands on the board and runs around the plate's front
    # corner; bend radii >= 20 mm. The inline isolator is fusion-spliced in, each splice protected by a
    # 40 mm heat-shrink sleeve lying on the board, the fiber running through it beside its steel rod.
    r = 0.45
    b0 = Vector((25.0 - 34.4, 25.0, 30.5))
    iso_c = Vector((-60, -45, 2.78))             # resting on its label band (r 2.78)
    iso_d = Vector((-1, 0.25, 0)).normalized()   # forward = away from the laser (toward the left)
    iso_in, iso_out = iso_c - iso_d * 25.5, iso_c + iso_d * 25.5
    zs = 1.9                                     # fiber height inside a sleeve (axis 1.5, rod below)

    def at(p, z):
        return Vector((p.x, p.y, z))

    isolator(m, iso_c, iso_d, A["iso_reverse"] == 1)
    rot_d = iso_d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    for k, c in enumerate((iso_in - iso_d * 31, iso_out + iso_d * 31)):
        sl = C.cylinder(f"splice{k}", 1.5, 40.0, (0, 0, 0), m["sleeve"], verts=32, bevel=0.4)
        sl.matrix_world = Matrix.Translation(at(c, 1.5)) @ rot_d
        rod = C.cylinder(f"splice_rod{k}", 0.5, 38.0, (0, 0, 0), m["ss"], verts=16)
        rod.matrix_world = Matrix.Translation(at(c, 1.5 - 0.6)) @ rot_d
    # pig1, plan view: left arc R 28 from the boot around the plate's front-left corner, tangent line,
    # right hairpin R 22 onto the isolator axis, straight through the splice sleeve into the isolator
    e0 = iso_in - iso_d * 80
    h_iso = math.degrees(math.atan2(iso_d.y, iso_d.x))
    pr = csc(b0, 180.0, 28.0, e0, h_iso, -22.0) + line(e0, h_iso, 80.0)
    tot = sum(_prim_len(p) for p in pr)
    zf = zprofile([(0.0, 85.0, b0.z, r, "S"), (tot - 80.0, tot - 63.0, r, zs, "S"),
                   (tot - 11.0, tot, zs, iso_in.z, "S")])
    nodes, _ = path_nodes(pr, zf)
    route("pig1", nodes, r, m["yellow"], touch=("boot", "splice0", "iso_boot-1"), start=(b0, (-1, 0, 0)),
          end=(iso_in, iso_d), hf=1 / 3)
    # FC/APC at the end of the pigtail, resting on its nut and boot, dust cap on
    cap_g = C.principled("cap_green", color=(0.05, 0.40, 0.10), rough=0.45)
    cap_b = C.principled("cap_blue", color=(0.03, 0.13, 0.55), rough=0.45)
    apc_tip, apc_dir = fc_rest((70.0, -96.0), (1, 0.12))
    apc_back = apc_tip - apc_dir * FC_LEN
    apc_h = Vector((apc_dir.x, apc_dir.y, 0)).normalized()
    # pig2: straight out of the isolator through the second sleeve, then a left loop R 22 (dropping to
    # the board over its first 34 mm) and a gentle left arc (R 150) onto the connector axis; the last
    # 13.8 mm rise to the tilted boot
    lz = 2 * (apc_back.z - r) / math.tan(math.asin(apc_dir.z))
    p3 = iso_out + iso_d * 51
    land = apc_back - apc_h * lz
    h_apc = math.degrees(math.atan2(apc_h.y, apc_h.x))
    pr = line(iso_out, h_iso, 51.0) + csc(p3, h_iso, 22.0, land, h_apc, 150.0) + line(land, h_apc, lz)
    tot = sum(_prim_len(p) for p in pr)
    zf = zprofile([(0.0, 11.0, iso_out.z, zs, "S"), (51.0, 85.0, zs, r, "S"), (tot - lz, tot, r, apc_back.z, "P0")])
    nodes, _ = path_nodes(pr, zf)
    route("pig2", nodes, r, m["yellow"], touch=("iso_boot1", "splice1", "fc_apc_boot"), start=(iso_out, iso_d),
          end=(apc_back, apc_dir), hf=1 / 3)
    fc_connector(m, "fc_apc", apc_tip, apc_dir, m["green"], apc=True, cap=cap_g)
    # FC/PC patch cord (2 mm jacket) beside it, also capped; the cord runs to the board edge and hangs
    # over it (out of frame)
    pc_tip, pc_dir = fc_rest((76.0, -122.0), (1, -0.05))
    fc_connector(m, "fc_pc", pc_tip, pc_dir, m["blue"], apc=False, cap=cap_b)
    pcb = pc_tip - pc_dir * FC_LEN
    pc_h = Vector((pc_dir.x, pc_dir.y, 0)).normalized()
    lz = 2 * (pcb.z - 1.0) / math.tan(math.asin(pc_dir.z))
    h_pc = math.degrees(math.atan2(-pc_h.y, -pc_h.x))
    p1 = pcb - pc_h * lz
    pr = line(pcb, h_pc, lz) + csc(p1, h_pc, 300.0, (-300.0, -130.0), 180.0, -300.0)
    tot = sum(_prim_len(p) for p in pr)
    nodes, _ = path_nodes(pr, zprofile([(0.0, lz, pcb.z, 1.0, "P1")]))
    # over the board edge (top corner at x = -300): R 24 arc around (-300, z -23), then hanging
    for k in range(1, 7):
        a = math.radians(15 * k)
        nodes.append((Vector((-300 - 24 * math.sin(a), -130, -23 + 24 * math.cos(a))), Vector((-math.cos(a), 0, -math.sin(a)))))
    nodes.append((Vector((-324, -130, -80)), Vector((0, 0, -1))))
    route("pc_cord", nodes, 1.0, m["yellow"], touch=("fc_pc_boot",), start=(pcb, -pc_dir), hf=1 / 3)
    dust = C.principled("dust", color=(0.5, 0.48, 0.45), rough=0.9)
    C.scatter_dust("dust", ((-150, 150), (-180, 120)), 0.0, 260, 0.03, 0.2, dust, seed=5, fibers=5,
                   fiber_len=(2.0, 6.0))
    if A["view"] == "iso_close":   # isolator and its arrow, laser side for context on the right
        tgt = Vector((-38, -30, 4))
        el, az, D, field = math.radians(36), math.radians(-80), 650.0, 135.0
    else:
        tgt = Vector((-8, -45, 6))
        el, az, D, field = math.radians(40), math.radians(-70), 650.0, 285.0
    loc = tgt + D * Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    cam = C.camera(loc, tgt, lens=36 * D / field)
    if A["view"] == "iso_close":
        C.set_aperture(cam, Vector((-60, -45, 3)), pupil_mm=7.0)
    else:
        C.set_aperture(cam, Vector((-40, -62, 3)), pupil_mm=9.0)
    C.area_light("key", (-250, 150, 380), (0, 0, 0), 300, 5.5e5, color=(1.0, 0.97, 0.93))
    C.area_light("fill", (350, -250, 220), (0, 0, 0), 300, 8.0e4, color=(0.95, 0.97, 1.0))
    C.area_light("rim", (100, 400, 120), (0, 0, 0), 200, 1.0e5, color=(1.0, 0.98, 0.95))
    C.ceiling(900.0, color=(0.05, 0.05, 0.05))
    C.compositor(vignette=0.22, dispersion=0.003)
    check_routes(ROUTES)


def scene_ferrules():
    m = mats()
    C.world_hdri("studio_small_09", strength=0.5, rot_z_deg=30, bg_color=(0.02, 0.02, 0.022))
    # both connectors lie on an anti-static mat, caps off, key up (APC angle in the key plane, so the
    # 8 degree face is seen in profile); APC behind, PC in front, camera low from the front
    C.box("table", (400, 400, 10), (0, 100, -5.0), C.principled("mat", color=(0.035, 0.038, 0.04), rough=0.75,
                                                              bump=0.2, bump_scale=2.0, bump_dist=0.01))
    # each connector rests on the rear edge of its knurled nut and on its boot end (fc_rest); the 2 mm
    # jacket drops from the boot onto the mat, then curves gently toward the back (R 400 mm arcs)
    for nm, y, mat, apc in (("fc_apc", 15.0, m["green"], True), ("fc_pc", 0.0, m["blue"], False)):
        tip, d = fc_rest((0.0, y), (1, 0))
        fc_connector(m, nm, tip, d, mat, apc=apc)
        b = tip - d * FC_LEN
        lz = 2 * (b.z - 1.0) / math.tan(math.asin(d.z))
        p1 = b - Vector((lz, 0, 0))
        pr = line(b, 180.0, lz) + csc(p1, 180.0, -400.0, (b.x - 140, y + 14), 173.5, 400.0)
        nodes, _ = path_nodes(pr, zprofile([(0.0, lz, b.z, 1.0, "P1")]))
        route(nm + "_jacket", nodes, 1.0, m["yellow"], touch=(nm + "_boot",), start=(b, -d), hf=1 / 3)
    tgt = Vector((-24.0, 7.5, 4.0))
    el = math.radians(28.0)
    loc = tgt + 220.0 * Vector((0, -math.cos(el), math.sin(el)))
    cam = C.camera(loc, tgt, lens=36 * 220 / 66.0)
    C.set_aperture(cam, Vector((-8, 7.5, 5.0)), pupil_mm=1.6)
    C.area_light("key", (-40, -60, 80), (0, 0, 4), 60, 1.5e4, color=(1.0, 0.97, 0.93))
    C.area_light("back", (40, 80, 30), (0, 0, 4), 60, 1.2e4, color=(0.95, 0.97, 1.0))
    C.area_light("fill", (80, -80, 10), (0, 0, 4), 60, 6.0e3)
    C.compositor(vignette=0.2, dispersion=0.004)
    check_routes(ROUTES, table_xy=((-200.0, 200.0), (-100.0, 300.0)), floor="table")


if __name__ == "__main__":
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    scene_ferrules() if A["view"] == "ferrules" else scene_main()
    if not A["check_only"]:
        C.render(A["out"])
