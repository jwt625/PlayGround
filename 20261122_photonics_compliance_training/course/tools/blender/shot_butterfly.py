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
           look="AgX - Medium High Contrast")
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
    parts.append(C.cylinder("boot", 1.6, 18.0, (-16.4 - 9.0, 0, 5.0), m["rubber"], verts=48,
                            rot=(0, math.radians(90), 0), r2=1.0))
    parts.append(C.text_mesh("lid_t1", "DFB  1550 nm", 1.5, (0, 1.6, 8.9), h=0.01, mat=m["white_mark"]))
    parts.append(C.text_mesh("lid_t2", "SN 24-0817", 1.1, (0, -1.4, 8.9), h=0.01, mat=m["white_mark"]))
    C.parent(parts, root)
    return root


def dsub(m, name, face_x, y, z, sgn, cable_pts):
    """D-sub 9 receptacle on an end face (normal +-x) with a mated plug, backshell and cable."""
    parts = []
    parts.append(C.box(name + "_shell", (1.0, 18.0, 10.0), (face_x + sgn * 0.5, y, z), m["nickel"], bevel=0.4))
    parts.append(C.box(name + "_plug", (8.0, 16.5, 8.5), (face_x + sgn * 5.0, y, z), m["nickel"], bevel=0.6))
    parts.append(C.box(name + "_back", (16.0, 31.0, 12.0), (face_x + sgn * 17.0, y, z), m["grey_plastic"],
                       bevel=1.5))
    for sy in (-1, 1):
        parts.append(C.cylinder(name + f"_jk{sy}", 1.6, 6.0, (face_x + sgn * 11.0, y + sy * 12.5, z), m["nickel"],
                                verts=24, rot=(0, math.radians(90), 0), bevel=0.2))
    parts.append(lay_cable(name + "_cable", [Vector((face_x + sgn * 25.0, y, z))] + [Vector(p) for p in cable_pts],
                           2.6, m["grey_plastic"]))
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
    parts += dsub(m, "db_ld", ox + 22.0, oy, 15.0, 1, [(ox + 60, oy + 10, 12), (ox + 75, oy + 60, 2.6),
                                                     (ox + 80, oy + 200, 2.6)])
    parts += dsub(m, "db_tec", ox - 22.0, oy, 15.0, -1, [(ox - 52, oy + 15, 8), (ox - 50, oy + 60, 2.6),
                                                       (ox - 40, oy + 200, 2.6)])
    return parts


FC_LEN = 52.0  # tip to boot end (mm): 24 mm plug + 28 mm boot


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


def isolator(m, center, direction, reverse):
    d = Vector(direction).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    root = C.empty("iso")
    root.matrix_world = Matrix.Translation(Vector(center)) @ rot
    parts = [C.cylinder("iso_body", 2.75, 35.0, (0, 0, 0), m["ss"], verts=64, bevel=0.3)]
    for s in (1, -1):
        parts.append(C.cylinder(f"iso_boot{s}", 1.2, 8.0, (0, 0, s * 21.5), m["rubber"], verts=32,
                                r2=0.7 if s > 0 else 1.2, rot=(0 if s > 0 else math.pi, 0, 0)))
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
    # pigtail (900 um tube): leaves the boot on axis, droops to the board with > 25 mm bend radii; the
    # inline isolator is fusion-spliced in, each splice protected by a 40 mm heat-shrink sleeve
    r = 0.45
    b0 = Vector((25.0 - 34.4, 25.0, 30.5))
    iso_c = Vector((-60, -45, 2.75))
    iso_d = Vector((-1, 0.25, 0)).normalized()   # forward = away from the laser (toward the left)
    iso_in, iso_out = iso_c - iso_d * 25.5, iso_c + iso_d * 25.5
    pig1 = [b0, b0 + Vector((-12, 0, -0.5)), Vector((-34, 4, 10)), Vector((0, -26, r)), Vector((50, -38, r)),
            Vector((66, -60, r)), iso_in - iso_d * 74, iso_in - iso_d * 50, iso_in - iso_d * 12, iso_in]
    lay_cable("pig1", pig1, r, m["yellow"])
    isolator(m, iso_c, iso_d, A["iso_reverse"] == 1)
    for k, c in enumerate((iso_in - iso_d * 31, iso_out + iso_d * 31)):
        sl = C.cylinder(f"splice{k}", 1.5, 40.0, (0, 0, 0), m["sleeve"], verts=32, bevel=0.4)
        sl.matrix_world = Matrix.Translation(c + Vector((0, 0, 1.5 - r))) @ iso_d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        rod = C.cylinder(f"splice_rod{k}", 0.5, 38.0, (0, 0, 0), m["ss"], verts=16)
        rod.matrix_world = Matrix.Translation(c + Vector((0, 0, 1.5 - r)) + Vector((0, 0, -0.6))) @ iso_d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    # FC/APC at the end of the pigtail, resting on its nut and boot, dust cap on
    cap_g = C.principled("cap_green", color=(0.05, 0.40, 0.10), rough=0.45)
    cap_b = C.principled("cap_blue", color=(0.03, 0.13, 0.55), rough=0.45)
    apc_dir = Vector((1, 0.12, 0.07)).normalized()
    apc_tip = Vector((70.0, -96.0, 5.1))
    apc_back = apc_tip - apc_dir * FC_LEN
    pig2 = [iso_out, iso_out + iso_d * 12, iso_out + iso_d * 50, iso_out + iso_d * 62 + Vector((0, 0, -0.5)),
            Vector((-152, -60, r)), Vector((-120, -104, r)), Vector((-50, -112, r)),
            apc_back - apc_dir * 14 + Vector((0, 0, -0.6)), apc_back]
    lay_cable("pig2", pig2, r, m["yellow"])
    fc_connector(m, "fc_apc", apc_tip, apc_dir, m["green"], apc=True, cap=cap_g)
    # FC/PC patch cord (2 mm jacket) beside it, also capped; cord runs off the board edge
    pc_dir = Vector((1, -0.05, 0.07)).normalized()
    pc_tip = Vector((76.0, -122.0, 5.1))
    fc_connector(m, "fc_pc", pc_tip, pc_dir, m["blue"], apc=False, cap=cap_b)
    pcb = pc_tip - pc_dir * FC_LEN
    lay_cable("pc_cord", [pcb, pcb - pc_dir * 14 + Vector((0, 0, -0.6)), Vector((-40, -136, 1.0)),
                          Vector((-110, -150, 1.0)), Vector((-200, -140, 1.0)), Vector((-320, -130, 1.0))], 1.0,
              m["yellow"])
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


def scene_ferrules():
    m = mats()
    C.world_hdri("studio_small_09", strength=0.5, rot_z_deg=30, bg_color=(0.02, 0.02, 0.022))
    # both connectors lie on an anti-static mat, caps off, key up (APC angle in the key plane, so the
    # 8 degree face is seen in profile); APC behind, PC in front, camera low from the front
    C.box("table", (400, 400, 10), (0, 100, -5.0), C.principled("mat", color=(0.035, 0.038, 0.04), rough=0.75,
                                                              bump=0.2, bump_scale=2.0, bump_dist=0.01))
    d = Vector((1, 0, 0.056)).normalized()  # resting on the knurled nut and the boot end
    d = Vector((1, 0, (5.0 - 1.35) / FC_LEN)).normalized()  # resting on the knurled nut and the boot end
    for nm, y, mat, apc in (("fc_apc", 15.0, m["green"], True), ("fc_pc", 0.0, m["blue"], False)):
        tip = Vector((0.0, y, 5.0))
        fc_connector(m, nm, tip, d, mat, apc=apc)
        b = tip - d * FC_LEN
        lay_cable(nm + "_jacket", [b + d * 1.0, b - d * 8.0, b - d * 25.0 + Vector((0, 2, -0.3)),
                                    Vector((b.x - 70, y + 6, 1.0)), Vector((b.x - 140, y + 14, 1.0))], 1.0, m["yellow"])
    tgt = Vector((-24.0, 7.5, 4.0))
    el = math.radians(28.0)
    loc = tgt + 220.0 * Vector((0, -math.cos(el), math.sin(el)))
    cam = C.camera(loc, tgt, lens=36 * 220 / 66.0)
    C.set_aperture(cam, Vector((-8, 7.5, 5.0)), pupil_mm=1.6)
    C.area_light("key", (-40, -60, 80), (0, 0, 4), 60, 1.5e4, color=(1.0, 0.97, 0.93))
    C.area_light("back", (40, 80, 30), (0, 0, 4), 60, 1.2e4, color=(0.95, 0.97, 1.0))
    C.area_light("fill", (80, -80, 10), (0, 0, 4), 60, 6.0e3)
    C.compositor(vignette=0.2, dispersion=0.004)


if __name__ == "__main__":
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    scene_ferrules() if A["view"] == "ferrules" else scene_main()
    C.render(A["out"])
