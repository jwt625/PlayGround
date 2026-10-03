"""Wafer geometry shared by wafer_300mm, wafer_300mm_siph and wafer_tape_frame builds (mm, asset space)."""
import math

import bpy
from mathutils import Vector

import ft

MM = ft.MM
R_WAFER = 150.0
T_WAFER = 0.775          # SEMI M1 nominal for 300 mm
NOTCH_DEPTH = 1.0        # SEMI M1: 1.00 +0.25/-0 mm, 90 deg included angle
FIELD_X, FIELD_Y = 26.0, 33.0
DIE_X, DIE_Y = 7.0, 9.0   # project PIC, about
SCRIBE = 0.1
PITCH_X, PITCH_Y = DIE_X + SCRIBE, DIE_Y + SCRIBE
DIES_PER_FIELD = (3, 3)
R_USE = 147.0            # 3 mm edge exclusion
FILM_T = 0.020           # exaggerated die film (real BEOL about 10 um)

PASS_RGB = (0.10, 0.90, 0.25)
FAIL_RGB = (0.80, 0.08, 0.08)
IDLE_RGB = (0.10, 0.14, 0.26)


def wafer_outline(R=R_WAFER, depth=NOTCH_DEPTH, N=1080, notch=True):
    """(x, y) outline points (mm) with a V notch at -Y (6 o'clock), counter-clockwise, star-shaped about the centre."""
    th0 = -math.pi / 2
    wa = depth / R if notch else 0.0
    angs = []
    for k in range(N):
        a = -math.pi + 2 * math.pi * k / N
        if notch and abs(a - th0) < wa * 1.2:
            continue
        angs.append((a, R))
    if notch:
        angs += [(th0 - wa, R), (th0, R - depth), (th0 + wa, R)]
    angs.sort()
    return [(r * math.cos(a), r * math.sin(a)) for a, r in angs]


def disc(A, name, mat, z0=0.0, T=T_WAFER, R=R_WAFER, notch=True, parent=None, coll=None, N=1080, center=(0, 0)):
    """Wafer disc with chamfered/rounded edge and a notch; origin ends at the bbox centre (bottom at z0)."""
    out = wafer_outline(R, N=N, notch=notch)
    prof = [(0.30, 0.0), (0.08, 0.10), (0.0, 0.25), (0.0, T - 0.25), (0.08, T - 0.10), (0.30, T)]
    verts = []
    ringidx = []
    nrm = []
    n0 = len(out)
    for i, (x, y) in enumerate(out):
        px, py = out[i - 1]
        qx, qy = out[(i + 1) % n0]
        tx, ty = qx - px, qy - py          # CCW tangent -> outward normal is (ty, -tx)
        L = math.hypot(tx, ty) or 1.0
        nrm.append((ty / L, -tx / L))
    for (d, z) in prof:
        idx = []
        for (x, y), (nx, ny) in zip(out, nrm):
            verts.append(((x - nx * d + center[0]) * MM, (y - ny * d + center[1]) * MM, (z0 + z) * MM))
            idx.append(len(verts) - 1)
        ringidx.append(idx)
    cb = len(verts)
    verts.append((center[0] * MM, center[1] * MM, z0 * MM))
    ct = len(verts)
    verts.append((center[0] * MM, center[1] * MM, (z0 + T) * MM))
    faces = []
    n = len(out)
    for i in range(len(prof) - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append((ringidx[i][j], ringidx[i][j2], ringidx[i + 1][j2], ringidx[i + 1][j]))
    for j in range(n):
        j2 = (j + 1) % n
        faces.append((cb, ringidx[0][j2], ringidx[0][j]))
        faces.append((ct, ringidx[-1][j], ringidx[-1][j2]))
    ob = A.obj_from_mesh(name, verts, faces, mat, parent=parent, coll=coll, smooth=True, sharp_deg=40.0)
    for p in ob.data.polygons:
        if abs(p.normal.z) > 0.9999:
            p.use_smooth = False
    return ob


# ---------------------------------------------------------------- die layout
def die_layout():
    """List of dicts for every full die inside the usable radius.
    Field grid: wafer centre at the centre of a 26 x 33 mm field; 3 x 3 dies at 7.1 x 9.1 pitch fill the lower-left
    21.3 x 27.3 mm of each field, the remaining right (4.7 mm) and top (5.7 mm) bands carry test structures."""
    dies = []
    nfx, nfy = 7, 7
    for fj in range(-nfy, nfy + 1):
        for fi in range(-nfx, nfx + 1):
            X0 = fi * FIELD_X - FIELD_X / 2
            Y0 = fj * FIELD_Y - FIELD_Y / 2
            for j in range(DIES_PER_FIELD[1]):
                for i in range(DIES_PER_FIELD[0]):
                    x0 = X0 + i * PITCH_X + SCRIBE / 2
                    y0 = Y0 + j * PITCH_Y + SCRIBE / 2
                    corners = [(x0, y0), (x0 + DIE_X, y0), (x0, y0 + DIE_Y), (x0 + DIE_X, y0 + DIE_Y)]
                    if all(math.hypot(cx, cy) <= R_USE for cx, cy in corners):
                        dies.append({"fi": fi, "fj": fj, "i": i, "j": j, "x": x0 + DIE_X / 2, "y": y0 + DIE_Y / 2})
    # serpentine probe order (rows from -Y to +Y, alternating direction)
    rows = sorted({round(d["y"], 3) for d in dies})
    order = 0
    for ri, ry in enumerate(rows):
        row = sorted([d for d in dies if round(d["y"], 3) == ry], key=lambda d: d["x"], reverse=(ri % 2 == 1))
        for d in row:
            d["order"] = order
            d["row"] = ri
            order += 1
    for d in dies:
        cols = sorted({round(e["x"], 3) for e in dies})
        d["col"] = cols.index(round(d["x"], 3))
    return sorted(dies, key=lambda d: d["order"])


def field_list(dies):
    return sorted({(d["fi"], d["fj"]) for d in dies})


# ---------------------------------------------------------------- die mesh
def die_boxes(level="full", si_thick=0.0, size=(DIE_X, DIE_Y)):
    """Boxes for one die in local coordinates (centre of the die, z = top of silicon is 0). Slot 0 film (tinted),
    slot 1 gold (pads, gratings), slot 2 silicon (diced tile body, only if si_thick > 0)."""
    sx, sy = size
    bx = []
    if si_thick > 0:
        bx.append((0, 0, -si_thick / 2, sx, sy, si_thick, 2))
    bx.append((0, 0, FILM_T / 2, DIE_X, DIE_Y, FILM_T, 0))
    zt = FILM_T
    npad = 12 if level == "full" else 6
    pp = (DIE_X - 1.0) / (npad - 1)
    for k in range(npad):
        x = -(DIE_X - 1.0) / 2 + k * pp
        for y in (DIE_Y / 2 - 0.35, -DIE_Y / 2 + 0.35):
            bx.append((x, y, zt + 0.003, 0.16, 0.16, 0.006, 1))
    if level == "full":
        for k in range(8):                        # grating coupler array along the left edge
            bx.append((-DIE_X / 2 + 0.5, -1.4 + k * 0.4, zt + 0.002, 0.45, 0.06, 0.004, 1))
        for yy in (0.6, -0.6):                    # bus waveguides
            bx.append((0.3, yy, zt + 0.0015, 5.6, 0.012, 0.003, 1))
        for k in range(12):                       # microring cells
            for yy, sg in ((0.6, 1), (-0.6, -1)):
                bx.append((-2.2 + k * 0.4, yy + sg * 0.1, zt + 0.0025, 0.11, 0.11, 0.005, 1))
    else:
        for k in range(4):
            bx.append((-DIE_X / 2 + 0.5, -0.6 + k * 0.4, zt + 0.002, 0.45, 0.06, 0.004, 1))
    return bx


def die_material(A, glow_gain=3.0):
    """MAT_fab_test_die_state: base colour and emission follow the OBJECT colour (r, g, b = state colour,
    a = glow 0..1). Per-die recolouring needs no extra materials."""
    full = "MAT_fab_test_die_state"
    if full in bpy.data.materials:
        return bpy.data.materials[full]
    m = bpy.data.materials.new(full)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = glow_gain
    mul.label = "glow gain (alpha * gain)"
    nt.links.new(oi.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(oi.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(oi.outputs["Alpha"], mul.inputs[0])
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    b.inputs["Metallic"].default_value = 0.0
    b.inputs["Roughness"].default_value = 0.35
    A.mats[full] = m
    return m


def build_die_proto(A, name, level, mats, si_thick=0.0, size=(DIE_X, DIE_Y), z_top=T_WAFER):
    ob = A.boxes_obj(name, die_boxes(level, si_thick, size), mats, pos=(0, 0, z_top), bottom=(si_thick > 0))
    ob.color = (*IDLE_RGB, 0.0)
    return ob


def place_dies(A, proto, dies, prefix, z_top=T_WAFER, parent=None, coll=None, center=(0, 0)):
    objs = []
    for d in dies:
        nm = "%s_die_r%02d_c%02d" % (prefix, d["row"], d["col"])
        ob = A.dup(proto, nm, (center[0] + d["x"], center[1] + d["y"], z_top), parent=parent, coll=coll)
        ob.color = (*IDLE_RGB, 0.0)
        ob.visible_shadow = False
        ob["die_row"] = d["row"]
        ob["die_col"] = d["col"]
        ob["die_x_mm"] = round(d["x"], 4)
        ob["die_y_mm"] = round(d["y"], 4)
        ob["die_probe_order"] = d["order"]
        ob["die_field"] = "%d,%d" % (d["fi"], d["fj"])
        objs.append(ob)
    return objs


def pcm_and_marks(A, dies, mats, z_top=T_WAFER, parent=None, coll=None, center=(0, 0)):
    """Merged meshes: scribe-band test structures, alignment crosses and reticle outlines for every field with a die."""
    fields = field_list(dies)
    pcm, marks, lines = [], [], []
    for (fi, fj) in fields:
        X0 = fi * FIELD_X - FIELD_X / 2
        Y0 = fj * FIELD_Y - FIELD_Y / 2
        # right band (x from 21.3 to 26): vertical column of test structure blocks
        bx0 = X0 + 3 * PITCH_X + 0.35
        bw = FIELD_X - 3 * PITCH_X - 0.7
        for k in range(10):
            yy = Y0 + 1.2 + k * 3.1
            if math.hypot(bx0 + bw / 2, yy) > R_USE:
                continue
            pcm.append((center[0] + bx0 + bw / 2, center[1] + yy, z_top + 0.004, bw * 0.8, 2.0, 0.008, 1))
            for q in range(3):
                pcm.append((center[0] + bx0 + 0.6 + q * 1.4, center[1] + yy, z_top + 0.0085, 0.5, 0.5, 0.003, 0))
        # top band (y from 27.3 to 33)
        by0 = Y0 + 3 * PITCH_Y + 0.35
        bh = FIELD_Y - 3 * PITCH_Y - 0.7
        for k in range(6):
            xx = X0 + 1.9 + k * 3.6
            if math.hypot(xx, by0 + bh / 2) > R_USE:
                continue
            pcm.append((center[0] + xx, center[1] + by0 + bh / 2, z_top + 0.004, 2.8, bh * 0.7, 0.008, 1))
        # alignment crosses at field corners
        for (cx, cy) in ((X0 + 0.5, Y0 + 0.5), (X0 + FIELD_X - 0.5, Y0 + FIELD_Y - 0.5)):
            if math.hypot(cx, cy) > R_USE:
                continue
            marks.append((center[0] + cx, center[1] + cy, z_top + 0.003, 0.9, 0.08, 0.006, 1))
            marks.append((center[0] + cx, center[1] + cy, z_top + 0.003, 0.08, 0.9, 0.006, 1))
        # reticle field outline (thin strips) only if the field fully lies in the usable radius
        ok = all(math.hypot(X0 + a * FIELD_X, Y0 + b * FIELD_Y) <= R_USE for a in (0, 1) for b in (0, 1))
        if ok:
            for (cx, cy, sx, sy) in ((X0 + FIELD_X / 2, Y0, FIELD_X, 0.08), (X0 + FIELD_X / 2, Y0 + FIELD_Y, FIELD_X, 0.08),
                                     (X0, Y0 + FIELD_Y / 2, 0.08, FIELD_Y), (X0 + FIELD_X, Y0 + FIELD_Y / 2, 0.08, FIELD_Y)):
                lines.append((center[0] + cx, center[1] + cy, z_top + 0.0015, sx, sy, 0.003, 1))
    o1 = A.boxes_obj(A.id + "_scribe_test_structures", pcm, [mats[0], mats[1]], parent=parent, coll=coll)
    o2 = A.boxes_obj(A.id + "_alignment_marks", marks, [mats[0], mats[1]], parent=parent, coll=coll)
    o3 = A.boxes_obj(A.id + "_reticle_outlines", lines, [mats[0], mats[1]], parent=parent, coll=coll)
    for o in (o1, o2, o3):
        o.visible_shadow = False
    return o1, o2, o3
