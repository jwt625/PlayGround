"""Shot 3: 100 mm lithium niobate wafers (x-cut), 0.5 mm thick.

scene=carrier : blank DOUBLE-SIDE-POLISHED wafer (optical grade): water-clear with a faint yellow
                tint, front and back reflections, the molded carrier floor seen through it. It rests
                on the ledge of an open single-wafer carrier (pocket 102 mm, 1 mm clearance all round).
scene=hotplate: processed SINGLE-SIDE-POLISHED device wafer: mirror-polished front carrying a grid of
                6 mm modulator dies in gold, lapped (frosted) back, so the body looks milky with no
                see-through; lying on the aluminium plate of a lab hot plate set to 115 C.

Flats (stated convention, SEMI-style for 100 mm wafers): primary flat 32.5 mm, secondary flat
18 mm, secondary 90 degrees clockwise from the primary seen from the polished face. LN vendors
differ in flat lengths and in which crystal axis the primary marks; this is a representative layout,
not a vendor drawing.
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

A = C.args(out="/tmp/wafer.jpg", samples=128, res=1920, noise=0.015, scene="carrier", look="AgX - Medium High Contrast")
RW, TW = 50.0, 0.5
PRIMARY, SECONDARY = 32.5, 18.0


def wafer_outline(n=360, prim_angle=-90.0):
    """Outline points: circle with a primary flat centred at prim_angle and a secondary flat at
    prim_angle + 90 deg (clockwise when viewed from +z means angle decreases)."""
    # secondary 90 degrees clockwise from the primary, seen from the polished (+z) face
    flats = [(math.radians(prim_angle), PRIMARY), (math.radians(prim_angle - 90.0), SECONDARY)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        p = Vector((RW * math.cos(a), RW * math.sin(a)))
        for fa, L in flats:
            d = math.sqrt(RW ** 2 - (L / 2) ** 2)
            u = Vector((math.cos(fa), math.sin(fa)))
            if p.dot(u) > d:
                p = p - u * (p.dot(u) - d)
        if not pts or (p - pts[-1]).length > 1e-4:
            pts.append(p)
    return [(p.x, p.y) for p in pts]


def ln_materials():
    m = {}
    # LN, n ~ 2.2 (about 14 % reflectance per polished face), colourless to faint yellow
    m["ln_front"] = C.principled("ln_polished", color=(1.0, 0.93, 0.74), rough=0.0, ior=2.2, transmission=1.0)
    m["ln_lapped"] = C.principled("ln_lapped", color=(1.0, 0.95, 0.82), rough=0.6, ior=2.2, transmission=1.0)
    m["ln_edge"] = C.principled("ln_edge", color=(1.0, 0.95, 0.82), rough=0.25, ior=2.2, transmission=1.0)
    return m


def build_wafer(m, name="wafer", loc=(0, 0, 0), rot_z=0.0, back="lapped"):
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0)) for x, y in wafer_outline()]
    f = bm.faces.new(vs)
    if f.normal.z < 0:
        f.normal_flip()
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv, vec=(0, 0, TW))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for face in bm.faces:
        if face.normal.z > 0.9:
            face.material_index = 0
        elif face.normal.z < -0.9:
            face.material_index = 1
        else:
            face.material_index = 2
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    me.materials.append(m["ln_front"])
    me.materials.append(m["ln_lapped"] if back == "lapped" else m["ln_front"])
    me.materials.append(m["ln_edge"])
    bv = ob.modifiers.new("bevel", "BEVEL")
    bv.width = 0.12
    bv.segments = 3
    bv.limit_method = "ANGLE"
    bv.angle_limit = math.radians(60)
    ob.location = loc
    ob.rotation_euler = (0, 0, math.radians(rot_z))
    # thin parallel plate: let light through for shadow rays (caustics are off, and a 0.5 mm plate
    # barely displaces transmitted light), otherwise everything under the wafer renders in shadow
    ob.visible_shadow = False
    return ob


def die_grid(au, parent, z, pitch=6.1, edge_excl=3.0):
    """Grid of the 6 mm modulator dies (electrodes and fiducials of tfln_die) on the polished face,
    rows parallel to the primary flat, 0.1 mm dicing streets, 3 mm edge exclusion."""
    import tfln_die
    polys = tfln_die.electrode_polys() + tfln_die.fiducial_polys()
    proto = C.extrude_polys("die_au_proto", polys, 0.0, 0.0015, au)
    me = proto.data
    bpy.data.objects.remove(proto)
    flat_d = math.sqrt(RW ** 2 - (PRIMARY / 2) ** 2)
    n = 0
    for i in range(-9, 10):
        for j in range(-9, 10):
            cx, cy = i * pitch, j * pitch
            corners = [(cx + sx * 3.0, cy + sy * 3.0) for sx in (-1, 1) for sy in (-1, 1)]
            if any(math.hypot(x, y) > RW - edge_excl for x, y in corners):
                continue
            if any(y < -(flat_d - edge_excl) for x, y in corners):
                continue
            ob = C.link(bpy.data.objects.new(f"die_au_{i}_{j}", me))
            ob.location = (cx, cy, z)
            ob.parent = parent
            n += 1
    return n


# ------------------------------------------------------------------ props
def bench(mat_kind="steel"):
    if mat_kind == "steel":
        m = C.principled("bench_ss", color=(0.50, 0.50, 0.51), metallic=1.0, rough=0.3, aniso=0.7)
    else:
        m = C.principled("bench_pp", color=(0.82, 0.82, 0.80), rough=0.45, rough_var=0.05, rough_scale=0.02)
    return C.box("bench", (1600, 1000, 20), (0, 150, -10), m)


def wafer_carrier(z0=0.0):
    """Black conductive (carbon-loaded) PP single-wafer carrier: 118 mm OD, 102 mm pocket, a 2.5 mm
    wide ledge ring the wafer edge rests on (1.5 mm above the floor), plain dark floor with a molded
    size mark. The dark floor lets the transparent DSP wafer read by its reflections and edge."""
    pp = C.principled("pp_black", color=(0.035, 0.035, 0.038), rough=0.4, ior=1.49,
                      rough_var=0.06, rough_scale=0.05)
    outer = C.cylinder("carrier_out", 59.0, 9.0, (0, 0, z0 + 4.5), pp, verts=192, bevel=0.6)
    inner = C.cylinder("carrier_in", 51.0, 9.0, (0, 0, z0 + 4.5 + 1.5), None, verts=192)
    md = outer.modifiers.new("b", "BOOLEAN")
    md.object = inner
    md.solver = "EXACT"
    outer.modifiers.move(len(outer.modifiers) - 1, 0)
    inner.hide_render = True
    inner.hide_viewport = True
    ring = C.cylinder("carrier_ledge", 51.0, 1.5, (0, 0, z0 + 1.5 + 0.75), pp, verts=192)
    hole = C.cylinder("carrier_ledge_h", 48.5, 3.0, (0, 0, z0 + 2.25), None, verts=192)
    md = ring.modifiers.new("b", "BOOLEAN")
    md.object = hole
    md.solver = "EXACT"
    ring.modifiers.move(len(ring.modifiers) - 1, 0)
    hole.hide_render = True
    hole.hide_viewport = True
    mark = C.principled("mold_mark", color=(0.30, 0.30, 0.31), rough=0.6)
    C.text_mesh("mold_t", "100 MM", 6.0, (8, -20, z0 + 1.5), h=0.2, mat=mark)
    lid = C.cylinder("lid", 59.6, 7.0, (128, 58, z0 + 3.5), pp, verts=192, bevel=0.6)
    lh = C.cylinder("lid_h", 58.0, 7.0, (128, 58, z0 + 3.5 + 1.2), None, verts=192)
    md = lid.modifiers.new("b", "BOOLEAN")
    md.object = lh
    md.solver = "EXACT"
    lid.modifiers.move(len(lid.modifiers) - 1, 0)
    lh.hide_render = True
    lh.hide_viewport = True
    return z0 + 3.0  # wafer bottom height (top of the ledge)


def wafer_tweezers(loc, rot_z):
    """Stainless wafer tweezers: two flat legs joined at the back, wide flat paddle tips."""
    ss = C.principled("tweezer_ss", color=(0.75, 0.75, 0.76), metallic=1.0, rough=0.22, aniso=0.4)
    root = C.empty("tweezers", loc=loc, rot=(0, 0, math.radians(rot_z)))
    parts = []
    L = 130.0
    th = 0.7
    n = 40
    for leg in (0, 1):  # lying on the bench: lower leg flat, upper leg opening toward the tips
        verts, faces = [], []
        for i in range(n + 1):
            t = i / n
            x = -L / 2 + L * t
            w = 8.0 - 4.5 * min(t / 0.75, 1.0) + 7.5 * max(0.0, (t - 0.8) / 0.2)
            z = 0.0 if leg == 0 else th + 4.0 * t ** 1.3
            verts += [(x, -w / 2, z), (x, w / 2, z), (x, w / 2, z + th), (x, -w / 2, z + th)]
        for i in range(n):
            o = 4 * i
            for j in range(4):
                faces.append((o + j, o + 4 + j, o + 4 + (j + 1) % 4, o + (j + 1) % 4))
        faces.append((0, 1, 2, 3))
        o = 4 * n
        faces.append((o + 3, o + 2, o + 1, o))
        ob = C.mesh_obj(f"tw_leg{leg}", verts, faces, ss)
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = 0.2
        md.segments = 2
        parts.append(ob)
    C.parent(parts, root)
    return root


def hotplate():
    body = C.principled("hp_body", color=(0.80, 0.80, 0.78), rough=0.4, coat=0.3, coat_rough=0.2,
                        rough_var=0.05, rough_scale=0.05)
    top = C.principled("hp_top", color=(0.72, 0.72, 0.73), metallic=1.0, rough=0.3, rough_var=0.04,
                       rough_scale=0.6, aniso=0.5, color_var=0.04, color_scale=0.08)
    panel = C.principled("hp_panel", color=(0.03, 0.03, 0.035), rough=0.25, coat=0.6, coat_rough=0.05)
    knob = C.principled("hp_knob", color=(0.02, 0.02, 0.02), rough=0.35)
    led = C.emission_mat("hp_led", (1.0, 0.08, 0.02), 6.0)
    label = C.principled("hp_label", color=(0.95, 0.75, 0.02), rough=0.5)
    ink = C.principled("hp_ink", color=(0.02, 0.02, 0.02), rough=0.6)
    C.box("hp_base", (200, 300, 85), (0, -40, 42.5), body, bevel=6.0, segments=6)
    C.box("hp_plate", (190, 190, 10), (0, 0, 90), top, bevel=1.5)
    # front panel (slanted), display and knobs
    pz, py = 40.0, -190.5
    C.box("hp_panel", (170, 2, 55), (0, py, pz), panel, bevel=0.8)
    # seven-segment LED digits "115" behind a smoked window, plus a small C indicator
    segs = {"1": "bc", "5": "afgcd"}
    dw, dh, t = 9.0, 16.0, 1.8
    seg_xy = {"a": (0, dh / 2, dw, t), "g": (0, 0, dw, t), "d": (0, -dh / 2, dw, t),
              "b": (dw / 2, dh / 4, t, dh / 2), "c": (dw / 2, -dh / 4, t, dh / 2),
              "f": (-dw / 2, dh / 4, t, dh / 2), "e": (-dw / 2, -dh / 4, t, dh / 2)}
    for i, ch in enumerate("115"):
        x0 = -48 + i * 14.0
        for sg in segs[ch]:
            sx_, sz_, w_, h_ = seg_xy[sg]
            C.box(f"seg{i}{sg}", (w_ * 0.86 if w_ > t else w_, 0.4, h_ * 0.86 if h_ > t else h_),
                  (x0 + sx_, py - 1.1, pz + 8 + sz_), led)
    C.text_mesh("hp_unit", "\u00b0C", 6.0, (-10, py - 1.2, pz + 12), h=0.3, mat=led, rot=(math.radians(90), 0, 0))
    smoked = C.principled("smoked_glass", color=(0.05, 0.02, 0.02), rough=0.05, ior=1.5, transmission=0.55)
    C.box("hp_window", (56, 1.0, 26), (-32, py - 2.0, pz + 8), smoked, bevel=0.3)
    C.text_mesh("hp_set", "SET  HEAT", 4.0, (45, py - 1.2, pz + 18), h=0.2,
                mat=C.principled("hp_w", color=(0.8, 0.8, 0.8), rough=0.5), rot=(math.radians(90), 0, 0))
    for x in (30, 62):
        C.cylinder("hp_knob", 9.0, 8.0, (x, py - 5, pz - 6), knob, verts=48, bevel=0.8,
                   rot=(math.radians(90), 0, 0))
    # caution label on the base front
    C.box("hp_lab", (48, 0.4, 16), (-50, py + 0.5, 8), label, bevel=0.3)
    C.text_mesh("hp_lab_t", "CAUTION  HOT SURFACE", 3.2, (-50, py + 0.1, 8), h=0.1, mat=ink,
                rot=(math.radians(90), 0, 0))
    return 95.0  # plate top


def scene():
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    m = ln_materials()
    dust = C.principled("dust", color=(0.6, 0.58, 0.55), rough=0.9)
    if A["scene"] == "carrier":
        C.world_hdri("hospital_room", strength=0.6, rot_z_deg=200)
        bench("steel")
        zb = wafer_carrier(0.0)
        # flats on the left (primary at -120 deg, secondary 90 deg clockwise at 150 deg), lit from the left
        w = build_wafer(m, loc=(0, 0, zb), rot_z=-30.0, back="polished")  # double-side polished
        w.visible_shadow = False
        C.scatter_dust("dust_w", ((-35, 35), (-35, 35)), zb + TW, 25, 0.01, 0.05, dust, seed=11, fibers=1,
                       fiber_len=(1.0, 2.0))
        # (wafer tweezers helper kept for reuse; not placed in this framing)
        C.scatter_dust("dust_b", ((-200, 200), (-150, 200)), 0.0, 200, 0.03, 0.15, dust, seed=12, fibers=4,
                       fiber_len=(2.0, 6.0))
        tgt = Vector((4, 2, 2))
        el, az, D, field = math.radians(60), math.radians(-80), 520.0, 170.0
        loc = tgt + D * Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        cam = C.camera(loc, tgt, lens=36 * D / field)
        C.set_aperture(cam, Vector((0, -10, 3)), pupil_mm=12.0)
        C.area_light("key", (-480, 120, 300), (0, 0, 0), 400, 3.0e5, color=(1.0, 0.98, 0.95))
        C.area_light("fill", (300, -300, 250), (0, 0, 0), 300, 5.0e4, color=(0.95, 0.97, 1.0))
        # glossy-only studio panels in the mirror direction: a large window-like panel with a mullion,
        # reflected by the polished front (white) and back (yellow, offset by the 0.5 mm thickness)
        mp = C.mirror_point(loc, (-12, 8, zb + TW), 700.0)
        em = C.emission_mat("panel_m", (1.0, 0.98, 0.95), 5.0)
        for k, off in enumerate((-40.0, 40.0)):
            pnl = C.box(f"panel{k}", (70, 520, 1), (0, 0, 0), em)
            rot = (Vector((-12, 8, zb)) - mp).to_track_quat("-Z", "Y")
            pnl.rotation_euler = rot.to_euler()
            pnl.location = mp + rot @ Vector((off - 170.0, 60.0, -1.0))
            pnl.visible_camera = False
            pnl.visible_diffuse = False
            pnl.visible_shadow = False
        # large dim ceiling panel behind them: the whole polished face reflects a soft grey, so the
        # wafer outline and both flats read against the black carrier ledge
        rot = (Vector((-12, 8, zb)) - mp).to_track_quat("-Z", "Y")
        big = C.box("panel_big", (1400, 1400, 1), (0, 0, 0), C.emission_mat("panel_big_m", (0.95, 0.96, 1.0), 0.3))
        big.rotation_euler = rot.to_euler()
        big.location = mp + rot @ Vector((0, 0, 10.0))
        big.visible_camera = False
        big.visible_diffuse = False
        big.visible_shadow = False
        C.ceiling(1000.0, color=(0.03, 0.03, 0.03))
        C.surround(800.0, 1200.0, color=(0.05, 0.05, 0.055))
        # low rim light from behind: makes the 0.5 mm edge and the two flats glint
        C.area_light("rim", (-260, -120, 90), (-40, -20, 3), 120, 9.0e5, color=(1.0, 0.97, 0.92), size_y=30)
        C.area_light("rim2", (-240, 150, 90), (-40, 20, 3), 120, 7.0e5, color=(1.0, 0.97, 0.92), size_y=30)
    else:
        C.world_hdri("hospital_room", strength=0.45, rot_z_deg=60)
        bpy.context.scene.view_settings.exposure = -0.4
        bench("pp")
        zt = hotplate()
        w = build_wafer(m, loc=(0, 0, zt + 0.01), rot_z=12.0, back="lapped")  # single-side polished
        au = C.principled("au_wafer", color=(1.0, 0.72, 0.30), metallic=1.0, rough=0.15)
        die_grid(au, w, TW)
        C.scatter_dust("dust_w", ((-35, 35), (-35, 35)), zt + TW, 30, 0.01, 0.05, dust, seed=21)
        tgt = Vector((0, -62, 64))
        el, az, D, field = math.radians(34), math.radians(-66), 900.0, 330.0
        loc = tgt + D * Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        cam = C.camera(loc, tgt, lens=36 * D / field)
        C.set_aperture(cam, Vector((0, -20, zt)), pupil_mm=22.0)
        C.area_light("key", (-300, 250, 600), (0, 0, 80), 500, 3.5e5, color=(1.0, 0.98, 0.95))
        C.area_light("fill", (400, -400, 300), (0, 0, 80), 400, 1.0e5, color=(0.95, 0.97, 1.0))
        mp = C.mirror_point(loc, (0, 0, zt + TW), 1200.0)
        # strip light in the mirror direction: a glossy highlight band across the polished front
        mp2 = C.mirror_point(loc, (15, 5, zt + TW), 1200.0)
        strip = C.box("strip", (1400, 90, 1), mp2, C.emission_mat("strip_m", (1.0, 0.98, 0.95), 12.0))
        strip.rotation_euler = (Vector((15, 5, zt)) - mp2).to_track_quat("-Z", "Y").to_euler()
        strip.rotation_euler.rotate_axis("Z", math.radians(35))
        strip.visible_camera = False
        strip.visible_diffuse = False   # seen only in glossy reflections (studio strip-light trick)
        strip.visible_shadow = False
        strip.visible_transmission = False
    C.compositor(vignette=0.22, dispersion=0.002)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
