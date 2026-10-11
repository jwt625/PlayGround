"""Environment helpers (workstream E, DevLog-013). Blender 4.2, headless. Units: metres, Z up.

MB (mesh builder: boxes, cylinders, tubes, ellipsoids into one mesh with per-face material index) and box-projected UVs are
copied from workstream R's `scripts/assets/datacenter/rk_lib.py` (itself adapted from the interconnect film) and extended
with a transform stack (`with mb.at(loc, rz):`) so sub-assemblies are written in local frames.
"""
import contextlib
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
import vfx  # noqa: E402

PI = math.pi
FONT_DIR = "/System/Library/Fonts/Supplemental"
OUT_ASSETS = os.path.join(ROOT, "assets", "components", "environment")
OUT_PREVIEWS = os.path.join(ROOT, "outputs", "assets", "environment")


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _MC.clear()
    _FONTS.clear()
    _MESH_CACHE.clear()
    return bpy.context.scene


def rel(path):
    """Repo-relative path for JSON/log text (no absolute home paths in committed text)."""
    return os.path.relpath(path, ROOT)


# ==================================================================== collections, hooks
def coll(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in [x.name for x in (parent.children if parent else bpy.context.scene.collection.children)]:
        (parent.children if parent else bpy.context.scene.collection.children).link(c)
    return c


def empty(name, c, loc=(0, 0, 0), rot=(0, 0, 0), size=0.2, kind="PLAIN_AXES", parent=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = kind
    e.empty_display_size = size
    c.objects.link(e)
    e.parent = parent
    e.location = loc
    e.rotation_euler = rot
    return e


HOOKS = []


def hook(name, c, loc, yaw=0.0, kind="landmark", parent=None, size=0.25, rot=None, **props):
    """HOOK_<name>: location + facing (local +X = facing direction after rotation Z = yaw). Recorded for the JSON."""
    h = empty("HOOK_" + name, c, loc, rot if rot is not None else (0, 0, yaw), size, "ARROWS", parent)
    h["kind"] = kind
    h["yaw_deg"] = round(math.degrees(yaw), 2)
    for k, v in props.items():
        h[k] = v
    HOOKS.append(h)
    return h


def hooks_json():
    out = []
    bpy.context.view_layer.update()
    for h in HOOKS:
        mw = h.matrix_world
        fx = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
        d = {"name": h.name, "kind": h["kind"], "location": [round(v, 4) for v in mw.translation],
             "facing": [round(v, 4) for v in fx], "yaw_deg": round(math.degrees(math.atan2(fx.y, fx.x)), 2)}
        for k in h.keys():
            if k not in ("kind", "yaw_deg") and not k.startswith("_"):
                v = h[k]
                d[k] = list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
        out.append(d)
    return out


# ==================================================================== materials
_MC = {}


def principled(name, base=(0.5, 0.5, 0.5), metal=0.0, rough=0.5, emit=None, emit_strength=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base[:3], 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[:3], 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    m.diffuse_color = (*base[:3], 1)
    return m


def _noisy(name, base, base2, rough, rough2, scale, metal=0.0, coords="Object", bump=0.0, bump_scale=30.0):
    """Principled with noise-varied colour/roughness (grime) in object or UV space."""
    m, b, out = vfx.nodes.new_material(name)
    tc = b.texcoord(coords)
    n = b.noise(tc, scale=scale, detail=4.0, rough=0.6)
    col = b.cmix((*base, 1), (*base2, 1), b.smooth(n, 0.35, 0.7))
    p = b.principled(Base_Color=col, Roughness=b.lerp(rough, rough2, n), Metallic=metal)
    if bump:
        bn = b.node("ShaderNodeBump")
        bn.inputs["Strength"].default_value = bump
        bn.inputs["Distance"].default_value = 0.003
        b.link(b.noise(tc, scale=bump_scale, detail=3.0), bn.inputs["Height"])
        b.link(bn.outputs["Normal"], p.inputs["Normal"])
    b.link(p.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*base, 1)
    return m


def _window(name, kind):
    """Window glass by UV (0..1 per pane, from MB.quad_uv): lit warm/cool with blinds and room gradient, TV flicker, or dark."""
    m, b, out = vfx.nodes.new_material(name)
    uv = b.sep(b.texcoord("UV"))
    u, v = uv[0], uv[1]
    glass = b.principled(Base_Color=(0.01, 0.012, 0.016, 1), Roughness=0.06)
    glass.inputs["Specular IOR Level"].default_value = 0.6
    if kind == "dark":
        b.link(glass.outputs[0], out.inputs["Surface"])
        return m
    col = {"warm": (1.0, 0.62, 0.3), "cool": (0.55, 0.75, 1.0), "tv": (0.35, 0.55, 1.0), "pink": (1.0, 0.35, 0.65),
           "shop": (1.0, 0.72, 0.45)}[kind]
    strength = {"warm": 1.1, "cool": 0.9, "tv": 0.9, "pink": 1.0, "shop": 0.9}[kind]
    room = b.lerp(0.45, 1.0, b.smooth(v, 0.0, 0.9))                          # brighter at the ceiling
    blinds = b.add(0.55, b.mul(0.45, b.smooth(b.math("FRACT", b.mul(v, 14.0)), 0.25, 0.4)))
    if kind == "shop":                                                        # shelves / counter silhouettes
        blinds = b.add(0.6, b.mul(0.4, b.smooth(b.math("FRACT", b.mul(v, 4.0)), 0.15, 0.3)))
    shade_down = b.smooth(v, 0.55, 0.6) if kind in ("warm", "pink") else 0.0     # half-drawn blind
    k = b.mul(b.mul(room, blinds), strength)
    if kind in ("warm", "pink"):
        k = b.mul(k, b.sub(1.0, b.mul(shade_down, 0.6)))
    if kind == "tv":
        t = b.time()
        fl = b.lerp(0.5, 1.2, b.noise(w=b.mul(t, 3.0), scale=1.0, detail=1.0, dims="1D"))
        k = b.mul(k, fl)
    side = b.mul(b.smooth(u, 0.0, 0.12), b.smooth(b.one_minus(u), 0.0, 0.12))
    k = b.mul(k, b.lerp(0.6, 1.0, side))
    em = b.emission((*col, 1), k)
    b.link(b.add_shader(glass.outputs[0], em), out.inputs["Surface"])
    m.diffuse_color = (*col, 1)
    return m


def _stripes(name, c1, c2, period=0.2, axis=0, rough=0.5, metal=0.0, diag=False, bump=0.0):
    """Stripes in UV (hazard diagonal / roller shutter slats)."""
    m, b, out = vfx.nodes.new_material(name)
    uv = b.sep(b.texcoord("UV"))
    s = b.add(uv[0], uv[1]) if diag else uv[axis]
    f = b.math("FRACT", b.math("DIVIDE", s, period))
    st = b.smooth(f, 0.45, 0.55)
    p = b.principled(Base_Color=b.cmix((*c1, 1), (*c2, 1), st), Roughness=rough, Metallic=metal)
    if bump:
        bn = b.node("ShaderNodeBump")
        bn.inputs["Strength"].default_value = bump
        bn.inputs["Distance"].default_value = 0.01
        b.link(b.math("SINE", b.mul(f, 2 * PI)), bn.inputs["Height"])
        b.link(bn.outputs["Normal"], p.inputs["Normal"])
    b.link(p.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*c1, 1)
    return m


def _cells(name):
    """Solar panel: dark blue cells with silver grid lines (UV, 10 x 4 cells per panel)."""
    m, b, out = vfx.nodes.new_material(name)
    uv = b.sep(b.texcoord("UV"))
    gx = b.smooth(b.math("PINGPONG", b.mul(uv[0], 10.0), 0.5), 0.0, 0.04)
    gy = b.smooth(b.math("PINGPONG", b.mul(uv[1], 4.0), 0.5), 0.0, 0.06)
    line = b.one_minus(b.mul(gx, gy))
    p = b.principled(Base_Color=b.cmix((0.02, 0.04, 0.12, 1), (0.6, 0.62, 0.65, 1), line), Roughness=0.15, Metallic=0.3)
    b.link(p.outputs[0], out.inputs["Surface"])
    return m


def _panels(name, base, seam, pw=1.2, ph=0.6):
    """Cladding panels: base metal with dark seams on a pw x ph grid (UV in metres)."""
    m, b, out = vfx.nodes.new_material(name)
    uv = b.sep(b.texcoord("UV"))
    sx = b.smooth(b.math("PINGPONG", b.math("DIVIDE", uv[0], pw), 0.5), 0.0, 0.012)
    sy = b.smooth(b.math("PINGPONG", b.math("DIVIDE", uv[1], ph), 0.5), 0.0, 0.025)
    s = b.mul(sx, sy)
    n = b.noise(b.texcoord("Object"), scale=2.0, detail=3.0)
    col = b.cmix(b.cmix((*seam, 1), (*base, 1), s), (*[c * 0.7 for c in base], 1), b.mul(b.smooth(n, 0.45, 0.7), 0.6))
    p = b.principled(Base_Color=col, Roughness=b.lerp(0.35, 0.6, n), Metallic=0.6)
    b.link(p.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (*base, 1)
    return m


def _foil(name):
    m, b, out = vfx.nodes.new_material(name)
    tc = b.texcoord("Object")
    n = b.voronoi(tc, scale=18.0, out="Distance")
    p = b.principled(Base_Color=(0.85, 0.62, 0.22, 1), Roughness=b.lerp(0.15, 0.35, n), Metallic=1.0)
    bn = b.node("ShaderNodeBump")
    bn.inputs["Strength"].default_value = 0.6
    b.link(n, bn.inputs["Height"])
    b.link(bn.outputs["Normal"], p.inputs["Normal"])
    b.link(p.outputs[0], out.inputs["Surface"])
    m.diffuse_color = (0.85, 0.62, 0.22, 1)
    return m


def M(name):
    """Environment material palette (created on first use, cached)."""
    if name in _MC:
        return _MC[name]
    mk = {
        "asphalt": lambda: vfx.wet_ground_material("ENV_asphalt_wet", (0.028, 0.03, 0.036), wetness=0.7, coverage=0.15, scale=0.3),
        "sidewalk": lambda: vfx.wet_ground_material("ENV_sidewalk_wet", (0.07, 0.07, 0.078), wetness=0.5, coverage=0.12, scale=0.5,
                                                    rough_dry=0.75, grain=0.15, seed=11.0),
        "puddle": lambda: vfx.puddle_material("ENV_puddle"),
        "concrete": lambda: _noisy("ENV_concrete", (0.075, 0.075, 0.082), (0.045, 0.045, 0.05), 0.75, 0.9, 1.5, bump=0.15),
        "concrete_light": lambda: _noisy("ENV_concrete_light", (0.16, 0.16, 0.17), (0.09, 0.09, 0.1), 0.7, 0.9, 1.2, bump=0.15),
        "plaster_a": lambda: _noisy("ENV_plaster_a", (0.085, 0.075, 0.08), (0.05, 0.045, 0.05), 0.8, 0.95, 0.8, bump=0.1),
        "plaster_b": lambda: _noisy("ENV_plaster_b", (0.06, 0.07, 0.08), (0.035, 0.04, 0.048), 0.8, 0.95, 0.8, bump=0.1),
        "plaster_c": lambda: _noisy("ENV_plaster_c", (0.1, 0.085, 0.07), (0.055, 0.05, 0.045), 0.8, 0.95, 0.8, bump=0.1),
        "tile_dark": lambda: _noisy("ENV_tile_dark", (0.04, 0.042, 0.05), (0.025, 0.026, 0.03), 0.35, 0.5, 3.0),
        "metal_dark": lambda: _noisy("ENV_metal_dark", (0.05, 0.052, 0.058), (0.03, 0.03, 0.033), 0.45, 0.65, 3.0, metal=0.7),
        "metal_paint": lambda: _noisy("ENV_metal_paint", (0.11, 0.115, 0.12), (0.06, 0.06, 0.065), 0.45, 0.7, 2.0, metal=0.3),
        "galv": lambda: _noisy("ENV_galvanized", (0.42, 0.43, 0.45), (0.25, 0.26, 0.27), 0.35, 0.55, 4.0, metal=0.9),
        "rail": lambda: principled("ENV_rail_steel", (0.32, 0.33, 0.36), 0.8, 0.35),
        "rust": lambda: _noisy("ENV_rust_iron", (0.07, 0.045, 0.035), (0.13, 0.06, 0.035), 0.6, 0.9, 6.0, metal=0.5),
        "wood": lambda: _noisy("ENV_wood_pole", (0.09, 0.065, 0.045), (0.05, 0.035, 0.025), 0.75, 0.9, 1.0, bump=0.3, bump_scale=8.0),
        "rubber": lambda: principled("ENV_rubber", (0.012, 0.012, 0.013), 0.0, 0.7),
        "plastic_white": lambda: principled("ENV_plastic_white", (0.6, 0.6, 0.62), 0.0, 0.4),
        "plastic_dark": lambda: principled("ENV_plastic_dark", (0.025, 0.025, 0.028), 0.0, 0.35),
        "ac_white": lambda: _noisy("ENV_ac_housing", (0.24, 0.24, 0.23), (0.13, 0.13, 0.12), 0.45, 0.7, 2.5),
        "grille": lambda: principled("ENV_grille", (0.04, 0.04, 0.045), 0.6, 0.5),
        "glass_dark": lambda: _window("ENV_glass_dark", "dark"),
        "win_warm": lambda: _window("ENV_win_warm", "warm"),
        "win_cool": lambda: _window("ENV_win_cool", "cool"),
        "win_tv": lambda: _window("ENV_win_tv", "tv"),
        "win_pink": lambda: _window("ENV_win_pink", "pink"),
        "shopfront": lambda: _window("ENV_shopfront", "shop"),
        "shutter": lambda: _stripes("ENV_roller_shutter", (0.16, 0.165, 0.17), (0.09, 0.09, 0.1), 0.035, 1, 0.5, 0.6, bump=0.3),
        "hazard": lambda: _stripes("ENV_hazard", (0.9, 0.7, 0.05), (0.02, 0.02, 0.02), 0.25, 0, 0.6, diag=True),
        "cladding": lambda: _panels("ENV_dc_cladding", (0.075, 0.08, 0.09), (0.015, 0.016, 0.02)),
        "cladding_light": lambda: _panels("ENV_dc_cladding_light", (0.16, 0.17, 0.18), (0.03, 0.03, 0.035), 0.6, 1.2),
        "paint_white": lambda: principled("ENV_road_paint", (0.55, 0.55, 0.5), 0.0, 0.5),
        "paint_yellow": lambda: principled("ENV_paint_yellow", (0.75, 0.55, 0.04), 0.0, 0.5),
        "red_body": lambda: principled("ENV_scooter_red", (0.45, 0.03, 0.04), 0.3, 0.25, coat=1.0),
        "chrome": lambda: principled("ENV_chrome", (0.8, 0.8, 0.82), 1.0, 0.12),
        "seat": lambda: principled("ENV_seat_leather", (0.02, 0.018, 0.017), 0.0, 0.45),
        "bin_green": lambda: _noisy("ENV_bin_green", (0.05, 0.12, 0.08), (0.03, 0.06, 0.045), 0.45, 0.7, 3.0, metal=0.4),
        "sign_back": lambda: principled("ENV_sign_backer", (0.012, 0.012, 0.015), 0.4, 0.4),
        "lamp_lens": lambda: principled("ENV_lamp_lens", (1.0, 0.85, 0.6), 0.0, 0.2, (1.0, 0.8, 0.55), 18.0),
        "lamp_lens_cool": lambda: principled("ENV_lamp_lens_cool", (0.85, 0.9, 1.0), 0.0, 0.2, (0.8, 0.88, 1.0), 14.0),
        "led_red": lambda: vfx.eye_glow_material("ENV_led_red", (1.0, 0.05, 0.03), strength=25.0, pulse=0.8, pulse_rate=0.7),
        "led_green": lambda: principled("ENV_led_green", (0.1, 1.0, 0.3), 0.0, 0.3, (0.1, 1.0, 0.3), 12.0),
        "led_cyan": lambda: principled("ENV_led_cyan", (0.1, 0.85, 1.0), 0.0, 0.3, (0.1, 0.85, 1.0), 8.0),
        "led_warm": lambda: principled("ENV_led_warm", (1.0, 0.8, 0.55), 0.0, 0.3, (1.0, 0.8, 0.55), 8.0),
        "camera_white": lambda: principled("ENV_cctv_housing", (0.55, 0.56, 0.58), 0.1, 0.35),
        "lens_black": lambda: principled("ENV_lens_black", (0.005, 0.005, 0.006), 0.0, 0.05, coat=1.0),
        "foil": lambda: _foil("ENV_mli_foil"),
        "solar": lambda: _cells("ENV_solar_cells"),
        "dish": lambda: principled("ENV_dish_white", (0.7, 0.7, 0.72), 0.0, 0.5),
        "fabric_awning": lambda: _stripes("ENV_awning", (0.35, 0.02, 0.05), (0.06, 0.01, 0.02), 0.3, 0, 0.8),
        "cable": lambda: principled("ENV_cable_black", (0.015, 0.015, 0.016), 0.0, 0.5),
        "insulator": lambda: principled("ENV_insulator", (0.25, 0.3, 0.28), 0.0, 0.15, coat=0.5),
        "transformer": lambda: _noisy("ENV_transformer", (0.2, 0.22, 0.2), (0.12, 0.12, 0.11), 0.5, 0.7, 3.0, metal=0.5),
        "curb": lambda: _noisy("ENV_curb", (0.14, 0.14, 0.15), (0.08, 0.08, 0.085), 0.7, 0.9, 2.0, bump=0.2),
        "water_tank": lambda: _noisy("ENV_water_tank", (0.08, 0.07, 0.06), (0.05, 0.04, 0.035), 0.7, 0.9, 1.0, bump=0.2),
    }
    if name.startswith("neon_"):
        col, fl = NEON_COLORS[name[5:]]
        _MC[name] = vfx.neon_material("ENV_" + name, col, strength=6.0, flicker=fl, seed=len(_MC))
        return _MC[name]
    _MC[name] = mk[name]()
    return _MC[name]


NEON_COLORS = {  # name: (colour, flicker)
    "mag": ((1.0, 0.1, 0.7), 0.0), "cyan": ((0.1, 0.85, 1.0), 0.0), "orange": ((1.0, 0.45, 0.05), 0.0),
    "red": ((1.0, 0.08, 0.06), 0.0), "green": ((0.2, 1.0, 0.35), 0.0), "violet": ((0.55, 0.2, 1.0), 0.0),
    "warm": ((1.0, 0.75, 0.4), 0.0), "mag_fl": ((1.0, 0.1, 0.7), 0.35), "cyan_fl": ((0.1, 0.85, 1.0), 0.25),
    "orange_fl": ((1.0, 0.45, 0.05), 0.3), "white": ((0.9, 0.95, 1.0), 0.0),
}


# ==================================================================== mesh builder
class MB:
    """Boxes, cylinders, tubes, ellipsoids into one mesh with per-face material index (metres). Transform stack via at()."""

    def __init__(self):
        self.v, self.f, self.mi, self.uv = [], [], [], []
        self.T = Matrix.Identity(4)

    @contextlib.contextmanager
    def at(self, loc=(0, 0, 0), rz=0.0, rx=0.0, ry=0.0, scale=1.0):
        old = self.T
        self.T = old @ Matrix.Translation(Vector(loc)) @ Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ \
            Matrix.Rotation(rx, 4, "X") @ Matrix.Scale(scale, 4)
        try:
            yield self
        finally:
            self.T = old

    def _add(self, verts, faces, m=0, uvs=None):
        o = len(self.v)
        T = self.T
        self.v.extend(tuple(T @ Vector(p)) for p in verts)
        for k, f in enumerate(faces):
            self.f.append(tuple(o + i for i in f))
            self.mi.append(m)
            self.uv.append(uvs[k] if uvs else None)

    def box(self, c, s, m=0):
        x, y, z = c
        dx, dy, dz = s[0] / 2, s[1] / 2, s[2] / 2
        v = [(x - dx, y - dy, z - dz), (x + dx, y - dy, z - dz), (x + dx, y + dy, z - dz), (x - dx, y + dy, z - dz),
             (x - dx, y - dy, z + dz), (x + dx, y - dy, z + dz), (x + dx, y + dy, z + dz), (x - dx, y + dy, z + dz)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self._add(v, f, m)

    def box2(self, lo, hi, m=0):
        self.box(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
                 (abs(hi[0] - lo[0]), abs(hi[1] - lo[1]), abs(hi[2] - lo[2])), m)

    def seg(self, p0, p1, w, h=None, m=0, up=(0, 0, 1)):
        """Rectangular bar from p0 to p1 (width w, height h)."""
        h = w if h is None else h
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        L = d.length
        if L < 1e-9:
            return
        x = d / L
        upv = Vector(up)
        if abs(x.dot(upv)) > 0.95:
            upv = Vector((1, 0, 0)) if abs(x.x) < 0.9 else Vector((0, 1, 0))
        y = upv.cross(x).normalized()
        z = x.cross(y)
        R = Matrix((x, y, z)).transposed().to_4x4()
        R.translation = (p0 + p1) / 2
        old = self.T
        self.T = old @ R
        self.box((0, 0, 0), (L, w, h), m)
        self.T = old

    def cyl(self, c, r, h, axis="z", seg=16, m=0, r2=None, cap=True):
        x, y, z = c
        r2 = r if r2 is None else r2
        v = []
        for end, rr in ((-1, r), (1, r2)):
            for i in range(seg):
                a = 2 * math.pi * i / seg
                p = (rr * math.cos(a), rr * math.sin(a), end * h / 2)
                if axis == "x":
                    p = (p[2], p[0], p[1])
                elif axis == "y":
                    p = (p[1], p[2], p[0])
                v.append((x + p[0], y + p[1], z + p[2]))
        f = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
        if cap:
            f.append(tuple(range(seg - 1, -1, -1)))
            f.append(tuple(seg + i for i in range(seg)))
        self._add(v, f, m)

    def rod(self, p0, p1, r, seg=10, m=0, r2=None, cap=True):
        """Cylinder between two points."""
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        L = d.length
        if L < 1e-9:
            return
        q = d.normalized().to_track_quat("Z", "Y")
        R = q.to_matrix().to_4x4()
        R.translation = (p0 + p1) / 2
        old = self.T
        self.T = old @ R
        self.cyl((0, 0, 0), r, L, "z", seg, m, r2, cap)
        self.T = old

    def ring(self, c, r_out, r_in, h, axis="z", seg=24, m=0):
        x, y, z = c
        v = []
        for end in (-1, 1):
            for rr in (r_out, r_in):
                for i in range(seg):
                    a = 2 * math.pi * i / seg
                    p = (rr * math.cos(a), rr * math.sin(a), end * h / 2)
                    if axis == "x":
                        p = (p[2], p[0], p[1])
                    elif axis == "y":
                        p = (p[1], p[2], p[0])
                    v.append((x + p[0], y + p[1], z + p[2]))
        bo, bi, to, ti = 0, seg, 2 * seg, 3 * seg
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f += [(bo + i, bo + j, to + j, to + i), (bi + j, bi + i, ti + i, ti + j),
                  (to + i, to + j, ti + j, ti + i), (bo + j, bo + i, bi + i, bi + j)]
        self._add(v, f, m)

    def ellipsoid(self, c, r, seg=16, rings=10, m=0):
        v = [(c[0], c[1], c[2] - r[2])]
        for j in range(1, rings):
            th = math.pi * j / rings - math.pi / 2
            for i in range(seg):
                a = 2 * math.pi * i / seg
                v.append((c[0] + r[0] * math.cos(th) * math.cos(a), c[1] + r[1] * math.cos(th) * math.sin(a),
                          c[2] + r[2] * math.sin(th)))
        v.append((c[0], c[1], c[2] + r[2]))
        f = []
        for i in range(seg):
            f.append((0, 1 + (i + 1) % seg, 1 + i))
        for j in range(rings - 2):
            a0, a1 = 1 + j * seg, 1 + (j + 1) * seg
            for i in range(seg):
                k = (i + 1) % seg
                f.append((a0 + i, a0 + k, a1 + k, a1 + i))
        top = len(v) - 1
        a0 = 1 + (rings - 2) * seg
        for i in range(seg):
            f.append((a0 + i, a0 + (i + 1) % seg, top))
        self._add(v, f, m)

    def dish(self, c, r, depth, axis_dir=(0, 0, 1), seg=24, rings=5, m=0):
        """Shallow paraboloid shell (open), apex at c, opening along axis_dir."""
        q = Vector(axis_dir).normalized().to_track_quat("Z", "Y")
        R = q.to_matrix().to_4x4()
        R.translation = Vector(c)
        old = self.T
        self.T = old @ R
        v = [(0, 0, 0)]
        for j in range(1, rings + 1):
            rr = r * j / rings
            zz = depth * (rr / r) ** 2
            for i in range(seg):
                a = 2 * math.pi * i / seg
                v.append((rr * math.cos(a), rr * math.sin(a), zz))
        f = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
        for j in range(rings - 1):
            a0, a1 = 1 + j * seg, 1 + (j + 1) * seg
            for i in range(seg):
                k = (i + 1) % seg
                f.append((a0 + i, a1 + i, a1 + k, a0 + k))
        self._add(v, f, m)
        self.T = old

    def tube(self, pts, r, seg=8, m=0, closed=False, cap=True):
        rings = []
        n = len(pts)
        P = [Vector(p) for p in pts]
        for k, p in enumerate(P):
            if closed:
                t = P[(k + 1) % n] - P[k - 1]
            elif k == 0:
                t = P[1] - p
            elif k == n - 1:
                t = p - P[k - 1]
            else:
                t = P[k + 1] - P[k - 1]
            t.normalize()
            ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
            u = t.cross(ref).normalized()
            w = t.cross(u).normalized()
            rings.append([tuple(p + (u * math.cos(2 * math.pi * i / seg) + w * math.sin(2 * math.pi * i / seg)) * r)
                          for i in range(seg)])
        v = [q for ring in rings for q in ring]
        f = []
        for k in range(n - 1 + (1 if closed else 0)):
            k2 = (k + 1) % n
            for i in range(seg):
                j = (i + 1) % seg
                f.append((k * seg + i, k * seg + j, k2 * seg + j, k2 * seg + i))
        if not closed and cap:
            f.append(tuple(range(seg - 1, -1, -1)))
            f.append(tuple((n - 1) * seg + i for i in range(seg)))
        self._add(v, f, m)

    def quad(self, a, b, c, d, m=0, uv=True):
        """Quad with its own 0..1 UV (window panes, signs, panels)."""
        self._add([a, b, c, d], [(0, 1, 2, 3)], m, [((0, 0), (1, 0), (1, 1), (0, 1))] if uv else None)

    def mesh(self, name, uv_tile=1.0):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        for p, m in zip(me.polygons, self.mi):
            p.material_index = m
        uvl = me.uv_layers.new(name="UVMap")
        for p, fuv in zip(me.polygons, self.uv):
            if fuv is not None:
                for li, q in zip(p.loop_indices, fuv):
                    uvl.data[li].uv = q
                continue
            n = p.normal
            ax = max(range(3), key=lambda i: abs(n[i]))
            a, b = [(1, 2), (0, 2), (0, 1)][ax]
            for li in p.loop_indices:
                co = me.vertices[me.loops[li].vertex_index].co
                uvl.data[li].uv = (co[a] / uv_tile, co[b] / uv_tile)
        return me


_MESH_CACHE = {}


def obj(name, mb, mats, c, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), smooth=False, key=None, shadow=True):
    """Object from an MB; mats = list of material names (palette) or materials; key = share mesh data across objects."""
    if not mb.f:
        return None
    if key and key in _MESH_CACHE:
        me = _MESH_CACHE[key]
    else:
        me = mb.mesh(name)
        for m in mats:
            me.materials.append(M(m) if isinstance(m, str) else m)
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        if key:
            _MESH_CACHE[key] = me
    o = bpy.data.objects.new(name, me)
    c.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_euler = rot
    o.visible_shadow = shadow
    return o


# ==================================================================== text
_FONTS = {}


def font(name):
    if name not in _FONTS:
        path = os.path.join(FONT_DIR, name + ".ttf")
        _FONTS[name] = bpy.data.fonts.load(path) if os.path.exists(path) else None
    return _FONTS[name]


def to_mesh(o, c, parent=None):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    name, mw, data = o.name, o.matrix_world.copy(), o.data
    mats = list(data.materials)
    bpy.data.objects.remove(o, do_unlink=True)
    if isinstance(data, bpy.types.Curve):
        bpy.data.curves.remove(data)
    if not me.materials:
        for m in mats:
            me.materials.append(m)
    n = bpy.data.objects.new(name, me)
    c.objects.link(n)
    n.matrix_world = mw
    if parent is not None:
        n.parent = parent
        n.matrix_parent_inverse = parent.matrix_world.inverted()
    return n


def text(name, body, size, c, loc=(0, 0, 0), rot=(PI / 2, 0, 0), mat="sign_back", fnt="Arial Black", extrude=0.0,
         bevel=0.0, tube=False, align="CENTER", spacing=1.0, parent=None):
    """Text converted to mesh. Default rot stands the text up facing -Y. tube=True: outline tubes of radius `bevel` (neon)."""
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.space_character = spacing
    f = font(fnt)
    if f:
        cu.font = f
    cu.extrude = extrude
    if tube:
        cu.fill_mode = "NONE"
        cu.bevel_depth = bevel
        cu.bevel_resolution = 2
    elif bevel:
        cu.bevel_depth = bevel
    cu.resolution_u = 6
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    o.data.materials.append(M(mat) if isinstance(mat, str) else mat)
    bpy.context.view_layer.update()
    n = to_mesh(o, c)
    if parent is not None:
        mw = n.matrix_world.copy()
        n.parent = parent
        n.matrix_world = parent.matrix_world @ mw
    n.visible_shadow = False
    return n


def text_dims(o):
    bb = [Vector(v) for v in o.bound_box]
    return Vector((max(v.x for v in bb) - min(v.x for v in bb), max(v.y for v in bb) - min(v.y for v in bb),
                   max(v.z for v in bb) - min(v.z for v in bb)))


# ==================================================================== curves
def catenary(p0, p1, sag, n=24):
    """Points of a catenary between p0 and p1 with mid-span sag `sag` (m, measured from the chord midpoint).
    Solves a in z = a cosh(x/a) for the given sag on the horizontal span (level-equivalent), adds the chord slope."""
    p0, p1 = Vector(p0), Vector(p1)
    span = (Vector((p1.x, p1.y, 0)) - Vector((p0.x, p0.y, 0))).length
    if sag <= 1e-6 or span < 1e-6:
        return [p0.lerp(p1, i / (n - 1)) for i in range(n)]
    h = span / 2
    lo, hi = 1e-3, 1e6
    for _ in range(200):                # sag(a) = a (cosh(h/a) - 1) decreases with a
        a = math.sqrt(lo * hi)
        s = a * (math.cosh(min(h / a, 700)) - 1)
        if s > sag:
            lo = a
        else:
            hi = a
    pts = []
    for i in range(n):
        u = i / (n - 1)
        x = -h + 2 * h * u
        drop = sag - a * (math.cosh(x / a) - 1)       # 0 at the ends, sag at mid-span
        pts.append(p0.lerp(p1, u) - Vector((0, 0, drop)))
    return pts


def catenary_z(p0, p1, sag, u):
    """Height on the catenary at span fraction u (0..1)."""
    p0, p1 = Vector(p0), Vector(p1)
    span = (Vector((p1.x, p1.y, 0)) - Vector((p0.x, p0.y, 0))).length
    h = span / 2
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        a = math.sqrt(lo * hi)
        if a * (math.cosh(min(h / a, 700)) - 1) > sag:
            lo = a
        else:
            hi = a
    x = -h + 2 * h * u
    return p0.z + (p1.z - p0.z) * u - (sag - a * (math.cosh(x / a) - 1))


# ==================================================================== lights
def light(name, kind, loc, energy, color=(1, 1, 1), c=None, size=None, rot=None, spot_deg=None, blend=0.3, target=None,
          radius=None, shadow=True):
    d = bpy.data.lights.new(name, kind)
    d.energy = energy
    d.color = color[:3]
    if size is not None and kind == "AREA":
        d.size = size
    if radius is not None and kind in ("POINT", "SPOT"):
        d.shadow_soft_size = radius
    if kind == "SPOT":
        d.spot_size = math.radians(spot_deg or 60)
        d.spot_blend = blend
    d.use_shadow = shadow
    o = bpy.data.objects.new(name, d)
    (c or bpy.context.scene.collection).objects.link(o)
    o.location = loc
    if target is not None:
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    elif rot is not None:
        o.rotation_euler = rot
    return o


# ==================================================================== io
def save_blend(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)


def write_json(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def count_tris(objs):
    n = 0
    for o in objs:
        if o.type == "MESH":
            n += sum(len(p.vertices) - 2 for p in o.data.polygons)
    return n


def append_collection(blend, name):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.collections = [n for n in src.collections if n == name]
    return bpy.data.collections.get(name)
