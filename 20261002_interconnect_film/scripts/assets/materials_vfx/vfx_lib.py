"""materials_vfx helper library (Blender 4.2): node-group DSL, drivers, scene/preview helpers.

Import from a build script in this folder:

    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "_common"))
    import common as C
    import vfx_lib as V

Also usable by the assembler at run time: V.set_param(obj, "Speed", 2.0) sets a Geometry Nodes modifier input by its
display name; V.append_library(...) appends node groups / materials / collections from the library blend.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

# ----------------------------------------------------------------------------- colour
def _s2l(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb(h, a=1.0):
    """'#RRGGBB' (sRGB) -> linear RGBA tuple (what Blender colour sockets store)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (_s2l(r), _s2l(g), _s2l(b), a)


# ----------------------------------------------------------------------------- node groups
_T = None  # current tree for the expression DSL


def use(tree):
    global _T
    _T = tree
    return tree


def new_group(name, kind="Shader", inputs=(), outputs=()):
    """Create a node group. kind: Shader | Geometry | Compositor.
    inputs: (name, socket_type, default, min, max[, description]); socket_type like 'Float', 'Color', 'Vector',
    'Int', 'Bool', 'Shader', 'Geometry', 'Object', 'Material'.  outputs: (name, socket_type).
    Returns (tree, group_input_node, group_output_node).
    """
    if name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[name])
    tree = bpy.data.node_groups.new(name, kind + "NodeTree")
    for o in outputs:
        tree.interface.new_socket(o[0], in_out="OUTPUT", socket_type="NodeSocket" + o[1])
    for i in inputs:
        nm, st, d = i[0], i[1], i[2]
        s = tree.interface.new_socket(nm, in_out="INPUT", socket_type="NodeSocket" + st)
        if d is not None:
            try:
                s.default_value = d
            except Exception:
                s.default_value = tuple(d) if hasattr(d, "__len__") else d
        if len(i) > 4 and i[3] is not None:
            try:
                s.min_value = i[3]
                s.max_value = i[4]
            except Exception:
                pass
        if len(i) > 5:
            s.description = i[5]
    gi = tree.nodes.new("NodeGroupInput")
    go = tree.nodes.new("NodeGroupOutput")
    use(tree)
    return tree, gi, go


def N(idname, **props):
    """Create a node in the current tree. Props starting with 'in_' set input defaults by name (in_Strength=2.0)."""
    n = _T.nodes.new(idname)
    for k, v in props.items():
        if k.startswith("in_"):
            setin(n, k[3:].replace("__", " "), v)
        else:
            setattr(n, k, v)
    return n


def _sock(node, key, outputs=False):
    col = node.outputs if outputs else node.inputs
    if isinstance(key, int):
        return col[key]
    for s in col:
        if s.name == key and s.enabled:
            return s
    for s in col:
        if s.name == key:
            return s
    raise KeyError("%s has no %s %r (have %s)" % (node.name, "output" if outputs else "input", key, [s.name for s in col]))


def setin(node, key, val):
    s = _sock(node, key)
    if hasattr(val, "outputs") or hasattr(val, "node"):
        L(val, s)
    else:
        try:
            s.default_value = val
        except TypeError:
            s.default_value = tuple(val)
    return s


def L(src, dst, *rest):
    """Link: src may be a node (first enabled output), an output socket, or X/Vv wrapper; dst a socket."""
    if isinstance(src, (X, Vv)):
        src = src.s
    if hasattr(src, "outputs") and not hasattr(src, "node"):
        src = [o for o in src.outputs if o.enabled][0]
    _T.links.new(src, dst)


def out(node, key=0):
    return _sock(node, key, outputs=True)


def auto_layout(tree, xgap=260, ygap=190):
    """Layer nodes left to right by longest path from group inputs; stack vertically."""
    depth = {}
    nodes = list(tree.nodes)
    by_in = {n: [] for n in nodes}
    for l in tree.links:
        by_in[l.to_node].append(l.from_node)

    def d(n, stack=()):
        if n in depth:
            return depth[n]
        if n in stack:
            return 0
        v = 0
        for p in by_in[n]:
            v = max(v, d(p, stack + (n,)) + 1)
        depth[n] = v
        return v
    for n in nodes:
        d(n)
    cols = {}
    for n in nodes:
        cols.setdefault(depth[n], []).append(n)
    for k, ns in cols.items():
        for j, n in enumerate(ns):
            n.location = (k * xgap, -j * ygap)


# ----------------------------------------------------------------------------- expression DSL (float / vector math)
class X:
    """Float expression wrapper: arithmetic builds ShaderNodeMath nodes in the current tree."""
    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s

    def _m(self, op, a, b=None, c=None):
        return mathop(op, a, b, c)

    def __add__(self, o): return self._m("ADD", self, o)
    __radd__ = __add__
    def __sub__(self, o): return self._m("SUBTRACT", self, o)
    def __rsub__(self, o): return self._m("SUBTRACT", o, self)
    def __mul__(self, o): return self._m("MULTIPLY", self, o)
    __rmul__ = __mul__
    def __truediv__(self, o): return self._m("DIVIDE", self, o)
    def __rtruediv__(self, o): return self._m("DIVIDE", o, self)
    def __pow__(self, o): return self._m("POWER", self, o)
    def __rpow__(self, o): return self._m("POWER", o, self)
    def __neg__(self): return self._m("MULTIPLY", self, -1.0)
    def __gt__(self, o): return self._m("GREATER_THAN", self, o)
    def __lt__(self, o): return self._m("LESS_THAN", self, o)
    def __ge__(self, o): return self._m("GREATER_THAN", self, o - 1e-6 if isinstance(o, float) else o)
    def __le__(self, o): return self._m("LESS_THAN", self, o + 1e-6 if isinstance(o, float) else o)


def wrapx(v):
    if isinstance(v, X):
        return v
    if hasattr(v, "node"):  # a socket
        return X(v)
    if hasattr(v, "outputs"):
        return X([o for o in v.outputs if o.enabled][0])
    n = _T.nodes.new("ShaderNodeValue")
    n.outputs[0].default_value = float(v)
    return X(n.outputs[0])


def mathop(op, a, b=None, c=None, clamp=False):
    n = _T.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            L(v, n.inputs[i])
    return X(n.outputs[0])


def sin(a): return mathop("SINE", a)
def cos(a): return mathop("COSINE", a)
def sqrt(a): return mathop("SQRT", a)
def absx(a): return mathop("ABSOLUTE", a)
def expx(a): return mathop("EXPONENT", a)
def minx(a, b): return mathop("MINIMUM", a, b)
def maxx(a, b): return mathop("MAXIMUM", a, b)
def fract(a): return mathop("FRACT", a)
def floorx(a): return mathop("FLOOR", a)
def modx(a, b): return mathop("FLOORED_MODULO", a, b)
def clamp01(a): return mathop("ADD", a, 0.0, clamp=True)
def clampx(a, lo, hi):
    return minx(maxx(a, lo), hi)


def lerp(a, b, t):
    return wrapx(a) + (wrapx(b) - wrapx(a)) * wrapx(t)


def smoothstep(e0, e1, x):
    n = _T.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP"
    n.clamp = True
    for i, v in zip((1, 2), (e0, e1)):
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            L(v, n.inputs[i])
    n.inputs[3].default_value = 0.0
    n.inputs[4].default_value = 1.0
    L(x, n.inputs[0]) if not isinstance(x, (int, float)) else setattr(n.inputs[0], "default_value", float(x))
    return X(n.outputs[0])


def mapr(x, a0, a1, b0, b1, clamp=True):
    n = _T.nodes.new("ShaderNodeMapRange")
    n.clamp = clamp
    for i, v in zip((1, 2, 3, 4), (a0, a1, b0, b1)):
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            L(v, n.inputs[i])
    L(x, n.inputs[0]) if not isinstance(x, (int, float)) else setattr(n.inputs[0], "default_value", float(x))
    return X(n.outputs[0])


def where(cond, a, b):
    """cond ? a : b for float expressions (cond is 0/1)."""
    return wrapx(b) + (wrapx(a) - wrapx(b)) * wrapx(cond)


def gi_x(gi, name):
    return X(gi.outputs[name])


class Vv:
    """Vector expression wrapper (ShaderNodeVectorMath)."""
    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s

    def __add__(self, o): return vop("ADD", self, o)
    __radd__ = __add__
    def __sub__(self, o): return vop("SUBTRACT", self, o)
    def __rsub__(self, o): return vop("SUBTRACT", o, self)
    def __mul__(self, o):
        if isinstance(o, (Vv,)) or (isinstance(o, tuple)):
            return vop("MULTIPLY", self, o)
        return vscale(self, o)
    __rmul__ = __mul__
    def __neg__(self): return vscale(self, -1.0)

    @property
    def x(self): return sepxyz(self)[0]
    @property
    def y(self): return sepxyz(self)[1]
    @property
    def z(self): return sepxyz(self)[2]


def wrapv(v):
    if isinstance(v, Vv):
        return v
    if hasattr(v, "node"):
        return Vv(v)
    n = _T.nodes.new("FunctionNodeInputVector")
    n.vector = tuple(v)
    return Vv(n.outputs[0])


def vop(op, a, b=None, c=None):
    n = _T.nodes.new("ShaderNodeVectorMath")
    n.operation = op
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (tuple, list)):
            n.inputs[i].default_value = tuple(v)
        else:
            L(v, n.inputs[i])
    return Vv(n.outputs[0])


def vscale(a, f):
    n = _T.nodes.new("ShaderNodeVectorMath")
    n.operation = "SCALE"
    L(a, n.inputs[0])
    if isinstance(f, (int, float)):
        n.inputs[3].default_value = float(f)
    else:
        L(f, n.inputs[3])
    return Vv(n.outputs[0])


def vnorm(a):
    n = _T.nodes.new("ShaderNodeVectorMath")
    n.operation = "NORMALIZE"
    L(a, n.inputs[0])
    return Vv(n.outputs[0])


def vlen(a):
    n = _T.nodes.new("ShaderNodeVectorMath")
    n.operation = "LENGTH"
    L(a, n.inputs[0])
    return X(n.outputs["Value"])


def vdot(a, b):
    n = _T.nodes.new("ShaderNodeVectorMath")
    n.operation = "DOT_PRODUCT"
    L(a, n.inputs[0])
    if isinstance(b, (tuple, list)):
        n.inputs[1].default_value = tuple(b)
    else:
        L(b, n.inputs[1])
    return X(n.outputs["Value"])


def vsin(a): return vop("SINE", a)


def sepxyz(v):
    n = _T.nodes.new("ShaderNodeSeparateXYZ")
    L(v, n.inputs[0])
    return X(n.outputs[0]), X(n.outputs[1]), X(n.outputs[2])


def comb(x, y, z):
    n = _T.nodes.new("ShaderNodeCombineXYZ")
    for i, v in enumerate((x, y, z)):
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = float(v)
        else:
            L(v, n.inputs[i])
    return Vv(n.outputs[0])


# ----------------------------------------------------------------------------- group usage
def group_node(tree, group, **inputs):
    """Add a group node to `tree` (a material/world/compositor tree) using node group `group` and set inputs by name."""
    prev = _T
    use(tree)
    t = "ShaderNodeGroup" if tree.bl_idname in ("ShaderNodeTree",) else (
        "CompositorNodeGroup" if tree.bl_idname == "CompositorNodeTree" else "GeometryNodeGroup")
    n = tree.nodes.new(t)
    n.node_tree = group
    for k, v in inputs.items():
        setin(n, k.replace("__", " "), v)
    use(prev) if prev else None
    return n


def new_material(name, use_nodes=True):
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    m = bpy.data.materials.new(name)
    m.use_nodes = use_nodes
    return m


def clear_nodes(tree):
    for n in list(tree.nodes):
        tree.nodes.remove(n)


def mat_output(m):
    for n in m.node_tree.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial":
            return n
    return None


def set_render_method(m, method="DITHERED", shadows=True):
    """EEVEE Next surface_render_method: DITHERED (default, fast, stochastic alpha) or BLENDED (sorted, no AA shadow)."""
    m.surface_render_method = method
    try:
        m.use_transparent_shadow = shadows
    except Exception:
        pass
    return m


# ----------------------------------------------------------------------------- drivers
def drive_prop(target_id, data_path, index, owner, props, expr):
    """Add a scripted (simple expression) driver. props = {var_name: custom_prop_name_on_owner}."""
    fc = target_id.driver_add(data_path, index) if index is not None else target_id.driver_add(data_path)
    d = fc.driver
    d.type = "SCRIPTED"
    for vn, pn in props.items():
        v = d.variables.new()
        v.name = vn
        v.type = "SINGLE_PROP"
        v.targets[0].id = owner
        v.targets[0].data_path = '["%s"]' % pn
    d.expression = expr
    return fc


def mod_socket_id(mod, name):
    """Identifier ('Socket_N') of a Geometry Nodes modifier input by its display name."""
    for it in mod.node_group.interface.items_tree:
        if getattr(it, "in_out", None) == "INPUT" and it.name == name:
            return it.identifier
    raise KeyError(name)


def set_param(obj, name, value, modifier="GeometryNodes"):
    """Assembler helper: set an input of an object's Geometry Nodes modifier by display name."""
    mod = obj.modifiers[modifier]
    ident = mod_socket_id(mod, name)
    try:
        mod[ident] = value
    except TypeError:
        mod[ident] = tuple(value)
    obj.update_tag()


def drive_mod(obj, mod, name, owner, props, expr):
    """Drive a Geometry Nodes modifier input (by display name) from custom properties of `owner`."""
    ident = mod_socket_id(mod, name)
    path = 'modifiers["%s"]["%s"]' % (mod.name, ident)
    fc = obj.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    for vn, pn in props.items():
        v = d.variables.new()
        v.name = vn
        v.type = "SINGLE_PROP"
        v.targets[0].id = owner
        v.targets[0].data_path = '["%s"]' % pn
    d.expression = expr
    return fc


# ----------------------------------------------------------------------------- scene helpers
def blank_scene(res=(900, 675), samples=16, engine="BLENDER_EEVEE_NEXT"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = engine
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.look = "None"
    scn.eevee.taa_render_samples = samples
    scn.render.fps = 30
    return scn


def prep_scene(res=(900, 675), samples=16, engine="BLENDER_EEVEE_NEXT"):
    """Configure the currently open scene for preview rendering (use after reopen())."""
    scn = bpy.context.scene
    scn.render.engine = engine
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.look = "None"
    scn.eevee.taa_render_samples = samples
    scn.render.fps = 30
    # hide everything that exists already (the asset), previews add their own objects
    return scn


def reopen(path):
    bpy.ops.wm.open_mainfile(filepath=path)
    return bpy.context.scene


def make_camera(loc, target, lens=50, name="CAM", sensor=36, ortho=None):
    scn = bpy.context.scene
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.01
    cd.clip_end = 1000
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
    cam = bpy.data.objects.new(name, cd)
    scn.collection.objects.link(cam)
    cam.location = loc
    tg = bpy.data.objects.new(name + "_target", None)
    tg.location = target
    scn.collection.objects.link(tg)
    c = cam.constraints.new("TRACK_TO")
    c.target = tg
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    scn.camera = cam
    return cam, tg


def simple_world(color=(0.6, 0.7, 0.85), strength=0.8):
    w = bpy.data.worlds.new("W_simple")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color[:3], 1)
    bg.inputs["Strength"].default_value = strength
    bpy.context.scene.world = w
    return w


def sun_light(name="sun", energy=3.0, rot=(0.9, 0.3, 0.7), angle=0.12, loc=(0, 0, 5), color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.angle = angle
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.rotation_euler = rot
    o.location = loc
    return o


def area_light(name, energy, size, loc, target=None, size_y=None, color=(1, 1, 1), rot=None, shape="RECTANGLE"):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.shape = shape if size_y else "SQUARE"
    ld.size = size
    if size_y:
        ld.size_y = size_y
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    if target is not None:
        d = Vector(target) - Vector(loc)
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    elif rot:
        o.rotation_euler = rot
    return o


def render_still(path, frame=None):
    scn = bpy.context.scene
    if frame is not None:
        scn.frame_set(frame)
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def label_text(body, loc, size=0.08, color=(1, 1, 1), name="label"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "TOP"
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (math.pi / 2, 0, 0)
    m = new_material("PREVIEW_label_" + name)
    t = m.node_tree
    clear_nodes(t)
    e = t.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color[:3], 1)
    e.inputs["Strength"].default_value = 1.0
    o_ = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(e.outputs[0], o_.inputs[0])
    o.data.materials.append(m)
    return o


def tile_pngs(paths, out_path, cols):
    """Tile PNG files into one contact sheet with ffmpeg (shell); returns out_path."""
    import subprocess
    n = len(paths)
    rows = (n + cols - 1) // cols
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    ins = "".join("[%d:v]" % i for i in range(n))
    cmd += ["-filter_complex", "%sxstack=inputs=%d:layout=%s" % (ins, n, "|".join(
        "%s_%s" % ("+".join(["w0"] * (i % cols)) if i % cols else "0", "+".join(["h0"] * (i // cols)) if i // cols else "0")
        for i in range(n))), out_path]
    subprocess.run(cmd, check=True)
    return out_path


# ----------------------------------------------------------------------------- previews / lighting for material sheets
def studio_lights(center=(0, 0, 0), scale=1.0, key=1.0, world=0.55):
    """Soft three-light studio for material previews (key softbox, fill, rim) + neutral gradient world."""
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    nt = w.node_tree
    clear_nodes(nt)
    use(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    stops = [(0.0, (0.03, 0.03, 0.035)), (0.30, (0.12, 0.12, 0.14)), (0.42, (0.7, 0.72, 0.8)), (0.50, (3.5, 3.5, 3.5)),
             (0.58, (0.55, 0.58, 0.68)), (0.75, (0.35, 0.40, 0.50)), (1.0, (1.1, 1.2, 1.4))]
    cr.elements[0].position = stops[0][0]
    cr.elements[0].color = (*stops[0][1], 1)
    cr.elements[1].position = stops[1][0]
    cr.elements[1].color = (*stops[1][1], 1)
    for pos_, col_ in stops[2:]:
        e = cr.elements.new(pos_)
        e.color = (*col_, 1)
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = world
    nt.links.new(ramp.outputs[0], bg.inputs[0])
    o = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], o.inputs[0])
    bpy.context.scene.world = w
    s = scale
    c = Vector(center)
    # energy = irradiance_target * pi * d^2 so the exposure does not depend on the rig scale
    for nm, pos, E, size, col in (
            ("PREVIEW_key", (-2.4, -3.0, 3.0), 2.6 * key, 2.0, (1.0, 0.96, 0.9)),
            ("PREVIEW_fill", (3.0, -2.5, 1.2), 0.9 * key, 3.0, (0.85, 0.92, 1.0)),
            ("PREVIEW_rim", (1.0, 3.2, 2.6), 1.6 * key, 2.0, (1, 1, 1))):
        p = c + Vector(pos) * s
        d = (p - c).length
        area_light(nm, E * math.pi * d * d, size * s, p, target=c, color=col)
    bpy.context.scene.eevee.shadow_ray_count = 3
    bpy.context.scene.eevee.shadow_step_count = 8


def grid_positions(n, cols, pitch):
    rows = (n + cols - 1) // cols
    pts = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = (c - (cols - 1) / 2.0) * pitch
        z = ((rows - 1) / 2.0 - r) * pitch
        pts.append((x, 0.0, z))
    return pts


def material_sheet(mats, out_png, cols=6, pitch=1.2, radius=0.5, res=(900, 675), samples=24, labels=True,
                   lights=True, backdrop=(0.5, 0.52, 0.55), shape="sphere", label_size=0.085, fit=1.0):
    """Render a labelled grid of material spheres (or cards) to out_png. Builds throwaway objects in the current scene;
    call on a scratch scene (blank_scene) -- the caller should not save the blend afterwards."""
    scn = bpy.context.scene
    for o0 in scn.objects:
        if o0.type not in {"LIGHT", "CAMERA"}:
            o0.hide_render = True
            o0.hide_viewport = True
    n = len(mats)
    rows = (n + cols - 1) // cols
    pos = grid_positions(n, cols, pitch)
    for m, p in zip(mats, pos):
        if shape == "card":
            bpy.ops.mesh.primitive_cube_add(size=1, location=p)
            o = bpy.context.active_object
            o.scale = (radius * 1.7, radius * 0.12, radius * 1.7)
            bpy.ops.object.transform_apply(scale=True)
            mod = o.modifiers.new("b", "BEVEL")
            mod.width = 0.02
            mod.segments = 3
        elif shape == "sphere_flat":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=64, ring_count=32, location=p)
            o = bpy.context.active_object
            smooth(o)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=64, ring_count=32, location=p)
            o = bpy.context.active_object
            smooth(o)
        o.name = "PREVIEW_" + m.name
        o.data.materials.append(m)
        if labels:
            label_text(m.name.replace("MAT_vfx_", ""), (p[0], -0.02, p[2] - radius * 1.25), size=label_size, name="L_" + m.name)
    # backdrop wall
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 2.2, 0), rotation=(math.pi / 2, 0, 0))
    bk = bpy.context.active_object
    bk.scale = (pitch * cols * 3, pitch * rows * 3, 1)
    bk.visible_shadow = False
    bm = new_material("PREVIEW_backdrop")
    bm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*backdrop, 1)
    bm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    bk.data.materials.append(bm)
    if lights:
        studio_lights(center=(0, 0, 0), scale=max(cols, rows) * pitch / 3.0)
    wide = cols * pitch
    high = rows * pitch
    # fit the grid inside the frame
    aspect = res[0] / res[1]
    need_w = wide
    need_h = high * aspect
    ortho = max(need_w, need_h) * 1.04 / fit
    cam, tg = make_camera((0, -20, 0), (0, 0, 0), lens=50, ortho=ortho)
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.eevee.taa_render_samples = samples
    return render_still(out_png)


def finish_lib(asset_id, blend_path, coll, meta, keep_objects=True):
    """Save the library blend + JSON metadata (no previews: build scripts render their own)."""
    import json
    import common as C
    meta = dict(meta)
    meta["asset_id"] = asset_id
    bb = C.bbox_mm(coll) if coll is not None else None
    meta["bbox_mm"] = bb
    if bb:
        meta["size_mm"] = [round(bb[1][k] - bb[0][k], 3) for k in range(3)]
    meta["triangles"] = C.count_tris(coll) if coll is not None else 0
    C.save(blend_path)
    C.write_meta(os.path.splitext(blend_path)[0] + ".json", meta)
    size = os.path.getsize(blend_path)
    print("ASSET", asset_id, "blend_bytes", size, "tris", meta["triangles"])
    return meta


def mark_asset(idblock, tags=(), description=""):
    try:
        idblock.asset_mark()
        idblock.asset_data.description = description
        for t in tags:
            idblock.asset_data.tags.new(t)
    except Exception as e:
        print("asset_mark failed", idblock.name, e)


def append_library(blend_path, node_groups=(), materials=(), collections=(), worlds=(), link=False):
    """Assembler helper: append datablocks by name from a library .blend."""
    with bpy.data.libraries.load(blend_path, link=link) as (src, dst):
        dst.node_groups = [n for n in node_groups if n in src.node_groups]
        dst.materials = [n for n in materials if n in src.materials]
        dst.collections = [n for n in collections if n in src.collections]
        dst.worlds = [n for n in worlds if n in src.worlds]
    return dst


def drive_time_input(mat, node, input_name="Time", fps=30.0):
    """Drive a group-node input in a material with scene time in seconds (frame / fps)."""
    nt = mat.node_tree
    path = 'nodes["%s"].inputs["%s"].default_value' % (node.name, input_name)
    fc = nt.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = "frame / %s" % float(fps)
    return fc
