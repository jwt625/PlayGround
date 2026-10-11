"""Shader node-graph helpers for the VFX library (workstream E, DevLog-013). Blender 4.2.

NB wraps a node tree (material or shader node group) with small arithmetic helpers that accept sockets or numbers.
Time inside a graph comes from a Value node driven by the simple expression `frame/<fps>` (evaluated without Python,
so it works with auto-run scripts disabled).
"""
import bpy

FPS = 30


def _is_sock(x):
    return isinstance(x, bpy.types.NodeSocket)


class NB:
    """Node builder over node_tree `nt`; nodes are laid out left to right by creation order (cosmetic)."""

    def __init__(self, nt):
        self.nt = nt
        self.n = 0

    def node(self, kind, **props):
        nd = self.nt.nodes.new(kind)
        nd.location = (-900 + 180 * (self.n % 8), 400 - 160 * (self.n // 8))
        self.n += 1
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def link(self, a, b):
        self.nt.links.new(a, b)

    def put(self, sock, val):
        if _is_sock(val):
            self.link(val, sock)
        elif val is not None:
            sock.default_value = val

    # ---------------------------------------------------------------- scalar math
    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.put(nd.inputs[0], a)
        if b is not None:
            self.put(nd.inputs[1], b)
        if c is not None:
            self.put(nd.inputs[2], c)
        return nd.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b, clamp=False):
        return self.math("MULTIPLY", a, b, clamp=clamp)

    def pow(self, a, b):
        return self.math("POWER", a, b)

    def clamp01(self, a):
        return self.math("ADD", a, 0.0, clamp=True)

    def one_minus(self, a):
        return self.math("SUBTRACT", 1.0, a, clamp=True)

    def mrange(self, v, a0, a1, b0=0.0, b1=1.0, interp="LINEAR", clamp=True):
        nd = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.put(nd.inputs["Value"], v)
        self.put(nd.inputs["From Min"], a0)
        self.put(nd.inputs["From Max"], a1)
        self.put(nd.inputs["To Min"], b0)
        self.put(nd.inputs["To Max"], b1)
        return nd.outputs["Result"]

    def smooth(self, v, a0, a1):
        return self.mrange(v, a0, a1, 0.0, 1.0, "SMOOTHSTEP")

    def lerp(self, a, b, t):
        """Float lerp a + (b - a) t."""
        return self.add(a, self.mul(self.sub(b, a), t))

    # ---------------------------------------------------------------- vectors / colours
    def sep(self, v):
        nd = self.node("ShaderNodeSeparateXYZ")
        self.put(nd.inputs[0], v)
        return nd.outputs

    def comb(self, x, y, z):
        nd = self.node("ShaderNodeCombineXYZ")
        for i, s in enumerate((x, y, z)):
            self.put(nd.inputs[i], s)
        return nd.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self.put(nd.inputs[0], a)
        if b is not None:
            self.put(nd.inputs[1], b)
        if scale is not None:
            self.put(nd.inputs["Scale"], scale)
        return nd.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else nd.outputs["Vector"]

    def cmix(self, a, b, fac, blend="MIX"):
        nd = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.put(nd.inputs["Factor"], fac)
        self.put(nd.inputs[6], a)
        self.put(nd.inputs[7], b)
        return nd.outputs[2]

    def cscale(self, col, s):
        nd = self.node("ShaderNodeVectorMath", operation="SCALE")
        self.put(nd.inputs[0], col)
        self.put(nd.inputs["Scale"], s)
        return nd.outputs["Vector"]

    # ---------------------------------------------------------------- inputs
    def texcoord(self, out="Object"):
        return self.node("ShaderNodeTexCoord").outputs[out]

    def geometry(self, out):
        return self.node("ShaderNodeNewGeometry").outputs[out]

    def objinfo(self, out):
        return self.node("ShaderNodeObjectInfo").outputs[out]

    def layer_weight(self, blend=0.5, out="Facing"):
        nd = self.node("ShaderNodeLayerWeight")
        self.put(nd.inputs["Blend"], blend)
        return nd.outputs[out]

    def time(self, fps=FPS):
        nd = self.node("ShaderNodeValue")
        nd.label = "time_s"
        fc = nd.outputs[0].driver_add("default_value")
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = "frame/%d" % fps
        return nd.outputs[0]

    def noise(self, vec=None, w=None, scale=5.0, detail=2.0, rough=0.5, dims="3D", out="Fac"):
        nd = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        if vec is not None:
            self.put(nd.inputs["Vector"], vec)
        if w is not None:
            self.put(nd.inputs["W"], w)
        self.put(nd.inputs["Scale"], scale)
        self.put(nd.inputs["Detail"], detail)
        self.put(nd.inputs["Roughness"], rough)
        return nd.outputs[out]

    def voronoi(self, vec=None, scale=5.0, feature="F1", dims="3D", out="Distance", randomness=1.0):
        nd = self.node("ShaderNodeTexVoronoi", voronoi_dimensions=dims, feature=feature)
        if vec is not None:
            self.put(nd.inputs["Vector"], vec)
        self.put(nd.inputs["Scale"], scale)
        self.put(nd.inputs["Randomness"], randomness)
        return nd.outputs[out]

    def white(self, vec, dims="3D", out="Value"):
        nd = self.node("ShaderNodeTexWhiteNoise", noise_dimensions=dims)
        self.put(nd.inputs["Vector"] if dims != "1D" else nd.inputs["W"], vec)
        return nd.outputs[out]

    # ---------------------------------------------------------------- shaders
    def emission(self, color, strength):
        nd = self.node("ShaderNodeEmission")
        self.put(nd.inputs["Color"], color)
        self.put(nd.inputs["Strength"], strength)
        return nd.outputs[0]

    def transparent(self, color=(1, 1, 1, 1)):
        nd = self.node("ShaderNodeBsdfTransparent")
        self.put(nd.inputs[0], color)
        return nd.outputs[0]

    def add_shader(self, a, b):
        nd = self.node("ShaderNodeAddShader")
        self.link(a, nd.inputs[0])
        self.link(b, nd.inputs[1])
        return nd.outputs[0]

    def mix_shader(self, fac, a, b):
        nd = self.node("ShaderNodeMixShader")
        self.put(nd.inputs[0], fac)
        self.link(a, nd.inputs[1])
        self.link(b, nd.inputs[2])
        return nd.outputs[0]

    def principled(self, **kw):
        nd = self.node("ShaderNodeBsdfPrincipled")
        for k, v in kw.items():
            self.put(nd.inputs[k.replace("_", " ")], v)
        return nd


# -------------------------------------------------------------------- node groups
def new_group(name, inputs, outputs):
    """Shader node group with sockets; inputs/outputs: list of (name, socket_type, default, min, max). Returns (ng, NB, gin, gout).
    An existing group of the same name is returned as is (second value None) so materials share one definition."""
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name], None, None, None
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for spec in inputs:
        nm, st = spec[0], spec[1]
        s = ng.interface.new_socket(nm, in_out="INPUT", socket_type=st)
        if len(spec) > 2 and spec[2] is not None:
            s.default_value = spec[2]
        if len(spec) > 3:
            s.min_value = spec[3]
        if len(spec) > 4:
            s.max_value = spec[4]
    for spec in outputs:
        ng.interface.new_socket(spec[0], in_out="OUTPUT", socket_type=spec[1])
    b = NB(ng)
    gin = b.node("NodeGroupInput")
    gout = b.node("NodeGroupOutput")
    return ng, b, gin, gout


def new_material(name, replace=True):
    """Empty node material (output node only). Returns (mat, NB, output node)."""
    m = bpy.data.materials.get(name)
    if m is not None and replace:
        m.node_tree.nodes.clear() if m.node_tree else None
    if m is None:
        m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    b = NB(m.node_tree)
    out = b.node("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return m, b, out


def group_node(b, ng, **inputs):
    nd = b.node("ShaderNodeGroup")
    nd.node_tree = ng
    for k, v in inputs.items():
        b.put(nd.inputs[k.replace("_", " ")], v)
    return nd


def set_blended(m, shadow=False, backface=False):
    """Alpha-blended EEVEE Next surface (additive effects: emission + transparent)."""
    m.surface_render_method = "BLENDED"
    for attr, val in (("use_transparency_overlap", True), ("use_backface_culling", backface)):
        try:
            setattr(m, attr, val)
        except AttributeError:
            pass
    try:
        m.use_transparent_shadow = not shadow
    except AttributeError:
        pass
