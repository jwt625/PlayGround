"""Clay materials, thumbprint bump node group and the procedural bullet-hole cutout node group."""
import bpy


def srgb(r, g=None, b=None):
    """sRGB 0..1 (or 0..255 ints or hex string) to linear."""
    if isinstance(r, str):
        h = r.lstrip("#")
        r, g, b = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    elif g is None:
        r, g, b = r
    if max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0

    def c(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    return (c(r), c(g), c(b))


def _n(nt, typ, x=0, y=0, **kw):
    n = nt.nodes.new(typ)
    n.location = (x, y)
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def _l(nt, a, b):
    nt.links.new(a, b)


def _driver(idblock, path, root, prop, expr="v", index=-1):
    fc = idblock.driver_add(path, index) if index >= 0 else idblock.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    var = d.variables.new()
    var.name = "v"
    var.targets[0].id_type = "OBJECT"
    var.targets[0].id = root
    var.targets[0].data_path = '["%s"]' % prop
    return fc


# ------------------------------------------------------------------ clay bump group
def clay_bump_group():
    """NG_clay_bump: thumbprint dimples (concentric ridges in random Voronoi cells), broad dents and fine grain.

    Inputs: Strength, Print Scale (cells per metre), Normal. Output: Normal. Uses the mesh attribute rest_pos
    (undeformed position, metres) so the pattern follows the skinned surface instead of swimming.
    """
    if "NG_clay_bump" in bpy.data.node_groups:
        return bpy.data.node_groups["NG_clay_bump"]
    ng = bpy.data.node_groups.new("NG_clay_bump", "ShaderNodeTree")
    ng.interface.new_socket("Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.35
    ng.interface.new_socket("Print Scale", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 30.0
    ng.interface.new_socket("Normal", in_out="INPUT", socket_type="NodeSocketVector")
    ng.interface.new_socket("Normal", in_out="OUTPUT", socket_type="NodeSocketVector")
    gi = _n(ng, "NodeGroupInput", -1400, 0)
    go = _n(ng, "NodeGroupOutput", 900, 0)
    at = _n(ng, "ShaderNodeAttribute", -1300, 300, attribute_name="rest_pos", attribute_type="GEOMETRY")
    vor = _n(ng, "ShaderNodeTexVoronoi", -1000, 400, voronoi_dimensions="3D", feature="F1", distance="EUCLIDEAN")
    vor.inputs["Randomness"].default_value = 1.0
    _l(ng, at.outputs["Vector"], vor.inputs["Vector"])
    _l(ng, gi.outputs["Print Scale"], vor.inputs["Scale"])
    rings = _n(ng, "ShaderNodeMath", -760, 500, operation="MULTIPLY")
    rings.inputs[1].default_value = 190.0
    _l(ng, vor.outputs["Distance"], rings.inputs[0])
    sine = _n(ng, "ShaderNodeMath", -600, 500, operation="SINE")
    _l(ng, rings.outputs[0], sine.inputs[0])
    mask = _n(ng, "ShaderNodeMapRange", -760, 340)
    mask.inputs["From Min"].default_value = 0.46
    mask.inputs["From Max"].default_value = 0.14
    _l(ng, vor.outputs["Distance"], mask.inputs["Value"])
    sep = _n(ng, "ShaderNodeSeparateColor", -760, 200)
    _l(ng, vor.outputs["Color"], sep.inputs[0])
    amp = _n(ng, "ShaderNodeMapRange", -600, 200)
    amp.inputs["From Min"].default_value = 0.25
    amp.inputs["From Max"].default_value = 0.9
    _l(ng, sep.outputs[0], amp.inputs["Value"])
    m1 = _n(ng, "ShaderNodeMath", -420, 450, operation="MULTIPLY")
    _l(ng, sine.outputs[0], m1.inputs[0])
    _l(ng, mask.outputs[0], m1.inputs[1])
    m2 = _n(ng, "ShaderNodeMath", -260, 450, operation="MULTIPLY")
    _l(ng, m1.outputs[0], m2.inputs[0])
    _l(ng, amp.outputs[0], m2.inputs[1])
    m3 = _n(ng, "ShaderNodeMath", -100, 450, operation="MULTIPLY")
    m3.inputs[1].default_value = 0.22
    _l(ng, m2.outputs[0], m3.inputs[0])
    # broad dents
    nz1 = _n(ng, "ShaderNodeTexNoise", -1000, 0, noise_dimensions="3D")
    nz1.inputs["Scale"].default_value = 11.0
    nz1.inputs["Detail"].default_value = 3.0
    _l(ng, at.outputs["Vector"], nz1.inputs["Vector"])
    nz2 = _n(ng, "ShaderNodeTexNoise", -1000, -250, noise_dimensions="3D")
    nz2.inputs["Scale"].default_value = 260.0
    nz2.inputs["Detail"].default_value = 1.0
    _l(ng, at.outputs["Vector"], nz2.inputs["Vector"])
    a1 = _n(ng, "ShaderNodeMath", -420, 0, operation="MULTIPLY")
    a1.inputs[1].default_value = 0.55
    _l(ng, nz1.outputs["Fac"], a1.inputs[0])
    a2 = _n(ng, "ShaderNodeMath", -420, -250, operation="MULTIPLY")
    a2.inputs[1].default_value = 0.10
    _l(ng, nz2.outputs["Fac"], a2.inputs[0])
    s1 = _n(ng, "ShaderNodeMath", 60, 200, operation="ADD")
    _l(ng, m3.outputs[0], s1.inputs[0])
    _l(ng, a1.outputs[0], s1.inputs[1])
    s2 = _n(ng, "ShaderNodeMath", 240, 100, operation="ADD")
    _l(ng, s1.outputs[0], s2.inputs[0])
    _l(ng, a2.outputs[0], s2.inputs[1])
    bump = _n(ng, "ShaderNodeBump", 480, 0)
    bump.inputs["Distance"].default_value = 0.004
    _l(ng, s2.outputs[0], bump.inputs["Height"])
    _l(ng, gi.outputs["Strength"], bump.inputs["Strength"])
    _l(ng, gi.outputs["Normal"], bump.inputs["Normal"])
    _l(ng, bump.outputs["Normal"], go.inputs["Normal"])
    return ng


# ------------------------------------------------------------------ bullet holes
def hole_group(holes, root, name="NG_bullet_holes"):
    """Procedural see-through cutouts.

    holes: list of dicts {name, empty (Object), base_radius (m), half_len (m), prop (root custom property name)}.
    Each hole is the capsule-free cylinder |local z| < half_len, local radius < base_radius * prop, evaluated in the
    space of its empty (local Z is the hole axis). Output 'Alpha' is 0 inside any hole, 1 elsewhere; connect it to
    a Principled Alpha. The radius Value nodes are driven by root["prop"].
    """
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    ng.interface.new_socket("Alpha", in_out="OUTPUT", socket_type="NodeSocketFloat")
    go = _n(ng, "NodeGroupOutput", 1400, 0)
    prev = None
    for k, h in enumerate(holes):
        y = -k * 330
        tc = _n(ng, "ShaderNodeTexCoord", -900, y)
        tc.object = h["empty"]
        sep = _n(ng, "ShaderNodeSeparateXYZ", -700, y)
        _l(ng, tc.outputs["Object"], sep.inputs[0])
        comb = _n(ng, "ShaderNodeCombineXYZ", -520, y)
        _l(ng, sep.outputs[0], comb.inputs[0])
        _l(ng, sep.outputs[1], comb.inputs[1])
        ln = _n(ng, "ShaderNodeVectorMath", -340, y, operation="LENGTH")
        _l(ng, comb.outputs[0], ln.inputs[0])
        rv = _n(ng, "ShaderNodeValue", -520, y - 140)
        rv.name = "R_%d" % k
        rv.label = h["name"] + " radius (m)"
        rv.outputs[0].default_value = h["base_radius"]
        lt = _n(ng, "ShaderNodeMath", -160, y, operation="LESS_THAN")
        _l(ng, ln.outputs["Value"], lt.inputs[0])
        _l(ng, rv.outputs[0], lt.inputs[1])
        az = _n(ng, "ShaderNodeMath", -340, y - 140, operation="ABSOLUTE")
        _l(ng, sep.outputs[2], az.inputs[0])
        lz = _n(ng, "ShaderNodeMath", -160, y - 140, operation="LESS_THAN")
        lz.inputs[1].default_value = h["half_len"]
        _l(ng, az.outputs[0], lz.inputs[0])
        both = _n(ng, "ShaderNodeMath", 20, y, operation="MULTIPLY")
        _l(ng, lt.outputs[0], both.inputs[0])
        _l(ng, lz.outputs[0], both.inputs[1])
        if prev is None:
            prev = both
        else:
            mx = _n(ng, "ShaderNodeMath", 200, y, operation="MAXIMUM")
            _l(ng, prev.outputs[0], mx.inputs[0])
            _l(ng, both.outputs[0], mx.inputs[1])
            prev = mx
        _driver(ng, 'nodes["R_%d"].outputs[0].default_value' % k, root, h["prop"], "v*%.6f" % h["base_radius"])
    inv = _n(ng, "ShaderNodeMath", 1200, 0, operation="SUBTRACT")
    inv.inputs[0].default_value = 1.0
    _l(ng, prev.outputs[0], inv.inputs[1])
    _l(ng, inv.outputs[0], go.inputs["Alpha"])
    return ng


# ------------------------------------------------------------------ materials
def _finish_principled(mat, nt, bsdf, base, rough, spec, bump_strength, print_scale, holes_group, sheen=0.0,
                       subsurface=0.0):
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular IOR Level"].default_value = spec
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
        bsdf.inputs["Sheen Roughness"].default_value = 0.6
    if subsurface:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (0.5, 0.2, 0.12)
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
    if bump_strength > 0:
        g = _n(nt, "ShaderNodeGroup", -500, -300)
        g.node_tree = clay_bump_group()
        g.inputs["Strength"].default_value = bump_strength
        g.inputs["Print Scale"].default_value = print_scale
        _l(nt, g.outputs["Normal"], bsdf.inputs["Normal"])
    if holes_group is not None:
        h = _n(nt, "ShaderNodeGroup", -500, -600)
        h.node_tree = holes_group
        _l(nt, h.outputs["Alpha"], bsdf.inputs["Alpha"])
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            pass
        try:
            mat.use_transparent_shadow = True
        except Exception:
            pass


def clay(name, color, rough=0.62, spec=0.35, bump=0.35, print_scale=30.0, holes=None, sheen=0.0, subsurface=0.0,
         linear=False):
    """Matte clay with thumbprint bump. color is sRGB (hex or tuple) unless linear=True."""
    base = color if linear else srgb(color)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    _finish_principled(mat, nt, b, base, rough, spec, bump, print_scale, holes, sheen, subsurface)
    return mat


def skin(name, color, root, flush_prop="p_flush", vein_prop=None, flush_color="#d8402f", rough=0.58, bump=0.22,
         holes=None, vein_region=None, flush_zmin=None):
    """Clay skin with a red-flush mix (driven by root[flush_prop]) and optional vein bump (driven by root[vein_prop])."""
    base = srgb(color)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    mix = _n(nt, "ShaderNodeMix", -700, 300, data_type="RGBA")
    mix.inputs[6].default_value = (*base, 1.0)
    mix.inputs[7].default_value = (*srgb(flush_color), 1.0)
    fv = _n(nt, "ShaderNodeValue", -900, 300)
    fv.name = "FLUSH"
    fv.label = "flush (driven)"
    _driver(mat.node_tree, 'nodes["FLUSH"].outputs[0].default_value', root, flush_prop, "v")
    if flush_zmin is not None:
        at0 = _n(nt, "ShaderNodeAttribute", -1300, 500, attribute_name="rest_pos", attribute_type="GEOMETRY")
        sp0 = _n(nt, "ShaderNodeSeparateXYZ", -1150, 500)
        _l(nt, at0.outputs["Vector"], sp0.inputs[0])
        mr = _n(nt, "ShaderNodeMapRange", -1000, 500)
        mr.inputs["From Min"].default_value = flush_zmin
        mr.inputs["From Max"].default_value = flush_zmin + 0.06
        _l(nt, sp0.outputs[2], mr.inputs["Value"])
        mu = _n(nt, "ShaderNodeMath", -820, 400, operation="MULTIPLY")
        _l(nt, fv.outputs[0], mu.inputs[0])
        _l(nt, mr.outputs[0], mu.inputs[1])
        _l(nt, mu.outputs[0], mix.inputs[0])
    else:
        _l(nt, fv.outputs[0], mix.inputs[0])
    col_out = mix.outputs[2]
    height_extra = None
    if vein_prop and vein_region:
        zc, xr, yfront, z0, z1 = vein_region  # head-space window (baseline metres)
        at = _n(nt, "ShaderNodeAttribute", -1500, -50, attribute_name="rest_pos", attribute_type="GEOMETRY")
        sep = _n(nt, "ShaderNodeSeparateXYZ", -1300, -50)
        _l(nt, at.outputs["Vector"], sep.inputs[0])
        vor = _n(nt, "ShaderNodeTexVoronoi", -1300, -250, voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = 75.0
        _l(nt, at.outputs["Vector"], vor.inputs["Vector"])
        # line profile: 1 on the cell edges
        lines = _n(nt, "ShaderNodeMapRange", -1100, -250)
        lines.inputs["From Min"].default_value = 0.09
        lines.inputs["From Max"].default_value = 0.0
        _l(nt, vor.outputs["Distance"], lines.inputs["Value"])
        # window masks: z in [z0, z1], |x| < xr, y < yfront
        mz = _n(nt, "ShaderNodeMapRange", -1100, 0)
        mz.inputs["From Min"].default_value = z0
        mz.inputs["From Max"].default_value = z0 + 0.025
        _l(nt, sep.outputs[2], mz.inputs["Value"])
        mz2 = _n(nt, "ShaderNodeMapRange", -1100, 130)
        mz2.inputs["From Min"].default_value = z1
        mz2.inputs["From Max"].default_value = z1 - 0.025
        _l(nt, sep.outputs[2], mz2.inputs["Value"])
        ax = _n(nt, "ShaderNodeMath", -1100, 260, operation="ABSOLUTE")
        _l(nt, sep.outputs[0], ax.inputs[0])
        mx = _n(nt, "ShaderNodeMapRange", -940, 260)
        mx.inputs["From Min"].default_value = xr
        mx.inputs["From Max"].default_value = xr - 0.02
        _l(nt, ax.outputs[0], mx.inputs["Value"])
        my = _n(nt, "ShaderNodeMapRange", -940, 400)
        my.inputs["From Min"].default_value = yfront + 0.01
        my.inputs["From Max"].default_value = yfront - 0.02
        _l(nt, sep.outputs[1], my.inputs["Value"])
        prod = lines.outputs[0]
        for other in (mz, mz2, mx, my):
            mm = _n(nt, "ShaderNodeMath", -700, -250, operation="MULTIPLY")
            _l(nt, prod, mm.inputs[0])
            _l(nt, other.outputs[0], mm.inputs[1])
            prod = mm.outputs[0]
        vv = _n(nt, "ShaderNodeValue", -700, -400)
        vv.name = "VEIN"
        vv.label = "vein amount (driven)"
        _driver(nt, 'nodes["VEIN"].outputs[0].default_value', root, vein_prop, "v")
        vm = _n(nt, "ShaderNodeMath", -540, -300, operation="MULTIPLY")
        _l(nt, prod, vm.inputs[0])
        _l(nt, vv.outputs[0], vm.inputs[1])
        height_extra = vm.outputs[0]
        # vein tint (purple-ish) on the colour
        tint = _n(nt, "ShaderNodeMix", -560, 200, data_type="RGBA")
        tint.inputs[7].default_value = (*srgb("#8a2a4a"), 1.0)
        _l(nt, vm.outputs[0], tint.inputs[0])
        _l(nt, col_out, tint.inputs[6])
        col_out = tint.outputs[2]
    _finish_principled(mat, nt, b, base, rough, 0.3, bump, 30.0, holes, subsurface=0.05)
    _l(nt, col_out, b.inputs["Base Color"])
    if height_extra is not None:
        # add vein height on top of the clay bump: second bump node chained after the clay one
        vb = _n(nt, "ShaderNodeBump", -250, -400)
        vb.inputs["Distance"].default_value = 0.003
        vb.inputs["Strength"].default_value = 1.0
        _l(nt, height_extra, vb.inputs["Height"])
        src = None
        for lk in nt.links:
            if lk.to_socket == b.inputs["Normal"]:
                src = lk.from_socket
        if src is not None:
            _l(nt, src, vb.inputs["Normal"])
        _l(nt, vb.outputs["Normal"], b.inputs["Normal"])
    return mat


def gloss(name, color, rough=0.12, spec=0.6, holes=None, coat=0.0):
    base = srgb(color)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    _finish_principled(mat, nt, b, base, rough, spec, 0.0, 30.0, holes)
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    return mat


def metal(name, color="#b8bcc4", rough=0.35, holes=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*srgb(color), 1.0)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    return mat
