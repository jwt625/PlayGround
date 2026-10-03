"""Effect building blocks: closed-form (stateless) Geometry Nodes particle groups, FX materials, effect roots."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bmesh
import bpy
import common as C
import vfx_lib as V
from mathutils import Vector

PI = math.pi

# ----------------------------------------------------------------------------- particle GN factory
def particle_group(name, shape="sphere", align_vel=False, tumble=False, defaults=None):
    """Create a stateless particle node group. shape: sphere | quad | cone | object.
    Position, scale and fade are closed-form functions of Time (seconds): scrubbing/jumping frames is exact."""
    D = dict(Count=20, Life=1.5, Spawn_Window=0.5, Loop=0, Emit_Radius=0.05, Emit_Squash_Z=1.0, Direction=(0, 0, 1),
             Spread=0.5, Speed=1.0, Speed_Var=0.3, Drag=1.0, Gravity=(0, 0, 0), Turbulence=0.0, Scale_Start=0.05,
             Scale_End=0.1, Scale_Var=0.2, Pop=0.15, Fade=0.3, Overshoot=0.0, Spin=0.0, Stretch=0.0, Floor_Z=-1e4,
             Squash=0.3, Intensity=1.0, Seed=1, Time=0.0)
    D.update(defaults or {})
    ins = [("Geometry", "Geometry", None), ("Time", "Float", D["Time"], -1e5, 1e5, "Seconds since the effect trigger (drive from root p_t)"),
           ("Seed", "Int", D["Seed"], 0, 100000, "Random seed"), ("Count", "Int", D["Count"], 1, 2000, "Number of particles"),
           ("Life", "Float", D["Life"], 0.01, 100.0, "Lifetime of each particle in seconds"),
           ("Spawn Window", "Float", D["Spawn_Window"], 0.0, 100.0, "Spawn times are spread over this many seconds (0 = burst)"),
           ("Loop", "Float", float(D["Loop"]), 0.0, 1.0, "1 = repeat forever"),
           ("Emit Radius", "Float", D["Emit_Radius"], 0.0, 100.0, "Half-size of the emission box (metres)"),
           ("Emit Squash Z", "Float", D["Emit_Squash_Z"], 0.0, 1.0, "Flatten the emission box vertically (0 = disc/plane)"),
           ("Direction", "Vector", D["Direction"], None, None, "Mean emission direction (local space)"),
           ("Spread", "Float", D["Spread"], 0.0, 5.0, "Random cone/sphere spread added to Direction"),
           ("Speed", "Float", D["Speed"], 0.0, 1000.0, "Initial speed (m/s)"),
           ("Speed Var", "Float", D["Speed_Var"], 0.0, 1.0, "Random speed variation"),
           ("Drag", "Float", D["Drag"], 0.0, 100.0, "Exponential velocity decay rate (1/s)"),
           ("Gravity", "Vector", D["Gravity"], None, None, "Constant acceleration (negative z = falls, positive z = buoyant)"),
           ("Turbulence", "Float", D["Turbulence"], 0.0, 10.0, "Wobble amplitude"),
           ("Scale Start", "Float", D["Scale_Start"], 0.0, 100.0, "Particle size at birth"),
           ("Scale End", "Float", D["Scale_End"], 0.0, 100.0, "Particle size at death"),
           ("Scale Var", "Float", D["Scale_Var"], 0.0, 1.0, "Random size variation"),
           ("Pop", "Float", D["Pop"], 0.01, 1.0, "Fraction of life used to grow in"),
           ("Fade", "Float", D["Fade"], 0.0, 1.0, "Fraction of life at the end spent shrinking / fading out"),
           ("Overshoot", "Float", D["Overshoot"], 0.0, 2.0, "Cartoon overshoot on pop-in"),
           ("Spin", "Float", D["Spin"], 0.0, 100.0, "Tumble rate (rad/s)"),
           ("Stretch", "Float", D["Stretch"], 0.0, 10.0, "Stretch along velocity per m/s of speed (needs align-to-velocity groups)"),
           ("Floor Z", "Float", D["Floor_Z"], -1e5, 1e5, "Particles stop and squash at this local z"),
           ("Squash", "Float", D["Squash"], 0.0, 1.0, "Vertical squash factor of particles resting on the floor"),
           ("Intensity", "Float", D["Intensity"], 0.0, 1000.0, "Emission multiplier stored as attribute fx_int"),
           ("Material", "Material", None, None, None, "Material for the particles"),
           ]
    if shape == "object":
        ins.append(("Instance Object", "Object", None, None, None, "Object used as particle shape"))
    tree, gi, go = V.new_group(name, "Geometry", ins, [("Geometry", "Geometry")])
    g = lambda n: V.X(gi.outputs[n])
    gv = lambda n: V.wrapv(gi.outputs[n])
    idx = V.X(V.N("GeometryNodeInputIndex").outputs[0])
    seed = V.X(gi.outputs["Seed"])

    def rnd(k, lo=0.0, hi=1.0):
        n = V.N("FunctionNodeRandomValue", data_type="FLOAT")
        V.setin(n, "Min", lo)
        V.setin(n, "Max", hi)
        V.L(idx, _in(n, "ID"))
        V.L(seed + float(k * 17), _in(n, "Seed"))
        return V.X(_out(n, "Value"))

    # time handling
    life = g("Life")
    delay = rnd(0) * g("Spawn Window")
    age0 = g("Time") - delay
    cyc = life + g("Spawn Window")
    age = V.where(g("Loop") > 0.5, V.modx(age0, cyc), age0)
    u = age / life
    vis = (age >= 0.0) * (u <= 1.0)
    uc = V.clamp01(u)
    # emission
    p0 = V.comb(rnd(1, -1, 1), rnd(2, -1, 1), rnd(3, -1, 1) * g("Emit Squash Z"))
    p0 = V.vscale(p0, g("Emit Radius"))
    dirn = V.vnorm(gi.outputs["Direction"])
    rv = V.comb(rnd(4, -1, 1), rnd(5, -1, 1), rnd(6, -1, 1))
    v0 = V.vscale(dirn + V.vscale(rv, g("Spread")), g("Speed") * (1.0 + (rnd(7) - 0.5) * 2.0 * g("Speed Var")))
    k = V.maxx(g("Drag"), 0.001)
    agec = V.maxx(age, 0.0)
    disp = V.vscale(v0, (1.0 - V.expx(-k * agec)) / k)
    grav = V.vscale(gv("Gravity"), 0.5 * agec * agec)
    ph = V.comb(rnd(8, 0, 6.28) , rnd(9, 0, 6.28), rnd(10, 0, 6.28))
    wob = V.vsin(ph + V.comb(agec * 5.0, agec * 4.3, agec * 5.7))
    turb = V.vscale(wob, g("Turbulence") * V.mathop("SQRT", agec) * 0.3)
    pr = p0 + disp + grav + turb
    zraw = pr.z
    landed = zraw < g("Floor Z")
    pz = V.maxx(zraw, g("Floor Z"))
    px, py, _ = V.sepxyz(pr)
    pos = V.comb(px, py, pz)
    # scale
    grow = V.lerp(g("Scale Start"), g("Scale End"), 1.0 - (1.0 - uc) * (1.0 - uc))
    sv = 1.0 + (rnd(11) - 0.5) * 2.0 * g("Scale Var")
    pu = V.clamp01(u / g("Pop"))
    pop = V.smoothstep(0.0, 1.0, pu) * (1.0 + g("Overshoot") * V.sin(pu * PI))
    fade = 1.0 - V.smoothstep(1.0 - g("Fade") - 1e-4, 1.0, uc)
    s = grow * sv * pop * fade * vis
    sz = s * V.where(landed, g("Squash"), 1.0)
    sxy = s * V.where(landed, 1.0 + (1.0 - g("Squash")) * 0.6, 1.0)
    # velocity-aligned stretch
    vel = V.vscale(v0, V.expx(-k * agec)) + V.vscale(gv("Gravity"), agec)
    speed_now = V.vlen(vel)
    if align_vel:
        sz = sz * (1.0 + g("Stretch") * speed_now)
        scale = V.comb(sxy, sxy, sz)
        al = V.N("FunctionNodeAlignRotationToVector")
        al.axis = "Z"
        V.L(vel, al.inputs["Vector"])
        rot = al.outputs["Rotation"]
    else:
        scale = V.comb(sxy, sxy, sz)
        e0 = V.comb(rnd(12, 0, 6.28), rnd(13, 0, 6.28), rnd(14, 0, 6.28))
        if tumble:
            e = e0 + V.vscale(V.comb(rnd(15, -1, 1), rnd(16, -1, 1), rnd(17, -1, 1)), g("Spin") * agec)
        else:
            e = e0 + V.vscale(V.comb(0.0, 0.0, rnd(15, -1, 1)), g("Spin") * agec)
        er = V.N("FunctionNodeEulerToRotation")
        V.L(e, er.inputs["Euler"])
        rot = er.outputs["Rotation"]
    # points
    pts = V.N("GeometryNodePoints")
    V.L(gi.outputs["Count"], pts.inputs["Count"])
    sp = V.N("GeometryNodeSetPosition")
    tree.links.new(pts.outputs[0], sp.inputs["Geometry"])
    V.L(pos, sp.inputs["Position"])
    last = sp.outputs[0]
    for an, ex in (("fx_fade", uc * 0.0 + (1.0 - V.smoothstep(0.45, 1.0, uc)) * vis), ("fx_tint", rnd(18)), ("fx_int", g("Intensity")),
                   ("fx_age", uc)):
        st = V.N("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
        st.inputs["Name"].default_value = an
        tree.links.new(last, st.inputs["Geometry"])
        V.L(ex, st.inputs["Value"])
        last = st.outputs[0]
    # instance geometry
    if shape == "sphere":
        sh = V.N("GeometryNodeMeshUVSphere", in_Segments=12, in_Rings=8, in_Radius=1.0)
        shg = sh.outputs["Mesh"]
        sm = V.N("GeometryNodeSetShadeSmooth")
        tree.links.new(shg, sm.inputs["Geometry"])
        shg = sm.outputs[0]
    elif shape == "quad":
        sh = V.N("GeometryNodeMeshGrid", in_Vertices__X=2, in_Vertices__Y=2, in_Size__X=1.0, in_Size__Y=0.6)
        shg = sh.outputs["Mesh"]
    elif shape == "cone":
        sh = V.N("GeometryNodeMeshCone", in_Vertices=6, in_Radius__Top=0.0, in_Radius__Bottom=0.35, in_Depth=1.0)
        shg = sh.outputs["Mesh"]
    else:
        oi = V.N("GeometryNodeObjectInfo", transform_space="RELATIVE")
        V.L(gi.outputs["Instance Object"], oi.inputs["Object"])
        oi.inputs["As Instance"].default_value = True
        shg = oi.outputs["Geometry"]
    ip = V.N("GeometryNodeInstanceOnPoints")
    tree.links.new(last, ip.inputs["Points"])
    tree.links.new(shg, ip.inputs["Instance"])
    tree.links.new(rot, ip.inputs["Rotation"])
    V.L(scale, ip.inputs["Scale"])
    ri = V.N("GeometryNodeRealizeInstances")
    tree.links.new(ip.outputs[0], ri.inputs[0])
    sm2 = V.N("GeometryNodeSetMaterial")
    tree.links.new(ri.outputs[0], sm2.inputs["Geometry"])
    tree.links.new(gi.outputs["Material"], sm2.inputs["Material"])
    tree.links.new(sm2.outputs[0], go.inputs["Geometry"])
    V.auto_layout(tree)
    return tree


def _in(node, name):
    for s in node.inputs:
        if s.name == name and s.enabled:
            return s
    raise KeyError(name)


def _out(node, name):
    for s in node.outputs:
        if s.name == name and s.enabled:
            return s
    raise KeyError(name)


# ----------------------------------------------------------------------------- ring (smoke ring / shockwave)
def ring_group(name, defaults=None):
    D = dict(Life=0.9, Radius_Start=0.04, Radius_End=0.32, Thickness_Start=0.045, Thickness_End=0.012, Travel=1.4, Drag=3.0, Time=0.0)
    D.update(defaults or {})
    ins = [("Geometry", "Geometry", None), ("Time", "Float", 0.0, -1e5, 1e5, "Seconds since trigger"),
           ("Life", "Float", D["Life"], 0.01, 100.0, "Lifetime"),
           ("Radius Start", "Float", D["Radius_Start"], 0.0, 100.0, "Ring radius at birth"),
           ("Radius End", "Float", D["Radius_End"], 0.0, 100.0, "Ring radius at death"),
           ("Thickness Start", "Float", D["Thickness_Start"], 0.0, 10.0, "Tube radius at birth"),
           ("Thickness End", "Float", D["Thickness_End"], 0.0, 10.0, "Tube radius at death (thins out)"),
           ("Travel", "Float", D["Travel"], 0.0, 100.0, "Initial speed along -Y (m/s); 0 for a stationary shockwave"),
           ("Drag", "Float", D["Drag"], 0.0, 100.0, "Speed decay rate"),
           ("Intensity", "Float", 1.0, 0.0, 1000.0, "Emission multiplier (attribute fx_int)"),
           ("Material", "Material", None, None, None, "Ring material")]
    tree, gi, go = V.new_group(name, "Geometry", ins, [("Geometry", "Geometry")])
    g = lambda n: V.X(gi.outputs[n])
    u = V.clamp01(g("Time") / g("Life"))
    vis = (g("Time") >= 0.0) * (g("Time") <= g("Life"))
    e = 1.0 - (1.0 - u) * (1.0 - u)
    R = V.lerp(g("Radius Start"), g("Radius End"), e) * vis
    r = V.lerp(g("Thickness Start"), g("Thickness End"), u) * vis
    k = V.maxx(g("Drag"), 0.001)
    ty = -g("Travel") * (1.0 - V.expx(-k * V.maxx(g("Time"), 0.0))) / k
    circ = V.N("GeometryNodeCurvePrimitiveCircle", mode="RADIUS", in_Resolution=32)
    V.L(R, circ.inputs["Radius"])
    prof = V.N("GeometryNodeCurvePrimitiveCircle", mode="RADIUS", in_Resolution=8)
    V.L(r, prof.inputs["Radius"])
    c2m = V.N("GeometryNodeCurveToMesh")
    tree.links.new(circ.outputs["Curve"], c2m.inputs["Curve"])
    tree.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    tf = V.N("GeometryNodeTransform")
    tree.links.new(c2m.outputs[0], tf.inputs["Geometry"])
    tf.inputs["Rotation"].default_value = (PI / 2, 0.0, 0.0)
    V.L(V.comb(0.0, ty, 0.0), tf.inputs["Translation"])
    st = V.N("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st.inputs["Name"].default_value = "fx_fade"
    tree.links.new(tf.outputs[0], st.inputs["Geometry"])
    V.L(1.0 - V.smoothstep(0.5, 1.0, u), st.inputs["Value"])
    st2 = V.N("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st2.inputs["Name"].default_value = "fx_int"
    tree.links.new(st.outputs[0], st2.inputs["Geometry"])
    V.L(g("Intensity"), st2.inputs["Value"])
    sm = V.N("GeometryNodeSetShadeSmooth")
    tree.links.new(st2.outputs[0], sm.inputs["Geometry"])
    sm2 = V.N("GeometryNodeSetMaterial")
    tree.links.new(sm.outputs[0], sm2.inputs["Geometry"])
    tree.links.new(gi.outputs["Material"], sm2.inputs["Material"])
    tree.links.new(sm2.outputs[0], go.inputs["Geometry"])
    V.auto_layout(tree)
    return tree


# ----------------------------------------------------------------------------- streak tunnel (motion streaks)
def streak_group(name):
    ins = [("Geometry", "Geometry", None), ("Time", "Float", 0.0, -1e5, 1e5, "Seconds; streaks loop"),
           ("Seed", "Int", 3, 0, 100000, ""), ("Count", "Int", 70, 1, 1000, "Streak count"),
           ("Radius Min", "Float", 0.25, 0.0, 100.0, "Inner radius of the streak tunnel around the -Z axis"),
           ("Radius Max", "Float", 1.6, 0.0, 100.0, "Outer radius"),
           ("Depth", "Float", 4.0, 0.1, 1000.0, "Length of the tunnel in front of the parent (local -Z)"),
           ("Start", "Float", 0.4, 0.0, 1000.0, "Distance from the parent where streaks begin"),
           ("Speed", "Float", 6.0, 0.0, 1000.0, "Streak speed in m/s toward the parent (higher = faster speed ramp)"),
           ("Length", "Float", 0.9, 0.0, 100.0, "Streak length"), ("Width", "Float", 0.012, 0.0, 10.0, "Streak width"),
           ("Intensity", "Float", 1.0, 0.0, 1000.0, "Emission multiplier"), ("Material", "Material", None, None, None, "")]
    tree, gi, go = V.new_group(name, "Geometry", ins, [("Geometry", "Geometry")])
    g = lambda n: V.X(gi.outputs[n])
    idx = V.X(V.N("GeometryNodeInputIndex").outputs[0])
    seed = V.X(gi.outputs["Seed"])

    def rnd(k, lo=0.0, hi=1.0):
        n = V.N("FunctionNodeRandomValue", data_type="FLOAT")
        V.setin(n, "Min", lo)
        V.setin(n, "Max", hi)
        V.L(idx, _in(n, "ID"))
        V.L(seed + float(k * 13), _in(n, "Seed"))
        return V.X(_out(n, "Value"))
    th = rnd(0, 0, 2 * PI)
    rr = V.sqrt(rnd(1)) * (g("Radius Max") - g("Radius Min")) + g("Radius Min")
    ph = V.fract(rnd(2) + g("Time") * g("Speed") / g("Depth"))
    z = -(g("Start") + (1.0 - ph) * g("Depth"))
    pos = V.comb(rr * V.cos(th), rr * V.sin(th), z)
    pts = V.N("GeometryNodePoints")
    V.L(gi.outputs["Count"], pts.inputs["Count"])
    sp = V.N("GeometryNodeSetPosition")
    tree.links.new(pts.outputs[0], sp.inputs["Geometry"])
    V.L(pos, sp.inputs["Position"])
    st = V.N("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st.inputs["Name"].default_value = "fx_fade"
    tree.links.new(sp.outputs[0], st.inputs["Geometry"])
    V.L(V.smoothstep(0.0, 0.15, ph) * (1.0 - V.smoothstep(0.7, 1.0, ph)), st.inputs["Value"])
    st2 = V.N("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st2.inputs["Name"].default_value = "fx_int"
    tree.links.new(st.outputs[0], st2.inputs["Geometry"])
    V.L(g("Intensity") * (0.5 + rnd(3)), st2.inputs["Value"])
    cone = V.N("GeometryNodeMeshCone", in_Vertices=4, in_Radius__Top=0.0, in_Radius__Bottom=0.5, in_Depth=1.0)
    ip = V.N("GeometryNodeInstanceOnPoints")
    tree.links.new(st2.outputs[0], ip.inputs["Points"])
    tree.links.new(cone.outputs["Mesh"], ip.inputs["Instance"])
    V.L(V.comb(g("Width"), g("Width"), g("Length") * (0.5 + rnd(4))), ip.inputs["Scale"])
    ri = V.N("GeometryNodeRealizeInstances")
    tree.links.new(ip.outputs[0], ri.inputs[0])
    sm = V.N("GeometryNodeSetMaterial")
    tree.links.new(ri.outputs[0], sm.inputs["Geometry"])
    tree.links.new(gi.outputs["Material"], sm.inputs["Material"])
    tree.links.new(sm.outputs[0], go.inputs["Geometry"])
    V.auto_layout(tree)
    return tree


# ----------------------------------------------------------------------------- materials
def fx_attr(t, name):
    a = t.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "GEOMETRY"
    a.attribute_name = name
    return V.X(a.outputs["Fac"])


def fx_material(name, color, emit=0.0, rough=0.9, alpha_mode="fade", alpha=1.0, color2=None, tint_mix=0.0, hue_tint=False,
                gradient=None, spec=0.2, facing_dark=0.0):
    """Generic FX material. alpha_mode: fade (alpha from fx_fade) | none. color2 + tint_mix: random per particle via fx_tint.
    hue_tint: base colour = HSV(fx_tint). gradient: ('Z'|'Y', ...) alpha and emission fade along object axis."""
    m = V.new_material(name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    pb = t.nodes.new("ShaderNodeBsdfPrincipled")
    out = t.nodes.new("ShaderNodeOutputMaterial")
    tint = fx_attr(t, "fx_tint")
    fade = fx_attr(t, "fx_fade")
    inten = fx_attr(t, "fx_int")
    # fx_int missing on plain meshes -> attribute returns 0; use max(inten, base)
    inten = V.maxx(inten, 0.0)
    col = t.nodes.new("ShaderNodeRGB")
    col.outputs[0].default_value = color
    base = col.outputs[0]
    if color2 is not None:
        c2 = t.nodes.new("ShaderNodeRGB")
        c2.outputs[0].default_value = color2
        mx = t.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        V.L(tint * tint_mix, mx.inputs["Factor"])
        t.links.new(base, mx.inputs[6])
        t.links.new(c2.outputs[0], mx.inputs[7])
        base = mx.outputs[2]
    if hue_tint:
        hsv = t.nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Saturation"].default_value = 1.0
        hsv.inputs["Value"].default_value = 1.0
        V.L(tint, hsv.inputs["Hue"])
        hsv.inputs["Color"].default_value = (1.0, 0.15, 0.1, 1.0)
        base = hsv.outputs[0]
    if facing_dark:
        geo = t.nodes.new("ShaderNodeNewGeometry")
        fac = V.absx(V.vdot(V.wrapv(geo.outputs["Normal"]), V.wrapv(geo.outputs["Incoming"])))
        mm = t.nodes.new("ShaderNodeMix")
        mm.data_type = "RGBA"
        mm.blend_type = "MULTIPLY"
        V.L(1.0 - V.lerp(1.0 - facing_dark, 1.0, V.smoothstep(0.1, 0.8, fac)), mm.inputs["Factor"])
        t.links.new(base, mm.inputs[6])
        mm.inputs[7].default_value = (0.55, 0.58, 0.68, 1.0)
        base = mm.outputs[2]
    t.links.new(base, pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = rough
    pb.inputs["Specular IOR Level"].default_value = spec
    if emit:
        t.links.new(base, pb.inputs["Emission Color"])
        V.L(inten * emit, pb.inputs["Emission Strength"])
    if alpha_mode == "fade":
        V.L(fade * alpha, pb.inputs["Alpha"])
        V.set_render_method(m, "DITHERED")
    if gradient:
        ax, p0, p1 = gradient
        tc = t.nodes.new("ShaderNodeTexCoord")
        co = V.wrapv(tc.outputs["Object"])
        a_ = co.y if ax == "Y" else co.z
        gfac = V.smoothstep(p0, p1, a_)
        V.L(gfac * alpha, pb.inputs["Alpha"])
        V.L(base, pb.inputs["Emission Color"]) if emit else None
        V.L(gfac * emit, pb.inputs["Emission Strength"]) if emit else None
        V.set_render_method(m, "BLENDED")
    t.links.new(pb.outputs["BSDF"], out.inputs["Surface"])
    V.auto_layout(t)
    m.diffuse_color = color
    return m


def flat_emission_material(name, color_inner, color_outer, strength=6.0, radial=0.5):
    """Mesh emission with a radial gradient in object XZ (white-hot core to coloured edge) for flash/star meshes."""
    m = V.new_material(name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    tc = t.nodes.new("ShaderNodeTexCoord")
    co = V.wrapv(tc.outputs["Object"])
    x, y, z = V.sepxyz(co)
    rr = V.sqrt(x * x + y * y + z * z) / radial
    ramp = t.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*color_inner, 1)
    ramp.color_ramp.elements[1].color = (*color_outer, 1)
    ramp.color_ramp.elements[0].position = 0.15
    V.L(rr, ramp.inputs["Fac"])
    em = t.nodes.new("ShaderNodeEmission")
    t.links.new(ramp.outputs[0], em.inputs["Color"])
    em.inputs["Strength"].default_value = strength
    out = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(em.outputs[0], out.inputs[0])
    V.auto_layout(t)
    m.diffuse_color = (*color_outer, 1)
    return m


def solid_emission(name, color, strength=3.0):
    m = V.new_material(name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    em = t.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color[:3], 1)
    em.inputs["Strength"].default_value = strength
    out = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(em.outputs[0], out.inputs[0])
    m.diffuse_color = (*color[:3], 1)
    return m


# ----------------------------------------------------------------------------- meshes
def star_mesh(name, points=5, r_out=1.0, r_in=0.45, thick=0.1, jitter=0.0, seed=0, plane="XZ"):
    import random
    rng = random.Random(seed)
    bm = bmesh.new()
    n = points * 2
    ring_top, ring_bot = [], []
    for i in range(n):
        a = 2 * PI * i / n + PI / 2
        r = (r_out if i % 2 == 0 else r_in) * (1 + jitter * rng.uniform(-1, 1))
        px, pz = r * math.cos(a), r * math.sin(a)
        if plane == "XZ":
            ring_top.append(bm.verts.new((px, -thick / 2, pz)))
            ring_bot.append(bm.verts.new((px, thick / 2, pz)))
        elif plane == "YZ":
            ring_top.append(bm.verts.new((-thick / 2, px, pz)))
            ring_bot.append(bm.verts.new((thick / 2, px, pz)))
        else:
            ring_top.append(bm.verts.new((px, pz, thick / 2)))
            ring_bot.append(bm.verts.new((px, pz, -thick / 2)))
    ct = bm.verts.new((0, -thick / 2, 0) if plane == "XZ" else ((-thick / 2, 0, 0) if plane == "YZ" else (0, 0, thick / 2)))
    cb = bm.verts.new((0, thick / 2, 0) if plane == "XZ" else ((thick / 2, 0, 0) if plane == "YZ" else (0, 0, -thick / 2)))
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((ct, ring_top[j], ring_top[i]))
        bm.faces.new((cb, ring_bot[i], ring_bot[j]))
        bm.faces.new((ring_top[i], ring_top[j], ring_bot[j], ring_bot[i]))
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False
    return me


def obj(name, mesh, coll, parent=None, mat=None, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    o = bpy.data.objects.new(name, mesh)
    coll.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    o.scale = scale
    if parent is not None:
        o.parent = parent
    if mat is not None and mesh is not None:
        mesh.materials.append(mat)
    return o


def gn_object(name, group, coll, parent, mats=None, values=None, loc=(0, 0, 0)):
    me = bpy.data.meshes.new(name + "_mesh")
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    mod = o.modifiers.new("GeometryNodes", "NODES")
    mod.node_group = group
    for k, v in (values or {}).items():
        ident = V.mod_socket_id(mod, k)
        if isinstance(v, (bpy.types.Material, bpy.types.Object)):
            mod[ident] = v
        else:
            try:
                mod[ident] = v
            except TypeError:
                mod[ident] = tuple(v)
    return o, mod


def make_root(coll_name, props=None):
    coll, root = C.new_asset(coll_name, accuracy="C")
    for k, v in {"p_t": 0.0, "p_auto": 0, "p_start": 1, "p_fps": 30.0, "p_seed": 1, "p_intensity": 1.0, **(props or {})}.items():
        root[k] = v
    return coll, root


def drive_time(obj_, mod, root, name="Time"):
    V.drive_mod(obj_, mod, name, root, {"t": "p_t", "a": "p_auto", "s": "p_start", "f": "p_fps"},
                "(frame - s) / f if a > 0.5 else t")


def drive_prop_mod(obj_, mod, name, root, prop):
    V.drive_mod(obj_, mod, name, root, {"v": prop}, "v")
