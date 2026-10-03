"""Build vfx_particles_and_effects: reusable effect assets (collections ASSET_fx_<name>) in one blend, with contact sheets.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_vfx_effects.py -- assets/components/materials_vfx [only_effect ...]
Effects are driven by custom properties on each root empty: p_t (seconds since trigger), p_auto (1 = time from the scene frame), p_start (frame of the
trigger when p_auto = 1), p_fps, p_seed, p_intensity, plus effect-specific props documented in the JSON.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bpy
import common as C
import vfx_lib as V
import fx_lib as F

PI = math.pi
args = C.argv_after_dashes()
OUT = os.path.abspath(args[0] if args else "assets/components/materials_vfx")
ONLY = set(args[1:])
PREV = os.path.join(OUT, "previews")
SCR = os.environ.get("VFX_SCRATCH", "/tmp")
os.makedirs(PREV, exist_ok=True)
BLEND = os.path.join(OUT, "vfx_particles_and_effects.blend") if not ONLY else os.path.join(SCR, "vfx_partial.blend")

EFFECTS = {}  # name -> dict(meta)


def want(n):
    return not ONLY or n in ONLY


def env_drivers(o, root, dur="p_dur", size="p_size", up=0.15, down_start=0.5, extra=""):
    """Scale an object by a pop-in / fade-out envelope of p_t over p_dur."""
    for i in range(3):
        fc = o.driver_add("scale", i)
        d = fc.driver
        d.type = "SCRIPTED"
        for vn, pn in (("t", "p_t"), ("d", dur), ("z", size), ("a", "p_auto"), ("s", "p_start"), ("f", "p_fps")):
            v = d.variables.new()
            v.name = vn
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["%s"]' % pn
        tt = "((frame - s) / f if a > 0.5 else t)"
        d.expression = ("z * (%s >= 0) * (%s < d) * min(%s / (%s * d), 1) * (1 - max((%s / d - %s) / %s, 0) ** 2)%s"
                        % (tt, tt, tt, up, tt, down_start, 1 - down_start, extra))


def drive_rot_z(o, root, rate):
    fc = o.driver_add("rotation_euler", 2)
    d = fc.driver
    d.type = "SCRIPTED"
    for vn, pn in (("t", "p_t"), ("a", "p_auto"), ("s", "p_start"), ("f", "p_fps")):
        v = d.variables.new()
        v.name = vn
        v.type = "SINGLE_PROP"
        v.targets[0].id = root
        v.targets[0].data_path = '["%s"]' % pn
    d.expression = "((frame - s) / f if a > 0.5 else t) * %s" % rate


def register(name, coll, root, desc, props, hooks, params_note, preview, groups=(), mats=()):
    EFFECTS[name] = dict(collection=coll.name, root=root.name, description=desc, custom_properties=props, hooks=hooks,
                         usage=params_note, preview=preview, node_groups=list(groups), materials=list(mats))


def particle_effect(name, desc, mats, group_kwargs, defaults, preview, shape="sphere", extra_props=None, hooks=None, note=""):
    coll, root = F.make_root("fx_" + name, extra_props)
    grp = F.particle_group("NG_fx_" + name, shape=shape, defaults=defaults, **group_kwargs)
    o, mod = F.gn_object("fx_%s_emitter" % name, grp, coll, root, values={"Material": mats})
    F.drive_time(o, mod, root)
    F.drive_prop_mod(o, mod, "Seed", root, "p_seed")
    F.drive_prop_mod(o, mod, "Intensity", root, "p_intensity")
    for h in (hooks or ["emit"]):
        C.hook(h, coll, root)
    props = {"p_t": "seconds since trigger (keyframe 0 -> Life + Spawn Window) or set p_auto = 1", "p_auto": "1 = time from scene frame",
             "p_start": "start frame when p_auto = 1", "p_fps": "frames per second (30)", "p_seed": "random seed (changes the look)",
             "p_intensity": "emission multiplier"}
    props.update({k: "effect specific" for k in (extra_props or {})})
    register(name, coll, root, desc, props, ["HOOK_" + h for h in (hooks or ["emit"])],
             "Emitter object %s carries the Geometry Nodes modifier 'GeometryNodes' (group %s); tune its inputs with vfx_lib.set_param(obj, name, value). %s"
             % (o.name, grp.name, note), preview, groups=[grp.name], mats=[mats.name])
    return coll, root, o, mod


# ----------------------------------------------------------------------------- materials
def build_materials():
    M = {}
    M["smoke"] = F.fx_material("MAT_vfx_fx_smoke", (0.92, 0.92, 0.94, 1), emit=0.25, rough=1.0, color2=(0.38, 0.40, 0.50, 1), tint_mix=0.6,
                               facing_dark=0.5)
    M["dust"] = F.fx_material("MAT_vfx_fx_dust", V.srgb("#D8C7A4"), emit=0.15, rough=1.0, color2=V.srgb("#9A8A6C"), tint_mix=0.6, facing_dark=0.3)
    M["steam"] = F.fx_material("MAT_vfx_fx_steam", (1, 1, 1, 1), emit=0.18, rough=1.0, alpha=0.9, color2=(0.62, 0.66, 0.78, 1), tint_mix=0.4, facing_dark=0.45)
    M["spark"] = F.fx_material("MAT_vfx_fx_spark", V.srgb("#FFB020"), emit=3.0, rough=0.5, alpha_mode="none", color2=V.srgb("#FFF0A0"), tint_mix=0.9)
    M["confetti"] = F.fx_material("MAT_vfx_fx_confetti", (1, 0, 0, 1), emit=0.0, rough=0.45, alpha_mode="none", hue_tint=True, spec=0.5)
    M["egg"] = F.fx_material("MAT_vfx_fx_egg", V.srgb("#F3F0E4"), emit=0.1, rough=0.25, alpha_mode="none", color2=V.srgb("#F2A60C"), tint_mix=0.0, spec=0.6)
    # egg: yolk chunks by tint threshold: handled by a second mix using tint
    # (kept simple: colour2 mix factor equals tint^3 via node edit below)
    M["star"] = F.fx_material("MAT_vfx_fx_star", V.srgb("#FFD21E"), emit=1.4, rough=0.5, alpha_mode="none", color2=V.srgb("#FFF5B0"), tint_mix=0.8)
    M["line"] = F.solid_emission("MAT_vfx_fx_line", (1.0, 0.95, 0.7), 2.0)
    M["streak"] = F.fx_material("MAT_vfx_fx_streak", (1, 1, 1, 1), emit=1.2, rough=0.5)
    M["flash_star"] = F.flat_emission_material("MAT_vfx_fx_flash_star", (1.0, 0.95, 0.7), (1.0, 0.3, 0.02), strength=1.3, radial=0.34)
    M["flash_glow"] = F.solid_emission("MAT_vfx_fx_flash_glow", (1.0, 0.7, 0.25), 1.6)
    M["shock"] = F.fx_material("MAT_vfx_fx_shockwave", (1, 1, 0.9, 1), emit=1.3, rough=0.5)
    M["bang_outer"] = F.solid_emission("MAT_vfx_fx_bang_outer", (1.0, 0.18, 0.05), 1.1)
    M["bang_inner"] = F.solid_emission("MAT_vfx_fx_bang_inner", (1.0, 0.85, 0.1), 1.2)
    M["bang_text"] = F.solid_emission("MAT_vfx_fx_bang_text", (1.0, 1.0, 1.0), 1.1)
    M["bang_text_edge"] = F.solid_emission("MAT_vfx_fx_bang_text_edge", (0.05, 0.02, 0.02), 0.5)
    for nm, col in (("cyan", (0.1, 0.85, 1.0)), ("orange", (1.0, 0.45, 0.08)), ("white", (1.0, 1.0, 1.0))):
        M["trail_" + nm] = F.solid_emission("MAT_vfx_fx_trail_" + nm, col, 5.0)
    # shimmer
    return M


def fix_egg_material(m):
    """yolk chunk = tint > 0.75: set the colour mix factor to smoothstep(0.7, 0.8, tint)."""
    t = m.node_tree
    mx = next(n for n in t.nodes if n.bl_idname == "ShaderNodeMix")
    for l in list(t.links):
        if l.to_node == mx and l.to_socket.name == "Factor":
            t.links.remove(l)
    V.use(t)
    tn = next(n for n in t.nodes if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == "fx_tint")
    V.L(V.smoothstep(0.7, 0.78, V.X(tn.outputs["Fac"])), mx.inputs["Factor"])


def trail_material_fix(m, col):
    """Trail: ribbon along object +Y (tail at +Y). Alpha / emission fade from head (y=0) to tail (y=1)."""
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    tc = t.nodes.new("ShaderNodeTexCoord")
    co = V.wrapv(tc.outputs["Object"])
    y = co.y
    fade = 1.0 - V.smoothstep(0.0, 1.0, y)
    fade = fade * fade
    em = t.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    V.L(fade * 6.0 + 0.2, em.inputs["Strength"])
    tr = t.nodes.new("ShaderNodeBsdfTransparent")
    mx = t.nodes.new("ShaderNodeMixShader")
    V.L(fade, mx.inputs["Fac"])
    t.links.new(tr.outputs[0], mx.inputs[1])
    t.links.new(em.outputs[0], mx.inputs[2])
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(mx.outputs[0], o.inputs[0])
    V.set_render_method(m, "BLENDED")
    V.auto_layout(t)


# ----------------------------------------------------------------------------- the effects
def build_all():
    M = build_materials()
    fix_egg_material(M["egg"])
    for nm, col in (("cyan", (0.1, 0.85, 1.0)), ("orange", (1.0, 0.45, 0.08)), ("white", (1.0, 1.0, 1.0))):
        trail_material_fix(M["trail_" + nm], col)

    # ---- smoke puff
    if want("smoke_puff"):
        particle_effect("smoke_puff", "Cartoon smoke puffs rising and expanding (white-gray clay-like spheres).", M["smoke"], {},
                        dict(Count=16, Life=2.4, Spawn_Window=1.0, Emit_Radius=0.07, Emit_Squash_Z=0.3, Direction=(0, 0, 1), Spread=0.35, Speed=0.5,
                             Speed_Var=0.4, Drag=1.1, Gravity=(0, 0, 0.18), Turbulence=0.25, Scale_Start=0.06, Scale_End=0.34, Scale_Var=0.35,
                             Pop=0.15, Fade=0.4, Overshoot=0.25),
                        dict(times=[0.2, 0.7, 1.2, 1.7, 2.2, 2.9], cam=((0, -3.2, 0.9), (0, 0, 0.8), 50), bg="sky"), hooks=["emit"])
    if want("steam"):
        particle_effect("steam", "Small white steam puffs (kettle / hot lid).", M["steam"], {},
                        dict(Count=22, Life=1.3, Spawn_Window=1.0, Emit_Radius=0.03, Emit_Squash_Z=0.2, Direction=(0, 0, 1), Spread=0.3, Speed=0.5,
                             Drag=0.8, Gravity=(0, 0, 0.35), Turbulence=0.2, Scale_Start=0.015, Scale_End=0.075, Scale_Var=0.3, Pop=0.2, Fade=0.5),
                        dict(times=[0.2, 0.6, 1.0, 1.4, 1.9, 2.4], cam=((0, -1.4, 0.35), (0, 0, 0.3), 50), bg="sky"), hooks=["emit"])
    if want("ear_steam"):
        coll, root = F.make_root("fx_ear_steam", {"p_ear_spacing": 0.46})
        for side, sx in (("L", -1), ("R", 1)):
            h = C.hook("ear_" + side, coll, root, loc=(sx * 0.23, 0, 0))
            grp = F.particle_group("NG_fx_ear_steam_" + side, shape="sphere", defaults=dict(
                Count=26, Life=0.9, Spawn_Window=0.9, Emit_Radius=0.025, Emit_Squash_Z=0.5, Direction=(sx * 1.0, 0, 0.55), Spread=0.25, Speed=1.4, Drag=1.4,
                Gravity=(0, 0, 0.8), Turbulence=0.2, Scale_Start=0.03, Scale_End=0.13, Scale_Var=0.3, Pop=0.2, Fade=0.45, Loop=1))
            o, mod = F.gn_object("fx_ear_steam_%s_emitter" % side, grp, coll, h, values={"Material": M["steam"]})
            F.drive_time(o, mod, root)
            F.drive_prop_mod(o, mod, "Seed", root, "p_seed")
            F.drive_prop_mod(o, mod, "Intensity", root, "p_intensity")
        register("ear_steam", coll, root, "Cartoon steam jets from both ears (Manager anger), looping, two emitters on HOOK_ear_L / HOOK_ear_R.",
                 {"p_t": "seconds; loops", "p_auto": "1 = scene time", "p_ear_spacing": "reference only: 0.46 m = Manager ear spacing in the crude film (ears at x = +/-0.23); move the hooks"},
                 ["HOOK_ear_L", "HOOK_ear_R"], "Parent the hooks to the Manager's head; set 'Loop' 0 on the modifiers for a one-shot blast.",
                 dict(times=[0.2, 0.5, 0.8, 1.1, 1.5, 2.0], cam=((0, -3.6, 0.2), (0, 0, 0.3), 50), bg="sky"), groups=["NG_fx_ear_steam_L", "NG_fx_ear_steam_R"],
                 mats=[M["steam"].name])
    if want("sparks_burst"):
        particle_effect("sparks_burst", "Sparks burst: stretched emissive streaks flying out under gravity.", M["spark"], dict(align_vel=True),
                        dict(Count=45, Life=0.9, Spawn_Window=0.06, Emit_Radius=0.01, Direction=(0, 0, 0.6), Spread=1.5, Speed=1.6, Speed_Var=0.5,
                             Drag=0.8, Gravity=(0, 0, -5.0), Scale_Start=0.03, Scale_End=0.012, Scale_Var=0.4, Pop=0.05, Fade=0.4, Stretch=0.1),
                        dict(times=[0.05, 0.15, 0.3, 0.5, 0.7, 0.95], cam=((0, -2.2, 0.2), (0, 0, 0.0), 50), bg="dark"), shape="cone")
    if want("dust_cloud"):
        coll, root, o, mod = particle_effect(
            "dust_cloud", "Comic fight dust cloud: a rolling clump of puffs with tumbling stars.", M["dust"], {},
            dict(Count=30, Life=1.6, Spawn_Window=0.15, Emit_Radius=0.2, Emit_Squash_Z=0.7, Direction=(0, 0, 0.3), Spread=1.6, Speed=0.35, Drag=2.0,
                 Gravity=(0, 0, 0.1), Turbulence=0.5, Scale_Start=0.09, Scale_End=0.2, Scale_Var=0.4, Pop=0.25, Fade=0.4, Overshoot=0.3, Loop=0),
            dict(times=[0.1, 0.35, 0.7, 1.0, 1.4, 1.8], cam=((0, -3.4, 0.4), (0, 0, 0.35), 50), bg="sky"))
        star = F.obj("fx_dust_cloud_star_src", F.star_mesh("fx_dust_cloud_star_src", 5, 1.0, 0.45, 0.2), coll, root, M["star"])
        star.hide_render = True
        star.hide_viewport = True
        grp = F.particle_group("NG_fx_dust_stars", shape="object", defaults=dict(
            Count=7, Life=1.2, Spawn_Window=0.2, Emit_Radius=0.12, Direction=(0, 0, 1), Spread=1.5, Speed=1.0, Drag=1.2, Gravity=(0, 0, -0.5),
            Scale_Start=0.05, Scale_End=0.07, Pop=0.15, Fade=0.3, Overshoot=0.6, Spin=5.0), tumble=True)
        o2, mod2 = F.gn_object("fx_dust_cloud_stars", grp, coll, root, values={"Material": M["star"], "Instance Object": star})
        F.drive_time(o2, mod2, root)
        F.drive_prop_mod(o2, mod2, "Seed", root, "p_seed")
        EFFECTS["dust_cloud"]["node_groups"].append(grp.name)
    if want("impact_stars"):
        coll, root = F.make_root("fx_impact_stars", {"p_dur": 0.5, "p_size": 1.0})
        star = F.obj("fx_impact_stars_star_src", F.star_mesh("fx_impact_stars_star_src", 5, 1.0, 0.45, 0.12), coll, root, M["star"])
        star.hide_render = True
        star.hide_viewport = True
        g1 = F.particle_group("NG_fx_impact_stars", shape="object", defaults=dict(
            Count=10, Life=0.9, Spawn_Window=0.0, Emit_Radius=0.02, Direction=(0, 0, 0.2), Spread=2.2, Speed=1.6, Drag=2.2, Gravity=(0, 0, -0.6),
            Scale_Start=0.05, Scale_End=0.09, Pop=0.12, Fade=0.35, Overshoot=0.7, Spin=7.0), tumble=True)
        o1, m1 = F.gn_object("fx_impact_stars_stars", g1, coll, root, values={"Material": M["star"], "Instance Object": star})
        g2 = F.particle_group("NG_fx_impact_lines", shape="cone", defaults=dict(
            Count=16, Life=0.28, Spawn_Window=0.0, Emit_Radius=0.12, Direction=(0, 0, 0.2), Spread=2.5, Speed=2.4, Drag=1.5, Scale_Start=0.012,
            Scale_End=0.004, Pop=0.1, Fade=0.5, Stretch=0.22), align_vel=True)
        o2, m2 = F.gn_object("fx_impact_stars_lines", g2, coll, root, values={"Material": M["line"]})
        for oo, mm in ((o1, m1), (o2, m2)):
            F.drive_time(oo, mm, root)
            F.drive_prop_mod(oo, mm, "Seed", root, "p_seed")
        burst = F.obj("fx_impact_stars_burst", F.star_mesh("fx_impact_stars_burst_m", 9, 0.35, 0.13, 0.02, jitter=0.15, seed=4), coll, root, M["flash_star"])
        env_drivers(burst, root, dur="p_dur", size="p_size", up=0.12, down_start=0.35)
        C.hook("impact", coll, root)
        register("impact_stars", coll, root, "Comic impact: tumbling stars, radial speed lines and a flash starburst.",
                 {"p_t": "seconds since impact", "p_dur": "starburst duration (0.5)", "p_size": "starburst size multiplier", "p_seed": "seed"},
                 ["HOOK_impact"], "Face the camera by rotating the root (stars/lines are 3D; the burst faces -Y).",
                 dict(times=[0.03, 0.1, 0.2, 0.35, 0.55, 0.8], cam=((0, -2.2, 0.0), (0, 0, 0), 50), bg="sky"),
                 groups=[g1.name, g2.name], mats=[M["star"].name, M["line"].name, M["flash_star"].name])
    if want("muzzle_flash"):
        coll, root = F.make_root("fx_muzzle_flash", {"p_dur": 0.12, "p_size": 1.0})
        pop = bpy.data.objects.new("fx_muzzle_flash_pop", None)
        coll.objects.link(pop)
        pop.parent = root
        pop.empty_display_size = 0.05
        for nm, pl, rot in (("xz", "XZ", 0.0), ("yz", "YZ", 0.0), ("xz2", "XZ", PI / 8)):
            F.obj("fx_muzzle_flash_star_" + nm, F.star_mesh("fx_muzzle_flash_star_" + nm, 8, 0.32, 0.09, 0.01, jitter=0.12, seed=2, plane=pl), coll, pop,
                  M["flash_star"], loc=(0, -0.1, 0), rot=(0, rot, 0) if pl == "XZ" else (0, 0, 0))
        glow = C.sphere("fx_muzzle_flash_glow", 0.09, loc=(0, -0.05, 0), mat=M["flash_glow"])
        C.add(glow, coll, pop)
        cone = C.cylinder("fx_muzzle_flash_flame", 0.06, 0.5, loc=(0, -0.3, 0), axis="Y", verts=12, mat=M["flash_star"])
        cone.scale = (1, 1, 1)
        # taper flame to a point
        for v in cone.data.vertices:
            if v.co.y < 0:
                v.co.x *= 0.05
                v.co.z *= 0.05
        C.add(cone, coll, pop)
        env_drivers(pop, root, up=0.12, down_start=0.35)
        C.hook("muzzle", coll, root)
        register("muzzle_flash", coll, root, "Muzzle flash: crossed star-shaped flash meshes, forward flame cone and a glow sphere, pop-in / fade scale envelope.",
                 {"p_t": "seconds since the shot (0 .. p_dur)", "p_dur": "flash duration, 0.12 s default", "p_size": "size multiplier", "p_auto": "1 = scene time"},
                 ["HOOK_muzzle (origin; flash points along -Y)"], "Parent ROOT to the gun's muzzle hook. Emissive HDR: enable the comp bloom.",
                 dict(times=[0.0, 0.015, 0.03, 0.05, 0.08, 0.11], cam=((1.1, -1.3, 0.35), (0, -0.2, 0), 50), bg="dark"),
                 mats=[M["flash_star"].name, M["flash_glow"].name])
    if want("smoke_ring"):
        coll, root = F.make_root("fx_smoke_ring")
        grp = F.ring_group("NG_fx_smoke_ring")
        o, mod = F.gn_object("fx_smoke_ring_ring", grp, coll, root, values={"Material": M["smoke"]})
        F.drive_time(o, mod, root)
        F.drive_prop_mod(o, mod, "Intensity", root, "p_intensity")
        C.hook("muzzle", coll, root)
        register("smoke_ring", coll, root, "Cartoon smoke ring: an expanding, thinning torus that travels along -Y.",
                 {"p_t": "seconds since the shot (0 .. Life 0.9)"}, ["HOOK_muzzle"], "Tune Radius End, Travel, Life in the modifier. Parent root to the muzzle.",
                 dict(times=[0.03, 0.15, 0.3, 0.5, 0.7, 0.88], cam=((-1.6, -2.3, 0.5), (0, -0.6, 0), 50), bg="sky"), groups=[grp.name], mats=[M["smoke"].name])
    if want("shockwave"):
        coll, root = F.make_root("fx_shockwave")
        grp = F.ring_group("NG_fx_shockwave", dict(Life=0.45, Radius_Start=0.05, Radius_End=0.9, Thickness_Start=0.05, Thickness_End=0.006, Travel=0.0))
        o, mod = F.gn_object("fx_shockwave_ring", grp, coll, root, values={"Material": M["shock"]})
        F.drive_time(o, mod, root)
        F.drive_prop_mod(o, mod, "Intensity", root, "p_intensity")
        C.hook("center", coll, root)
        register("shockwave", coll, root, "Flat expanding shockwave ring for CRACK / SLAP impacts (ring plane normal along Y; rotate the root to aim it).",
                 {"p_t": "seconds since impact (0 .. 0.45)"}, ["HOOK_center"], "Emissive: pair with the comp bloom.",
                 dict(times=[0.02, 0.08, 0.15, 0.25, 0.35, 0.44], cam=((0, -2.6, 0.0), (0, 0, 0), 50), bg="dark"), groups=[grp.name], mats=[M["shock"].name])
    if want("confetti"):
        particle_effect("confetti", "Confetti burst: tumbling coloured flakes with flutter.", M["confetti"], dict(tumble=True),
                        dict(Count=140, Life=3.0, Spawn_Window=0.25, Emit_Radius=0.08, Direction=(0, 0, 1), Spread=0.7, Speed=3.5, Speed_Var=0.5, Drag=1.6,
                             Gravity=(0, 0, -1.6), Turbulence=0.8, Scale_Start=0.06, Scale_End=0.06, Scale_Var=0.4, Pop=0.04, Fade=0.1, Spin=9.0,
                             Floor_Z=-0.9, Squash=1.0),
                        dict(times=[0.1, 0.4, 0.8, 1.3, 2.0, 2.8], cam=((0, -3.6, 0.0), (0, 0, -0.1), 50), bg="sky"), shape="quad")
    if want("egg_splatter"):
        particle_effect("egg_splatter", "Egg splatter: translucent white blobs with yolk chunks that fly, land and squash flat.", M["egg"], {},
                        dict(Count=26, Life=6.0, Spawn_Window=0.04, Emit_Radius=0.04, Direction=(0, -1, 0.35), Spread=0.8, Speed=2.4, Speed_Var=0.5,
                             Drag=0.3, Gravity=(0, 0, -9.8), Scale_Start=0.025, Scale_End=0.025, Scale_Var=0.6, Pop=0.02, Fade=0.0, Floor_Z=-0.4,
                             Squash=0.2),
                        dict(times=[0.05, 0.15, 0.3, 0.5, 0.8, 1.6], cam=((1.8, -2.6, 0.2), (0, -0.6, -0.15), 50), bg="sky"))
    if want("glow_trail"):
        coll, root = F.make_root("fx_glow_trail", {"p_length": 1.0, "p_width": 0.06})
        me = bpy.data.meshes.new("fx_glow_trail_ribbon")
        import bmesh as bm_
        b = bm_.new()
        ns = 24
        vl, vr = [], []
        for i in range(ns + 1):
            y = i / ns
            vl.append(b.verts.new((-0.5 * (1 - y) ** 0.8, y, 0)))
            vr.append(b.verts.new((0.5 * (1 - y) ** 0.8, y, 0)))
        for i in range(ns):
            b.faces.new((vl[i], vr[i], vr[i + 1], vl[i + 1]))
        b.to_mesh(me)
        b.free()
        rib = F.obj("fx_glow_trail_ribbon_a", me, coll, root, M["trail_cyan"])
        me2 = me.copy()
        me2.name = "fx_glow_trail_ribbon_b"
        me2.materials.clear()
        rib2 = F.obj("fx_glow_trail_ribbon_b", me2, coll, root, M["trail_cyan"], rot=(0, PI / 2, 0))
        for r_ in (rib, rib2):
            for i, (vn, pn) in enumerate((("w", "p_width"), ("l", "p_length"), ("w", "p_width"))):
                fc = r_.driver_add("scale", i)
                d = fc.driver
                d.type = "SCRIPTED"
                v = d.variables.new()
                v.name = "v"
                v.type = "SINGLE_PROP"
                v.targets[0].id = root
                v.targets[0].data_path = '["%s"]' % pn
                d.expression = "v"
        C.hook("head", coll, root)
        register("glow_trail", coll, root, "Glow trail: two crossed tapered emissive ribbons trailing behind (+Y) the head at the origin; head moves along -Y.",
                 {"p_length": "trail length in metres (drives scale Y)", "p_width": "head width in metres (drives scale X/Z)"}, ["HOOK_head"],
                 "Parent ROOT to the moving object (e.g. Pulse) so the trail points opposite its travel; swap material MAT_vfx_fx_trail_orange / _white for colour. "
                 "Emissive: pair with bloom.",
                 dict(times=[0], cam=((0.9, -1.6, 0.6), (0, 0.4, 0), 50), bg="dark", single=True), mats=[M["trail_cyan"].name, M["trail_orange"].name, M["trail_white"].name])
    if want("motion_streaks"):
        coll, root = F.make_root("fx_motion_streaks")
        grp = F.streak_group("NG_fx_motion_streaks")
        o, mod = F.gn_object("fx_motion_streaks_tunnel", grp, coll, root, values={"Material": M["streak"]})
        F.drive_time(o, mod, root)
        F.drive_prop_mod(o, mod, "Seed", root, "p_seed")
        F.drive_prop_mod(o, mod, "Intensity", root, "p_intensity")
        C.hook("camera_mount", coll, root)
        register("motion_streaks", coll, root, "Speed-line streaks for speed-ramp / whip shots: thin emissive streaks flowing past in a tunnel around the camera's -Z axis.",
                 {"p_t": "seconds; loops. Animate Speed in the modifier for ramps (or keyframe p_t non-linearly)"}, ["HOOK_camera_mount"],
                 "Parent ROOT to the camera (identity transform): streaks live in front of the camera along local -Z. Raise Speed / Length for faster ramps.",
                 dict(times=[0.0, 0.07, 0.14, 0.21, 0.28, 0.35], cam=((0, 0, 0.0), (0, 0, -3.0), 35), bg="dark"), groups=[grp.name], mats=[M["streak"].name])
    if want("heat_shimmer"):
        coll, root = F.make_root("fx_heat_shimmer", {"p_strength": 1.0})
        for nm, rf, x in (("haze", False, 0.0),):
            me = bpy.data.meshes.new("fx_heat_shimmer_" + nm)
            b = bmesh_grid(0.35, 0.8)
            b.to_mesh(me)
            b.free()
            m = shimmer_material(rf)
            F.obj("fx_heat_shimmer_" + nm, me, coll, root, m, rot=(PI / 2, 0, 0), loc=(x, 0, 0.4))
        C.hook("base", coll, root)
        register("heat_shimmer", coll, root, "Heat shimmer: a vertical plane with rising translucent haze streaks and a soft edge (cheap approximation of heat distortion; works with ray tracing off).",
                 {"p_t": "seconds (scroll time)", "p_strength": "refraction bend strength"}, ["HOOK_base"],
                 "Place between the camera and the hot object, plane faces -Y; scale the root to cover the object. True refraction was tried (Refraction BSDF with ray tracing) and renders dark on a one-sided plane in EEVEE Next, so it was dropped; if real distortion is wanted add a compositor Displace node (see the lighting_and_world comp group notes).",
                 dict(times=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5], cam=((0, -2.6, 0.4), (0, 0, 0.35), 50), bg="checker"), mats=["MAT_vfx_fx_heat_shimmer_haze"])
    if want("bang_starburst"):
        coll, root = F.make_root("fx_bang_starburst", {"p_dur": 0.7, "p_size": 1.0})
        pop = bpy.data.objects.new("fx_bang_starburst_pop", None)
        coll.objects.link(pop)
        pop.parent = root
        outer = F.obj("fx_bang_starburst_outer", F.star_mesh("fx_bang_outer_m", 14, 0.55, 0.3, 0.02, jitter=0.18, seed=7), coll, pop, M["bang_outer"])
        inner = F.obj("fx_bang_starburst_inner", F.star_mesh("fx_bang_inner_m", 14, 0.45, 0.24, 0.02, jitter=0.18, seed=11), coll, pop, M["bang_inner"],
                      loc=(0, -0.012, 0), rot=(0, 0.1, 0))
        cu = bpy.data.curves.new("fx_bang_text_c", "FONT")
        cu.body = "BANG"
        cu.size = 0.22
        cu.align_x = "CENTER"
        cu.align_y = "CENTER"
        cu.extrude = 0.01
        cu.bevel_depth = 0.006
        tx = bpy.data.objects.new("fx_bang_starburst_text", cu)
        coll.objects.link(tx)
        tx.parent = pop
        tx.location = (0, -0.03, 0)
        tx.rotation_euler = (PI / 2, 0, 0.1)
        cu.materials.append(M["bang_text"])
        env_drivers(pop, root, up=0.2, down_start=0.7, extra="")
        C.hook("center", coll, root)
        register("bang_starburst", coll, root, "Comic 'BANG' starburst: jagged red / yellow starburst with extruded text, pop-in with fast scale-up and fade-out.",
                 {"p_t": "seconds since the shot (0 .. p_dur 0.7)", "p_dur": "duration", "p_size": "size multiplier"}, ["HOOK_center (faces -Y)"],
                 "Parent to the camera or place near the muzzle; replace the text body (object fx_bang_starburst_text) for CRACK / SLAP. Emissive.",
                 dict(times=[0.0, 0.06, 0.12, 0.25, 0.5, 0.65], cam=((0, -1.5, 0.0), (0, 0, 0), 50), bg="sky"),
                 mats=[M["bang_outer"].name, M["bang_inner"].name, M["bang_text"].name])
    return M


def bmesh_grid(w, h):
    import bmesh as bm_
    b = bm_.new()
    bm_.ops.create_grid(b, x_segments=16, y_segments=24, size=0.5)
    for v in b.verts:
        v.co.x *= w * 2
        v.co.y *= h * 2
    uvl = b.loops.layers.uv.new("UVMap")
    for f in b.faces:
        for l in f.loops:
            l[uvl].uv = ((l.vert.co.x / w) * 0.5 + 0.5, (l.vert.co.y / h) * 0.5 + 0.5)
    return b


def shimmer_material(refract):
    m = V.new_material("MAT_vfx_fx_heat_shimmer_" + ("refract" if refract else "haze"))
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    tc = t.nodes.new("ShaderNodeTexCoord")
    tm = t.nodes.new("ShaderNodeValue")
    tm.name = "ShimmerTime"
    st = t.nodes.new("ShaderNodeValue")
    st.name = "ShimmerStrength"
    st.outputs[0].default_value = 1.0
    uv = V.wrapv(tc.outputs["UV"])
    x, y, _ = V.sepxyz(uv)
    n = t.nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = "3D"
    n.inputs["Detail"].default_value = 3.0
    V.L(V.comb(x * 3.0, y * 1.5 - V.X(tm.outputs[0]) * 0.8, V.X(tm.outputs[0]) * 0.2), n.inputs["Vector"])
    n.inputs["Scale"].default_value = 2.5
    streak = V.smoothstep(0.58, 0.78, V.X(n.outputs["Fac"])) * V.smoothstep(0.0, 0.25, y) * (1.0 - V.smoothstep(0.55, 1.0, y)) * 0.45
    edge = V.smoothstep(0.0, 0.2, x) * (1.0 - V.smoothstep(0.8, 1.0, x))
    tr = t.nodes.new("ShaderNodeBsdfTransparent")
    wh = t.nodes.new("ShaderNodeEmission")
    wh.inputs["Strength"].default_value = 0.8
    mx = t.nodes.new("ShaderNodeMixShader")
    V.L(streak * edge, mx.inputs["Fac"])
    if refract:
        bump = t.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = 0.05
        V.L(V.X(n.outputs["Fac"]), bump.inputs["Height"])
        V.L(V.X(st.outputs[0]) * 1.6, bump.inputs["Strength"])
        rf = t.nodes.new("ShaderNodeBsdfRefraction")
        rf.inputs["IOR"].default_value = 1.08
        rf.inputs["Roughness"].default_value = 0.0
        t.links.new(bump.outputs["Normal"], rf.inputs["Normal"])
        t.links.new(rf.outputs[0], mx.inputs[1])
        m.use_raytrace_refraction = True
    else:
        t.links.new(tr.outputs[0], mx.inputs[1])
    t.links.new(wh.outputs[0], mx.inputs[2])
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(mx.outputs[0], o.inputs[0])
    V.set_render_method(m, "BLENDED")
    V.auto_layout(t)
    return m


def hook_shimmer_drivers():
    root = bpy.data.objects.get("ROOT_fx_heat_shimmer")
    if root is None:
        return
    for m in [bpy.data.materials.get("MAT_vfx_fx_heat_shimmer_haze")]:
        _shimmer_drv(m, root)


def _shimmer_drv(m, root):
    nt = m.node_tree
    for nodename, props, expr in (("ShimmerTime", {"t": "p_t", "a": "p_auto", "s": "p_start", "f": "p_fps"}, "(frame - s) / f if a > 0.5 else t"),
                                  ("ShimmerStrength", {"v": "p_strength"}, "v")):
        fc = nt.driver_add('nodes["%s"].outputs[0].default_value' % nodename)
        d = fc.driver
        d.type = "SCRIPTED"
        for vn, pn in props.items():
            v = d.variables.new()
            v.name = vn
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["%s"]' % pn
        d.expression = expr


def previews():
    spec_all = {k: v["preview"] for k, v in EFFECTS.items()}
    paths_done = []
    for name, spec in spec_all.items():
        if ONLY and name not in ONLY:
            continue
        V.reopen(BLEND)
        V.prep_scene(res=(450, 338), samples=16)
        scn = bpy.context.scene
        keep = bpy.data.collections["ASSET_fx_" + name]
        keep_objs = set(keep.all_objects)
        for o in scn.objects:
            if o not in keep_objs:
                o.hide_render = True
                o.hide_viewport = True
        for c in list(scn.collection.children):
            c.hide_render = c.name != keep.name
        bg = spec.get("bg", "sky")
        if bg == "sky":
            V.simple_world((0.35, 0.6, 0.95), 0.9)
        elif bg == "checker":
            V.simple_world((0.4, 0.5, 0.65), 0.8)
        else:
            V.simple_world((0.05, 0.05, 0.07), 0.6)
        cam_loc, cam_tgt, lens = spec["cam"]
        # backdrop
        if bg != "dark" or name in ("sparks_burst",):
            bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 4.0, 0), rotation=(PI / 2, 0, 0))
            bk = bpy.context.active_object
            bk.scale = (30, 30, 1)
            bm = V.new_material("TMP_bk")
            nt = bm.node_tree
            b = nt.nodes["Principled BSDF"]
            if bg == "checker":
                ck = nt.nodes.new("ShaderNodeTexChecker")
                ck.inputs["Scale"].default_value = 18
                ck.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
                ck.inputs["Color2"].default_value = (0.2, 0.25, 0.35, 1)
                nt.links.new(ck.outputs[0], b.inputs["Base Color"])
            else:
                b.inputs["Base Color"].default_value = (0.12, 0.45, 0.85, 1) if bg == "sky" else (0.03, 0.03, 0.04, 1)
            b.inputs["Roughness"].default_value = 1.0
            bk.data.materials.append(bm)
            bk.visible_shadow = False
        V.studio_lights(center=(0, 0, 0.3), scale=1.5, key=1.0, world=0.6)
        if bg == "sky":
            V.simple_world((0.35, 0.6, 0.95), 0.9)
        if spec.get("rt"):
            scn.eevee.use_raytracing = True
        V.make_camera(cam_loc, cam_tgt, lens=lens)
        root = bpy.data.objects[EFFECTS[name]["root"]]
        paths = []
        times = spec["times"]
        for i, t in enumerate(times):
            root["p_t"] = float(t)
            bpy.context.view_layer.update()
            scn.frame_set(1)
            p = os.path.join(SCR, "fx_%s_%d.png" % (name, i))
            V.render_still(p)
            paths.append(p)
        if len(paths) == 1:
            import shutil
            out = os.path.join(PREV, "vfx_%s.png" % name)
            shutil.copy(paths[0], out)
        else:
            out = os.path.join(PREV, "vfx_%s_contact_sheet.png" % name)
            V.tile_pngs(paths, out, 3)
        for p in paths:
            os.remove(p)
        paths_done.append(out)
        print("PREVIEW", out)
    return paths_done


def main():
    t0 = time.time()
    C.reset()
    # carry over previous blend contents when building a subset
    build_all()
    hook_shimmer_drivers()
    # view layer: leave all collections visible; save
    meta = {
        "category": "materials_vfx",
        "description": "Reusable closed-form effects as collections ASSET_fx_<name>, each driven by custom properties on its root empty.",
        "sources": [],
        "effects": EFFECTS,
        "accuracy_notes": "Art-directed cartoon effects, no physical dimension claims; sizes are for human-scale scenes (1 unit = 1 m); scale the ROOT for hardware-scale uses.",
        "build_seconds": round(time.time() - t0, 1),
        "usage": [
            "Append a collection ASSET_fx_<name> (collection append with its objects, node groups and materials); then parent its ROOT_fx_<name> empty to the hook that should emit it.",
            "Time: keyframe root['p_t'] from 0 over the event, or set root['p_auto'] = 1 and root['p_start'] = trigger frame (time = (frame - p_start) / p_fps). All particle positions are closed-form in time, so scrubbing and re-rendering any frame is exact.",
            "Tune the look with vfx_lib.set_param(emitter_object, 'Speed', ...) (Geometry Nodes modifier inputs by display name); Seed/Intensity are driven by p_seed/p_intensity.",
            "Emissive effects (flash, sparks, streaks, trails, shockwave, bang) are HDR: use the compositor glow in lighting_and_world for the bloom.",
            "Particles are real geometry (realized instances), triangle counts in the metadata; effect cost is negligible next to materials/lights.",
        ],
    }
    coll_all = None
    for c in bpy.data.collections:
        pass
    C.save(BLEND)
    with open(os.path.splitext(BLEND)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    print("SAVED", BLEND, "effects", list(EFFECTS))
    previews()


main()
