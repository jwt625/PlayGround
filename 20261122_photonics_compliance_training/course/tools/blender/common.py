"""Shared helpers for the course Blender renders (Blender 4.2 LTS, Cycles, background mode).

Run any shot script through the slot wrapper, e.g.
  bslot.sh -b --factory-startup --python shot_die_fixture.py -- out=/path/x.jpg samples=128 variant=0

Units: unless a script says otherwise, 1 Blender unit = 1 mm. Cycles does not rescale the
camera aperture with the scene unit scale, so depth of field is set from the physical
entrance-pupil diameter with `set_aperture`.
"""
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
HDRI_DIR = os.environ.get("BLENDER_HDRI_DIR", os.path.expanduser("~/.cache/photonics_compliance_hdri"))


# ------------------------------------------------------------------ args / scene
def args(**defaults):
    """Parse key=value pairs after '--'. Values are cast to the type of the default."""
    out = dict(defaults)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            d = defaults.get(k)
            if isinstance(d, bool):
                out[k] = v.lower() in ("1", "true", "yes")
            elif isinstance(d, int):
                out[k] = int(v)
            elif isinstance(d, float):
                out[k] = float(v)
            else:
                out[k] = v
    return out


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    return sc


def setup_render(samples=128, res=(1920, 1080), noise=0.015, look="AgX - Medium High Contrast",
                 exposure=0.0, max_bounces=12):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    c = sc.cycles
    c.device = "GPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = noise
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.denoising_prefilter = "ACCURATE"
    c.max_bounces = max_bounces
    c.diffuse_bounces = 4
    c.glossy_bounces = 8
    c.transmission_bounces = 12
    c.transparent_max_bounces = 16
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.blur_glossy = 0.5
    c.sample_clamp_indirect = 10.0
    c.filter_width = 1.5
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = look
    except TypeError:
        pass
    sc.view_settings.exposure = exposure
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 88
    sc.render.film_transparent = False
    return sc


def world_hdri(name="studio_small_09", strength=1.0, rot_z_deg=0.0, bg_strength=None, bg_color=None):
    """HDRI world. If bg_color is given, the camera sees that flat color while lighting uses the HDRI."""
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    path = os.path.join(HDRI_DIR, f"{name}_2k.hdr")
    if os.path.exists(path):
        env.image = bpy.data.images.load(path)
    else:  # bundled fallback
        p2 = os.path.join(os.path.dirname(bpy.app.binary_path), "..", "Resources",
                          f"{bpy.app.version[0]}.{bpy.app.version[1]}", "datafiles", "studiolights",
                          "world", "interior.exr")
        env.image = bpy.data.images.load(os.path.abspath(p2))
        print("WARNING: HDRI missing, using bundled interior.exr; run fetch_hdri.sh")
    mp.inputs["Rotation"].default_value[2] = math.radians(rot_z_deg)
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    if bg_color is None:
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    else:
        bg2 = nt.nodes.new("ShaderNodeBackground")
        bg2.inputs["Color"].default_value = (*bg_color, 1)
        bg2.inputs["Strength"].default_value = 1.0 if bg_strength is None else bg_strength
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs["Background"], mix.inputs[1])
        nt.links.new(bg2.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return w


# ------------------------------------------------------------------ materials
def _node(nt, kind, loc=(0, 0)):
    n = nt.nodes.new(kind)
    n.location = loc
    return n


def principled(name, color=(0.8, 0.8, 0.8), metallic=0.0, rough=0.5, ior=1.5, transmission=0.0,
               coat=0.0, coat_rough=0.03, spec=0.5, film_nm=0.0, film_ior=1.33, alpha=1.0,
               emission=None, emission_strength=0.0, rough_var=0.0, rough_scale=40.0,
               bump=0.0, bump_scale=200.0, bump_dist=0.002, aniso=0.0, color_var=0.0,
               color_scale=5.0, coords="Object"):
    """Principled BSDF with optional procedural roughness / color variation and micro bump.
    Scales are in 1/BU of the chosen texture coordinates."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    p.inputs["IOR"].default_value = ior
    p.inputs["Transmission Weight"].default_value = transmission
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = coat_rough
    p.inputs["Specular IOR Level"].default_value = spec
    p.inputs["Thin Film Thickness"].default_value = film_nm
    p.inputs["Thin Film IOR"].default_value = film_ior
    p.inputs["Alpha"].default_value = alpha
    p.inputs["Anisotropic"].default_value = aniso
    if emission is not None:
        p.inputs["Emission Color"].default_value = (*emission, 1)
        p.inputs["Emission Strength"].default_value = emission_strength
    tc = _node(nt, "ShaderNodeTexCoord", (-900, 0))
    if rough_var > 0:
        nz = _node(nt, "ShaderNodeTexNoise", (-600, 200))
        nz.inputs["Scale"].default_value = rough_scale
        nz.inputs["Detail"].default_value = 6
        nt.links.new(tc.outputs[coords], nz.inputs["Vector"])
        mr = _node(nt, "ShaderNodeMapRange", (-350, 200))
        mr.inputs["From Min"].default_value = 0.3
        mr.inputs["From Max"].default_value = 0.7
        mr.inputs["To Min"].default_value = max(0.0, rough - rough_var)
        mr.inputs["To Max"].default_value = min(1.0, rough + rough_var)
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], p.inputs["Roughness"])
    if color_var > 0:
        nz = _node(nt, "ShaderNodeTexNoise", (-600, 500))
        nz.inputs["Scale"].default_value = color_scale
        nz.inputs["Detail"].default_value = 3
        nt.links.new(tc.outputs[coords], nz.inputs["Vector"])
        mix = _node(nt, "ShaderNodeMix", (-300, 500))
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["A"].default_value = (*color, 1)
        mr = _node(nt, "ShaderNodeMapRange", (-450, 650))
        mr.inputs["To Min"].default_value = 1 - color_var
        mr.inputs["To Max"].default_value = 1 + color_var
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        cm = _node(nt, "ShaderNodeCombineColor", (-350, 650))
        for i in range(3):
            nt.links.new(mr.outputs["Result"], cm.inputs[i])
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(cm.outputs["Color"], mix.inputs["B"])
        nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    if bump > 0:
        nz = _node(nt, "ShaderNodeTexNoise", (-600, -250))
        nz.inputs["Scale"].default_value = bump_scale
        nz.inputs["Detail"].default_value = 8
        nt.links.new(tc.outputs[coords], nz.inputs["Vector"])
        b = _node(nt, "ShaderNodeBump", (-300, -250))
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = bump_dist
        nt.links.new(nz.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
        if coat > 0:
            pass
    return m


def emission_mat(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def gold(name="gold", rough=0.07):
    # linear-space gold reflectance (approx. F0 of Au); evaporated films are near-mirror
    return principled(name, color=(1.0, 0.70, 0.27), metallic=1.0, rough=rough, rough_var=0.03,
                      rough_scale=60.0, bump=0.3, bump_scale=900.0, bump_dist=0.00002)


# ------------------------------------------------------------------ geometry
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def set_mat(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def mesh_obj(name, verts, faces, mat=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = link(bpy.data.objects.new(name, me))
    if mat is not None:
        set_mat(ob, mat)
    return ob


def extrude_polys(name, polys, z0, h, mat=None, bevel=0.0, bevel_segments=2):
    """Each poly is a list of (x, y) (counterclockwise preferred). Extruded from z0 to z0+h."""
    bm = bmesh.new()
    for poly in polys:
        vs = [bm.verts.new((x, y, z0)) for x, y in poly]
        try:
            f = bm.faces.new(vs)
        except ValueError:
            continue
        if f.normal.z < 0:
            f.normal_flip()
    bm.normal_update()
    faces = list(bm.faces)
    res = bmesh.ops.extrude_face_region(bm, geom=faces)
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv, vec=(0, 0, h))
    for f in faces:  # original faces are now bottoms
        f.normal_flip()
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    if mat is not None:
        set_mat(ob, mat)
    if bevel > 0:
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = bevel
        md.segments = bevel_segments
        md.limit_method = "ANGLE"
        md.harden_normals = False
    return ob


def box(name, size, loc, mat=None, bevel=0.0, segments=3, rot=(0, 0, 0)):
    sx, sy, sz = size
    x, y, z = 0, 0, 0
    v = [(x + a * sx / 2, y + b * sy / 2, z + c * sz / 2) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ob = mesh_obj(name, v, f, mat)
    ob.location = loc
    ob.rotation_euler = rot
    if bevel > 0:
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = bevel
        md.segments = segments
    return ob


def cylinder(name, r, h, loc, mat=None, verts=64, rot=(0, 0, 0), bevel=0.0, r2=None):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r, radius2=r if r2 is None else r2,
                                    depth=h, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.name = name
    if mat is not None:
        set_mat(ob, mat)
    if bevel > 0:
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = bevel
        md.segments = 3
        md.limit_method = "ANGLE"
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    return ob


def smooth(ob, angle=35):
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
    return ob


def text_mesh(name, text, size, loc, rot_z=0.0, h=0.001, mat=None, align="CENTER", font=None,
              rot=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = text
    cu.size = size
    cu.extrude = h / 2
    cu.align_x = align
    cu.align_y = "CENTER"
    if font is not None:
        cu.font = font
    ob = link(bpy.data.objects.new(name, cu))
    ob.location = (loc[0], loc[1], loc[2] + h / 2)
    ob.rotation_euler = rot if rot is not None else (0, 0, rot_z)
    if mat is not None:
        cu.materials.append(mat)
    return ob


def curve_tube(name, pts, radius, mat=None, resolution=24, bevel_res=8):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = bevel_res
    cu.resolution_u = resolution
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = link(bpy.data.objects.new(name, cu))
    if mat is not None:
        cu.materials.append(mat)
    return ob


def parent(children, par):
    for c in children:
        c.parent = par
    return par


def empty(name, loc=(0, 0, 0), rot=(0, 0, 0)):
    e = link(bpy.data.objects.new(name, None))
    e.location = loc
    e.rotation_euler = rot
    return e


# ------------------------------------------------------------------ camera / lights
def camera(loc, target, lens=100.0, sensor=36.0, focus=None, pupil_mm=0.0, ortho_scale=None,
           clip=(0.1, 1e5), shift=(0, 0)):
    cd = bpy.data.cameras.new("Cam")
    cam = link(bpy.data.objects.new("Cam", cd))
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = sensor
    cd.lens = lens
    cd.clip_start, cd.clip_end = clip
    cd.shift_x, cd.shift_y = shift
    if ortho_scale is not None:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho_scale
    if focus is not None and pupil_mm > 0:
        set_aperture(cam, focus, pupil_mm)
    bpy.context.scene.camera = cam
    return cam


def set_aperture(cam, focus, pupil_mm, blades=8):
    """Thin-lens DOF with entrance pupil diameter `pupil_mm` scene units, focused at point/distance."""
    cd = cam.data
    bpy.context.view_layer.update()
    cd.dof.use_dof = True
    if isinstance(focus, (int, float)):
        cd.dof.focus_distance = float(focus)
    else:
        v = Vector(focus) - cam.location
        fwd = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cd.dof.focus_distance = v.dot(fwd)
    # Cycles: aperture radius [BU] = lens[mm]*1e-3 / (2 * fstop)
    cd.dof.aperture_fstop = cd.lens * 1e-3 / pupil_mm
    cd.dof.aperture_blades = blades
    cd.dof.aperture_rotation = math.radians(10)


def area_light(name, loc, target, size, power, color=(1, 1, 1), shape="RECTANGLE", size_y=None,
               spread=180.0, cam_visible=False):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = power
    ld.color = color
    ld.spread = math.radians(spread)
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    ob.visible_camera = cam_visible
    return ob


def ceiling(z, size=2000.0, color=(0.03, 0.03, 0.03), name="ceiling"):
    """Camera-invisible dark plane above the set: controls what flat metal and mirrors reflect."""
    ob = box(name, (size, size, 1.0), (0, 0, z), principled(name + "_m", color=color, rough=0.9))
    ob.visible_camera = False
    ob.visible_shadow = False
    return ob


def surround(radius=600.0, height=800.0, color=(0.025, 0.025, 0.027), name="surround"):
    """Camera-invisible dark cylinder around the set (blocks bright HDRI horizons in reflections)."""
    ob = cylinder(name, radius, height, (0, 0, height / 2 - 100), None, verts=64)
    md = ob.modifiers.new("solid", "SOLIDIFY")
    md.thickness = 2.0
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for f in [f for f in bm.faces if abs(f.normal.z) > 0.9]:
        bm.faces.remove(f)
    bm.to_mesh(ob.data)
    bm.free()
    set_mat(ob, principled(name + "_m", color=color, rough=0.8))
    ob.visible_camera = False
    return ob


def mirror_point(cam_loc, surf_pt, dist, normal=(0, 0, 1)):
    """Point at distance `dist` from surf_pt along the camera ray reflected by a surface of `normal`."""
    n = Vector(normal).normalized()
    v = (Vector(surf_pt) - Vector(cam_loc)).normalized()
    r = v - 2 * v.dot(n) * n
    return Vector(surf_pt) + r * dist


def edge_softbox(name, loc, target, size, strength, edge=10.0, angle_deg=0.0, low=0.08,
                 color=(1.0, 0.97, 0.93)):
    """Large emissive card whose emission steps from `low` to 1 across a soft edge through its
    centre (local x = 0, width `edge`). Placed so a mirror-like surface reflects the edge, it gives
    the graded studio reflection seen on polished dies and gold. Invisible to the camera."""
    ob = box(name, (size, size, 0.5), loc)
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    ob.rotation_euler.rotate_axis("Z", math.radians(angle_deg))
    m = bpy.data.materials.new(name + "_m")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    e = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = -edge / 2
    mr.inputs["From Max"].default_value = edge / 2
    mr.inputs["To Min"].default_value = low * strength
    mr.inputs["To Max"].default_value = strength
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], e.inputs["Strength"])
    e.inputs["Color"].default_value = (*color, 1)
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    set_mat(ob, m)
    ob.visible_camera = False
    ob.visible_shadow = False
    return ob


def point_light(name, loc, power, radius=1.0, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = power
    ld.shadow_soft_size = radius
    ld.color = color
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    return ob


# ------------------------------------------------------------------ dirt
def scatter_dust(name, region, z, n, rmin, rmax, mat, seed=1, fibers=0, fiber_len=(0.2, 1.0), flat=0.5):
    """Small specks (flattened icospheres) and lint fibers lying on a plane at height z."""
    rnd = random.Random(seed)
    (x0, x1), (y0, y1) = region
    bm = bmesh.new()
    for _ in range(n):
        r = rmin * (rmax / rmin) ** (rnd.random() ** 2.2)
        c = (rnd.uniform(x0, x1), rnd.uniform(y0, y1), z + r * flat * 0.8)
        g = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
        sx, sy = rnd.uniform(0.6, 1.4), rnd.uniform(0.6, 1.4)
        m = Matrix.Translation(c) @ Matrix.Rotation(rnd.uniform(0, 6.28), 4, "Z") @ Matrix.Diagonal((sx, sy, flat, 1))
        bmesh.ops.transform(bm, matrix=m, verts=g["verts"])
    for _ in range(fibers):
        L = rnd.uniform(*fiber_len)
        r = rmin * 0.6
        a = rnd.uniform(0, 6.28)
        cx, cy = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        k = rnd.uniform(-1.5, 1.5) / L
        prev = None
        segs = 10
        for i in range(segs + 1):
            t = i / segs
            ang = a + k * L * (t - 0.5)
            p = Vector((cx + math.cos(ang) * L * (t - 0.5), cy + math.sin(ang) * L * (t - 0.5), z + r))
            g = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
            bmesh.ops.translate(bm, verts=g["verts"], vec=p)
            prev = p
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me))
    set_mat(ob, mat)
    return ob


# ------------------------------------------------------------------ compositor
def compositor(vignette=0.25, dispersion=0.0, glare=0.0, grain=0.0, distort=0.0):
    sc = bpy.context.scene
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    cur = rl.outputs["Image"]
    if glare > 0:
        g = nt.nodes.new("CompositorNodeGlare")
        g.glare_type = "FOG_GLOW"
        g.quality = "HIGH"
        g.mix = -1 + glare
        g.threshold = 0.9
        nt.links.new(cur, g.inputs["Image"])
        cur = g.outputs["Image"]
    if dispersion > 0 or distort != 0:
        ld = nt.nodes.new("CompositorNodeLensdist")
        ld.inputs["Dispersion"].default_value = dispersion
        ld.inputs["Distortion"].default_value = distort
        nt.links.new(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    if vignette > 0:
        em = nt.nodes.new("CompositorNodeEllipseMask")
        em.width, em.height = 1.25, 1.25 * 16 / 9 * 0 + 1.35
        bl = nt.nodes.new("CompositorNodeBlur")
        bl.filter_type = "GAUSS"
        bl.use_relative = True
        bl.factor_x = bl.factor_y = 30
        bl.size_x = bl.size_y = 400
        nt.links.new(em.outputs["Mask"], bl.inputs["Image"])
        mr = nt.nodes.new("CompositorNodeMapRange")
        mr.inputs["To Min"].default_value = 1 - vignette
        mr.inputs["To Max"].default_value = 1.0
        nt.links.new(bl.outputs["Image"], mr.inputs["Value"])
        mul = nt.nodes.new("CompositorNodeMixRGB")
        mul.blend_type = "MULTIPLY"
        nt.links.new(cur, mul.inputs[1])
        nt.links.new(mr.outputs["Value"], mul.inputs[2])
        cur = mul.outputs["Image"]
    nt.links.new(cur, comp.inputs["Image"])


def render(out):
    import time
    sc = bpy.context.scene
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sc.render.filepath = out
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print(f"RENDER_SECONDS {dt:.1f} {out}")
    return dt
