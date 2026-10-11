"""Verification renders of the RX optics + TIA subassembly (Blender 4.2, Cycles 64 spp + denoise, headless).

  scripts/bslot.sh -b assets/components/optics_rx/optics_rx.blend --python scripts/assets/optics_rx/render_optics_rx.py \
      -- [shot ...]        (default: all shots)
Raw PNGs -> outputs/assets/optics_rx/work/<shot>_raw.png; label anchors -> work/<shot>_labels.json.
Camera numbers are in the subassembly frame, mm.
"""
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(PROJ, "outputs", "assets", "optics_rx", "work")
os.makedirs(WORK, exist_ok=True)
# EEVEE shadow maps misbehave at true mm scale (jagged self-shadow edges), so the root is scaled x1000 for rendering
# only: 1 mm renders as 1 m.  Camera numbers stay in mm.  The saved asset is untouched (true scale).
S = 1.0
SCN = bpy.context.scene
bpy.data.objects["RX_OPTICS_ROOT"].scale = (1000.0, 1000.0, 1000.0)
for _m in bpy.data.materials:
    if _m.node_tree:
        for _n in _m.node_tree.nodes:
            if _n.type == "BUMP":
                _n.inputs["Distance"].default_value *= 1000.0
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 32
SCN.eevee.use_raytracing = True
SCN.eevee.shadow_ray_count = 3
SCN.eevee.shadow_step_count = 12
SCN.eevee.ray_tracing_options.resolution_scale = "1"
SCN.render.resolution_x, SCN.render.resolution_y = 1920, 1200
SCN.render.resolution_percentage = 100
SCN.cycles.device = "GPU"
try:
    _cp = bpy.context.preferences.addons["cycles"].preferences
    _cp.compute_device_type = "METAL"
    _cp.get_devices()
    for _d in _cp.devices:
        _d.use = True
except Exception as _e:
    print("cycles device setup:", _e)
SCN.cycles.samples = 64
SCN.cycles.use_denoising = True
SCN.cycles.max_bounces = 24
SCN.cycles.transmission_bounces = 24
SCN.cycles.glossy_bounces = 8
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Medium High Contrast"

world = bpy.data.worlds.new("rx_dark")
SCN.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.018, 0.022, 0.032, 1.0)
bg.inputs["Strength"].default_value = 1.0


def sun(name, rot, strength, angle=0.45, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "SUN")
    d.energy = strength
    d.angle = angle
    d.color = color
    o = bpy.data.objects.new(name, d)
    o.rotation_euler = [math.radians(a) for a in rot]
    d.shadow_maximum_resolution = 0.002
    SCN.collection.objects.link(o)
    return o


sun("key", (40, 0, 135), 2.6, color=(1.0, 0.97, 0.92))
sun("fill", (65, 0, -40), 1.0, color=(0.75, 0.85, 1.0))
sun("rim", (110, 0, 80), 1.4, color=(0.85, 0.95, 1.0))
# soft ambient so shadowed die areas stay readable
bg.inputs["Strength"].default_value = 1.0
amb = world.node_tree.nodes.new("ShaderNodeBackground")
amb.inputs["Color"].default_value = (0.22, 0.24, 0.28, 1.0)
amb.inputs["Strength"].default_value = 0.9
lp = world.node_tree.nodes.new("ShaderNodeLightPath")
mix = world.node_tree.nodes.new("ShaderNodeMixShader")
out = world.node_tree.nodes["World Output"]
world.node_tree.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
world.node_tree.links.new(amb.outputs["Background"], mix.inputs[1])
world.node_tree.links.new(bg.outputs["Background"], mix.inputs[2])
world.node_tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])

cam_d = bpy.data.cameras.new("cam")
cam_d.clip_start = 1e-3
cam_d.clip_end = 1000.0
CAM = bpy.data.objects.new("cam", cam_d)
SCN.collection.objects.link(CAM)
SCN.camera = CAM

BASE = {o.name: o.location.copy() for o in bpy.data.objects}
COLLS = {c.name: c for c in bpy.data.collections}


def place(loc, tgt, lens, dof=None):
    cam_d.type = "PERSP"
    cam_d.shift_x = 0.0
    CAM.location = Vector(loc) * S
    d = (Vector(tgt) - Vector(loc))
    CAM.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam_d.lens = lens
    cam_d.dof.use_dof = False


def explode(on):
    for o in bpy.data.objects:
        if "explode" in o.keys() and o.name in BASE:
            e = Vector(o["explode"]) if on else Vector((0, 0, 0))
            o.location = BASE[o.name] + e


def show(coll, vis):
    COLLS[coll].hide_render = not vis
    COLLS[coll].hide_viewport = not vis


GLASS = [n for n in ("fused_silica", "lenslet_fused_silica_ar", "uv_epoxy") if n in bpy.data.materials]


def viz_glass(on):
    """beam shots: alpha-blended glass (no refraction smear of the beam lines); Si plates as tinted translucent."""
    for nm in GLASS + ["silicon_plate_ar"]:
        m = bpy.data.materials[nm]
        b = m.node_tree.nodes["Principled BSDF"]
        if on:
            b.inputs["Transmission Weight"].default_value = 0.0
            b.inputs["Metallic"].default_value = 0.0
            b.inputs["Alpha"].default_value = 0.30 if nm == "silicon_plate_ar" else 0.10
            if nm == "silicon_plate_ar":
                b.inputs["Base Color"].default_value = (0.45, 0.40, 0.80, 1)
            m.surface_render_method = "BLENDED"
        else:
            b.inputs["Alpha"].default_value = 1.0
            b.inputs["Transmission Weight"].default_value = 0.0 if nm == "silicon_plate_ar" else 1.0
            b.inputs["Metallic"].default_value = 0.75 if nm == "silicon_plate_ar" else 0.0
            if nm == "silicon_plate_ar":
                b.inputs["Base Color"].default_value = (0.30, 0.30, 0.38, 1)
            m.surface_render_method = "DITHERED"


for nm in GLASS:
    bpy.data.materials[nm].node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.25


def translucent_si(on):
    m = bpy.data.materials["silicon_plate_ar"]
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Transmission Weight"].default_value = 0.85 if on else 0.0
    b.inputs["Metallic"].default_value = 0.0 if on else 0.75
    b.inputs["Base Color"].default_value = (0.55, 0.50, 0.75, 1) if on else (0.30, 0.30, 0.38, 1)
    m.use_raytrace_refraction = True


def anchors(labels, path):
    """project each labelled object's visible vertex centroid to pixels."""
    W, H = SCN.render.resolution_x, SCN.render.resolution_y
    out = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        lab = o.get("part_label")
        if o.type != "MESH" or not lab or (labels and o.name not in labels):
            continue
        if any(c.hide_render for c in o.users_collection):
            continue
        me = o.evaluated_get(dg).data
        step = max(1, len(me.vertices) // 400)
        pts = [o.matrix_world @ me.vertices[i].co for i in range(0, len(me.vertices), step)]
        cam = CAM.matrix_world.translation
        vis = []
        for w in pts:                                   # keep only vertices the camera actually sees (ray test)
            d = w - cam
            hit, loc, _n, _i, ob, _m = SCN.ray_cast(dg, cam, d.normalized(), distance=d.length * 1.001)
            if hit and ob.name == o.name and (loc - w).length < d.length * 2e-3:
                vis.append(w)
        if not vis:
            print("label skipped (hidden):", o.name)
            continue
        cen = sum(vis, Vector()) / len(vis)
        c = min(vis, key=lambda w: (w - cen).length)
        v = world_to_camera_view(SCN, CAM, c)
        if 0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0:
            out.append(dict(label=labels.get(o.name, lab) if labels else lab, x=v.x * W, y=(1 - v.y) * H))
    json.dump(out, open(path, "w"), indent=1)


EXPL_LABELS = {
    "rx_lenslet_array_2x4": "2x4 lenslet array (fused silica)",
    "rx_si_interleaver_plate_top": "Si plate, top row (+30 deg): shift -0.25",
    "rx_si_interleaver_plate_bot": "Si plate, bottom row (-30 deg): shift +0.25",
    "rx_interleaver_shelf_posts": "glass shelf + posts",
    "rx_optical_bench": "optical bench",
    "rx_turning_block_stepped": "stepped glass turning block, Au-coated 45 deg facets",
    "rx_pin_pd_array_1x8": "1x8 PIN PD array, 0.5 mm pitch",
    "rx_wirebonds_pd_to_tia_diff": "differential bonds PD -> TIA (G-S-S-G, 32)",
    "rx_octal_tia": "octal TIA (spiral inductors)",
    "rx_wirebonds_tia_out_gssg": "GSSG output bonds (32)",
    "rx_wirebonds_dc_ctrl": "DC / control bonds",
    "rx_decoupling_caps": "decoupling SLC + 0201",
    "rx_carrier_aln": "AlN carrier",
    "rx_carrier_traces_au": "differential CPW + DC traces",
    "rx_flex_pads": "flex pads (8 x GSSG + 6 DC)",
}

SHOTS = {
    # name: (camera loc, target, lens, exploded, beams, extra)
    "overview": ((-0.6, 9.6, 7.4), (-3.3, 0.0, -0.9), 50, False, False, {"lens_alpha": True}),
    "exploded": ((12.0, 30.0, 24.0), (-0.8, 0.0, 0.6), 60, True, False, {"shift_x": -0.04}),
    "tia_closeup": ((-4.50, 0.95, 0.55), (-4.68, 0.15, -1.20), 60, False, False, {"hide": ["rx_micro_optics"]}),
    "bond_closeup": ((-3.05, 1.55, -0.45), (-4.12, 0.25, -1.20), 50, False, False, {"hide": ["rx_micro_optics"]}),
    "optics_closeup": ((2.2, 7.6, 3.6), (-2.1, 0.0, -0.35), 48, False, True, {"si_glass": True}),
    "optics_top": ((-3.0, 0.0, 9.5), (-3.0, 0.0, -1.0), 50, False, True, {"si_glass": True, "up_y": True, "ortho": 8.4}),
}

todo = argv or list(SHOTS)
for name in todo:
    loc, tgt, lens, ex, beams, extra = SHOTS[name]
    explode(ex)
    show("viz_beams", beams)
    show("rx_hooks", False)
    show("rx_micro_optics", "rx_micro_optics" not in extra.get("hide", []))
    viz_glass(extra.get("si_glass", False))
    if extra.get("lens_alpha"):          # thick refractive slab seen from behind renders black in Cycles; draw it clear
        _m = bpy.data.materials["lenslet_fused_silica_ar"]
        _b = _m.node_tree.nodes["Principled BSDF"]
        _b.inputs["Transmission Weight"].default_value = 0.0
        _b.inputs["Alpha"].default_value = 0.22
        _m.surface_render_method = "BLENDED"
    SCN.eevee.use_raytracing = not extra.get("no_rt", False)
    SCN.eevee.ray_tracing_method = extra.get("rt", "SCREEN")
    SCN.render.engine = "BLENDER_EEVEE_NEXT" if extra.get("eevee") else "CYCLES"     # Cycles: clean edges + glass
    place(loc, tgt, lens)
    if extra.get("up_y"):
        CAM.rotation_euler = (0.0, 0.0, 0.0)     # look straight down; +X (out of the eye) right, +Y up
        CAM.location = Vector(loc) * S
    cam_d.shift_x = extra.get("shift_x", 0.0)
    meta = dict(shot=name, beams=beams)
    if extra.get("ortho"):
        cam_d.type = "ORTHO"
        cam_d.ortho_scale = extra["ortho"] * S
        meta["px_per_mm"] = SCN.render.resolution_x / extra["ortho"]
    json.dump(meta, open(os.path.join(WORK, "%s_meta.json" % name), "w"))
    SCN.render.filepath = os.path.join(WORK, "%s_raw.png" % name)
    bpy.ops.render.render(write_still=True)
    if name == "exploded":
        anchors(EXPL_LABELS, os.path.join(WORK, "exploded_labels.json"))
    print("RENDERED", name)
explode(False)
