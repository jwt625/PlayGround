"""Verification renders of the hero pigeon asset (Blender 4.2 EEVEE, headless).

Run from the project folder:
  scripts/bslot.sh -b assets/hero/pigeon_osfp_dr8.blend --python scripts/assets/render_hero_pigeon.py -- outputs/hero [views]
views (comma list, default profile,transparent,exploded,face; "review" adds macro review renders)
Writes outputs/hero/pigeon_osfp_dr8_{profile,shell_transparent,face_closeup,review_*}.png,
outputs/hero/work/pigeon_osfp_dr8_exploded_raw.png and outputs/hero/work/pigeon_osfp_dr8_exploded_labels.json
(2D anchors for scripts/assets/label_overlay.py). Pigeon frame: -Y beak, +Z up, -X = pigeon's right (TX).
"""
import json
import os
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["outputs/hero"]
OUT = ARGS[0]
VIEWS = (ARGS[1] if len(ARGS) > 1 else "profile,transparent,exploded,face").split(",")
RES = (1600, 1200)
os.makedirs(os.path.join(OUT, "work"), exist_ok=True)
SCN = bpy.context.scene
C0 = Vector((0.0, -0.006, 0.047))      # pigeon body centre (m)

SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 64
SCN.render.resolution_x, SCN.render.resolution_y = RES
SCN.render.resolution_percentage = 100
SCN.render.film_transparent = False
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Punchy"
SCN.render.image_settings.file_format = "PNG"
SCN.render.image_settings.color_mode = "RGB"
world = SCN.world or bpy.data.worlds.new("dark")
SCN.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.008, 0.010, 0.015, 1)
bg.inputs["Strength"].default_value = 1.0

SCN.use_nodes = True
nt = SCN.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type = "FOG_GLOW"
gl.quality = "HIGH"
gl.threshold = 1.2
gl.size = 7
gl.mix = -0.55
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])


def light(name, loc, power, color, size, target=C0):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.color = color
    ld.size = size
    ob = bpy.data.objects.new(name, ld)
    SCN.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).normalized().to_track_quat("-Z", "Y").to_euler()
    return ob


light("key", (-0.22, -0.22, 0.30), 9.0, (1.0, 0.96, 0.92), 0.16)
light("rim_cyan", (0.24, 0.20, 0.16), 6.0, (0.35, 0.8, 1.0), 0.20)
light("rim_magenta", (-0.20, 0.25, 0.10), 1.5, (1.0, 0.35, 0.8), 0.20)
light("fill", (0.20, -0.25, 0.08), 2.0, (0.8, 0.85, 1.0), 0.20)
light("top", (0.0, 0.0, 0.35), 2.5, (0.9, 0.95, 1.0), 0.30)

cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
SCN.collection.objects.link(cam)
SCN.camera = cam
cam_data.clip_start = 0.002
cam_data.clip_end = 5.0


def aim(loc, target, lens):
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).normalized().to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = lens
    bpy.context.view_layer.update()


def render(path):
    SCN.render.filepath = os.path.abspath(path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print("RENDERED %s in %.1f s" % (path, dt))
    return dt


def coll_objs(name):
    out = []

    def walk(c):
        out.extend(c.objects)
        for ch in c.children:
            walk(ch)
    walk(bpy.data.collections[name])
    return [o for o in out if o.type == "MESH"]


SKIN = coll_objs("shell_top") + coll_objs("shell_bottom")
_orig = {}


def xray(objs, alpha):
    cache = {}
    for ob in objs:
        m0 = ob.data.materials[0]
        _orig.setdefault(ob.name, m0)
        key = (m0.name, alpha)
        if key not in cache:
            m = m0.copy()
            m.name = "%s_xray%02d" % (m0.name, int(alpha * 100))
            m.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = alpha
            m.surface_render_method = "BLENDED"
            m.use_transparency_overlap = False
            m.use_backface_culling = True
            cache[key] = m
        ob.data.materials[0] = cache[key]
        ob.visible_shadow = False


def restore():
    for nm, m0 in _orig.items():
        ob = bpy.data.objects[nm]
        ob.data.materials[0] = m0
        ob.visible_shadow = True
    _orig.clear()


TIMES = {}
if "profile" in VIEWS:
    xray(SKIN, 0.72)
    aim((-0.46, 0.012, 0.050), Vector((0.0, 0.012, 0.046)), 85)
    TIMES["profile"] = render(os.path.join(OUT, "pigeon_osfp_dr8_profile.png"))
    restore()

if "transparent" in VIEWS:
    xray(SKIN, 0.5)
    aim((-0.215, -0.235, 0.165), Vector((0.0, -0.008, 0.050)), 72)
    camf = light("cam_fill", (-0.17, -0.20, 0.13), 5.0, (0.95, 0.95, 1.0), 0.12, target=Vector((0.0, -0.01, 0.05)))
    TIMES["transparent"] = render(os.path.join(OUT, "pigeon_osfp_dr8_shell_transparent.png"))
    bpy.data.objects.remove(camf, do_unlink=True)
    restore()

if "face" in VIEWS:
    xray(SKIN, 0.5)
    aim((-0.075, -0.115, 0.105), Vector((0.0, -0.046, 0.080)), 85)
    TIMES["face"] = render(os.path.join(OUT, "pigeon_osfp_dr8_face_closeup.png"))
    restore()

if "review" in VIEWS:
    xray(SKIN, 0.35)
    aim((-0.060, -0.020, 0.030), Vector((0.0, -0.012, 0.006)), 70)
    TIMES["feet"] = render(os.path.join(OUT, "pigeon_osfp_dr8_review_feet.png"))
    aim((0.075, -0.080, 0.035), Vector((0.0, -0.018, 0.036)), 70)
    TIMES["chest"] = render(os.path.join(OUT, "pigeon_osfp_dr8_review_chest_left.png"))
    for o in SKIN:
        o.hide_render = True
    aim((-0.050, -0.075, 0.120), Vector((0.0, -0.040, 0.082)), 80)
    TIMES["head"] = render(os.path.join(OUT, "pigeon_osfp_dr8_review_head_organs.png"))
    for o in SKIN:
        o.hide_render = False
    restore()

if "pins" in VIEWS:
    # orthographic view from below (beak at the top of the image, pigeon's left = image right); pin anchors -> JSON
    pm = json.load(open(os.path.join(os.path.dirname(bpy.data.filepath), "pigeon_osfp_dr8_pinmap.json")))
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 0.050
    cam.location = (0.0, -0.0125, -0.060)
    cam.rotation_euler = (3.14159265, 0.0, 0.0)
    bpy.context.view_layer.update()
    pins_light = light("pins_fill", (0.03, -0.03, -0.12), 0.9, (1.0, 1.0, 1.0), 0.10, target=Vector((0, -0.012, 0.0)))
    for o in SKIN:
        o.hide_render = True
    gl.mix = -1.0
    TIMES["pins"] = render(os.path.join(OUT, "work", "pigeon_osfp_dr8_feet_pins_raw.png"))
    rows = []
    for r in pm["pins"]:
        c = Vector(r["centre_mm"]) * 0.001
        v = world_to_camera_view(SCN, cam, c)
        c2 = c + Vector((r["label_dir"][0], r["label_dir"][1], 0.0)) * 0.0021
        v2 = world_to_camera_view(SCN, cam, c2)
        rows.append(dict(pin=r["pin"], cls=r["cls"], anchor=[v.x * RES[0], (1 - v.y) * RES[1]],
                         label=[v2.x * RES[0], (1 - v2.y) * RES[1]]))
    with open(os.path.join(OUT, "work", "pigeon_osfp_dr8_feet_pins_labels.json"), "w") as fh:
        json.dump(dict(res=RES, pins=rows), fh, indent=1)
    bpy.data.objects.remove(pins_light, do_unlink=True)
    for o in SKIN:
        o.hide_render = False
    gl.mix = -0.55
    cam_data.type = "PERSP"

if "flex" in VIEWS:
    xray(SKIN, 0.18)
    wings = [o for o in bpy.data.objects if o.name.startswith("wing_")]
    for o in wings:
        o.hide_render = True
    fl = light("flex_fill", (-0.15, -0.14, 0.10), 5.0, (1.0, 0.97, 0.92), 0.12, target=Vector((0, -0.02, 0.045)))
    SCN.render.resolution_x = 800
    aim((-0.082, -0.075, 0.086), Vector((0.0, -0.036, 0.068)), 55)
    TIMES["flex_neck"] = render(os.path.join(OUT, "work", "flex_panel_neck.png"))
    aim((-0.060, -0.052, 0.034), Vector((-0.004, -0.010, 0.012)), 55)
    TIMES["flex_leg"] = render(os.path.join(OUT, "work", "flex_panel_leg.png"))
    SCN.render.resolution_x = RES[0]
    bpy.data.objects.remove(fl, do_unlink=True)
    for o in wings:
        o.hide_render = False
    restore()

LABELS = [   # shared with scripts/assets/animate_hero_explode.py
    ("DSP (brain)", ["dsp_lid_nickel"]), ("MCU", ["elec_mcu_ufqfpn48_7x7"]),
    ("TX PIC + 8 MZMs", ["tx_pic_siph_die"]), ("2x CW DFB", ["tx_dfb_laser_die_LD1", "tx_dfb_laser_die_LD2"]),
    ("TX eye: 2x4 plate + TIR prism", ["tx_eye_plate_2x4", "tx_staircase_tir_prism"]),
    ("RX eye: 2x4 lenslets + Si plates", ["rx_lenslet_array_2x4"]),
    ("PIN PD array + submount", ["rx_pin_pd_array_1x8", "rx_pd_submount_aln"]), ("octal TIA", ["rx_octal_tia"]),
    ("PMIC", ["elec_pmic_qfn_4x4"]), ("buck + inductor", ["elec_buck_mp28164_qfn_2x3", "elec_power_inductor_L1"]),
    ("load switch", ["elec_load_switch_dfn_2x3"]), ("crystal (REFCLK)", ["elec_crystal_2016"]),
    ("TX flexes: 8 RF (GSSG) + 32 DC", ["flex_tx_rf_dsp_to_pic", "flex_tx_dc_pic_laser"]),
    ("RX flex: 8 RF (GSSG) + 6 DC", ["flex_rx_tia_to_dsp"]),
    ("EEPROM leg band", ["leg_tx_band_eeprom_carrier"]), ("neck flex (stacked)", ["flex_power_i2c_neck"]),
    ("leg flex: 8 pairs + GND", ["flex_tx_host_lanes_neck_leg"]), ("feet: 52 of 60 contacts", ["foot_tx_edge_pads_x26"]),
    ("belly: 4 low-speed + 4 VCC", ["pcb_belly_lowspeed_vcc_pads_x8", "pcb_belly_contact_plate"]), ("wing heatsink fins", ["wing_right_flight_feather_fins"]),
    ("shell top half", ["skin_back_body"]), ("shell belly half", ["skin_belly_body"]),
    ("beak latch", ["beak_upper_mandible"]), ("leash (pull tab)", ["leash_end_loop"]),
]

if "exploded" in VIEWS:
    def ex_of(o):
        return Vector(o.get("explode", (0, 0, 0)))

    def anc_ex(o):
        p = o.parent
        while p is not None and "explode" not in p:
            p = p.parent
        return ex_of(p) if p is not None else Vector((0, 0, 0))

    moved = []
    for ob in bpy.data.objects:
        if "explode" in ob and ob.parent is not None:
            d = ex_of(ob) - anc_ex(ob)
            if d.length > 0:
                ob.location = ob.location + d
                moved.append((ob, d))
    bpy.context.view_layer.update()
    aim((-0.360, -0.200, 0.170), Vector((0.0, 0.004, 0.058)), 55)
    TIMES["exploded"] = render(os.path.join(OUT, "work", "pigeon_osfp_dr8_exploded_raw.png"))
    rows = []
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anchor_vis import visible_anchor
    glass = {o.name for o in bpy.data.objects if o.type == "MESH" and o.data.materials and
             o.data.materials[0].name.startswith("glass")}
    for text, names in LABELS:
        a, vis = visible_anchor(SCN, cam, [bpy.data.objects[nm] for nm in names], transparent=glass)
        v = world_to_camera_view(SCN, cam, a)
        rows.append(dict(text=text, anchor=[v.x * RES[0], (1 - v.y) * RES[1]], depth=v.z, visible=vis))
        print("ANCHOR %-32s visible=%s" % (text, vis))
    with open(os.path.join(OUT, "work", "pigeon_osfp_dr8_exploded_labels.json"), "w") as fh:
        json.dump(dict(res=RES, labels=rows), fh, indent=1)
    for ob, d in moved:
        ob.location = ob.location - d

with open(os.path.join(OUT, "work", "pigeon_osfp_dr8_render_times.json"), "w") as fh:
    json.dump(TIMES, fh, indent=1)
print("RENDER_TIMES", TIMES)
