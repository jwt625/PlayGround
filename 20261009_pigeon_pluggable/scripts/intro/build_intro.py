"""Tree intro (F8): bird photo cards + simple form-factor models, clade frames, numbered pluggable lineage, hero card.

Build + previews (one Blender process):
  scripts/bslot.sh -b --factory-startup --python scripts/intro/build_intro.py -- [--no-render] [--stills-only] [--agx]
Writes assets/components/intro/intro_tree.blend (+ intro_tree.json) and previews in outputs/assets/intro/.

API for the scene agent (inside Blender, any scene):
  import sys; sys.path.insert(0, "<project>/scripts/intro"); import build_intro as BI
  root = BI.append_intro(location=(0, -800, 0))          # appends collection INTRO_tree under empty INTRO_root
  BI.key_camera(cam, root, f_start=0)                     # keys cam along the hooks: move 0-135, end on hero card
  BI.offset_animation(root, df)                           # if the intro does not start at frame 0 (model spin keys)
  BI.FADE                                                 # recommended timing (frames at 30 fps)
Everything is in INTRO_tree (cards, frames, lineage, dark backdrop 60 x 60 m), all emission materials (no lights),
so the intro renders the same at any remote location and does not light the host scene.
v3 (owner, 2026-10-10): cards show bird photo + bird name + form-factor name only; the module models were removed
(scripts/intro/form_factor_models.py is no longer used).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

INTRO_BLEND = ROOT / "assets" / "components" / "intro" / "intro_tree.blend"
PHOTOS = ROOT / "assets" / "components" / "intro" / "photos"
PREVIEW = ROOT / "outputs" / "assets" / "intro"
COLL = "INTRO_tree"
FPS = 30
F_MOVE_END = 135          # 4.5 s camera move, ends on the hero card
F_INTRO_END = 159         # last intro frame used (hold on the hero through the dissolve)
FADE = {
    "fps": FPS,
    "camera_move": (0, F_MOVE_END),
    "hold_on_hero": (F_MOVE_END, F_INTRO_END),
    "cross_dissolve": (F_MOVE_END, F_MOVE_END + 24),   # 0.8 s: intro fully gone at 5.3 s
    "note": "Cross-dissolve (not fade-through-black) from the intro camera to the street camera over frames 135-159 "
            "(4.5-5.3 s). Render the intro and the street as separate passes and mix in the compositor or with ffmpeg "
            "xfade=transition=fade:duration=0.8:offset=4.5; do not overlay the tree in the street.",
}

# ------------------------------------------------------------------ content (DevLog-006 rev 3; no years)
# slug, species label, form factor label, form-factor model key, branch, lineage number, extinct bird
LEAVES = {
    "theropod": ("THEROPOD", "1x9 SOLDERED", "one_by_nine", "L", 0, True),
    "archaeopteryx": ("ARCHAEOPTERYX", "GBIC", "gbic", "L", 1, True),
    "moa": ("MOA", "XENPAK", "xenpak", None, None, True),
    "elephant_bird": ("ELEPHANT BIRD", "CFP", "cfp", "C", None, True),
    "emu": ("EMU", "CFP2-DCO", "cfp2_dco", "C", None, False),
    "ostrich": ("OSTRICH", "XPO", "xpo", "L", 6, False),
    "chicken": ("CHICKEN", "SFP", "sfp", "L", 2, False),
    "hummingbird": ("HUMMINGBIRD", "QSFP-DD", "qsfp_dd", "L", 5, False),
    "albatross": ("ALBATROSS", "400ZR", "zr400", "C", None, False),
    "parrot": ("PARROT", "COMPATIBLE", "compatible", None, None, False),
    "cuckoo": ("CUCKOO", "AEC (COPPER)", "aec", "U", None, False),
    "passenger_pigeon": ("PASSENGER PIGEON", "QSFP+", "qsfp_plus", "L", 3, True),
    "rock_pigeon": ("ROCK PIGEON", "QSFP28", "qsfp28", "L", 4, False),
    "dodo": ("DODO", "CPO (CO-PACKAGED)", "cpo", "P", None, True),
    "cyber_pigeon": ("CYBER PIGEON", "800G OSFP DR8", "osfp", "L", 5, False),
}
GRID = {  # slug: (column, row); hero handled separately
    "theropod": (0, 0), "archaeopteryx": (1, 0),
    "moa": (0, 1), "elephant_bird": (1, 1), "emu": (2, 1), "ostrich": (3, 1),
    "chicken": (0, 2), "hummingbird": (1, 2), "albatross": (2, 2), "parrot": (3, 2),
    "passenger_pigeon": (0, 3), "rock_pigeon": (1, 3), "cuckoo": (2, 3), "dodo": (3, 3),
}

# ------------------------------------------------------------------ geometry (set units = metres)
CW, CHt = 1.0, 1.30          # card (v3: no model band)
PH_TOP = CHt / 2 - 0.04      # photo top edge (card-local z)
GX, GZ = 0.34, 0.62          # gutters
PX, PZ = CW + GX, CHt + GZ   # pitch
HERO_S = 1.5
Y_LINE = 0.012               # connectors and frames just behind the cards
Y_WALL = 0.05

C_BG = (0.016, 0.017, 0.020)
C_CARD = (0.80, 0.80, 0.78)
C_TEXT = (0.012, 0.012, 0.015)
C_LIGHT = (0.80, 0.82, 0.85)
C_FRAME = (0.20, 0.22, 0.25)
C_GOLD = (0.95, 0.60, 0.06)
C_MAG = (0.80, 0.10, 0.50)
C_COPPER = (0.78, 0.33, 0.10)
C_GREY = (0.26, 0.28, 0.31)
C_RED = (0.95, 0.10, 0.07)
BRANCH = {"L": (C_GOLD, C_TEXT), "C": (C_MAG, (1, 1, 1)), "U": (C_COPPER, (1, 1, 1)), "P": (C_GREY, (1, 1, 1)),
          None: (C_GREY, (1, 1, 1))}


def card_xz(slug):
    c, r = GRID[slug]
    return (c - 1.5) * PX, r * PZ


HERO_XZ = (0.0, 3 * PZ + CHt / 2 + GZ + HERO_S * CHt / 2)


# ------------------------------------------------------------------ helpers
_MATS: dict = {}


def emit_mat(name, color, strength=1.0):
    key = (name, tuple(round(c, 4) for c in color), strength)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(f"INTRO_{name}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    m.diffuse_color = (*color, 1)
    _MATS[key] = m
    return m


def photo_mat(slug):
    m = bpy.data.materials.new(f"INTRO_photo_{slug}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    p = PHOTOS / (f"{slug}.png" if slug == "cyber_pigeon" else f"{slug}.jpg")
    img = bpy.data.images.load(str(p), check_existing=True)
    img.pack()
    tex.image = img
    tex.interpolation = "Cubic"
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def link(o, coll, parent=None):
    coll.objects.link(o)
    if parent is not None:
        o.parent = parent
    return o


def empty(name, coll, loc=(0, 0, 0), parent=None, kind="PLAIN_AXES", size=0.2):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = kind
    e.empty_display_size = size
    e.location = loc
    return link(e, coll, parent)


def rect(name, coll, parent, cx, cz, w, h, mat, y=0.0):
    """Axis-aligned rectangle in the XZ plane (faces -Y)."""
    me = bpy.data.meshes.new(name)
    x0, x1, z0, z1 = cx - w / 2, cx + w / 2, cz - h / 2, cz + h / 2
    me.from_pydata([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UV")
    for li, uv in zip(range(4), ((0, 0), (1, 0), (1, 1), (0, 1))):
        me.uv_layers[0].data[li].uv = uv
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll, parent)


def seg(name, coll, parent, a, b, width, mat, y=Y_LINE):
    """Horizontal or vertical line segment (a, b are (x, z)) as a rectangle, with square caps."""
    (ax, az), (bx, bz) = a, b
    w = abs(bx - ax) + width
    h = abs(bz - az) + width
    return rect(name, coll, parent, (ax + bx) / 2, (az + bz) / 2, w, h, mat, y)


def polyline(name, coll, parent, pts, width, mat, y=Y_LINE, closed=False):
    pts = list(pts) + ([pts[0]] if closed else [])
    return [seg(f"{name}_{i}", coll, parent, pts[i], pts[i + 1], width, mat, y) for i in range(len(pts) - 1)]


_FONT = None


def font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = bpy.data.fonts.load("/System/Library/Fonts/Supplemental/Arial Bold.ttf", check_existing=True)
            _FONT.pack()
        except Exception:  # noqa: BLE001
            _FONT = bpy.data.fonts.get("Bfont Regular") or bpy.data.fonts.load("<builtin>")
    return _FONT


def text(name, coll, parent, body, x, z, size, color, align="CENTER", y=-0.002, valign="CENTER", strength=1.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font()
    cu.size = size
    cu.align_x = align
    cu.align_y = valign
    cu.materials.append(emit_mat("text", color, strength))
    o = bpy.data.objects.new(name, cu)
    o.location = (x, y, z)
    o.rotation_euler = (math.pi / 2, 0, 0)
    return link(o, coll, parent)


def fit(body, size_max, width):
    """Text size that keeps a one-line caps label inside `width` (Arial Bold caps ~0.72 em average advance)."""
    return min(size_max, width / (0.72 * max(len(body), 1)))


# ------------------------------------------------------------------ card
def build_card(slug, coll, root, cx, cz, s=1.0, hero=False):
    species, ff_label, ff_key, branch, lin, extinct = LEAVES[slug]
    g = empty(f"CARD_{slug}", coll, (cx, 0, cz), root)
    g.scale = (s, s, s)
    g["species"], g["form_factor"], g["branch"] = species, ff_label, branch or ""
    rect(f"{slug}_face", coll, g, 0, 0, CW, CHt, emit_mat("card", C_CARD))
    # v3 (owner, 2026-10-10): bird photo + bird name + form-factor name tag only (no module models)
    rect(f"{slug}_photo", coll, g, 0, PH_TOP - 0.345, 0.92, 0.69, photo_mat(slug), y=-0.001)
    text(f"{slug}_species", coll, g, species, 0, -0.195, fit(species, 0.12, 0.92), C_TEXT)
    bar_c, bar_t = BRANCH[branch]
    rect(f"{slug}_ffbar", coll, g, 0, -0.43, 0.92, 0.19, emit_mat("bar", bar_c), y=-0.001)
    text(f"{slug}_ff", coll, g, ff_label, 0, -0.43, fit(ff_label, 0.115, 0.86), bar_t, y=-0.003)
    if lin is not None:  # lineage badge (gold square, number)
        rect(f"{slug}_badge", coll, g, -0.46 + 0.09, PH_TOP - 0.09, 0.19, 0.19, emit_mat("gold", C_GOLD), y=-0.003)
        text(f"{slug}_badge_n", coll, g, str(lin), -0.46 + 0.09, PH_TOP - 0.09, 0.135, C_TEXT, y=-0.005)
    if extinct:
        rect(f"{slug}_ext", coll, g, 0.46 - 0.16, PH_TOP - 0.055, 0.32, 0.09, emit_mat("tagbg", (0.02, 0.02, 0.022)), y=-0.003)
        text(f"{slug}_ext_t", coll, g, "EXTINCT", 0.46 - 0.16, PH_TOP - 0.055, 0.058, (0.9, 0.9, 0.9), y=-0.005)
    if hero:
        for k, (pad, wd, stg) in enumerate(((0.03, 0.02, 1.0),)):
            polyline(f"hero_frame{k}", coll, g, [(-CW / 2 - pad, -CHt / 2 - pad), (CW / 2 + pad, -CHt / 2 - pad),
                                                 (CW / 2 + pad, CHt / 2 + pad), (-CW / 2 - pad, CHt / 2 + pad)],
                     wd, emit_mat("hero_red", C_RED, stg), y=-0.002, closed=True)
    return g


# ------------------------------------------------------------------ set
def edges(slug):
    x, z = (HERO_XZ if slug == "cyber_pigeon" else card_xz(slug))
    s = HERO_S if slug == "cyber_pigeon" else 1.0
    return dict(l=x - s * CW / 2, r=x + s * CW / 2, b=z - s * CHt / 2, t=z + s * CHt / 2, x=x, z=z)


def build_set():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    coll = bpy.data.collections.new(COLL)
    scn.collection.children.link(coll)
    root = empty("INTRO_root", coll, kind="ARROWS", size=1.0)
    root["role"] = "intro tree root: move/scale this to place the intro"

    # backdrop
    rect("INTRO_backdrop", coll, root, 0, 4.0, 60, 60, emit_mat("bg", C_BG), y=Y_WALL)

    for slug in GRID:
        build_card(slug, coll, root, *card_xz(slug))
    build_card("cyber_pigeon", coll, root, *HERO_XZ, s=HERO_S, hero=True)

    E = {s: edges(s) for s in LEAVES}
    half_gx, half_gz = GX / 2, GZ / 2
    gold, mag = emit_mat("gold", C_GOLD), emit_mat("mag", C_MAG)
    W_L, W_C = 0.035, 0.028
    # pluggable lineage (gold), routed in the gutters behind the cards
    t, a = E["theropod"], E["archaeopteryx"]
    polyline("lin01", coll, root, [(t["r"], t["z"]), (a["l"], a["z"])], W_L, gold)
    ch = E["chicken"]
    xg = a["l"] - half_gx
    polyline("lin12", coll, root, [(a["x"], a["t"]), (a["x"], a["t"] + half_gz), (xg, a["t"] + half_gz),
                                   (xg, ch["z"]), (ch["r"], ch["z"])], W_L, gold)
    pp, rp, hb, hero, os_ = E["passenger_pigeon"], E["rock_pigeon"], E["hummingbird"], E["cyber_pigeon"], E["ostrich"]
    polyline("lin23", coll, root, [(ch["x"], ch["t"]), (pp["x"], pp["b"])], W_L, gold)
    polyline("lin34", coll, root, [(pp["r"], pp["z"]), (rp["l"], rp["z"])], W_L, gold)
    polyline("lin45_dd", coll, root, [(rp["x"], rp["b"]), (hb["x"], hb["t"])], W_L, gold)
    polyline("lin45_osfp", coll, root, [(rp["x"], rp["t"]), (rp["x"], hero["b"])], W_L, gold)
    xm = os_["r"] + GX * 0.9
    polyline("lin56", coll, root, [(hero["r"], hero["z"]), (xm, hero["z"]), (xm, os_["z"]), (os_["r"], os_["z"])],
             W_L, gold)
    # coherent branch (magenta)
    eb, emu, al = E["elephant_bird"], E["emu"], E["albatross"]
    polyline("coh1", coll, root, [(eb["r"], eb["z"] - 0.25), (emu["l"], emu["z"] - 0.25)], W_C, mag)
    polyline("coh2", coll, root, [(emu["x"] + 0.25, emu["t"]), (al["x"] + 0.25, al["b"])], W_C, mag)

    # clade frames (faint) + legends
    fm = emit_mat("frame", C_FRAME)
    m1, m2 = 0.10, 0.19

    def box(name, l, r, b, t_, label, lx, lz, align="LEFT"):
        polyline(name, coll, root, [(l, b), (r, b), (r, t_), (l, t_)], 0.012, fm, y=Y_LINE + 0.004, closed=True)
        text(f"{name}_label", coll, root, label, lx, lz, 0.085, (0.62, 0.65, 0.70), align=align, y=-0.002)

    mo, ost = E["moa"], E["ostrich"]
    box("clade_palaeognaths", mo["l"] - m1, ost["r"] + m1, mo["b"] - m1, mo["t"] + m1,
        "PALAEOGNATHS", mo["l"] - m1, mo["b"] - m1 - 0.075)
    box("clade_galloanserae", ch["l"] - m1, ch["r"] + m1, ch["b"] - m1, ch["t"] + m1,
        "GALLOANSERAE", ch["l"] - m1, ch["b"] - m1 - 0.075)
    dd = E["dodo"]
    # Neoaves: unresolved row (hummingbird, albatross, parrot) + Columbaves (row 3 + hero), one outline
    pa = E["parrot"]
    pts = [(hb["l"] - m2, hb["b"] - m2), (pa["r"] + m2, hb["b"] - m2), (pa["r"] + m2, dd["t"] + m2),
           (hero["r"] + m2, dd["t"] + m2), (hero["r"] + m2, hero["t"] + m2), (hero["l"] - m2, hero["t"] + m2),
           (hero["l"] - m2, pp["t"] + m2), (pp["l"] - m2, pp["t"] + m2), (pp["l"] - m2, pp["b"] - m2),
           (hb["l"] - m2, pp["b"] - m2)]
    polyline("clade_neoaves", coll, root, pts, 0.012, fm, y=Y_LINE + 0.004, closed=True)
    text("clade_neoaves_label", coll, root, "NEOAVES (UNRESOLVED)", hb["l"] - m2, hb["b"] - m2 - 0.075, 0.085,
         (0.62, 0.65, 0.70), align="LEFT")
    pts = [(pp["l"] - m1, pp["b"] - m1), (dd["r"] + m1, pp["b"] - m1), (dd["r"] + m1, dd["t"] + m1),
           (hero["r"] + m1, dd["t"] + m1), (hero["r"] + m1, hero["t"] + m1), (hero["l"] - m1, hero["t"] + m1),
           (hero["l"] - m1, pp["t"] + m1), (pp["l"] - m1, pp["t"] + m1)]
    polyline("clade_columbaves", coll, root, pts, 0.012, fm, y=Y_LINE + 0.004, closed=True)
    text("clade_columbaves_label", coll, root, "COLUMBAVES", pp["l"] - m1, pp["t"] + m1 + 0.075, 0.085,
         (0.62, 0.65, 0.70), align="LEFT")

    # title + legend in the empty space right of the root cards (row 0)
    lx = E["archaeopteryx"]["r"] + 0.13
    text("title1", coll, root, "BIRD TREE", lx, 0.52, 0.135, (0.95, 0.95, 0.95), align="LEFT")
    text("title2", coll, root, "+ PLUGGABLE", lx, 0.36, 0.095, C_GOLD, align="LEFT")
    text("title3", coll, root, "LINEAGE", lx, 0.245, 0.095, C_GOLD, align="LEFT")
    for k, (lab, col) in enumerate((("GOLD = LINEAGE", C_GOLD), ("MAGENTA = COHERENT", (0.95, 0.35, 0.70)),
                                    ("FRAMES = CLADES", (0.62, 0.65, 0.70)), ("TRAIT-MATCHED,", (0.62, 0.65, 0.70)),
                                    ("SIMPLIFIED", (0.62, 0.65, 0.70)))):
        text(f"legend{k}", coll, root, lab, lx, 0.05 - k * 0.105, 0.064, col, align="LEFT")

    hooks(coll, root)
    return scn, coll, root


# ------------------------------------------------------------------ camera hooks
LENS = 50.0
KEYS = (("start", 0), ("a", 45), ("b", 92), ("end", F_MOVE_END))   # hook name, frame (30 fps)
FRAME_W = {"start": 3.5, "a": 3.9, "b": 3.6, "end": 2.45}   # visible width (set m) at the card plane


def _cam_dist(width, lens=LENS):
    return width / 2 / math.tan(math.atan(18.0 / lens))   # sensor fit horizontal, 36 mm


def hooks(coll, root):
    """HOOK_intro_cam_<k> = camera location, HOOK_intro_tgt_<k> = look-at point, k in start / a / b / end (both children of INTRO_root).
    Content is centred in the top 82% of a 4:5 frame (bottom 18% kept for captions)."""
    hz = HERO_XZ[1]
    tgt = {"start": Vector((-0.92, 0, 0.39 * PZ)), "a": Vector((-0.70, 0, 1.45 * PZ)), "b": Vector((-0.30, 0, 2.67 * PZ)),
           "end": Vector((0.0, 0, hz))}
    side = {"start": -0.15, "a": 0.0, "b": 0.15, "end": 0.0}   # small lateral parallax for the 3D models
    lift = {"start": -0.2, "a": -0.2, "b": -0.15, "end": 0.0}  # camera slightly below: looking up the tree
    out = {}
    for k, fk in KEYS:
        w = FRAME_W[k]
        h = w * 5 / 4
        t = tgt[k] - Vector((0, 0, 0.09 * h))           # aim below the content: content sits in the top 82%
        d = _cam_dist(w)
        c = Vector((t.x + side[k], -d, t.z + lift[k]))
        ce = empty(f"HOOK_intro_cam_{k}", coll, tuple(c), root, kind="CONE", size=0.3)
        te = empty(f"HOOK_intro_tgt_{k}", coll, tuple(t), root, kind="SPHERE", size=0.15)
        ce["lens_mm"], ce["frame"] = LENS, fk
        out[k] = (ce, te)
    return out


def _hook_world(root, name):
    for o in root.children_recursive:
        if o.name.split(".")[0] == name:
            return o.matrix_world.translation.copy()
    raise KeyError(name)


def key_camera(cam, root, f_start=0, speed=1.0, target_name=None):
    """Key `cam` (and a Track-To target empty) through the intro hooks of `root` (INTRO_root, possibly moved/scaled).
    Keys at f_start + KEYS frames / speed (start 0, a 45, b 92, end 135); holds after. Returns the target empty."""
    scn = bpy.context.scene
    tname = target_name or f"{cam.name}_intro_target"
    tgt = bpy.data.objects.get(tname) or bpy.data.objects.new(tname, None)
    if tgt.name not in scn.collection.objects and not tgt.users_collection:
        scn.collection.objects.link(tgt)
    con = next((c for c in cam.constraints if c.type == "TRACK_TO" and c.name == "INTRO_track"), None)
    if con is None:
        con = cam.constraints.new("TRACK_TO")
        con.name = "INTRO_track"
        con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    con.target = tgt
    cam.data.lens = LENS
    cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
    for k, f in KEYS:
        fr = f_start + round(f / speed)
        cam.location = _hook_world(root, f"HOOK_intro_cam_{k}")
        tgt.location = _hook_world(root, f"HOOK_intro_tgt_{k}")
        cam.keyframe_insert("location", frame=fr)
        tgt.keyframe_insert("location", frame=fr)
    for ob in (cam, tgt):
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
            fc.keyframe_points[0].easing = "EASE_IN"
    return tgt


def offset_animation(root, df):
    """Shift the intro's own keys (model yaw) by df frames."""
    for o in [root, *root.children_recursive]:
        ad = o.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.co.x += df
                    kp.handle_left.x += df
                    kp.handle_right.x += df


def append_intro(blend=INTRO_BLEND, location=(0.0, -800.0, 0.0), scale=1.0, scene=None):
    """Append collection INTRO_tree from intro_tree.blend into `scene` and place INTRO_root. Returns INTRO_root."""
    scene = scene or bpy.context.scene
    with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
        dst.collections = [COLL]
    coll = dst.collections[0]
    scene.collection.children.link(coll)
    root = next(o for o in coll.objects if o.name.split(".")[0] == "INTRO_root")
    root.location = location
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return root


# ------------------------------------------------------------------ preview scene + renders
def preview_scene(scn, root, agx=False):
    cam = bpy.data.objects.new("INTRO_cam", bpy.data.cameras.new("INTRO_cam"))
    pc = bpy.data.collections.new("INTRO_preview")
    scn.collection.children.link(pc)
    pc.objects.link(cam)
    scn.camera = cam
    bpy.context.view_layer.update()
    tgt = key_camera(cam, root, 0)
    for c in list(tgt.users_collection):
        c.objects.unlink(tgt)
    pc.objects.link(tgt)
    w = bpy.data.worlds.new("INTRO_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*C_BG, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scn.world = w
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = 8
    scn.render.resolution_x, scn.render.resolution_y = int(os.environ.get("INTRO_W", "540")), int(os.environ.get("INTRO_H", "674"))
    scn.render.fps = FPS
    scn.frame_start, scn.frame_end = 0, F_INTRO_END
    scn.view_settings.view_transform = "AgX" if agx else "Standard"
    scn.view_settings.look = "None"
    scn.render.image_settings.file_format = "PNG"
    return cam


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    agx = "--agx" in argv
    scn, coll, root = build_set()
    preview_scene(scn, root, agx=agx)
    INTRO_BLEND.parent.mkdir(parents=True, exist_ok=True)
    side = {"fade": FADE, "collection": COLL, "root": "INTRO_root", "lens_mm": LENS,
            "hooks": {o.name: list(o.matrix_world.translation) for o in root.children if o.name.startswith("HOOK_")},
            "hero_card_centre": [HERO_XZ[0], 0.0, HERO_XZ[1]], "cards": {s: list(edges(s).values()) for s in LEAVES},
            "colour_management": "previews use Standard; under AgX the emission colours lose some contrast"}
    (INTRO_BLEND.with_suffix(".json")).write_text(json.dumps(side, indent=1))
    bpy.ops.wm.save_as_mainfile(filepath=str(INTRO_BLEND), compress=True)
    print("saved", INTRO_BLEND)
    if "--no-render" in argv:
        return
    PREVIEW.mkdir(parents=True, exist_ok=True)
    tag = ("_agx" if agx else "") + ("" if scn.render.resolution_x == 540 else f"_{scn.render.resolution_x}x{scn.render.resolution_y}")
    for name, f in (("start", 0), ("mid", 68), ("end", F_MOVE_END)):
        scn.frame_set(f)
        scn.render.filepath = str(PREVIEW / f"intro_tree_{name}{tag}.png")
        bpy.ops.render.render(write_still=True)
        print("still", scn.render.filepath)
    if "--stills-only" in argv:
        return
    frames = Path(os.environ.get("INTRO_FRAMES", str(PREVIEW / "_frames")))
    frames.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(frames / "f_")
    bpy.ops.render.render(animation=True)
    mp4 = PREVIEW / f"intro_tree_preview{tag}_20261010.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-start_number", "0", "-i", str(frames / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(mp4)], check=True)
    print("mp4", mp4)
    for f in frames.glob("f_*.png"):   # own intermediates
        f.unlink()
    if not any(frames.iterdir()):
        frames.rmdir()


if __name__ == "__main__":
    main()
