"""Camera-attached surveillance HUD for the pigeon POV (workstream E, DevLog-013 section 8; owner item F4, DevLog-015).

Generic "battlefield intel / stealth game" look built from scratch: target brackets that lock on a world object, ID card
with typing effect, range / bearing / heading readouts, heading compass tape, heading-up proximity radar with blips,
link bars, scrolling telemetry ticker, blinking alert banner, scanlines + vignette. Parody text only, no game names,
logos or fonts (system Courier New Bold). Everything is emission / transparent geometry on one holder empty parented to the
camera (5 cm in front of the lens), so it renders at the fast tier and follows lens changes (holder scale driver).

HUD units: x in [-1, 1] across the frame width, y in [-A, A] with A = res_y / res_x. The bottom 18 % of the frame
(y < -A + 0.36 A) is left free for captions; target brackets are clamped above it.
"""
import math
import os
import random

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from .materials import key_obj_alpha
from .nodes import NB, new_material, set_blended

FONT_PATH = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
FONT_SCALE = 1.3            # Courier glyphs are small for their em; fonts with overlapping contours (Andale, SF Mono) drop glyph stems
HUD_D = 0.05                 # holder distance in front of the lens (m); camera clip_start must be < 0.05
CAPTION_FRAC = 0.18
STYLES = {
    "intel": dict(main=(0.35, 1.0, 0.55), accent=(1.0, 0.78, 0.25), alert=(1.0, 0.12, 0.08), dark=0.6, tint=(0.05, 0.25, 0.1)),
    "stealth": dict(main=(0.35, 0.85, 1.0), accent=(0.92, 0.95, 1.0), alert=(1.0, 0.15, 0.1), dark=0.6, tint=(0.03, 0.15, 0.25)),
}
DEFAULT_CARD = [("SUBJECT", "GARY"), ("CLEARANCE", "COFFEE ONLY"), ("THREAT", "CAFFEINATED"), ("AFFIL.", "HUMAN (PROB.)"),
                ("HEART RATE", "96 BPM HIGH"), ("STATUS", "OBSERVED")]
DEFAULT_TICKER = ("TEMP 41.2 C  |  VCC 3.29 V  |  TX BIAS 62 mA  |  TX PWR +2.1 dBm  |  RX PWR -3.4 dBm  |  LANES 8/8 UP  |  "
                  "SUBJECT HEART RATE 96 BPM  |  COFFEE LEVEL 37 %  |  FLOCK MESH 14 NODES  |  UPLINK OK  |  ")


def _rgba(c):
    return (*c[:3], 1.0)


# ==================================================================== materials
_M = {}


def _ink(color, strength=2.2, clip=None):
    """Additive ink. Typing reveal: Generated X <= object colour R. Fade: object colour alpha. clip=(holder, xmin, xmax)."""
    key = ("ink", tuple(round(c, 3) for c in color), strength, None if clip is None else (clip[0].name, clip[1], clip[2]))
    if key in _M:
        return _M[key]
    m, b, out = new_material("HUD_ink_%d" % len(_M))
    gen = b.sep(b.texcoord("Generated"))[0]
    rcol = b.node("ShaderNodeSeparateColor")
    b.link(b.objinfo("Color"), rcol.inputs[0])
    vis = b.math("LESS_THAN", gen, b.add(rcol.outputs[0], 0.0005))
    k = b.mul(b.mul(vis, b.objinfo("Alpha")), strength)
    if clip is not None:
        tc = b.node("ShaderNodeTexCoord")
        tc.object = clip[0]
        x = b.sep(tc.outputs["Object"])[0]
        k = b.mul(k, b.mul(b.math("GREATER_THAN", x, clip[1]), b.math("LESS_THAN", x, clip[2])))
    b.link(b.add_shader(b.transparent(), b.emission(_rgba(color), k)), out.inputs["Surface"])
    set_blended(m)
    _M[key] = m
    return m


def _panel(dark, tint):
    key = ("panel", dark, tuple(tint))
    if key in _M:
        return _M[key]
    m, b, out = new_material("HUD_panel_%d" % len(_M))
    a = b.objinfo("Alpha")
    t = b.sub(1.0, b.mul(dark, a))
    tr = b.transparent(b.comb(t, t, t))
    b.link(b.add_shader(tr, b.emission(_rgba(tint), b.mul(a, 0.15))), out.inputs["Surface"])
    set_blended(m)
    _M[key] = m
    return m


def _scan(tint, A, lines=220.0, dark=0.22, vignette=0.55):
    """Full-frame multiply layer: scanlines (moving slowly) + radial vignette + faint tint."""
    m, b, out = new_material("HUD_scan_%d" % len(_M))
    _M[("scan", len(_M))] = m
    t = b.time()
    p = b.sep(b.texcoord("Object"))
    s = b.add(b.mul(b.math("SINE", b.mul(b.sub(p[1], b.mul(t, 0.02)), lines)), 0.5), 0.5)
    sc = b.sub(1.0, b.mul(dark, s))
    r2 = b.add(b.mul(p[0], p[0]), b.mul(b.math("DIVIDE", p[1], A), b.math("DIVIDE", p[1], A)))
    vg = b.sub(1.0, b.mul(vignette, b.smooth(r2, 0.35, 1.6)))
    a = b.objinfo("Alpha")
    f = b.lerp(1.0, b.mul(sc, vg), a)
    b.link(b.add_shader(b.transparent(b.comb(f, f, f)), b.emission(_rgba(tint), b.mul(a, 0.03))), out.inputs["Surface"])
    set_blended(m)
    return m


# ==================================================================== 2D geometry (HUD units, z = 0 on the holder)
class G2:
    def __init__(self):
        self.v, self.f = [], []

    def quad(self, a, b, c, d):
        o = len(self.v)
        self.v += [(a[0], a[1], 0.0), (b[0], b[1], 0.0), (c[0], c[1], 0.0), (d[0], d[1], 0.0)]
        self.f.append((o, o + 1, o + 2, o + 3))

    def rect(self, x0, y0, x1, y1):
        self.quad((x0, y0), (x1, y0), (x1, y1), (x0, y1))

    def line(self, p0, p1, w):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy) or 1e-9
        nx, ny = -dy / L * w / 2, dx / L * w / 2
        self.quad((p0[0] - nx, p0[1] - ny), (p1[0] - nx, p1[1] - ny), (p1[0] + nx, p1[1] + ny), (p0[0] + nx, p0[1] + ny))

    def frame(self, x0, y0, x1, y1, w):
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            self.line(a, b, w)

    def arc(self, cx, cy, r, w, a0=0.0, a1=2 * math.pi, seg=48):
        for i in range(seg):
            t0 = a0 + (a1 - a0) * i / seg
            t1 = a0 + (a1 - a0) * (i + 1) / seg
            ro, ri = r + w / 2, r - w / 2
            self.quad((cx + ri * math.cos(t0), cy + ri * math.sin(t0)), (cx + ro * math.cos(t0), cy + ro * math.sin(t0)),
                      (cx + ro * math.cos(t1), cy + ro * math.sin(t1)), (cx + ri * math.cos(t1), cy + ri * math.sin(t1)))

    def fan(self, cx, cy, r, a0, a1, seg=8):
        for i in range(seg):
            t0 = a0 + (a1 - a0) * i / seg
            t1 = a0 + (a1 - a0) * (i + 1) / seg
            self.quad((cx, cy), (cx + r * math.cos(t0), cy + r * math.sin(t0)), (cx + r * math.cos(t1), cy + r * math.sin(t1)), (cx, cy))


# ==================================================================== HUD builder
class _Hud:
    def __init__(self, scene, cam, f0, f1, style, seed, collection):
        self.scn, self.cam, self.f0, self.f1 = scene, cam, f0, f1
        self.S = STYLES[style]
        self.rng = random.Random(seed)
        self.A = scene.render.resolution_y / scene.render.resolution_x
        self.ysafe = -self.A + 2 * self.A * CAPTION_FRAC + 0.02
        self.coll = collection or scene.collection
        self.objs = []
        h = bpy.data.objects.new("HUD_holder", None)
        self.coll.objects.link(h)
        h.empty_display_size = 0.01
        h.parent = cam
        h.location = (0, 0, -HUD_D)
        for i in range(3):
            fc = h.driver_add("scale", i)
            d = fc.driver
            d.type = "SCRIPTED"
            for nm, path in (("lens", "lens"), ("sw", "sensor_width")):
                v = d.variables.new()
                v.name = nm
                v.type = "SINGLE_PROP"
                v.targets[0].id_type = "CAMERA"
                v.targets[0].id = cam.data
                v.targets[0].data_path = path
            d.expression = "%g*sw*0.5/lens" % HUD_D
        self.holder = h
        self.font = bpy.data.fonts.load(FONT_PATH, check_existing=True) if os.path.exists(FONT_PATH) else None
        self.ink = _ink(self.S["main"])
        self.acc = _ink(self.S["accent"])
        self.alert = _ink(self.S["alert"], 2.6)
        self.panel = _panel(self.S["dark"], self.S["tint"])

    # ---------------------------------------------------------------- object helpers
    def _reg(self, o, parent=None, boot=True):
        o.parent = parent or self.holder
        o.visible_shadow = False
        self.objs.append(o)
        if boot:
            fb = self.f0 + self.rng.randint(0, 6)
            key_obj_alpha(o, self.f0 - 1, 0.0, "CONSTANT")
            key_obj_alpha(o, fb, 1.0, "CONSTANT")
            key_obj_alpha(o, fb + 1, 0.0, "CONSTANT")
            key_obj_alpha(o, fb + 2, 1.0, "CONSTANT")
        return o

    def mesh(self, name, g, mat, loc=(0, 0, 0), parent=None, z=0.0, boot=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(g.v, [], g.f)
        me.materials.append(mat)
        o = bpy.data.objects.new(name, me)
        self.coll.objects.link(o)
        o.location = (loc[0], loc[1], z)
        return self._reg(o, parent, boot)

    def text(self, body, size, loc, mat, align="LEFT", name="HUD_txt", parent=None, boot=True):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size * FONT_SCALE
        cu.align_x = align
        cu.align_y = "CENTER"
        if self.font:
            cu.font = self.font
        cu.materials.append(mat)
        o = bpy.data.objects.new(name, cu)
        self.coll.objects.link(o)
        o.location = (loc[0], loc[1], 0.0)
        return self._reg(o, parent, boot)

    def windows(self, seq, size, loc, mat, align="LEFT", name="HUD_val", parent=None):
        """seq: [(frame, string)] -> one text object per distinct consecutive string, visible only in its window."""
        merged = []
        for f, s in seq:
            if not merged or merged[-1][1] != s:
                merged.append((f, s))
        out = []
        for i, (f, s) in enumerate(merged):
            fe = merged[i + 1][0] if i + 1 < len(merged) else self.f1 + 1
            o = self.text(s, size, loc, mat, align, name, parent, boot=False)
            key_obj_alpha(o, self.f0 - 1, 0.0, "CONSTANT")
            key_obj_alpha(o, max(f, self.f0 + 4), 1.0, "CONSTANT")
            key_obj_alpha(o, fe, 0.0, "CONSTANT")
            out.append(o)
        return out

    def type_in(self, o, f_start, cps):
        """Typing reveal: object colour R steps 0 -> 1 at cps characters per second (monospace)."""
        n = max(1, len(o.data.body))
        step = max(1, round(self.scn.render.fps / cps))
        o.color = (0.0, 1, 1, o.color[3])
        o.keyframe_insert("color", index=0, frame=self.f0 - 1)
        for i in range(n + 1):
            o.color[0] = i / n
            o.keyframe_insert("color", index=0, frame=f_start + i * step)
        for fc in o.animation_data.action.fcurves:
            if fc.data_path == "color" and fc.array_index == 0:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
        return f_start + n * step

    def blink(self, o, fa, fb, period=6, duty=0.6):
        f = fa
        while f < fb:
            key_obj_alpha(o, f, 1.0, "CONSTANT")
            key_obj_alpha(o, f + max(1, int(period * duty)), 0.0, "CONSTANT")
            f += period


def _sample(scn, cam, target, blips, f0, f1, step):
    frames = list(range(f0, f1 + 1, step))
    if frames[-1] != f1:
        frames.append(f1)
    data = []
    fcur = scn.frame_current
    for f in frames:
        scn.frame_set(f)
        mw = cam.matrix_world
        loc = mw.translation.copy()
        fwd = (mw.to_3x3() @ Vector((0, 0, -1))).normalized()
        right = (mw.to_3x3() @ Vector((1, 0, 0))).normalized()
        d = {"f": f, "cam": loc, "fwd": fwd, "right": right}
        if target is not None:
            corners = [target.matrix_world @ Vector(c) for c in target.bound_box]
            if target.type == "EMPTY" or all((corners[0] - c).length < 1e-6 for c in corners):
                p = target.matrix_world.translation
                corners = [p + Vector((dx, dy, dz)) for dx in (-0.3, 0.3) for dy in (-0.3, 0.3) for dz in (0.0, 1.75)]
            uv = [world_to_camera_view(scn, cam, c) for c in corners]
            d["tgt"] = sum(corners, Vector()) / 8.0
            d["uv"] = [u for u in uv if u.z > 0]
        d["blips"] = [(b.matrix_world.translation.copy() if hasattr(b, "matrix_world") else Vector(b)) for b in (blips or [])]
        data.append(d)
    scn.frame_set(fcur)
    return data


def hud_overlay(scene, cam, target_obj, f0, f1, style="intel", seed=0, blips=None, card=None, ticker=None, target_name="GARY",
                step=2, lock_at=None, every=4, collection=None, radar_range=8.0, feed_label="RX-L // SURVEILLANCE",
                alert_text="ALERT // SUBJECT IN VIEW", typing_cps=45.0, scanlines=True):
    """Build the HUD for frames [f0, f1] on `cam`, locking on `target_obj` (any object; empties use a 0.6 x 0.6 x 1.75 m
    box). blips: objects or world points shown on the radar (birds). card: [(field, value)] for the ID panel. Values
    (range, bearing, heading) are measured from the scene (sampled every `step` frames; text values change every
    `every` frames). Returns the created objects; [0] is the holder empty (parented to cam). All objects are hidden
    outside [f0, f1] (hide_render keys)."""
    H = _Hud(scene, cam, f0, f1, style, seed, collection)
    S, A, rng = H.S, H.A, H.rng
    lock_at = lock_at if lock_at is not None else f0 + 14
    data = _sample(scene, cam, target_obj, blips, f0, f1, step)
    fps = scene.render.fps

    # ---------------------------------------------------------------- scanlines + vignette (multiply layer, behind the ink)
    if scanlines:
        g = G2()
        g.rect(-1.05, -A - 0.05, 1.05, A + 0.05)
        H.mesh("HUD_scan", g, _scan(S["tint"], A), z=-0.002)
    # ---------------------------------------------------------------- frame corners, reticle, feed label, REC
    g = G2()
    m = 0.04
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * (1 - m), sy * (A - m) if sy > 0 else H.ysafe + 0.0
            g.line((cx, cy), (cx - sx * 0.12, cy), 0.006)
            g.line((cx, cy), (cx, cy - sy * 0.12), 0.006)
    g.line((-0.05, 0), (-0.015, 0), 0.004)
    g.line((0.015, 0), (0.05, 0), 0.004)
    g.line((0, -0.05), (0, -0.015), 0.004)
    g.line((0, 0.015), (0, 0.05), 0.004)
    H.mesh("HUD_frame", g, H.ink)
    H.text(feed_label, 0.036, (-0.94, A - 0.1), H.ink, name="HUD_feed")
    g = G2()
    g.fan(0, 0, 0.014, 0, 2 * math.pi, 16)
    rec = H.mesh("HUD_rec_dot", g, H.alert, loc=(-0.92, A - 0.17), boot=False)
    H.blink(rec, f0, f1 + 1, period=16, duty=0.55)
    H.text("REC", 0.04, (-0.895, A - 0.17), H.alert, name="HUD_rec")
    H.windows([(d["f"], "T+%05.2f" % ((d["f"] - f0) / fps + 36.0)) for d in data[::max(1, every // max(1, step))]], 0.04,
              (-0.78, A - 0.17), H.ink, name="HUD_clock")

    # ---------------------------------------------------------------- compass tape (heading), clipped to |x| < 0.42
    def az(v):
        return math.degrees(math.atan2(v.x, v.y)) % 360.0
    heads, prev = [], None
    for d in data:
        h = az(d["fwd"])
        if prev is not None:
            while h - prev > 180:
                h -= 360
            while h - prev < -180:
                h += 360
        heads.append(h)
        prev = h
    k = 0.007
    clip = _ink(S["main"], 2.2, (H.holder, -0.34, 0.34))
    tape = bpy.data.objects.new("HUD_tape_root", None)
    H.coll.objects.link(tape)
    tape.location = (0, A - 0.14, 0)
    H._reg(tape, boot=False)
    g = G2()
    lo, hi = int(min(heads)) - 120, int(max(heads)) + 120
    for deg in range(lo - lo % 5, hi, 5):
        L = 0.05 if deg % 30 == 0 else (0.03 if deg % 10 == 0 else 0.015)
        g.line((deg * k, 0), (deg * k, -L), 0.004)
    H.mesh("HUD_tape_ticks", g, clip, parent=tape, boot=False)
    names = {0: "N", 90: "E", 180: "S", 270: "W"}
    for deg in range(lo - lo % 30, hi, 30):
        lab = names.get(deg % 360, "%03d" % (deg % 360))
        H.text(lab, 0.04, (deg * k, 0.04), clip, align="CENTER", name="HUD_tape_lbl", parent=tape, boot=False)
    for d, h in zip(data, heads):
        tape.location.x = -h * k
        tape.keyframe_insert("location", index=0, frame=d["f"])
    g = G2()
    g.line((-0.34, A - 0.2), (0.34, A - 0.2), 0.003)
    g.line((0, A - 0.2), (-0.015, A - 0.225), 0.004)
    g.line((0, A - 0.2), (0.015, A - 0.225), 0.004)
    H.mesh("HUD_tape_base", g, H.ink)
    H.windows([(d["f"], "HDG %03d" % (round(h) % 360)) for d, h in zip(data, heads)][::max(1, every // max(1, step))], 0.042,
              (0, A - 0.265), H.acc, align="CENTER", name="HUD_hdg")

    # ---------------------------------------------------------------- left readouts panel
    g = G2()
    g.rect(-0.97, 0.12, -0.5, 0.62)
    H.mesh("HUD_left_panel", g, H.panel, z=-0.001)
    g = G2()
    g.frame(-0.97, 0.12, -0.5, 0.62, 0.004)
    H.mesh("HUD_left_frame", g, H.ink)
    H.text("TARGET TRACK", 0.042, (-0.95, 0.575), H.acc, name="HUD_lt_hdr")
    rows = []
    for d in data[::max(1, every // max(1, step))]:
        if "tgt" not in d:
            continue
        v = d["tgt"] - d["cam"]
        rows.append((d["f"], "RNG %5.2f m" % v.length, "BRG %05.1f" % (math.degrees(math.atan2(v.x, v.y)) % 360.0),
                     "ELV %+5.1f" % math.degrees(math.atan2(v.z, math.hypot(v.x, v.y))), "ALT %4.2f m" % d["cam"].z))
    for i in range(4):
        H.windows([(r[0], r[1 + i]) for r in rows], 0.045, (-0.95, 0.49 - 0.095 * i), H.ink, name="HUD_lt_%d" % i)

    # ---------------------------------------------------------------- target brackets (acquire -> lock), label, "!" flash
    bl = []
    for d in data:
        if not d.get("uv"):
            bl.append(None)
            continue
        xs = [(u.x - 0.5) * 2 for u in d["uv"]]
        ys = [(u.y - 0.5) * 2 * A for u in d["uv"]]
        x0, x1 = max(-0.98, min(xs) - 0.03), min(0.98, max(xs) + 0.03)
        y0, y1 = max(H.ysafe, min(ys) - 0.03), min(A - 0.3, max(ys) + 0.03)
        if y1 < y0 + 0.05:
            y1 = y0 + 0.05
        bl.append((x0, y0, x1, y1))
    corners = {}
    for tag, mat in (("acq", H.ink), ("lock", H.alert)):
        for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1)):
            g = G2()
            sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
            g.line((0, 0), (sx * 0.06, 0), 0.007)
            g.line((0, 0), (0, sy * 0.06), 0.007)
            o = H.mesh("HUD_br_%s_%d%d" % (tag, cx, cy), g, mat, boot=False)
            corners[(tag, cx, cy)] = o
    for tag in ("acq", "lock"):
        for cx in (0, 1):
            for cy in (0, 1):
                o = corners[(tag, cx, cy)]
                key_obj_alpha(o, f0 - 1, 0.0, "CONSTANT")
                key_obj_alpha(o, f0 + 2 if tag == "acq" else lock_at, 1.0, "CONSTANT")
                if tag == "acq":
                    key_obj_alpha(o, lock_at, 0.0, "CONSTANT")
                last = None
                for d, b in zip(data, bl):
                    if b is None:
                        b = last or (-0.2, -0.3, 0.2, 0.3)
                    last = b
                    u = min(1.0, max(0.0, (d["f"] - f0) / max(1, lock_at - f0)))
                    e = 1 - (1 - u) ** 3
                    full = (-0.85, H.ysafe + 0.05, 0.85, A - 0.35)
                    bb = [full[i] + (b[i] - full[i]) * e for i in range(4)] if tag == "acq" else b
                    o.location = (bb[2] if cx else bb[0], bb[3] if cy else bb[1], 0)
                    o.keyframe_insert("location", frame=d["f"])
    lab_root = bpy.data.objects.new("HUD_tgt_label_root", None)
    H.coll.objects.link(lab_root)
    H._reg(lab_root, boot=False)
    last = None
    for d, b in zip(data, bl):
        b = b or last or (-0.2, -0.3, 0.2, 0.3)
        last = b
        lab_root.location = (b[0], b[3] + 0.045, 0)
        lab_root.keyframe_insert("location", frame=d["f"])
    t = H.text("TGT-01 // %s" % target_name, 0.042, (0, 0.0), H.ink, name="HUD_tgt_name", parent=lab_root, boot=False)
    key_obj_alpha(t, f0 - 1, 0.0, "CONSTANT")
    key_obj_alpha(t, f0 + 6, 1.0, "CONSTANT")
    H.type_in(t, f0 + 6, typing_cps)
    lk = H.text("LOCKED", 0.042, (0, 0.055), H.alert, name="HUD_locked", parent=lab_root, boot=False)
    key_obj_alpha(lk, f0 - 1, 0.0, "CONSTANT")
    H.blink(lk, lock_at, min(f1 + 1, lock_at + 30), period=6, duty=0.6)
    key_obj_alpha(lk, min(f1 + 1, lock_at + 30), 1.0, "CONSTANT")
    ex = H.text("!", 0.18, (0.0, 0.2), H.alert, align="CENTER", name="HUD_excl", parent=lab_root, boot=False)
    ex.location.x = 0.05
    key_obj_alpha(ex, f0 - 1, 0.0, "CONSTANT")
    key_obj_alpha(ex, lock_at, 1.0, "CONSTANT")
    key_obj_alpha(ex, lock_at + 12, 0.0, "CONSTANT")
    for i, ff in enumerate((lock_at, lock_at + 2, lock_at + 5)):
        ex.scale = (1.0 + 0.4 * (2 - i) / 2,) * 3
        ex.keyframe_insert("scale", frame=ff)

    # ---------------------------------------------------------------- ID card (types in after the lock)
    card = card or DEFAULT_CARD
    if card and card[0][0] == "SUBJECT" and target_name != "GARY":
        card = [("SUBJECT", target_name)] + list(card[1:])
    x0, x1, yt = 0.33, 0.98, 0.64
    yb = yt - 0.13 - 0.075 * len(card)
    g = G2()
    g.rect(x0, yb, x1, yt)
    H.mesh("HUD_card_panel", g, H.panel, z=-0.001)
    g = G2()
    g.frame(x0, yb, x1, yt, 0.004)
    g.line((x0, yt - 0.07), (x1, yt - 0.07), 0.003)
    g.rect(x0, yt - 0.07, x0 + 0.012, yt)
    H.mesh("HUD_card_frame", g, H.ink)
    H.text("TARGET PROFILE", 0.044, (x0 + 0.03, yt - 0.035), H.acc, name="HUD_card_hdr")
    fcur = lock_at + 4
    for i, (fld, val) in enumerate(card):
        line = "%-10s %s" % (fld, val)
        o = H.text(line, 0.04, (x0 + 0.02, yt - 0.125 - 0.075 * i), H.acc if fld in ("THREAT", "HEART RATE") else H.ink,
                   name="HUD_card_%d" % i, boot=False)
        key_obj_alpha(o, f0 - 1, 0.0, "CONSTANT")
        key_obj_alpha(o, fcur, 1.0, "CONSTANT")
        fcur = H.type_in(o, fcur, typing_cps) + 2

    # ---------------------------------------------------------------- link bars (top right)
    bx, by = 0.66, A - 0.16
    H.text("LINK 8x100G", 0.034, (bx - 0.02, by - 0.075), H.ink, name="HUD_link")
    for i in range(5):
        g = G2()
        g.rect(bx + 0.045 * i, by - 0.03, bx + 0.045 * i + 0.03, by - 0.03 + 0.018 * (i + 1))
        o = H.mesh("HUD_bar_%d" % i, g, H.ink, boot=False)
        key_obj_alpha(o, f0 - 1, 0.0, "CONSTANT")
        for ff in range(f0 + 4, f1 + 1, 5):
            lvl = rng.choice((3, 4, 4, 5, 5))
            key_obj_alpha(o, ff, 1.0 if i < lvl else 0.15, "CONSTANT")
    H.windows([(ff, "SNR %4.1f dB" % rng.uniform(19.5, 23.5)) for ff in range(f0, f1 + 1, every * 2)], 0.034, (bx - 0.02, by - 0.13),
              H.ink, name="HUD_snr")

    # ---------------------------------------------------------------- alert banner (blinks after the lock)
    g = G2()
    g.rect(-0.6, A - 0.45, 0.6, A - 0.35)
    pb = H.mesh("HUD_alert_bg", g, _panel(0.35, (0.4, 0.02, 0.02)), z=-0.001, boot=False)
    at = H.text(alert_text, 0.046, (0, A - 0.4), H.alert, align="CENTER", name="HUD_alert", boot=False)
    for o in (pb, at):
        key_obj_alpha(o, f0 - 1, 0.0, "CONSTANT")
        H.blink(o, lock_at + 8, f1 + 1, period=10, duty=0.7)

    # ---------------------------------------------------------------- radar (heading-up), bottom left above the caption zone
    R = 0.19
    cx, cy = -0.76, H.ysafe + R + 0.02
    g = G2()
    for rr in (R, R * 0.66, R * 0.33):
        g.arc(cx, cy, rr, 0.004)
    g.line((cx - R, cy), (cx + R, cy), 0.003)
    g.line((cx, cy - R), (cx, cy + R), 0.003)
    g.frame(cx - R - 0.01, cy - R - 0.01, cx + R + 0.01, cy + R + 0.01, 0.004)
    H.mesh("HUD_radar_grid", g, H.ink)
    g = G2()
    g.rect(cx - R - 0.01, cy - R - 0.01, cx + R + 0.01, cy + R + 0.01)
    H.mesh("HUD_radar_bg", g, H.panel, z=-0.001)
    H.text("PROX %d m" % radar_range, 0.034, (cx - R, cy + R + 0.04), H.ink, name="HUD_radar_lbl")
    g = G2()
    g.fan(0, 0, R, math.pi / 2 - 0.45, math.pi / 2, 10)
    sw = H.mesh("HUD_radar_sweep", g, _ink(S["main"], 0.6), loc=(cx, cy), boot=True)
    sw.rotation_euler = (0, 0, 0)
    sw.keyframe_insert("rotation_euler", index=2, frame=f0)
    sw.rotation_euler = (0, 0, -2 * math.pi)
    sw.keyframe_insert("rotation_euler", index=2, frame=f0 + 45)
    for fc in sw.animation_data.action.fcurves:
        if fc.data_path == "rotation_euler":
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.modifiers.new("CYCLES")

    def radar_xy(d, p):
        v = p - d["cam"]
        fw = Vector((d["fwd"].x, d["fwd"].y, 0)).normalized()
        rt = Vector((fw.y, -fw.x, 0))
        a, b = v.dot(rt), v.dot(fw)
        r = math.hypot(a, b)
        return a / radar_range * R, b / radar_range * R, r <= radar_range
    pts = [("blip_%d" % i, i, H.ink) for i in range(len(blips or []))]
    if target_obj is not None:
        pts.append(("blip_tgt", -1, H.alert))
    for nm, i, mat in pts:
        g = G2()
        g.rect(-0.009, -0.009, 0.009, 0.009)
        o = H.mesh("HUD_" + nm, g, mat, boot=False)
        key_obj_alpha(o, f0 - 1, 0.0, "CONSTANT")
        for d in data:
            p = d["tgt"] if i < 0 else d["blips"][i]
            if p is None:
                continue
            x, y, inside = radar_xy(d, p)
            o.location = (cx + x, cy + y, 0)
            o.keyframe_insert("location", frame=d["f"])
            key_obj_alpha(o, d["f"], 1.0 if inside and d["f"] >= f0 + 6 else 0.0, "CONSTANT")
    H.windows([(d["f"], "CONTACTS %02d" % sum(1 for p in d["blips"] if radar_xy(d, p)[2])) for d in data][::max(1, every // max(1, step))],
              0.034, (cx - R, cy + R + 0.09), H.ink, name="HUD_contacts")

    # ---------------------------------------------------------------- telemetry ticker (scrolls right -> left, clipped)
    tx0, tx1, ty = -0.5, 0.98, H.ysafe + 0.04
    g = G2()
    g.rect(tx0, ty - 0.04, tx1, ty + 0.04)
    H.mesh("HUD_ticker_bg", g, H.panel, z=-0.001)
    g = G2()
    g.frame(tx0, ty - 0.04, tx1, ty + 0.04, 0.003)
    H.mesh("HUD_ticker_frame", g, H.ink)
    tclip = _ink(S["main"], 2.2, (H.holder, tx0 + 0.01, tx1 - 0.01))
    body = (ticker or DEFAULT_TICKER) * 2
    tk = H.text(body, 0.04, (tx1, ty), tclip, name="HUD_ticker", boot=False)
    key_obj_alpha(tk, f0 - 1, 0.0, "CONSTANT")
    key_obj_alpha(tk, f0 + 6, 1.0, "CONSTANT")
    speed = 0.6
    tk.location.x = tx1 - 0.3
    tk.keyframe_insert("location", index=0, frame=f0)
    tk.location.x = tx1 - 0.3 - speed * (f1 - f0) / fps
    tk.keyframe_insert("location", index=0, frame=f1)
    for fc in tk.animation_data.action.fcurves:
        if fc.data_path == "location":
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    # ---------------------------------------------------------------- visibility window
    for o in H.objs + [H.holder]:
        o.hide_render = True
        o.keyframe_insert("hide_render", frame=f0 - 1)
        o.hide_render = False
        o.keyframe_insert("hide_render", frame=f0)
        o.hide_render = True
        o.keyframe_insert("hide_render", frame=f1 + 1)
        for fc in o.animation_data.action.fcurves:
            if fc.data_path == "hide_render":
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
    return [H.holder] + H.objs
