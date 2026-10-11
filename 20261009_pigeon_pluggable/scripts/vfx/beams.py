"""Beam animation helpers (workstream E, DevLog-013 section 8; owner item F6, DevLog-015).

Beams are objects from materials.make_beam / datafx.make_data_beam (origin = source, local +Z toward the target).
Sources and targets may be world points (Vector / tuple), objects (world translation, optionally + an offset), or
callables f(frame) -> Vector. Object positions are sampled with scene.frame_set (once per keyed frame), so keep the
windows short or pass callables for heavy scenes.
"""
import math

import bpy
from mathutils import Vector

from .materials import key_beam_endpoints, key_obj_alpha


class _Sampler:
    def __init__(self, scene):
        self.scn = scene or bpy.context.scene
        self.cur = None

    def at(self, src, f):
        if callable(src):
            return Vector(src(f))
        if hasattr(src, "matrix_world"):
            if self.cur != f:
                self.scn.frame_set(f)
                self.cur = f
            return src.matrix_world.translation.copy()
        if isinstance(src, tuple) and len(src) == 2 and hasattr(src[0], "matrix_world"):
            o, off = src
            if self.cur != f:
                self.scn.frame_set(f)
                self.cur = f
            return o.matrix_world @ Vector(off)
        return Vector(src)


def _frames(f0, f1, step):
    fr = list(range(f0, f1 + 1, step))
    if fr[-1] != f1:
        fr.append(f1)
    return fr


def beam_track(beam, source, targets, f0, f1, handoffs=None, step=1, jump=3, ramp=4, scene=None):
    """Lock the beam on targets[0], follow it, then hand off to targets[1] at handoffs[0], ... . At each handoff the beam
    blinks off for one frame and re-grows from the source onto the new target over `jump` frames (a visible "jump").
    Fades/grows in over `ramp` frames at f0 and out at f1. handoffs=None splits [f0, f1] evenly.
    Returns [(frame, target_index)] for the hand-off schedule."""
    S = _Sampler(scene)
    n = len(targets)
    if handoffs is None:
        handoffs = [f0 + round((f1 - f0) * (i + 1) / n) for i in range(n - 1)]
    sched = [(f0, 0)] + [(h, i + 1) for i, h in enumerate(handoffs)]
    cur_f0 = f0
    keys = set(_frames(f0, f1, step)) | {h + j for h in handoffs for j in range(-1, jump + 1)} | {f0 + j for j in range(ramp + 1)}
    for f in sorted(k for k in keys if f0 <= k <= f1):
        idx = max(i for (h, i) in sched if h <= f)
        start = max(h for (h, i) in sched if h <= f)
        p0 = S.at(source, f)
        tgt = S.at(targets[idx], f)
        g = 1.0
        grow = ramp if start == f0 else jump
        if f - start < grow:
            g = max(0.02, (f - start + 1) / (grow + 1))
        key_beam_endpoints(beam, f, p0, p0 + (tgt - p0) * g)
    key_obj_alpha(beam, f0 - 1, 0.0, "CONSTANT")
    key_obj_alpha(beam, f0, 1.0, "CONSTANT")
    for h in handoffs:
        key_obj_alpha(beam, h - 1, 0.0, "CONSTANT")
        key_obj_alpha(beam, h, 1.0, "CONSTANT")
    key_obj_alpha(beam, f1, 1.0, "LINEAR")
    key_obj_alpha(beam, f1 + 2, 0.0, "CONSTANT")
    return sched


def beam_sweep(beam, source, f0, f1, yaw0, yaw1, pitch0=0.0, pitch1=0.0, length=12.0, step=1, wobble=0.0, ease=True,
               path=None, scene=None, seed=0.0):
    """Searchlight sweep: the beam keeps `length` m and turns from (yaw0, pitch0) to (yaw1, pitch1) (radians; yaw about
    world Z from +X, pitch up from the horizon), eased, with an optional small wobble. path=[world points] sweeps the
    beam end through those points instead (linear in arc length). Fades in/out over 3 frames."""
    S = _Sampler(scene)
    fr = _frames(f0, f1, step)
    if path:
        P = [Vector(p) for p in path]
        seglen = [(P[i + 1] - P[i]).length for i in range(len(P) - 1)]
        tot = sum(seglen) or 1.0
    for f in fr:
        u = (f - f0) / max(1, f1 - f0)
        if ease:
            u = u * u * (3 - 2 * u)
        p0 = S.at(source, f)
        if path:
            s = u * tot
            i = 0
            while i < len(seglen) - 1 and s > seglen[i]:
                s -= seglen[i]
                i += 1
            end = P[i].lerp(P[i + 1], s / (seglen[i] or 1.0))
        else:
            yw = yaw0 + (yaw1 - yaw0) * u + wobble * math.sin(f * 0.37 + seed)
            pt = pitch0 + (pitch1 - pitch0) * u + wobble * 0.5 * math.sin(f * 0.53 + 1.3 + seed)
            end = p0 + Vector((math.cos(pt) * math.cos(yw), math.cos(pt) * math.sin(yw), math.sin(pt))) * length
        key_beam_endpoints(beam, f, p0, end)
    key_obj_alpha(beam, f0 - 1, 0.0, "CONSTANT")
    key_obj_alpha(beam, f0, 0.0, "LINEAR")
    key_obj_alpha(beam, f0 + 3, 1.0, "LINEAR")
    key_obj_alpha(beam, f1 - 3, 1.0, "LINEAR")
    key_obj_alpha(beam, f1, 0.0, "CONSTANT")


def beam_hops(beam, nodes, f0, hop=6, hold=8, step=1, scene=None):
    """Beam jumping among birds: nodes = [n0, n1, n2, ...] (eye points / objects / callables). Hop k grows from n_k to
    n_(k+1) over `hop` frames, holds `hold` frames, then the next hop starts from n_(k+1). Returns the end frame."""
    S = _Sampler(scene)
    f = f0
    key_obj_alpha(beam, f0 - 1, 0.0, "CONSTANT")
    for k in range(len(nodes) - 1):
        key_obj_alpha(beam, f, 1.0, "CONSTANT")
        for j in range(0, hop + hold + 1, step):
            fr = f + j
            a, b = S.at(nodes[k], fr), S.at(nodes[k + 1], fr)
            g = min(1.0, (j + 1) / (hop + 1))
            key_beam_endpoints(beam, fr, a, a + (b - a) * max(0.02, g))
        f += hop + hold
        key_obj_alpha(beam, f, 0.0, "CONSTANT")
        f += 1
    return f


# ==================================================================== v3: MIMO uplinks + street choreography (DevLog-013 section 9)
def _session(beam, S, src, dst, fs, fe, fade, step):
    """Key one link session: grow from the source over `fade` frames at fs, hold, fade out over `fade` frames at fe."""
    frames = sorted(set(list(range(fs, fe + fade + 1, step)) + [fs + j for j in range(fade + 1)] + [fe, fe + fade]))
    for f in frames:
        a, b = S.at(src, f), S.at(dst, f)
        g = min(1.0, (f - fs + 1) / (fade + 1))
        key_beam_endpoints(beam, f, a, a + (b - a) * max(0.02, g))
    key_obj_alpha(beam, fs - 1, 0.0, "CONSTANT")
    key_obj_alpha(beam, fs, 1.0, "LINEAR")
    key_obj_alpha(beam, fe, 1.0, "LINEAR")
    key_obj_alpha(beam, fe + fade, 0.0, "CONSTANT")


def _new_beam(name, material, radius, color, collection):
    from .materials import make_beam
    return make_beam(name, (0, 0, 0), (0, 0, 1), radius=radius, material=material, color=color, collection=collection)


def mimo_links(sources, sats, f0, f1, k=2, step=2, min_elev_deg=20.0, reconfig_every=(12, 20), fade=4, cam=None,
               seed=0, radius=0.05, material=None, color=None, collection=None, isl=False, isl_material=None, isl_radius=0.02,
               scene=None, name="mimo"):
    """MIMO uplinks: every source (bird eye) keeps k simultaneous links to satellites. Links drop when a satellite falls
    below min_elev_deg (from that source) or leaves the camera view (cam given), are replaced by the best visible
    candidate (elevation + seeded preference - load from other sources), and each source re-configures one link every
    reconfig_every frames (seeded, staggered) so the pattern keeps changing as the array moves. isl=True adds
    inter-satellite links between neighbouring visible satellites. One beam object per link session.
    Returns {"sessions": [(src_i, sat_j, f_start, f_end, beam)], "isl": [beams]}."""
    import random
    from bpy_extras.object_utils import world_to_camera_view
    from .materials import TX_RED
    rng = random.Random(seed)
    S = _Sampler(scene)
    color = color or TX_RED
    pref = {(i, j): rng.uniform(0.0, 8.0) for i in range(len(sources)) for j in range(len(sats))}
    nxt = [f0 + rng.randint(*reconfig_every) for _ in sources]
    links = [dict() for _ in sources]          # j -> session start
    sessions = []
    frames = list(range(f0, f1 + 1, step))

    def visible(i, j, f, ps, pj):
        v = pj - ps
        el = math.degrees(math.atan2(v.z, math.hypot(v.x, v.y)))
        if el < min_elev_deg:
            return False, el
        if cam is not None:
            c = world_to_camera_view(S.scn, cam, pj)
            if not (0.02 < c.x < 0.98 and 0.02 < c.y < 0.98 and c.z > 0):
                return False, el
        return True, el

    for f in frames:
        ps = [S.at(s, f) for s in sources]
        pj = [S.at(t, f) for t in sats]
        load = {}
        for L in links:
            for j in L:
                load[j] = load.get(j, 0) + 1
        for i in range(len(sources)):
            vis = {}
            for j in range(len(sats)):
                ok, el = visible(i, j, f, ps[i], pj[j])
                if ok:
                    vis[j] = el + pref[(i, j)] - 6.0 * load.get(j, 0)
            for j in list(links[i]):
                if j not in vis or f >= f1:
                    sessions.append((i, j, links[i].pop(j), f))
            just = set()
            if f >= nxt[i] and links[i] and f < f1 - 2 * fade:
                cand = [j for j in vis if j not in links[i]]
                if cand:
                    worst = min(links[i], key=lambda j: vis.get(j, -1e9))
                    sessions.append((i, worst, links[i].pop(worst), f))
                    just.add(worst)
                nxt[i] = f + rng.randint(*reconfig_every)
            while len(links[i]) < k and f < f1 - 2 * fade:
                cand = sorted((j for j in vis if j not in links[i] and j not in just), key=lambda j: -vis[j])
                if not cand:
                    break
                j = cand[0]
                links[i][j] = f + (fade if any(s_[0] == i and s_[3] == f for s_ in sessions) else 0)   # stagger after a drop
                load[j] = load.get(j, 0) + 1
    for i, L in enumerate(links):
        for j, fs in L.items():
            sessions.append((i, j, fs, f1))
    out = []
    for n, (i, j, fs, fe) in enumerate(sorted(sessions, key=lambda s: s[2])):
        if fe - fs < 3:
            continue
        b = _new_beam("%s_link_%02d" % (name, n), material, radius, color, collection)
        _session(b, S, sources[i], sats[j], fs, fe, fade, step)
        out.append((i, j, fs, fe, b))
    isls = []
    if isl:
        from .materials import RX_CYAN
        for j in range(len(sats) - 1):
            b = _new_beam("%s_isl_%02d" % (name, j), isl_material, isl_radius, RX_CYAN, collection)
            _session(b, S, sats[j], sats[j + 1], f0 + 2 * j % 12, f1 - fade, fade, step)
            isls.append(b)
    return {"sessions": out, "isl": isls}


def street_beam_choreo(birds, f0, f1, n_sweep=4, n_p2p=3, switch_every=(15, 26), seed=0, sweep_len=(5.0, 9.0),
                       sweep_period=(45, 80), sweep_amp_deg=40.0, pitch_deg=(-20.0, 15.0), radius_sweep=0.015,
                       radius_p2p=0.012, mat_sweep=None, mat_p2p=None, step=2, fade=4, collection=None, scene=None,
                       name="street"):
    """Street beam choreography: n_sweep birds sweep searchlight beams back and forth (sinusoidal yaw about a centre
    direction, gentle pitch bob, seeded period/phase/length) for the whole window, and n_p2p point-to-point links run
    between other birds, each switching partner every switch_every frames (fade out, re-grow to the new partner).
    birds: list of sources, or (source, yaw_centre_rad) pairs (default centre: toward the street axis x = 0).
    Returns {"sweeps": [beam], "p2p": [(a, b, f_start, f_end, beam)]}."""
    import random
    from .materials import RX_CYAN, TX_RED
    rng = random.Random(seed)
    S = _Sampler(scene)
    items = [(b if (isinstance(b, tuple) and len(b) == 2 and isinstance(b[1], (int, float))) else (b, None)) for b in birds]
    srcs = [it[0] for it in items]
    idx = list(range(len(srcs)))
    rng.shuffle(idx)
    sw_idx, rest = idx[:n_sweep], idx[n_sweep:] or idx
    sweeps = []
    for n, i in enumerate(sw_idx):
        src, yc = items[i]
        p_first = S.at(src, f0)
        if yc is None:
            yc = math.atan2(4.0, -p_first.x) if abs(p_first.x) > 0.3 else math.pi / 2
        per = rng.uniform(*sweep_period)
        ph = rng.uniform(0, 2 * math.pi)
        L = rng.uniform(*sweep_len)
        p0d = math.radians(rng.uniform(*pitch_deg))
        b = _new_beam("%s_sweep_%d" % (name, n), mat_sweep, radius_sweep, TX_RED if n % 2 == 0 else RX_CYAN, collection)
        for f in list(range(f0, f1 + 1, step)) + [f1]:
            u = (f - f0) / per * 2 * math.pi
            yw = yc + math.radians(sweep_amp_deg) * math.sin(u + ph)
            pt = p0d + 0.12 * math.sin(1.7 * u + ph)
            a = S.at(src, f)
            key_beam_endpoints(b, f, a, a + Vector((math.cos(pt) * math.cos(yw), math.cos(pt) * math.sin(yw), math.sin(pt))) * L)
        key_obj_alpha(b, f0 - 1, 0.0, "CONSTANT")
        key_obj_alpha(b, f0 + n * 2, 0.0, "LINEAR")
        key_obj_alpha(b, f0 + n * 2 + fade, 1.0, "LINEAR")
        key_obj_alpha(b, f1 - fade, 1.0, "LINEAR")
        key_obj_alpha(b, f1, 0.0, "CONSTANT")
        sweeps.append(b)
    p2p = []
    busy = {}
    for n in range(n_p2p):
        f = f0 + 3 * n
        prev = None
        while f < f1 - 2 * fade:
            dur = rng.randint(*switch_every)
            fe = min(f1 - fade, f + dur)
            free = [i for i in rest if busy.get(i, -1) < f]
            if len(free) < 2:
                free = list(rest)
            for _ in range(20):
                a, c = rng.sample(free, 2)
                if (a, c) != prev and (c, a) != prev:
                    break
            prev = (a, c)
            busy[a] = busy[c] = fe + fade
            b = _new_beam("%s_p2p_%d_%d" % (name, n, f), mat_p2p, radius_p2p, TX_RED, collection)
            _session(b, S, srcs[a], srcs[c], f, fe, fade, step)
            p2p.append((a, c, f, fe, b))
            f = fe + fade + 1
    return {"sweeps": sweeps, "p2p": p2p}
