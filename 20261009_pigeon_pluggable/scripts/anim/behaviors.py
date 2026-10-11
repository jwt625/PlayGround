"""Crowd behaviour scheduler: continuous pigeon head motion, link re-pairing (TX right eye -> partner's RX left eye),
occasional mutual exchanges, and periodic observation of the Subject. Deterministic for a given seed.

Core (`schedule`) is pure Python (no bpy): it can be unit-tested with `uv run python scripts/anim/behaviors.py`.
`apply(sched, rigs)` writes the result into pigeon_rig rigs (bpy).

Geometry: see pgeom.py (bird frame, head yaw/roll conventions). Frames are absolute scene frames.
"""
import json
import math
import random
from dataclasses import dataclass, field

try:
    from . import pgeom as G
except ImportError:
    import pgeom as G

D2R = math.pi / 180.0


@dataclass
class BirdSpec:
    """One bird. pos = world feet origin, yaw = body yaw (beak direction (cos, sin)), scale = world size factor.
    kind 'rack' = head-only (cage), 'street' = full body (may hop-turn to bring a partner or the Subject into range).
    yaw_lim / roll_lim: head limits (rad). group: birds only pair within the same group (None = any).
    blocked: frame windows where the scene animates the bird itself (no moves, not a partner)."""
    id: str
    pos: tuple
    yaw: float
    scale: float = 2.6
    kind: str = "rack"
    yaw_lim: float = 100 * D2R
    roll_lim: float = 40 * D2R
    group: str = None
    blocked: list = field(default_factory=list)
    eye_tx: tuple = None
    eye_rx: tuple = None
    can_turn: bool = None          # default: True for street birds
    neck_ext: float = 0.0          # forward head offset (world m), e.g. hall_layout cell 'neck_extension_m'

    def __post_init__(self):
        self.pivot = G.HEAD_PIVOT
        if self.neck_ext:
            self.pivot, e = G.neck_shift(self.neck_ext, self.scale)
            self.eye_tx = self.eye_tx or e["tx"]
            self.eye_rx = self.eye_rx or e["rx"]
        self.eyes = {"tx": self.eye_tx or G.EYE["tx"], "rx": self.eye_rx or G.EYE["rx"]}

    def eye_world(self, k, yaw=0.0, roll=0.0, body_yaw=None):
        return G.eye_world(self.pos, self.yaw if body_yaw is None else body_yaw, self.scale, k, yaw, roll, eye=self.eyes[k],
                           pivot=self.pivot)

    def geo(self, yaw=None):
        return {"pos": self.pos, "yaw": self.yaw if yaw is None else yaw, "scale": self.scale, "eye_tx": self.eyes["tx"],
                "eye_rx": self.eyes["rx"], "pivot": self.pivot}


@dataclass
class Params:
    tick: int = 2                       # scheduler step (frames)
    link_dist: tuple = (0.25, 4.5)      # partner distance range (world m, eye to eye)
    p_link: float = 0.30                # chance a free bird starts a link at a decision
    p_mutual: float = 0.30              # chance a link is answered (reverse link right after)
    link_hold: tuple = (24, 48)         # frames a link is held once both heads are aligned
    reply_hold: tuple = (18, 36)
    react: tuple = (2, 6)               # partner reaction delay (frames)
    snap: tuple = (2, 4)                # head saccade duration (frames)
    idle_hold: tuple = (8, 30)
    observe_every: tuple = (75, 180)    # frames between Subject observations (per bird)
    observe_dur: tuple = (24, 54)
    observe_step: tuple = (6, 12)       # stepwise re-aim while observing
    observe_react: tuple = (0, 8)       # stagger at the start of a forced observation window
    max_aim_err: float = 3 * D2R
    turn_frames: int = 8                # street hop-turn duration
    avoid_last: int = 2                 # do not re-pick the last N partners
    idle_yaw_frac: float = 0.65
    los_clear: float = 0.045            # line of sight: link beam must clear other birds' head centres by this (world m at
                                        # scale 2.6; scaled with the bird) and any static obstacle sphere


@dataclass
class HeadMove:
    f: int          # snap start (pose before is held until f)
    yaw: float
    roll: float
    nod: float
    dur: int        # snap frames; pose settles at f + dur + 2
    kind: str       # idle | link | reply | observe | rx | forced


@dataclass
class LinkEvent:
    f0: int         # both heads aligned
    f1: int         # link released
    tx: str         # transmitting bird (right eye)
    rx: str         # receiving bird (left eye)
    mutual: bool
    pair: int       # pair id (a reply shares the id of its link)
    tx_eye: tuple   # world eye positions at f0 (for VFX)
    rx_eye: tuple

    def as_dict(self, fps=30, frame1=1):
        d = dict(self.__dict__)
        d["t0"], d["t1"] = (self.f0 - frame1) / fps, (self.f1 - frame1) / fps
        return d


class Track:
    """Piecewise-linear target track from [(frame, xyz), ...] (e.g. subject.head_track); callable frame -> xyz."""

    def __init__(self, samples):
        self.s = sorted(samples, key=lambda x: x[0])

    def __call__(self, f):
        s = self.s
        if f <= s[0][0]:
            return tuple(s[0][1])
        if f >= s[-1][0]:
            return tuple(s[-1][1])
        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if s[m][0] <= f:
                lo = m
            else:
                hi = m
        (fa, a), (fb, b) = s[lo], s[hi]
        u = (f - fa) / ((fb - fa) or 1)
        return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


@dataclass
class Schedule:
    f0: int
    f1: int
    seed: int
    moves: dict      # id -> [HeadMove]
    turns: dict      # id -> [(f, yaw1, frames)]
    links: list      # [LinkEvent]
    observes: list   # [(f0, f1, id, eye, exact)]

    def stats(self):
        n = len(self.moves)
        return dict(birds=n, moves=sum(len(v) for v in self.moves.values()), links=len(self.links),
                    mutual=sum(1 for e in self.links if e.mutual), observes=len(self.observes),
                    turns=sum(len(v) for v in self.turns.values()),
                    birds_linked=len({e.tx for e in self.links} | {e.rx for e in self.links}))

    def events_json(self, path, fps=30, frame1=1):
        with open(path, "w") as fh:
            json.dump(dict(f0=self.f0, f1=self.f1, seed=self.seed, fps=fps,
                           links=[e.as_dict(fps, frame1) for e in self.links],
                           observes=[dict(f0=a, f1=b, id=i, eye=k, exact=x) for a, b, i, k, x in self.observes]), fh, indent=1)


class _St:
    def __init__(self, spec, f0, rng, P):
        self.free = f0 + rng.randint(0, 12)
        self.yaw = 0.0
        self.roll = 0.0
        self.nod = 0.0
        self.byaw = spec.yaw
        self.next_obs = f0 + rng.randint(0, P.observe_every[1])
        self.last = []


def _seg_dist(p, a, b):
    ab = G.sub(b, a)
    L2 = sum(x * x for x in ab) or 1e-12
    u = max(0.0, min(1.0, sum(x * y for x, y in zip(G.sub(p, a), ab)) / L2))
    return G.dist(p, G.add(a, G.mul(ab, u)))


def schedule(birds, f0, f1, seed=0, subject=None, params=None, force_observe=(), forced_links=(), obstacles=(), log=None):
    """Schedule head motion + links for `birds` (list of BirdSpec) over [f0, f1].

    subject: callable frame -> world xyz of the Subject's head (or a list of (frame, xyz) samples), or None.
    force_observe: [(fa, fb)] windows where every bird stares at the Subject (staggered start, stepwise tracking).
    forced_links: [(f, tx_id, rx_id, hold)] links the story needs (scheduled first; birds reserved around them).
    obstacles: [(centre xyz, radius)] static spheres a link beam must not cross (rack posts, props); other birds' heads
    are always tested (Params.los_clear). Moving obstacles (the Subject) are not tested.
    Returns a Schedule."""
    P = params or Params()
    rng = random.Random(seed)
    if subject is not None and not callable(subject):
        subject = Track(subject)
    n = len(birds)
    idx = {b.id: i for i, b in enumerate(birds)}
    st = [_St(b, f0, random.Random(seed * 1000003 + i), P) for i, b in enumerate(birds)]
    moves = {b.id: [] for b in birds}
    turns = {b.id: [] for b in birds}
    links, observes = [], []
    pair_id = [0]
    for i, b in enumerate(birds):
        moves[b.id].append(HeadMove(f0, 0.0, 0.0, 0.0, 0, "init"))
    # neighbour lists (eye-to-eye distance at rest)
    eyes = [b.eye_world("tx") for b in birds]
    nb = []
    for i, a in enumerate(birds):
        lst = []
        for j, b in enumerate(birds):
            if i == j or (a.group != b.group):
                continue
            d = G.dist(eyes[i], eyes[j])
            if P.link_dist[0] <= d <= P.link_dist[1]:
                lst.append((j, d))
        nb.append(lst)
    heads = [G.mul(G.add(b.eye_world("tx"), b.eye_world("rx")), 0.5) for b in birds]

    def clear_path(i, j, pa, pb):
        for k, h in enumerate(heads):
            if k == i or k == j:
                continue
            if _seg_dist(h, pa, pb) < P.los_clear * birds[k].scale / 2.6:
                return False
        return all(_seg_dist(c, pa, pb) >= r for c, r in obstacles)
    reserved = [[] for _ in birds]      # (fa, fb) windows held for forced links
    for b in birds:
        reserved[idx[b.id]] += list(b.blocked)

    def is_reserved(i, fa, fb):
        return any(not (fb < a or fa > b_) for a, b_ in reserved[i])

    def next_force(f):
        for a, b_ in force_observe:
            if a > f:
                return a
        return None

    def in_force(f):
        return any(a <= f < b_ for a, b_ in force_observe)

    def can_turn(spec):
        return spec.can_turn if spec.can_turn is not None else spec.kind == "street"

    def snap(i, f, yaw, roll, nod, kind, dur=None):
        s = st[i]
        d = dur or rng.randint(*P.snap)
        moves[birds[i].id].append(HeadMove(int(f), yaw, roll, nod, d, kind))
        s.yaw, s.roll, s.nod = yaw, roll, nod
        return int(f) + d + 2

    def maybe_turn(i, f, yaw_needed):
        """Street bird: hop-turn so the needed head yaw drops to ~45 deg. Returns frame when done."""
        s, spec = st[i], birds[i]
        sgn = 1.0 if yaw_needed > 0 else -1.0
        new = G.wrap(s.byaw + yaw_needed - sgn * rng.uniform(35, 55) * D2R)
        turns[spec.id].append((int(f), new, P.turn_frames))
        s.byaw = new
        return int(f) + P.turn_frames + 1

    def link_solution(i, j, yi=None, yj=None):
        A, B = birds[i], birds[j]
        ga, gb = A.geo(st[i].byaw if yi is None else yi), B.geo(st[j].byaw if yj is None else yj)
        ya, ra, yb, rb = G.solve_link(ga, gb)
        ok = (abs(ya) <= A.yaw_lim and abs(ra) <= A.roll_lim and abs(yb) <= B.yaw_lim and abs(rb) <= B.roll_lim)
        if ok:
            ta = B.eye_world("rx", yb, rb, gb["yaw"])
            if G.aim_error(ga["pos"], ga["yaw"], ga["scale"], "tx", ta, ya, ra, eye=A.eyes["tx"], pivot=A.pivot) > P.max_aim_err:
                ok = False
            elif not clear_path(i, j, A.eye_world("tx", ya, ra, ga["yaw"]), ta):
                ok = False
        return ok, (ya, ra, yb, rb), ga, gb

    def do_link(i, j, f, hold, kind="link", mutual=False, pid=None):
        ok, (ya, ra, yb, rb), ga, gb = link_solution(i, j)
        if not ok:
            return None
        ea = snap(i, f, ya, ra, rng.uniform(-0.05, 0.05), kind)
        eb = snap(j, f + rng.randint(*P.react), yb, rb, rng.uniform(-0.05, 0.05), "rx")
        fa = max(ea, eb)
        fb = fa + hold
        if pid is None:
            pair_id[0] += 1
            pid = pair_id[0]
        tx_eye = birds[i].eye_world("tx", ya, ra, ga["yaw"])
        rx_eye = birds[j].eye_world("rx", yb, rb, gb["yaw"])
        links.append(LinkEvent(fa, fb, birds[i].id, birds[j].id, mutual, pid, tx_eye, rx_eye))
        st[i].free = st[j].free = fb
        st[i].last = (st[i].last + [j])[-P.avoid_last:]
        st[j].last = (st[j].last + [i])[-P.avoid_last:]
        return fb, pid

    # forced links first
    for f, a, b, hold in forced_links:
        i, j = idx[a], idx[b]
        r = do_link(i, j, f, hold, "forced")
        if r is None:
            if log:
                log("forced link %s -> %s at %d has no solution within limits" % (a, b, f))
            continue
        for k in (i, j):
            reserved[k].append((f - 4, r[0] + 2))
            st[k].free = f0   # the scheduler fills the time before; the window itself is reserved
    # restore states (the forced snaps were appended out of order: keep moves sorted at the end)
    for s in st:
        s.yaw = s.roll = s.nod = 0.0

    def observe(i, f, f_end, kind="observe"):
        spec, s = birds[i], st[i]
        eye = None
        g = f
        exact_any = False
        while g < f_end - 4:
            tgt = subject(g + 3)
            k, y, r, exact = G.stare(spec.pos, s.byaw, spec.scale, tgt, spec.yaw_lim, spec.roll_lim, prefer=eye, eyes=spec.eyes,
                                     pivot=spec.pivot)
            if not exact and can_turn(spec) and g == f:
                ky, _ = G.aim(spec.pos, s.byaw, spec.scale, k, tgt, eye=spec.eyes[k], pivot=spec.pivot)
                g = maybe_turn(i, g, ky)
                continue
            eye = k
            exact_any = exact_any or exact
            e = snap(i, g, y, r, rng.uniform(-0.08, 0.12), kind, rng.choice((2, 3, 3)))
            g = max(e, g + rng.randint(*P.observe_step))
        observes.append((int(f), int(f_end), spec.id, eye, exact_any))
        s.free = max(g, int(f_end))
        s.next_obs = s.free + rng.randint(*P.observe_every)

    t = f0
    while t < f1 - 4:
        order = list(range(n))
        rng.shuffle(order)
        nf = next_force(t)
        for i in order:
            s, spec = st[i], birds[i]
            if s.free > t:
                continue
            if is_reserved(i, t, t + 10):
                s.free = t + P.tick
                continue
            # cap any new activity so it ends before the next forced-observation window / reservation
            cap = f1
            if nf is not None:
                cap = min(cap, nf + rng.randint(*P.observe_react))
            res_next = [a for a, b_ in reserved[i] if a > t]
            if res_next:
                cap = min(cap, min(res_next) - 2)
            if subject is not None and in_force(t):
                fb = next(b_ for a, b_ in force_observe if a <= t < b_)
                observe(i, t, min(fb, cap if cap > t + 10 else fb), "forced_observe")
                continue
            if subject is not None and t >= s.next_obs and cap - t > 30:
                observe(i, t, min(t + rng.randint(*P.observe_dur), cap))
                continue
            r = rng.random()
            if r < P.p_link and cap - t > 30 and nb[i]:
                cands = [(j, d) for j, d in nb[i] if st[j].free <= t + 2 and j not in s.last
                         and not is_reserved(j, t, t + P.link_hold[1] + 12)]
                rng.shuffle(cands)
                done = False
                for j, d in cands[:6]:
                    hold = rng.randint(*P.link_hold)
                    if t + hold + 12 > cap:
                        hold = cap - t - 12
                        if hold < 12:
                            break
                    ok, _, _, _ = link_solution(i, j)
                    if not ok and can_turn(spec):
                        continue
                    if not ok:
                        continue
                    res = do_link(i, j, t, hold)
                    if res is None:
                        continue
                    fb, pid = res
                    done = True
                    if rng.random() < P.p_mutual and fb + 50 < cap:
                        h2 = rng.randint(*P.reply_hold)
                        r2 = do_link(j, i, fb + 1, h2, "reply", True, pid)
                        if r2 is not None:
                            for e in links:
                                if e.pair == pid:
                                    e.mutual = True
                    break
                if done:
                    continue
            # idle look-around: pigeon saccade, then hold
            if rng.random() < 0.6:
                y = rng.uniform(-1, 1) * spec.yaw_lim * P.idle_yaw_frac
            else:
                y = G.clamp(s.yaw + rng.uniform(-0.45, 0.45), spec.yaw_lim)
            ro = rng.uniform(-0.45, 0.45) * spec.roll_lim
            nd = rng.choice((0.0, 0.0, rng.uniform(-0.12, 0.25)))
            e = snap(i, t, y, ro, nd, "idle")
            s.free = min(e + rng.randint(*P.idle_hold), max(e, cap))
        t += P.tick
    for k in moves:
        moves[k].sort(key=lambda m: m.f)
        # drop overlaps (a forced move may land inside another snap): keep the later-starting, ensure spacing
        clean = []
        for m in moves[k]:
            if clean and m.f < clean[-1].f + clean[-1].dur + 2:
                if m.kind in ("forced", "rx") and clean[-1].kind not in ("forced", "rx", "init"):
                    clean[-1] = m
                    continue
                m = HeadMove(clean[-1].f + clean[-1].dur + 2, m.yaw, m.roll, m.nod, m.dur, m.kind)
            clean.append(m)
        moves[k] = clean
    links.sort(key=lambda e: e.f0)
    return Schedule(f0, f1, seed, moves, turns, links, observes)


def spec_from_rig(bid, r, f=1, **kw):
    """BirdSpec matching a pigeon_rig rig after placement (root location/yaw at frame f, scale, neck extension)."""
    loc = tuple(r.root.location)
    yaw = r.root.rotation_euler[2]
    ad = r.root.animation_data
    if ad and ad.action:
        vals = []
        for i in range(3):
            fc = ad.action.fcurves.find("location", index=i)
            vals.append(fc.evaluate(f) if fc else loc[i])
        loc = tuple(vals)
        fc = ad.action.fcurves.find("rotation_euler", index=2)
        yaw = fc.evaluate(f) if fc else yaw
    return BirdSpec(bid, loc, yaw, scale=r.root.scale[0], neck_ext=float(r.root.get("pg_neck_ext", 0.0)), **kw)


# ================================================================================================ bpy apply
def apply(sched, rigs, overshoot=0.12, breathe=False, blinks=True, seed=None, log=print):
    """Write a Schedule into pigeon_rig rigs ({id: PigeonRig}). Head moves -> keys on the head pivot (hold, snap with
    overshoot, settle); street turns -> pigeon_rig.turn_in_place (hop-turn); optional breathing and random blinks."""
    try:
        from . import pigeon_rig as PR
        from . import keys as KY
    except ImportError:
        import pigeon_rig as PR
        import keys as KY
    seed = sched.seed if seed is None else seed
    for n, (bid, mv) in enumerate(sched.moves.items()):
        r = rigs.get(bid)
        if r is None:
            continue
        prev = (0.0, 0.0, 0.0)
        for m in mv:
            y0, r0, n0 = prev
            if m.dur == 0:
                PR.head_pose(r, m.f, m.yaw, m.roll, m.nod)
            else:
                o = overshoot
                PR.head_pose(r, m.f, y0, r0, n0)
                PR.head_pose(r, m.f + m.dur, m.yaw + o * (m.yaw - y0), m.roll + o * (m.roll - r0), m.nod + o * (m.nod - n0))
                PR.head_pose(r, m.f + m.dur + 2, m.yaw, m.roll, m.nod)
            prev = (m.yaw, m.roll, m.nod)
        for f, y1, fr in sched.turns.get(bid, []):
            PR.turn_in_place(r, f, y1, frames=fr)
        if breathe and r.full:
            PR.breathe(r, sched.f0, sched.f1, seed=seed * 7919 + n)
        if blinks and r.lid:
            PR.blinks(r, sched.f0, sched.f1, seed=seed * 104729 + n)
    KY.flush()
    if log:
        log("BEHAVIOR applied", sched.stats())


# ================================================================================================ self-test (no bpy)
def _selftest():
    import time
    rng = random.Random(1)
    birds = []
    # two rack faces across a 2.2 m aisle, 10 racks x 10 rows of cells (200 birds), facing the aisle
    for side, x, yaw in (("L", -1.16, 0.0), ("R", 1.16, math.pi)):
        for k in range(10):
            for row in range(10):
                birds.append(BirdSpec("%s%02d_%d" % (side, k, row), (x, 1.4 + 0.8 * k + rng.uniform(-0.2, 0.2), 0.2 + 0.21 * row), yaw,
                                      group="rack"))
    for k in range(20):
        birds.append(BirdSpec("st%02d" % k, (rng.uniform(-4, 4), 18 + rng.uniform(-3, 3), rng.uniform(0, 5)), rng.uniform(-3, 3),
                              kind="street", yaw_lim=110 * D2R, roll_lim=50 * D2R, group="street"))
    subj = Track([(1, (0.0, 0.0, 1.55)), (900, (0.0, 12.0, 1.55)), (1800, (0.0, 22.0, 1.55))])
    t = time.time()
    sc = schedule(birds, 1, 1861, seed=7, subject=subj, force_observe=[(1650, 1720)], forced_links=[(900, "L05_4", "R05_4", 40)])
    dt = time.time() - t
    s = sc.stats()
    print("SELFTEST", s, "%.2f s" % dt)
    # checks: link aim errors, overlaps, limits, determinism
    bad = 0
    for e in sc.links:
        A = next(b for b in birds if b.id == e.tx)
        B = next(b for b in birds if b.id == e.rx)
        if A.kind == "rack" and B.kind == "rack":
            ya, ra, yb, rb = G.solve_link(A.geo(), B.geo())
            err = G.aim_error(A.pos, A.yaw, A.scale, "tx", e.rx_eye, ya, ra)
            bad += err > 3 * D2R
    hc = {b.id: G.mul(G.add(G.eye_world(b.pos, b.yaw, b.scale, "tx"), G.eye_world(b.pos, b.yaw, b.scale, "rx")), 0.5) for b in birds}
    los = sum(1 for e in sc.links for b in birds if b.id not in (e.tx, e.rx) and _seg_dist(hc[b.id], e.tx_eye, e.rx_eye) < 0.045)
    over = 0
    lim = 0
    for b in birds:
        mv = sc.moves[b.id]
        for m0, m1 in zip(mv[:-1], mv[1:]):
            over += m1.f < m0.f + m0.dur + 2
        lim += sum(1 for m in mv if abs(m.yaw) > b.yaw_lim + 1e-6 or abs(m.roll) > b.roll_lim + 1e-6)
    per_bird = {}
    for e in sc.links:
        per_bird[e.tx] = per_bird.get(e.tx, 0) + 1
        per_bird[e.rx] = per_bird.get(e.rx, 0) + 1
    sc2 = schedule(birds, 1, 1861, seed=7, subject=subj, force_observe=[(1650, 1720)], forced_links=[(900, "L05_4", "R05_4", 40)])
    same = [(e.f0, e.tx, e.rx) for e in sc.links] == [(e.f0, e.tx, e.rx) for e in sc2.links]
    forced = [e for e in sc.links if e.tx == "L05_4" and e.rx == "R05_4" and abs(e.f0 - 905) < 12]
    in_window = sum(1 for o in sc.observes if o[0] >= 1650 and o[0] <= 1660)
    print("CHECK aim_err>3deg", bad, "overlaps", over, "limit_violations", lim, "deterministic", same,
          "forced_link", len(forced), "los_violations", los, "links/bird mean %.1f" % (sum(per_bird.values()) / max(1, len(per_bird))),
          "birds_without_link", len(birds) - len(per_bird), "forced_observe_starts", in_window)


if __name__ == "__main__":
    _selftest()
