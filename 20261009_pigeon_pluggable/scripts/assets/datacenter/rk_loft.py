"""Tiered "caged farm" bird chassis (rack_bird_q3400 v2, owner decision P2 in DevLog-010), workstream R (DevLog-007).

One chassis (700 W x 640 H x 850 D mm, Q3400-style face language: black square-hole grille, champagne label strips,
side panels with LED bars, true-scale FNM OSFP port, parody badge, rear PSUs + fans) holds 3 rows of bird slots stepped
back in depth (bleacher / stair-step per chassis, like a caged farm):
  tier 0 (bottom row, front):   3 slots, face at x = 0
  tier 1 (middle row, -80 mm):  2 slots offset half a column (brick bond: heads of the row below sit between them)
  tier 2 (top row, -160 mm):    3 slots
Every bird stands with its feet 30 mm behind its own tier face, so breast, neck and head stick out of its opening;
each upper tier is recessed so the heads of the row below (in front) never touch it. Slot interiors are ONE shared mesh
(instancing-friendly); faces/treads/strips/panels are one mesh per chassis (shared by the 3 chassis of a rack).
Geometry checks against the crude-film bird primitive x 2.6 are in DevLog-007 section 10.

Chassis-local frame (mm): tier-0 face plane x = 0 (+x toward the aisle), y centred (+y = right seen from the front),
z = 0 at the chassis bottom.
"""
import rk_lib as K
import rk_q3400 as Q
from rk_lib import MB, Part

m = lambda x: x * 0.001

W, H, D = 700.0, 640.0, 850.0
PITCH_Y = 190.0                       # slot pitch across
SLOT_W, SLOT_H, SLOT_D = 172.0, 176.0, 300.0    # slot interior (wall 18 mm between slots); 176 clears the hero model back (172 max)
FLOOR_T = 12.0                        # floor / tread slab thickness (front edge = champagne label strip)
TIER_X = (0.0, -80.0, -160.0)         # tier face planes
FLOOR_Z = (30.0, 218.0, 406.0)        # slot floor tops (pitch 188 mm)
ROW_Y = ((-190.0, 0.0, 190.0), (-95.0, 95.0), (-190.0, 0.0, 190.0))
INNER_Y = 285.0                       # |y| of the stepped bay; side panels from 285 to 350
FEET_IN = 30.0                        # bird feet origin behind its tier face
S_BIRD = 2.6
HEAD_AHEAD = 44.0 * S_BIRD            # head centre ahead of the feet origin (mm, crude-film primitive)
HEAD_UP = 81.5 * S_BIRD
HEAD_TARGET_X = 50.0                  # suggested head-centre x (in front of the tier-0 face) when the neck is extended
RJ45_RIGHT = [(328.0, 308.0), (328.0, 292.0)]   # (y, z) of the 2 mgmt RJ-45 on the right panel (chassis-local mm)


def slots():
    """Slot records in chassis-local mm (feet origin) with head-pose data."""
    out = []
    for t in range(3):
        for c, y in enumerate(ROW_Y[t]):
            fx = TIER_X[t] - FEET_IN
            head_x = fx + HEAD_AHEAD
            out.append(dict(tier=t, col=c, y=y, feet=(fx, y, FLOOR_Z[t]), face_x=TIER_X[t],
                            head_center=(head_x, y, FLOOR_Z[t] + HEAD_UP),
                            head_out_own_face_mm=round(head_x - TIER_X[t], 1),
                            head_x_rel_front_face_mm=round(head_x, 1),
                            neck_extension_mm=round(max(0.0, HEAD_TARGET_X - head_x), 1)))
    return out


def slot_mesh(L):
    """One bird slot (shared mesh). Slot-local mm: opening plane x = 0, y centred, z = 0 floor top."""
    sw, sh, sd = SLOT_W, SLOT_H, SLOT_D
    wall = MB()    # 0 zinc wall (matte), 1 pcb floor, 2 dark back, 3 black connector, 4 gold, 5 led
    def fb(mb, x0, x1, y0, y1, z0, z1, mi):
        mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)
    for s in (-1, 1):
        fb(wall, -sd, -2.5, s * sw / 2, s * (sw / 2 + 2), 0, sh, 0)                 # side walls
    fb(wall, -sd, 0, -sw / 2, sw / 2, -2, 0, 1)                                    # host board floor
    fb(wall, -sd - 2, -sd, -sw / 2, sw / 2, 0, sh, 2)                              # back wall
    fb(wall, -sd, -sd + 30, -0.3 * sw, 0.3 * sw, 0, 45, 3)                         # host connector
    for k in range(8):
        yy = -0.26 * sw + k * 0.52 * sw / 7
        fb(wall, -sd + 30, -sd + 30.6, yy - 2, yy + 2, 10, 36, 4)
    fb(wall, -sd, -sd + 1.5, -6, 6, sh - 20, sh - 12, 5)                           # in-slot status LED
    for k in range(14):                                                              # gold contact fingers under the feet
        yy = -42 + k * 6.46
        fb(wall, -95, -8, yy - 2, yy + 2, 0, 0.6, 4)
    lip = MB()
    fb(lip, -4, 2, -sw / 2 - 5, -sw / 2, -5, sh, 0)                                # lips: sides + bottom (top = tread)
    fb(lip, -4, 2, sw / 2, sw / 2 + 5, -5, sh, 0)
    fb(lip, -4, 2, -sw / 2 - 5, sw / 2 + 5, -5, 0, 0)
    for k in range(9):                                                               # EMI fingers along the bottom lip
        yy = -sw / 2 + (k + 0.5) * sw / 9
        fb(lip, -24, -6, yy - 6, yy + 6, 0, 1.6, 0)
    bars = MB()                                                                      # cage wires flanking the opening
    for s_ in (-1, 1):
        for yy in (72.0, 80.0):                                                  # hero body is +-66 mm wide
            bars.cyl((m(1.0), m(s_ * yy), m(sh / 2)), m(1.6), m(sh), "z", 6, 0)
        for zz in (55.0, 115.0):
            bars.cyl((m(1.0), m(s_ * (72.0 + sw / 2) / 2), m(zz)), m(1.4), m(sw / 2 - 72.0), "y", 6, 0)
    return [Part("slot_bars", bars, [L["steel"]]),
            Part("slot_walls", wall, [L["cage_wall_lt"], L["pcb"], L["dark_steel"], L["black"], L["gold"], L["led_g"]]),
            Part("slot_lips", lip, [L["nickel_dk"]])]


def chassis_shell(L):
    """Faces, treads, label strips, side panels, body and rear of one tiered chassis (shared by all chassis)."""
    gr = MB()      # perforated grille
    st = MB()      # champagne strips
    bd = MB()      # dark steel body / treads / cover
    pl = MB()      # side-panel features: 0 black, 1 nickel, 2 port_hole, 3 led_g, 4 champagne, 5 rj45, 6 led_a, 7 badge
    def fb(mb, x0, x1, y0, y1, z0, z1, mi=0):
        mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)
    for t in range(3):
        x = TIER_X[t]
        z0, z1 = FLOOR_Z[t], FLOOR_Z[t] + SLOT_H
        edges = [-INNER_Y]
        for y in ROW_Y[t]:
            edges += [y - SLOT_W / 2, y + SLOT_W / 2]
        edges += [INNER_Y]
        for a, b in zip(edges[0::2], edges[1::2]):                                  # grille between openings
            if b - a > 0.5:
                fb(gr, x - 2.5, x, a, b, z0, z1)
        # floor / tread slab: from the slot backs to the face of the tier below (tier 0: its own face)
        xf = TIER_X[t - 1] if t > 0 else x
        fb(bd, x - SLOT_D - 10, xf - 2.5, -INNER_Y, INNER_Y, z0 - FLOOR_T, z0 - 0.5)
        fb(st, xf - 2.5, xf + 0.4, -INNER_Y, INNER_Y, z0 - FLOOR_T, z0)               # label strip on the front edge
    # top cover over tier 2 (behind its face) and bottom bezel
    fb(bd, -D, TIER_X[2] - 2.5, -W / 2 + 1, W / 2 - 1, FLOOR_Z[2] + SLOT_H, FLOOR_Z[2] + SLOT_H + 12)
    fb(gr, -2.5, 0, -INNER_Y, INNER_Y, 0, FLOOR_Z[0] - FLOOR_T)
    fb(bd, -D, -2.5, -W / 2 + 1, W / 2 - 1, 0, 2)
    for s in (-1, 1):
        fb(bd, -D, -2.5, s * (W / 2 - 2), s * (W / 2 - 1), 0, H)                     # chassis side skins
        fb(bd, -D, -2.5, s * (INNER_Y + 0.5), s * (INNER_Y + 2.5), 0, H)            # bay side walls
    fb(bd, -D, -470, -W / 2 + 1, W / 2 - 1, 2, H - 1)                                 # electronics body
    # side panels (front plane, full height)
    for s in (-1, 1):
        fb(gr, -2.5, 0, *sorted((s * INNER_Y, s * W / 2)), 0, H)
    # LED bar + port LED beside every slot row (left panel: cols 0-1, right panel: last col)
    for t in range(3):
        zc = FLOOR_Z[t] + 120
        n = len(ROW_Y[t])
        for c in range(n):
            left = c < n - 1
            yy = (-W / 2 + 14 + c * 16) if left else (W / 2 - 22)
            fb(pl, -0.5, 0.9, yy - 5, yy + 5, zc - 1.5, zc + 1.5, 4)
            fb(pl, -0.5, 1.0, yy - 2, yy + 2, zc + 4, zc + 8, 3)
    ylc = -(W / 2 + INNER_Y) / 2
    Q.osfp_cage(pl, ylc, 250.0, 1, 2)                                               # FNM port (true scale)
    fb(pl, -0.5, 0.6, ylc - 10, ylc + 10, 262, 264, 4)
    fb(pl, -0.5, 1.0, ylc - 26, ylc + 26, 84, 120, 7)                               # badge plate
    for k in range(5):
        yy = 308 + k * 6
        fb(pl, -0.5, 0.9, yy - 1.2, yy + 1.2, 360, 362.5, 3 if k != 2 else 6)
    Q.usb(pl, 305.0, 300.0, 5, 2)
    for (yy, zz) in RJ45_RIGHT:
        Q.rj45(pl, yy, zz, 5, 2)
    parts = [Part("grille", gr, [L["perf_black"]], uv_tile=m(9.0)),
             Part("label_strips", st, [L["champagne"]]),
             Part("body", bd, [L["dark_steel"]]),
             Part("panels", pl, [L["black"], L["nickel"], L["port_hole"], L["led_g"], L["champagne"], L["rj45"], L["led_a"],
                                 L["badge"]])]
    rear = [p for p in Q.chassis(Q.BIRD, L)[0] if p.name == "rear"]                 # same rear as v1 (PSUs + fans)
    return parts + rear


def label_texts(L, coll, parent, prefix, first_label, badge_text):
    """Slot numbers on the strip below each opening (tier 0 strip at its own face; tiers 1/2 on the tread edge)."""
    import math
    out = []
    n = first_label
    for t in range(3):
        xf = TIER_X[t - 1] if t > 0 else TIER_X[t]
        for y in ROW_Y[t]:
            out.append(K.text("%s_lbl_%d" % (prefix, n), str(n), m(10.0), (m(xf + 0.8), m(y), m(FLOOR_Z[t] - FLOOR_T / 2)),
                              (math.pi / 2, 0, math.pi / 2), L["champ_txt"], coll, parent))
            n += 1
    ylc = -(W / 2 + INNER_Y) / 2
    out.append(K.text(prefix + "_badge", badge_text, m(5.2), (m(1.2), m(ylc), m(102)), (math.pi / 2, 0, math.pi / 2),
                      L["champagne"], coll, parent))
    return out
