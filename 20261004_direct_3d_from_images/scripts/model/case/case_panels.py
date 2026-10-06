"""Photo-texture panels of the case (pure Python, shared by build.py and bake_case.py).

A panel is a planar quad in case-local mm (corners in image orientation TL, TR, BR, BL as seen from the outside
of the surface), the object whose faces it textures, and the outward normal. build.py assigns every face of
that object whose normal and centroid match the panel to the panel's image material, with UVs from the
projection onto the quad; bake_case.py bakes assets/textures/case_<panel>.png from train views (ID-owned
pixels, per-texel median of the best views)."""

from __future__ import annotations

import math


def window_rings(W: dict):
    """Top, bottom and flange rings (case-local mm) of the window shell, long walls split into nseg segments so
    each wall quad is nearly planar (the bottom is tilted along X). Ring order: near wall -X->+X, far wall +X->-X."""
    x0, x1, hw, zt, ins, n = W["x0"], W["x1"], W["half_w"], W["z_top"], W["inset"], int(W.get("nseg", 6))

    def zb(x):
        return W["z_bot_mx"] + (W["z_bot_px"] - W["z_bot_mx"]) * (x - x0) / (x1 - x0)

    xt = [x0 + (x1 - x0) * k / n for k in range(n + 1)]
    xb = [x0 + ins + (x1 - x0 - 2 * ins) * k / n for k in range(n + 1)]
    top = [(x, -hw, zt) for x in xt] + [(x, hw, zt) for x in xt[::-1]]
    bot = [(x, -hw + ins, zb(x)) for x in xb] + [(x, hw - ins, zb(x)) for x in xb[::-1]]
    fl = W["flange"]
    flg = [(x + (fl if x == x1 else -fl if x == x0 else 0), y + (fl if y > 0 else -fl), z) for x, y, z in top]
    return top, bot, flg


def _window_panels(W: dict) -> list[dict]:
    top, bot, _ = window_rings(W)
    out = []
    m = len(top)
    for i in range(m):
        j = (i + 1) % m
        ti, tj, bi, bj = top[i], top[j], bot[i], bot[j]
        d = [((bi[k] + bj[k]) - (ti[k] + tj[k])) / 2 for k in range(3)]
        tl, tr = ti, tj
        bl = tuple(tl[k] + d[k] for k in range(3))
        br = tuple(tr[k] + d[k] for k in range(3))
        eu = [tr[k] - tl[k] for k in range(3)]
        n = [eu[1] * d[2] - eu[2] * d[1], eu[2] * d[0] - eu[0] * d[2], eu[0] * d[1] - eu[1] * d[0]]
        if n[2] < 0:
            n = [-q for q in n]
        out.append(dict(name=f"win_w{i:02d}", obj="case.window", n=tuple(n), c=[tl, tr, br, bl]))
    fl = W["flange"]
    x0, x1, hw, zt, ins = W["x0"], W["x1"], W["half_w"], W["z_top"], W["inset"]
    out.append(dict(name="win_flange", obj="case.window", n=(0, 0, 1),
                    c=[(x0 - fl, hw + fl, zt), (x1 + fl, hw + fl, zt), (x1 + fl, -hw - fl, zt), (x0 - fl, -hw - fl, zt)]))
    za, zb_ = W["z_bot_mx"], W["z_bot_px"]
    zA = za + (zb_ - za) * ins / (x1 - x0)
    zB = za + (zb_ - za) * (x1 - x0 - ins) / (x1 - x0)
    out.append(dict(name="win_bottom", obj="case.window", n=(zA - zB, 0, x1 - x0 - 2 * ins),
                    c=[(x0 + ins, hw - ins, zA), (x1 - ins, hw - ins, zB), (x1 - ins, -hw + ins, zB),
                       (x0 + ins, -hw + ins, zA)]))
    return out


def panels(P: dict) -> list[dict]:
    b, s, t, br, E, W = P["body"], P["screen"], P["tray"], P["bracket"], P["window_edge"], P["window"]
    L, hw, w = b["length"], b["width"] / 2, b["wall"]
    x0, x2 = -L / 2, L / 2
    xj = x0 + s["length"]
    H = s["height"]
    hn, hf, he = t["near_height"], t["far_height"], t["end_height"]
    ht = max(hf, he)
    fs = b["flare"] / b["flare_ref_z"]
    fH, fT, fN = fs * H, fs * ht, fs * hn
    fl = t.get("floor", b["floor"])
    out = [
        # screen section outer walls and rim top
        dict(name="scr_near", obj="case.screen_box", n=(0, -1, 0),
             c=[(x0, -hw - fH, H), (xj, -hw - fH, H), (xj, -hw, 0), (x0, -hw, 0)]),
        dict(name="scr_far", obj="case.screen_box", n=(0, 1, 0),
             c=[(xj, hw + fH, H), (x0, hw + fH, H), (x0, hw, 0), (xj, hw, 0)]),
        dict(name="scr_end", obj="case.screen_box", n=(-1, 0, 0),
             c=[(x0 - fH, hw, H), (x0 - fH, -hw, H), (x0, -hw, 0), (x0, hw, 0)]),
        dict(name="scr_top", obj="case.screen_box", n=(0, 0, 1),
             c=[(x0 - fH, hw + fH, H), (xj, hw + fH, H), (xj, -hw - fH, H), (x0 - fH, -hw - fH, H)]),
        # tray outer walls
        dict(name="tray_near", obj="case.tray", n=(0, -1, 0),
             c=[(xj, -hw - fN, hn), (x2, -hw - fN, hn), (x2, -hw, 0), (xj, -hw, 0)]),
        dict(name="tray_far", obj="case.tray", n=(0, 1, 0),
             c=[(x2, hw + fs * hf, hf), (xj, hw + fs * hf, hf), (xj, hw, 0), (x2, hw, 0)]),
        dict(name="tray_end", obj="case.tray", n=(1, 0, 0),
             c=[(x2 + fs * he, -hw, he), (x2 + fs * he, hw, he), (x2, hw, 0), (x2, -hw, 0)]),
        # tray rim tops (thin strips; separate heights)
        dict(name="tray_rim_near", obj="case.tray", n=(0, 0, 1),
             c=[(xj, -hw + w, hn), (x2 + fT, -hw + w, hn), (x2 + fT, -hw - fN, hn), (xj, -hw - fN, hn)]),
        dict(name="tray_rim_far", obj="case.tray", n=(0, 0, 1),
             c=[(xj, hw + fT, hf), (x2 + fT, hw + fT, hf), (x2 + fT, hw - w, hf), (xj, hw - w, hf)]),
        dict(name="tray_rim_end", obj="case.tray", n=(0, 0, 1),
             c=[(x2 - w, hw - w, he), (x2 + fT, hw - w, he), (x2 + fT, -hw + w, he), (x2 - w, -hw + w, he)]),
        # tray interior walls (seen from inside)
        dict(name="tray_in_far", obj="case.tray", n=(0, -1, 0),
             c=[(xj, hw - w, hf), (x2 - w, hw - w, hf), (x2 - w, hw - w, fl), (xj, hw - w, fl)]),
        dict(name="tray_in_near", obj="case.tray", n=(0, 1, 0),
             c=[(x2 - w, -hw + w, hn), (xj, -hw + w, hn), (xj, -hw + w, fl), (x2 - w, -hw + w, fl)]),
        dict(name="tray_in_end", obj="case.tray", n=(-1, 0, 0),
             c=[(x2 - w, hw - w, he), (x2 - w, -hw + w, he), (x2 - w, -hw + w, fl), (x2 - w, hw - w, fl)]),
        # bracket bar top and pads
        dict(name="bracket_top", obj="case.bracket", n=(0, 0, 1),
             c=[(br["x"] - br["width"] / 2, br["tab_y0"], br["z_top"]), (br["x"] + br["width"] / 2, br["tab_y0"], br["z_top"]),
                (br["x"] + br["width"] / 2, -br["tab_y0"], br["z_top"]), (br["x"] - br["width"] / 2, -br["tab_y0"], br["z_top"])]),
        dict(name="tab_near_top", obj="case.bracket_tab_near", n=(0, 0, 1),
             c=[(br["x"] - br["width"] / 2, -br["tab_y0"], br["tab_z_top"]), (br["x"] + br["width"] / 2, -br["tab_y0"], br["tab_z_top"]),
                (br["x"] + br["width"] / 2, -hw, br["tab_z_top"]), (br["x"] - br["width"] / 2, -hw, br["tab_z_top"])]),
        dict(name="tab_far_top", obj="case.bracket_tab_far", n=(0, 0, 1),
             c=[(br["x"] - br["width"] / 2, hw, br["tab_z_top"]), (br["x"] + br["width"] / 2, hw, br["tab_z_top"]),
                (br["x"] + br["width"] / 2, br["tab_y0"], br["tab_z_top"]), (br["x"] - br["width"] / 2, br["tab_y0"], br["tab_z_top"])]),
        # dark band at the window's +X end
        dict(name="wedge_top", obj="case.window_edge_px", n=(0, 0, 1),
             c=[(E["x0"], E["half_w"], E["z_top"]), (E["x1"], E["half_w"], E["z_top"]),
                (E["x1"], -E["half_w"], E["z_top"]), (E["x0"], -E["half_w"], E["z_top"])]),
    ]
    # junction ramps (outer faces), lugs (outer faces + tops), screw heads
    rp, lg = P["ramp"], P["lugs"]
    for k, sy, hz in (("near", -1, hn), ("far", 1, hf)):
        fo = fs * (hz + H) / 2
        y = sy * (hw + fo)
        xa, xb = (xj, rp["x1"]) if sy < 0 else (rp["x1"], xj)
        out.append(dict(name=f"ramp_{k}", obj=f"case.ramp_{k}", n=(0, sy, 0),
                        c=[(xa, y, H), (xb, y, H), (xb, y, hz - 1), (xa, y, hz - 1)]))
        yo = sy * lg["y_out"]
        (za, zb_), (la, lb) = lg["z"], (lg["x"] if sy < 0 else lg["x"][::-1])
        out.append(dict(name=f"lug_{k}", obj=f"case.lug_{k}", n=(0, sy, 0),
                        c=[(la, yo, zb_), (lb, yo, zb_), (lb, yo, za), (la, yo, za)]))
        yi = sy * (hw - 0.5)
        y_hi, y_lo = (max(yo, yi), min(yo, yi))
        out.append(dict(name=f"lug_{k}_top", obj=f"case.lug_{k}", n=(0, 0, 1),
                        c=[(lg["x"][0], y_hi, zb_), (lg["x"][1], y_hi, zb_), (lg["x"][1], y_lo, zb_), (lg["x"][0], y_lo, zb_)]))
        r, zs = br["screw_r"], br["tab_z_top"] + br["screw_h"]
        cx, cy = br["screw_x"], sy * br["screw_y"]
        out.append(dict(name=f"screw_{k}", obj=f"case.screw_{k}", n=(0, 0, 1),
                        c=[(cx - r, cy + r, zs), (cx + r, cy + r, zs), (cx + r, cy - r, zs), (cx - r, cy - r, zs)]))
    # white card top (tilted plane, higher at -X)
    C = P["card"]
    out.append(dict(name="card_top", obj="case.card", n=(C["z_mx"] - C["z_px"], 0, C["x1"] - C["x0"]),
                    c=[(C["x0"], C["y1"], C["z_mx"]), (C["x1"], C["y1"], C["z_px"]),
                       (C["x1"], C["y0"], C["z_px"]), (C["x0"], C["y0"], C["z_mx"])]))
    if P["material"].get("window_tex", False):
        out += _window_panels(P["window"])
    if not P["material"].get("window_top", False):
        return out
    # window appearance quad: covers the window opening + flange at the flange height (glass + card look)
    fl_w = W["flange"]
    wx0, wx1, wy = W["x0"] - fl_w, W["x1"] + fl_w, W["half_w"] + fl_w
    zt = W["z_top"] + 0.05
    out.append(dict(name="window_top", obj="case.window_top", n=(0, 0, 1),
                    c=[(wx0, wy, zt), (wx1, wy, zt), (wx1, -wy, zt), (wx0, -wy, zt)]))
    return out


def to_world(p, f):
    """Case-local mm -> world mm with the [frame] placement (yaw about Z, then translation)."""
    a = math.radians(f["yaw_deg"])
    x, y, z = p
    return (f["cx"] + math.cos(a) * x - math.sin(a) * y, f["cy"] + math.sin(a) * x + math.cos(a) * y, z)
