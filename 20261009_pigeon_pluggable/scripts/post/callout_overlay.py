"""2D callouts, scene headers and the beam note on rendered frames, in the hero-loop style
(scripts/assets/anim_label_overlay.py): dark box, 1 px light-cyan outline, Helvetica Neue text, 2 px leader from a 4 px
anchor dot through a knee to the box edge. No colours or angles beyond that palette (owner, 2026-10-10: "Similar
callout and text box as the short hero pigeon vid"). Headers and the beam note moved to post because the render DOF
blurred the in-scene headers and the beam note doubled in one assembly shot.

Run (project root):
  uv run --no-project --with pillow python scripts/post/callout_overlay.py [--json outputs/film_v4_callouts.json] \
      [--texts outputs/film_v4_texts.json] [--frames-in outputs/film_v4_frames] [--frames-out outputs/film_v4_frames_lab] \
      [--only 1290,1291]

Callout JSON (S): {"W":1080,"H":1350,"frames":{"<frame>":[{"text":..,"x":..,"y":..,"side":"L"|"R","a":0..1}, ...]}}
Texts JSON (S):   {"W":1080,"H":1350,"frames":{"<frame>":[{"kind":"header"|"beamnote","text":..,"a":0..1}, ...]}}
Reads <frames-in>/fNNNN.png, writes labelled copies of only the frames listed in either JSON to <frames-out>/fNNNN.png
(same names); the clean frames are never written.

Callouts only inside --callout-range (default 1201-1436, the S5 exploded view; owner 2026-10-10); callouts elsewhere
in the JSON are ignored. Callouts: two fixed columns 14 px from the frame edges; a label keeps the column it has in its first frame and the order
inside each column is fixed (anchor height at each label's first frame), so boxes never swap; heights follow the
anchors with a minimum gap, limited to MAX_STEP px per frame (slight lag, no jitter), inside the frame and above the
caption band (box bottom < 0.82 H).
Header: top left, 22 px from the top and left edges, about 30 px font, wrapped to <= 60 percent of the frame width.
Beam note: bottom left, about 20 px font, box top at 0.955 H (below the caption block; burn_captions.py keeps every
caption block above 0.945 H).
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

LINE, TEXT, BOX = (150, 210, 230), (235, 240, 245), (8, 12, 18)
BOX_A = 0.85
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
EDGE, GAP, TOP = 14, 8, 10
MAX_STEP = 6.0          # px per frame
CAPTION_BAND = 0.82     # callout box bottoms stay above this fraction of H
HDR_EDGE, HDR_PX, HDR_MAXW = 22, 30, 0.60
NOTE_PX, NOTE_TOP, NOTE_EDGE = 20, 0.955, 22


def wrap(text, font, maxw):
    words, rows, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw or not cur:
            cur = t
        else:
            rows.append(cur)
            cur = w
    if cur:
        rows.append(cur)
    return rows


def text_box(dr, x, y, rows, font, fpx, a):
    """box style shared by headers and the beam note; returns (x0, y0, x1, y1)."""
    padx, pady = int(round(fpx / 3)), int(round(fpx * 0.19))
    lh = int(round(fpx * 1.25))
    w = max(font.getlength(r) for r in rows) + 2 * padx
    h = lh * len(rows) + 2 * pady + int(round(fpx * 0.18))
    al = int(255 * a)
    dr.rectangle([x, y, x + w, y + h], fill=BOX + (int(255 * BOX_A * a),), outline=LINE + (al,), width=1)
    for i, r in enumerate(rows):
        dr.text((x + padx, y + pady + i * lh), r, font=font, fill=TEXT + (al,))
    return x, y, x + w, y + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="callout JSON")
    ap.add_argument("--texts", default=None, help="header / beam note JSON")
    ap.add_argument("--frames-in", default="outputs/film_v4_frames")
    ap.add_argument("--frames-out", default="outputs/film_v4_frames_lab")
    ap.add_argument("--only", default=None, help="comma-separated frame numbers (test)")
    ap.add_argument("--font-px", type=int, default=27)
    ap.add_argument("--callout-range", default="1201-1436",
                    help="frames where callouts are allowed (owner 2026-10-10: S5 exploded view only); 'all' disables the guard")
    args = ap.parse_args()
    if not (args.json or args.texts):
        sys.exit("need --json and/or --texts")
    if os.path.abspath(args.frames_out) == os.path.abspath(args.frames_in):
        sys.exit("refusing to write over the clean frames")
    W = H = None
    frames, texts = {}, {}
    if args.json:
        J = json.load(open(args.json))
        W, H = int(J["W"]), int(J["H"])
        frames = {int(k): v for k, v in J["frames"].items()}
        if args.callout_range != "all":
            lo, hi = (int(x) for x in args.callout_range.split("-"))
            dropped = sorted(f for f in frames if not lo <= f <= hi)
            frames = {f: v for f, v in frames.items() if lo <= f <= hi}
            if dropped:
                print("callout guard %d-%d: ignored callouts on %d frames (%d-%d)" % (lo, hi, len(dropped), dropped[0], dropped[-1]))
    if args.texts:
        T = json.load(open(args.texts))
        if W is not None and (int(T["W"]), int(T["H"])) != (W, H):
            sys.exit("callout and texts JSON sizes differ")
        W, H = int(T["W"]), int(T["H"])
        texts = {int(k): v for k, v in T["frames"].items()}
    fpx = args.font_px
    font = ImageFont.truetype(FONT, fpx)
    hfont, nfont = ImageFont.truetype(FONT, HDR_PX), ImageFont.truetype(FONT, NOTE_PX)
    hgt = int(round(fpx * 1.43))                      # hero: 21 px font, 30 px box
    padx, pady = int(round(fpx / 3)), int(round(fpx * 0.19))
    knee_dx = int(round(fpx * 0.76))                  # hero: 16 px at 21 px
    ybot = CAPTION_BAND * H - hgt - 4

    first_side, first_y, first_f = {}, {}, {}
    for f in sorted(frames):
        for r in frames[f]:
            if r["text"] not in first_side:
                first_side[r["text"]], first_y[r["text"]], first_f[r["text"]] = r.get("side", "L"), r["y"], f
    order = {s: sorted([t for t in first_side if first_side[t] == s], key=lambda t: (first_y[t], first_f[t])) for s in ("L", "R")}

    only = set(int(x) for x in args.only.split(",")) if args.only else None
    os.makedirs(args.frames_out, exist_ok=True)
    prev_y, prev_f = {}, None
    n, cut, n_hdr, n_note = 0, [], 0, 0
    for f in sorted(set(frames) | set(texts)):
        if only and f not in only:
            continue
        src = os.path.join(args.frames_in, "f%04d.png" % f)
        if not os.path.exists(src):
            cut.append((f, "missing frame"))
            continue
        img = Image.open(src).convert("RGBA")
        if img.size != (W, H):
            sys.exit("frame %s is %s, JSON says %dx%d" % (src, img.size, W, H))
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)

        if f in frames:                                # ---- callouts
            rows = {r["text"]: r for r in frames[f]}
            contiguous = prev_f is not None and f == prev_f + 1
            new_y = {}
            for s in ("L", "R"):
                names = [t for t in order[s] if t in rows]
                if not names:
                    continue
                ys = []
                for t in names:                        # target from the anchor, rate-limited from the last frame
                    tgt = rows[t]["y"] - hgt / 2
                    if contiguous and t in prev_y:
                        p = prev_y[t]
                        tgt = p + max(-MAX_STEP, min(MAX_STEP, tgt - p))
                    ys.append(tgt)
                for i in range(1, len(ys)):            # minimum gap, fixed order
                    ys[i] = max(ys[i], ys[i - 1] + hgt + GAP)
                over = ys[-1] - ybot
                if over > 0:                           # keep above the caption band
                    ys = [y - over for y in ys]
                ys[0] = max(TOP, ys[0])
                for i in range(1, len(ys)):
                    ys[i] = max(ys[i], ys[i - 1] + hgt + GAP)
                for t, y in zip(names, ys, strict=True):
                    new_y[t] = y
                    r = rows[t]
                    a = max(0.0, min(1.0, float(r.get("a", 1.0))))
                    if a <= 0:
                        continue
                    w = font.getlength(t) + 2 * padx
                    x = EDGE if s == "L" else W - EDGE - w
                    ax, ay = r["x"], r["y"]
                    ex = x + w if s == "L" else x
                    knee = ex + (knee_dx if s == "L" else -knee_dx)
                    al = int(255 * a)
                    dr.line([(ax, ay), (knee, y + hgt / 2), (ex, y + hgt / 2)], fill=LINE + (al,), width=2)
                    dr.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=LINE + (al,))
                    dr.rectangle([x, y, x + w, y + hgt], fill=BOX + (int(255 * BOX_A * a),), outline=LINE + (al,), width=1)
                    dr.text((x + padx, y + pady), t, font=font, fill=TEXT + (al,))
                    if x < 0 or x + w > W or y < 0 or y + hgt > CAPTION_BAND * H or not (0 <= ax <= W and 0 <= ay <= H):
                        cut.append((f, t))
            prev_y, prev_f = new_y, f

        for r in texts.get(f, []):                     # ---- header / beam note
            a = max(0.0, min(1.0, float(r.get("a", 1.0))))
            if a <= 0:
                continue
            if r["kind"] == "header":
                rows_h = wrap(r["text"], hfont, HDR_MAXW * W - 2 * round(HDR_PX / 3))
                box = text_box(dr, HDR_EDGE, HDR_EDGE, rows_h, hfont, HDR_PX, a)
                n_hdr += 1
            elif r["kind"] == "beamnote":
                box = text_box(dr, NOTE_EDGE, int(round(NOTE_TOP * H)), [r["text"]], nfont, NOTE_PX, a)
                n_note += 1
            else:
                cut.append((f, "unknown kind %s" % r["kind"]))
                continue
            if box[2] > W or box[3] > H:
                cut.append((f, r["text"][:24]))

        Image.alpha_composite(img, ov).convert("RGB").save(os.path.join(args.frames_out, "f%04d.png" % f))
        n += 1
    print("OVERLAY_DONE frames %d, callout labels %d (L %d, R %d), headers %d, beam notes %d, outside/cut: %d %s"
          % (n, len(first_side), len(order["L"]), len(order["R"]), n_hdr, n_note, len(cut), cut[:6]))


if __name__ == "__main__":
    main()
