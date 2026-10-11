"""Burn narration captions onto the silent picture in post (DevLog-020 W2; DevLog-014 section 14).

The local ffmpeg has no libass / drawtext, so captions are rendered as RGBA PNGs with Pillow and composited with the
overlay filter (one input per caption, alpha fades, enable='between(t,a,b)').

Run (project root):
  uv run --no-project --with pillow python scripts/post/burn_captions.py --video outputs/film_v4_<date>.mp4 \
      --out outputs/film_v4_<date>_av.mp4 [--audio outputs/audio/mix_v4_<date>.wav] [--narration scripts/audio/narration.json]
      [--skip s7a] [--t0 A --t1 B] [--keep-png DIR]

Captions: one narration line at a time ("show" text), from narration.json t0/t1 (the measured VO clock): on at t0 - LEAD,
off at min(t1 + HOLD, next t0 - GAP), 0.12 s alpha fades, never overlapping. Lines wider than 88 percent of the frame
are wrapped into two balanced rows. Text: Arial Bold, white, solid black stroke max(6, 0.09 x font px) all round, soft shadow under it, rendered at 2x and
downsampled; overlay in planar RGB (gbrp), yuv420p only at the end; in the bottom 18 percent band, ink kept above 0.945 H
(the post beam note sits at 0.955 H, callout_overlay.py). Fast lines (the S7 disclaimer, its text is on the rendered card) are skipped by default.
Output: H.264 yuv420p (CRF 18, preset medium), same fps; with --audio: AAC 192 kb/s 48 kHz, +faststart.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
LEAD, HOLD, GAP, FADE = 0.05, 0.35, 0.08, 0.12
BAND = 0.18          # bottom band, fraction of frame height
SS = 2               # supersampling factor for the caption render
CAP_BOTTOM = 0.945   # caption ink (incl. stroke and shadow) ends above this fraction of H; the beam note box starts at 0.955 H
MAXW = 0.88          # max text width, fraction of frame width


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den, float(j["format"]["duration"])


def wrap(text, font, maxw, draw):
    if draw.textlength(text, font=font) <= maxw:
        return [text]
    words = text.split()
    best = None
    for i in range(1, len(words)):                       # most balanced two-row split that fits
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa, wb = draw.textlength(a, font=font), draw.textlength(b, font=font)
        if max(wa, wb) <= maxw and (best is None or abs(wa - wb) < best[0]):
            best = (abs(wa - wb), [a, b])
    return best[1] if best else [text]


def render_png(text, W, H, path):
    """Caption strip (W x band) as straight-alpha RGBA. Rendered at SS x and downsampled (Lanczos, premultiplied
    internally by Pillow) so the stroke edge is smooth; solid black stroke of uniform width all around every glyph
    (max(6, 0.09 x font px) at 1x); soft drop shadow drawn UNDER the stroke (never instead of it); the canvas is the
    full band, and the text block is checked to sit inside it with stroke + shadow padding so nothing is clipped."""
    band_h = int(round(BAND * H))
    size = int(round(0.05 * W))                          # 54 px at 1080 wide
    measure = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    for scale in (1.0, 0.88, 0.78):                      # shrink if two rows still do not fit the width
        fs = int(round(size * scale))
        font = ImageFont.truetype(FONT, fs)
        rows = wrap(text, font, MAXW * W, measure)
        if max(measure.textlength(r, font=font) for r in rows) <= MAXW * W and len(rows) <= 2:
            break
    sw = max(6, round(0.09 * fs))                        # stroke at 1x
    sh_off, sh_blur = (3, 4), 4
    pad = sw + sh_blur * 2 + max(sh_off)
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * 1.08) + sw
    block_h = lh * len(rows) + 2 * pad
    assert block_h <= band_h, "caption block %d px taller than band %d px: %r" % (block_h, band_h, text)
    k = SS
    fk = ImageFont.truetype(FONT, fs * k)
    img = Image.new("RGBA", (W * k, band_h * k), (0, 0, 0, 0))
    limit = int(CAP_BOTTOM * H) - (H - band_h)        # strip coords: caption ink must end above the beam-note row
    y0 = min((band_h - lh * len(rows)) // 2, limit - lh * len(rows) - sw)
    assert y0 >= pad, "caption block does not fit between the band top and %.3f H: %r" % (CAP_BOTTOM, text)
    pos = []
    for i, r in enumerate(rows):
        wpx = measure.textlength(r, font=font)
        x = (W - wpx) / 2
        assert x >= pad, "caption row too wide for padding: %r" % r
        pos.append((x * k, (y0 + i * lh) * k, r))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    for x, y, r in pos:
        ds.text((x + sh_off[0] * k, y + sh_off[1] * k), r, font=fk, fill=(0, 0, 0, 255), stroke_width=sw * k, stroke_fill=(0, 0, 0, 255))
    shadow = shadow.filter(ImageFilter.GaussianBlur(sh_blur * k))
    alpha = shadow.getchannel("A").point(lambda v: int(v * 0.55))
    shadow.putalpha(alpha)
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img)
    for x, y, r in pos:
        d.text((x, y), r, font=fk, fill=(255, 255, 255, 255), stroke_width=sw * k, stroke_fill=(0, 0, 0, 255))
    img = img.convert("RGBa").resize((W, band_h), Image.LANCZOS).convert("RGBA")   # premultiplied resize, straight alpha out
    m = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    assert m is None or m[3] <= limit, "caption ink reaches %d px, limit %d: %r" % (m[3], limit, text)
    img.save(path)
    return band_h, len(rows)


def schedule(lines, skip, t_end):
    caps = [L for L in lines if L["id"] not in skip and not L.get("fast")]
    out = []
    for i, L in enumerate(caps):
        a = max(0.0, L["t0"] - LEAD)
        nxt = caps[i + 1]["t0"] - LEAD - GAP if i + 1 < len(caps) else t_end
        b = min(L["t1"] + HOLD, nxt, t_end)
        out.append((L["id"], L["show"], round(a, 3), round(b, 3)))
    for (_, _, _, b), (_, _, a2, _) in zip(out, out[1:], strict=False):
        assert b <= a2, "caption overlap"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--narration", default=os.path.join(PROJ, "scripts", "audio", "narration.json"))
    ap.add_argument("--skip", default="s7a", help="comma-separated line ids not to caption")
    ap.add_argument("--t0", type=float, default=None, help="test: only encode from t0")
    ap.add_argument("--t1", type=float, default=None, help="test: only encode to t1")
    ap.add_argument("--keep-png", default=None)
    args = ap.parse_args()
    W, H, fps, dur = probe(args.video)
    lines = json.load(open(args.narration))["lines"]
    caps = schedule(lines, set(filter(None, args.skip.split(","))), dur)
    tmp = args.keep_png or tempfile.mkdtemp(prefix="caps_")
    os.makedirs(tmp, exist_ok=True)
    ins, fc, last = ["-i", args.video], ["[0:v]format=gbrp[base]"], "base"
    if args.t0 is not None or args.t1 is not None:
        lo, hi = args.t0 or 0.0, args.t1 or dur
        caps = [c for c in caps if c[3] > lo and c[2] < hi]
    for k, (cid, text, a, b) in enumerate(caps):
        png = os.path.join(tmp, "%02d_%s.png" % (k, cid))
        band_h, nrows = render_png(text, W, H, png)
        d = b - a
        ins += ["-loop", "1", "-framerate", "%g" % fps, "-t", "%.3f" % d, "-i", png]
        fc.append("[%d:v]format=rgba,fade=t=in:st=0:d=%g:alpha=1,fade=t=out:st=%.3f:d=%g:alpha=1,setpts=PTS-STARTPTS+%.3f/TB[c%d]"
                  % (k + 1, FADE, max(0.0, d - FADE), FADE, a, k))
        fc.append("[%s][c%d]overlay=x=0:y=%d:format=gbrp:eof_action=pass:enable='between(t,%.3f,%.3f)'[v%d]" % (last, k, H - band_h, a, b, k))
        last = "v%d" % k
        print("%-4s %6.2f-%6.2f  rows %d  %s" % (cid, a, b, nrows, text))
    fc.append("[%s]format=yuv420p[vout]" % last)
    last = "vout"
    fcs = ";".join(fc)
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y"]
    cmd += ins
    amap = []
    if args.audio:
        cmd += ["-i", args.audio]
        amap = ["-map", "%d:a:0" % (len(caps) + 1), "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-filter_complex", fcs, "-map", "[%s]" % last, *amap, "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "%g" % fps, "-movflags", "+faststart"]
    if args.t0 is not None or args.t1 is not None:
        cmd += ["-ss", "%.3f" % (args.t0 or 0.0), "-to", "%.3f" % (args.t1 or dur)]
    else:
        cmd += ["-t", "%.3f" % dur]
    cmd += [args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    print("wrote", os.path.relpath(args.out, PROJ), "(%d captions)" % len(caps))
    if not args.keep_png:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
