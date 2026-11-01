"""Composite the CPO silhouette / reveal onto the Who's That Pokemon template.

Usage: uv run --with numpy --with pillow python scripts/make_video.py [--preview f1,f2,...]
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = next(ROOT.glob("Who*Template V2.mp4"))
OUT = ROOT / "outputs"
W, H, FPS, NF = 1280, 720, 30, 521

CENTER = (365, 312)  # burst center (px), measured from template frames
WIDTH = 600  # steady-state CPO width (px)
APPEAR_F = 20  # burst first fully formed
REVEAL_F = 259  # sting-2 onset, 8.64 s
# burst zoom overshoot: measured white-core width / steady width (~355 px), f20-29
ZOOM = {20: 0.92, 21: 0.92, 22: 1.55, 23: 1.66, 24: 1.61, 25: 1.39, 26: 1.22, 27: 1.24, 28: 1.10, 29: 1.0}
ZOOM_DAMP = 0.4  # CPO is wider than the burst; follow only 40% of its overshoot
# silence trims (template frame indices; cut on frame boundaries to keep A/V sync)
# template digital silence: 5.872-8.573 s and 14.370-17.438 s; both trimmed to ~0.5 s
GAP_CUT = (179, 245)  # drop f179-244 (2.200 s); f179 picked as least-visible jump in the silent window
END_F = 446  # last kept frame is f445 -> 0.497 s tail silence
# narrator name line, mixed at unity gain into the template name slot (output timing)
NARRATOR = ["narrator_clips/marvell_1m29.wav", "narrator_clips/cpo_0m47.wav"]  # "Marvell" + "co-package optics"
NAME_T = 7.60  # s; template "It's" voice ends 9.80 s -> 7.60 s after GAP_CUT
NAME_GAP = 0.10  # s between the two clips
FLASH = [1.0, 0.6, 0.25]  # white blend on the CPO for the first reveal frames


def load(name: str, width: int) -> np.ndarray:
    im = Image.open(OUT / name)
    h = round(im.height * width / im.width)
    return np.asarray(im.resize((width, h), Image.LANCZOS)).astype(np.float32) / 255


def audio_filter() -> str:
    t = [GAP_CUT[0] / FPS, GAP_CUT[1] / FPS, END_F / FPS]
    return (
        f"[1:a]atrim=0:{t[0]:.6f},asetpts=PTS-STARTPTS[a0];"
        f"[1:a]atrim={t[1]:.6f}:{t[2]:.6f},asetpts=PTS-STARTPTS[a1];"
        "[a0][a1]concat=n=2:v=0:a=1[bed];"
        "[2:a]aresample=44100,aformat=channel_layouts=stereo[n0];"
        f"[3:a]aresample=44100,aformat=channel_layouts=stereo,adelay={NAME_GAP * 1000:.0f}:all=1[n1];"
        f"[n0][n1]concat=n=2:v=0:a=1,adelay={NAME_T * 1000:.0f}:all=1[name];"
        "[bed][name]amix=inputs=2:duration=first:normalize=0[a]"
    )


def over(frame: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    x0, y0 = CENTER[0] - w // 2, CENTER[1] - h // 2
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + w, W), min(y0 + h, H)
    sub = rgba[ys - y0 : ye - y0, xs - x0 : xe - x0]
    a = sub[..., 3:4]
    region = frame[ys:ye, xs:xe]
    frame[ys:ye, xs:xe] = region * (1 - a) + sub[..., :3] * a
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated frame numbers to dump as PNG instead of encoding")
    args = ap.parse_args()
    preview = {int(x) for x in args.preview.split(",")} if args.preview else None

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(TEMPLATE), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    enc = None
    if preview is None:
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-i", str(TEMPLATE), *[a for c in NARRATOR for a in ("-i", str(OUT / c))], "-filter_complex", audio_filter(), "-map", "0:v", "-map", "[a]",
             "-c:a", "aac", "-b:a", "192k",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-r", str(FPS), "-movflags", "+faststart", str(OUT / "whos_that_cpo.mp4")],
            stdin=subprocess.PIPE,
        )

    sil_full = load("cpo_silhouette.png", WIDTH)
    col_full = load("cpo_cutout.png", WIDTH)
    for n in range(NF):
        raw = dec.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            break
        f8 = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        frame = f8.astype(np.float32) / 255
        if n >= APPEAR_F:
            sil = load("cpo_silhouette.png", round(WIDTH * (1 + ZOOM_DAMP * (ZOOM[n] - 1)))) if n in ZOOM else sil_full
            if n < REVEAL_F:
                frame = over(frame, sil)
            else:
                k = n - REVEAL_F
                c = col_full.copy()
                if k < len(FLASH):
                    c[..., :3] = c[..., :3] * (1 - FLASH[k]) + FLASH[k]
                frame = over(frame, c)
        if GAP_CUT[0] <= n < GAP_CUT[1] or n >= END_F:
            continue
        out = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        if preview is not None:
            if n in preview:
                Image.fromarray(out).save(OUT / f"preview_{n:03d}.png")
        else:
            enc.stdin.write(out.tobytes())
    dec.stdout.close()
    dec.wait()
    if enc is not None:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    main()
