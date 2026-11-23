# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Verify the built sound files and write the manifest, the credits registry and the audition page.

Run from the project root (after build_music.py and make_sfx.py):  uv run course/tools/sound/publish.py
Writes:
  course/media/sound/manifest.js        window.LMS_SOUND (paths relative to course/)
  course/media/credits_sound.js         window.LMS_MEDIA entries, kind 'audio'
  research/sound_samples/index.html     static audition page (players + role + numbers)
  course/tools/sound/logs/verify_<timestamp>.json   ffprobe + loudness of every file
Loudness: BS.1770 integrated (music) and max momentary (SFX) measured on the decoded mp3, plus ffmpeg ebur128
integrated / true peak as an independent check for the music.
"""
import datetime as dt
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sound_common import COURSE, HERE, SAMPLES_DIR, SOUND_DIR, decode, ffmpeg_ebur128, loudness, probe  # noqa: E402

MACLEOD_LICENSE = "CC BY 4.0"
LIC_URL = "http://creativecommons.org/licenses/by/4.0/"

MUSIC = {
    "intro": {
        "file": "music_intro_loop.mp3", "loop": True,
        "role": "Intro bed (course title and welcome slides), seamless loop",
        "title": "Life of Riley",
        "credit": '"Life of Riley" Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 '
                  '(http://creativecommons.org/licenses/by/4.0/); excerpt looped and level adjusted',
        "source": "https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400054",
    },
    "outro": {
        "file": "music_outro.mp3", "loop": False,
        "role": "Outro with ending (certificate and final pages)",
        "title": "Cheery Monday",
        "credit": '"Cheery Monday" Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 '
                  '(http://creativecommons.org/licenses/by/4.0/); final section excerpted and level adjusted',
        "source": "https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1700065",
    },
}
ALTS = {
    "alt_intro_loop_carefree.mp3": ("Alternate intro loop (not in the course)", '"Carefree" Kevin MacLeod (incompetech.com), CC BY 4.0; excerpt looped',
                                    "https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400037"),
    "alt_outro_easy_lemon_30.mp3": ("Alternate outro (not in the course)", '"Easy Lemon (30 second)" Kevin MacLeod (incompetech.com), CC BY 4.0',
                                    "https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1200078"),
}
SFX = {
    "correct": ("sfx_correct.mp3", "Correct answer: two-note chime (A5 to E6, marimba and glockenspiel)"),
    "wrong": ("sfx_wrong.mp3", "Wrong answer: soft low wooden bonk"),
    "reveal": ("sfx_reveal.mp3", "Stage reveal / click to reveal: light rising whoosh with a soft pop"),
    "message": ("sfx_message.mp3", "New chat message: two quick bubble pops"),
    "popup": ("sfx_popup.mp3", "Ask popup appears: soft single ding"),
    "sortPlace": ("sfx_sort_place.mp3", "Sort item placed: soft wooden click"),
    "moduleDone": ("sfx_module_done.mp3", "Module complete: marimba arpeggio into a glockenspiel chord"),
    "certificate": ("sfx_certificate.mp3", "Certificate awarded: short brass fanfare with timpani"),
}
MEME_KEYS = ("memeBoom", "sadTrombone", "perfectScore")     # to be supplied by the course owner; null until then
GAIN = {"music": 0.28, "sfx": 0.5}


def measure(path):
    p = probe(path)
    x = decode(path, p["sample_rate"], p["channels"])
    L = loudness(x, p["sample_rate"])
    p.update(I=None if L["I"] is None else round(L["I"], 1), M_max=round(L["M_max"], 1), peak_dbfs=round(L["peak_dbfs"], 1))
    return p


def main():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, report = [], {}
    for key, m in MUSIC.items():
        f = SOUND_DIR / m["file"]
        v = measure(f)
        v["ffmpeg"] = ffmpeg_ebur128(f)
        report[f"music.{key}"] = v
        rows.append((f"music.{key}", m["role"], f"media/sound/{m['file']}", v, m["credit"], m["source"]))
    for key, (fn, role) in SFX.items():
        f = SOUND_DIR / fn
        v = measure(f)
        report[f"sfx.{key}"] = v
        rows.append((f"sfx.{key}", role, f"media/sound/{fn}", v, "Synthesized in code for this course", "Made for this course"))
    alt_rows = []
    for fn, (role, credit, src) in ALTS.items():
        f = SAMPLES_DIR / fn
        if f.exists():
            v = measure(f)
            v["ffmpeg"] = ffmpeg_ebur128(f)
            report[f"alt.{fn}"] = v
            alt_rows.append((fn, role, fn, v, credit, src))

    # ---- manifest
    man = {
        "music": {k: f"media/sound/{m['file']}" for k, m in MUSIC.items()},
        "loop": {k: m["loop"] for k, m in MUSIC.items()},
        "sfx": {**{k: f"media/sound/{fn}" for k, (fn, _) in SFX.items()}, **{k: None for k in MEME_KEYS}},
        "gain": GAIN,
    }
    js = ("/* Course sound manifest (paths relative to course/). Music: integrated -20 LUFS, stereo 128 kbps mp3; the intro\n"
          "   is a seamless loop (for a gapless loop decode with Web Audio and loop the AudioBuffer; HTMLAudioElement.loop\n"
          "   may leave a short gap). SFX: mono 96 kbps mp3, max momentary loudness -24 LUFS (sort click about -29.5,\n"
          "   peak-capped). Gains are relative to narration at 1.0 (narration measured about -18 LUFS integrated):\n"
          "   music 0.28 puts the bed near -31 LUFS, about 13 LU under the voice; sfx 0.5 puts effects near -30 LUFS\n"
          "   momentary. Meme effects (memeBoom, sadTrombone, perfectScore) are null until the course owner supplies them.\n"
          f"   Generated by tools/sound/publish.py, {dt.date.today().isoformat()}. */\n"
          f"window.LMS_SOUND = {json.dumps(man, indent=2)};\n")
    (SOUND_DIR / "manifest.js").write_text(js)

    # ---- credits registry
    reg = {}
    for key, m in MUSIC.items():
        reg[f"media/sound/{m['file']}"] = {
            "alt": f"Music: {m['role'].split(' (')[0].lower()}, \"{m['title']}\" by Kevin MacLeod",
            "credit": m["credit"], "license": MACLEOD_LICENSE, "source": m["source"], "kind": "audio",
            "duration": report[f"music.{key}"]["duration"]}
    for key, (fn, role) in SFX.items():
        reg[f"media/sound/{fn}"] = {
            "alt": f"Sound effect: {role.split(':')[0].lower()}",
            "credit": "Sound effect synthesized in code for this course", "license": "Made for this course",
            "source": "Made for this course", "kind": "audio", "duration": report[f"sfx.{key}"]["duration"]}
    cj = ("/* Credits for course music and sound effects (media/sound/). Music license verified 2026-11-22 on\n"
          "   https://incompetech.com/music/royalty-free/licenses/ (Creative Commons: By Attribution 4.0). */\n"
          f"window.LMS_MEDIA = Object.assign(window.LMS_MEDIA || {{}}, {json.dumps(reg, indent=2, ensure_ascii=False)});\n")
    (COURSE / "media" / "credits_sound.js").write_text(cj)

    # ---- audition page
    def tr(rid, role, src, v, credit, source):
        lufs = f"{v['I']} LUFS int." if rid.startswith(("music", "alt")) else f"{v['M_max']} LUFS mom. max"
        link = f'<a href="{html.escape(source)}">source</a>' if source.startswith("http") else html.escape(source)
        return (f"<tr><td><code>{html.escape(rid)}</code></td><td>{html.escape(role)}</td>"
                f"<td><audio controls preload=\"none\" src=\"{html.escape(src)}\"></audio></td>"
                f"<td>{v['duration']:.2f} s</td><td>{v['channels']} ch, {v['sample_rate'] // 1000} kHz</td>"
                f"<td>{lufs}, peak {v['peak_dbfs']} dBFS</td><td>{html.escape(credit)} ({link})</td></tr>")
    body = "\n".join(tr(r[0], r[1], "../../course/" + r[2], *r[3:]) for r in rows)
    alts = "\n".join(tr(r[0], r[1], r[2], *r[3:]) for r in alt_rows)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Course Sound Samples</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--mut:#5b6474;--line:#d9dde4;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e6e8ec;--mut:#9aa3b2;--line:#2b313b;--card:#1b1f26}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:15px;margin:20px 0 6px}} p{{color:var(--mut);margin:4px 0 12px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;background:var(--card);width:100%}}
td,th{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}}
th{{font-weight:600;color:var(--mut)}} audio{{height:32px;width:240px}} code{{font-size:12px}} a{{color:inherit}}
</style></head><body>
<h1>Course sound samples</h1>
<p>Files listed in <code>course/media/sound/manifest.js</code>. Suggested gains relative to narration 1.0: music {GAIN['music']}, sfx {GAIN['sfx']}. Narration clips measure about -18 LUFS. Meme effects (memeBoom, sadTrombone, perfectScore): not yet supplied.</p>
<div class="wrap"><table><thead><tr><th>Key</th><th>Role</th><th>Play</th><th>Length</th><th>Format</th><th>Level</th><th>Credit</th></tr></thead>
<tbody>
{body}
</tbody></table></div>
<h2>Alternates (audition only, not used by the course)</h2>
<div class="wrap"><table><thead><tr><th>File</th><th>Role</th><th>Play</th><th>Length</th><th>Format</th><th>Level</th><th>Credit</th></tr></thead>
<tbody>
{alts}
</tbody></table></div>
<p>Loop test: press play on the intro and let it wrap (the player loops it).</p>
<script>document.querySelectorAll('audio').forEach(a=>{{if(/loop/.test(a.src))a.loop=true}});</script>
</body></html>
"""
    (SAMPLES_DIR / "index.html").write_text(page)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"verify_{stamp}.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
