"""Writes media/cast.js and media/credits_avatars.js from the cast table below and draw.CHANNELS.
Usage (from this folder): uv run --no-project --with cairosvg --with pillow python write_cast.py ../../media
"""
import json, sys, importlib.util
MEDIA = sys.argv[1]
spec = importlib.util.spec_from_file_location("draw", "draw.py"); draw = importlib.util.module_from_spec(spec); spec.loader.exec_module(draw)
A = "media/avatars/"
DRAWN = "Drawn for this course"
credits = {}
def reg(fn, alt, credit=DRAWN, license=DRAWN, source="Drawn in code for this course", kind="drawing"):
    credits[A + fn] = dict(alt=alt, credit=credit, license=license, source=source, kind=kind, w=256, h=256)
reg("gary_bunnysuit.jpg", "Profile picture: a cleanroom worker in a full white hood, seen from behind, leaning into a microscope (cropped)",
    credit="MgCastellano, via Wikimedia Commons, public domain", license="Public domain",
    source="https://commons.wikimedia.org/wiki/File:Operator_at_microscope.JPG", kind="photo")
reg("newhire_corgi.jpg", "Profile picture: a Welsh Corgi puppy standing on its hind legs with its tongue out (cropped)",
    credit="Huoadg5888 (Pixabay), minor edits by Subsidiary account, via Wikimedia Commons, CC0", license="CC0",
    source="https://commons.wikimedia.org/wiki/File:Fawn_and_white_Welsh_Corgi_puppy_standing_on_rear_legs_and_sticking_out_the_tongue.jpg", kind="photo")
reg("prevshift_moon.jpg", "Profile picture: the full moon on a black sky",
    credit="Gregory H. Revera, via Wikimedia Commons, CC BY-SA 3.0", license="CC BY-SA 3.0",
    source="https://commons.wikimedia.org/wiki/File:FullMoon2010.jpg", kind="photo")
reg("dana_okafor.jpg", "Illustrated profile picture: a smiling person with dark brown skin, round glasses, natural hair, a teal sweater and a lanyard")
reg("morgan_manager.jpg", "Illustrated profile picture: a man in a navy suit, white shirt and red tie with neatly parted brown hair, on a grey studio background")
reg("pat_reynolds.jpg", "Illustrated profile picture: a woman with a silver bob, glasses, pearl earrings and a dark blue blazer, on a lavender background")
reg("org_five_sigma.jpg", "Company emblem: a white bell curve on navy, a dashed amber line at five sigma and one amber dot beyond it")
reg("org_office_photonic_conduct.jpg", "Office emblem: a gold balance scale inside a cream and maroon seal")
reg("org_partner_lab.jpg", "Laboratory crest: a navy shield with a gold star and a lens focusing light")
reg("vendor_lumenfold_lasers.jpg", "Vendor emblem: an amber laser beam folded into a Z by two mirrors, on teal")
reg("vendor_northcut_crystal.jpg", "Supplier emblem: a pale yellow crystal wafer with a flat and a triangular lattice pattern, on dark blue-grey")
ICON_ALT = dict(chart="a bar chart with a rising line", wrench="a wrench", gear="a gear", suit="a hooded cleanroom suit",
    layout="a waveguide bend in a layout frame", scope="an oscilloscope trace", rf="an antenna with radio waves", book="an open book",
    mesh="a triangular simulation mesh", megaphone="a megaphone", screen="a presentation screen with a chart", laser="a laser emitting a beam",
    mzm="a Mach-Zehnder interferometer", magnifier="a magnifying glass with a check mark", badge="a shield with a check mark", box="a shipping box",
    flask="a laboratory flask", trophy="a trophy", lifebuoy="a lifebuoy", ring="a ring resonator beside a bus waveguide")
cast = {}
for name, (fn, bg, icon) in draw.CHANNELS.items():
    reg(fn + ".jpg", f"Channel icon: {ICON_ALT[icon]}, white on a colored square")
    bare = name.lstrip("#")
    cast[name] = dict(avatar=A + fn + ".jpg", aliases=[bare, "# " + bare])
people = {
    "Gary": dict(avatar=A + "gary_bunnysuit.jpg", aliases=["Notes to self", "Gary (Engineer)", "Gary (me)"]),
    "Dana Okafor (Labmate)": dict(avatar=A + "dana_okafor.jpg", aliases=["Dana", "Dana Okafor", "Dana (Labmate)", "Dana Okafor, Gary's labmate"]),
    "Morgan (Manager)": dict(avatar=A + "morgan_manager.jpg", aliases=["Manager", "Morgan", "Gary's manager"]),
    "Pat Reynolds": dict(avatar=A + "pat_reynolds.jpg", aliases=["Pat Reynolds (Host)", "Host", "Pat"]),
    "New hire": dict(avatar=A + "newhire_corgi.jpg", aliases=["New hire (Process)"]),
    "Previous shift": dict(avatar=A + "prevshift_moon.jpg", aliases=["Night shift"]),
    "Partner lab": dict(avatar=A + "org_partner_lab.jpg", aliases=["Partner Lab", "Collaborator lab"]),
    "Office of Photonic Conduct": dict(avatar=A + "org_office_photonic_conduct.jpg", aliases=["Photonic Conduct", "OPC"]),
    "Five Sigma Photonics": dict(avatar=A + "org_five_sigma.jpg", aliases=["Five Sigma Photonics, Inc.", "Five Sigma"]),
    "Lumenfold Lasers": dict(avatar=A + "vendor_lumenfold_lasers.jpg", aliases=["Laser vendor", "Vendor", "Vendor support"]),
    "Northcut Crystal": dict(avatar=A + "vendor_northcut_crystal.jpg", aliases=["Wafer supplier", "Supplier", "New supplier"]),
}
cast = {**people, **cast}
hdr = "/* Chat profile pictures (media/avatars/). Keys are exact chat names (message from, participants, chat headers); aliases are matched case-insensitively. Code-drawn portraits and emblems; non-face photos and a hooded cleanroom crop from Wikimedia Commons (credits in credits_avatars.js). 2026-11-22. */\n"
with open(f"{MEDIA}/cast.js", "w") as f:
    f.write(hdr + "window.LMS_CAST = Object.assign(window.LMS_CAST || {}, " + json.dumps(cast, indent=2) + ");\n")
hdr2 = "/* Credits for chat profile pictures (media/avatars/). Drawings are generated by deterministic code for this course; photos verified from Wikimedia Commons file metadata (extmetadata), 2026-11-22. */\n"
with open(f"{MEDIA}/credits_avatars.js", "w") as f:
    f.write(hdr2 + "window.LMS_MEDIA = Object.assign(window.LMS_MEDIA || {}, " + json.dumps(credits, indent=2) + ");\n")
print(len(cast), "cast keys;", len(credits), "credits")
