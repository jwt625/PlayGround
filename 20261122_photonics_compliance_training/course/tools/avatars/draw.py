"""Deterministic code-drawn avatars (flat vector portraits and emblems) -> 256x256 JPEG.
Usage: uv run --no-project --with cairosvg --with pillow python draw.py OUTDIR [only_name]
"""
import io
import sys

import cairosvg
from PIL import Image

W = 512


def svg(body, defs=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}">'
            f'<defs>{defs}</defs>{body}</svg>')


def shade(hexc, f):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    if f < 1:
        r, g, b = (int(c * f) for c in (r, g, b))
    else:
        t = f - 1
        r, g, b = (int(c + (255 - c) * t) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------- portraits
HEAD = "M178 206 C176 118 336 118 334 206 C334 262 312 318 256 322 C200 318 178 262 178 206 Z"


def portrait(p):
    skin = p["skin"]
    sk_d = shade(skin, 0.84)
    hair = p["hair_color"]
    defs = ""
    out = []
    # background
    if p.get("bg_grad"):
        a, b = p["bg_grad"]
        defs += (f'<radialGradient id="bg" cx="50%" cy="38%" r="75%"><stop offset="0" stop-color="{a}"/>'
                 f'<stop offset="1" stop-color="{b}"/></radialGradient>')
        out.append(f'<rect width="{W}" height="{W}" fill="url(#bg)"/>')
    else:
        out.append(f'<rect width="{W}" height="{W}" fill="{p["bg"]}"/>')
    for extra in p.get("bg_extra", []):
        out.append(extra)
    out.append('<g transform="translate(256 252) scale(1.24) translate(-256 -232)">')
    # back hair (behind head and neck)
    style = p["hair"]
    if style == "afro":
        out.append(f'<circle cx="256" cy="176" r="128" fill="{hair}"/>'
                   f'<circle cx="160" cy="214" r="62" fill="{hair}"/><circle cx="352" cy="214" r="62" fill="{hair}"/>')
    elif style == "bob":
        out.append(f'<path d="M160 214 C150 92 362 92 352 214 L360 304 C360 318 340 322 330 316 L182 316 '
                   f'C172 322 152 318 152 304 Z" fill="{hair}"/>')
    # neck
    out.append(f'<path d="M222 286 L222 372 C236 384 276 384 290 372 L290 286 Z" fill="{sk_d}"/>')
    # body / clothing
    body = "M40 512 C44 430 110 386 214 368 L298 368 C402 386 468 430 472 512 Z"
    top = p["top"]
    if top == "suit":
        jacket = p["jacket"]
        out.append(f'<path d="{body}" fill="{jacket}"/>')
        # shirt V
        out.append(f'<path d="M212 368 L256 470 L300 368 Z" fill="{p["shirt"]}"/>')
        # collar points
        out.append(f'<path d="M218 362 L246 410 L256 384 L240 360 Z" fill="{shade(p["shirt"], 0.92)}"/>'
                   f'<path d="M294 362 L266 410 L256 384 L272 360 Z" fill="{shade(p["shirt"], 0.92)}"/>')
        if p.get("tie"):
            t = p["tie"]
            out.append(f'<path d="M246 392 L266 392 L262 410 L272 470 L256 492 L240 470 L250 410 Z" fill="{t}"/>'
                       f'<path d="M244 384 L268 384 L264 404 L248 404 Z" fill="{shade(t, 0.8)}"/>')
        # lapels
        jd = shade(jacket, 0.78)
        out.append(f'<path d="M206 372 L256 500 L232 512 L184 420 L200 404 L178 392 Z" fill="{jd}"/>'
                   f'<path d="M306 372 L256 500 L280 512 L328 420 L312 404 L334 392 Z" fill="{jd}"/>')
        if p.get("pin"):
            out.append(f'<circle cx="320" cy="430" r="7" fill="{p["pin"]}"/>')
    elif top == "sweater":
        c = p["sweater"]
        out.append(f'<path d="{body}" fill="{c}"/>')
        out.append(f'<path d="M212 366 C222 404 290 404 300 366" fill="none" stroke="{shade(c, 0.8)}" stroke-width="16" stroke-linecap="round"/>')
        if p.get("lanyard"):
            ly = p["lanyard"]
            out.append(f'<path d="M206 378 L246 486 M306 378 L266 486" stroke="{ly}" stroke-width="10" fill="none"/>'
                       f'<rect x="232" y="478" width="48" height="40" fill="#f4f4f0" stroke="{shade(ly, 0.8)}" stroke-width="3"/>'
                       f'<rect x="240" y="488" width="32" height="7" fill="{ly}"/>')
    elif top == "blouse":
        jacket = p["jacket"]
        out.append(f'<path d="{body}" fill="{jacket}"/>')
        out.append(f'<path d="M214 368 L256 452 L298 368 Z" fill="{p["shirt"]}"/>')
        jd = shade(jacket, 0.8)
        out.append(f'<path d="M210 372 L256 470 L240 512 L196 420 L184 390 Z" fill="{jd}"/>'
                   f'<path d="M302 372 L256 470 L272 512 L316 420 L328 390 Z" fill="{jd}"/>')
        if p.get("necklace"):
            out.append(f'<path d="M226 376 C236 420 276 420 286 376" fill="none" stroke="{p["necklace"]}" stroke-width="4"/>'
                       f'<circle cx="256" cy="414" r="7" fill="{p["necklace"]}"/>')
    # ears
    out.append(f'<ellipse cx="180" cy="222" rx="16" ry="24" fill="{skin}"/><ellipse cx="332" cy="222" rx="16" ry="24" fill="{skin}"/>')
    out.append(f'<ellipse cx="182" cy="224" rx="7" ry="12" fill="{sk_d}"/><ellipse cx="330" cy="224" rx="7" ry="12" fill="{sk_d}"/>')
    if p.get("earring"):
        out.append(f'<circle cx="178" cy="252" r="6" fill="{p["earring"]}"/><circle cx="334" cy="252" r="6" fill="{p["earring"]}"/>')
    # head
    out.append(f'<path d="{HEAD}" fill="{skin}"/>')
    # jaw shadow on neck
    out.append(f'<path d="M222 300 C236 326 276 326 290 300 L290 318 C276 334 236 334 222 318 Z" fill="{shade(skin, 0.74)}" opacity="0.5"/>')
    # front hair
    if style == "afro":
        out.append(f'<path d="M180 196 C176 112 336 112 332 196 C318 160 290 146 256 146 C222 146 194 160 180 196 Z" fill="{hair}"/>')
    elif style == "sidepart":
        out.append(f'<path d="M174 214 C162 110 238 82 300 98 C346 110 352 170 338 214 C334 186 326 166 316 152 '
                   f'C282 166 226 162 198 150 C188 168 180 190 174 214 Z" fill="{hair}"/>'
                   f'<path d="M222 104 C248 118 280 126 316 152" fill="none" stroke="{shade(hair, 1.25)}" stroke-width="4" opacity="0.7"/>')
    elif style == "bob":
        out.append(f'<path d="M176 230 C164 120 250 96 300 108 C346 120 354 170 336 230 C330 180 304 150 262 146 '
                   f'C240 172 206 186 184 190 C180 204 178 216 176 230 Z" fill="{hair}"/>'
                   f'<path d="M262 146 C238 176 210 188 186 192" fill="none" stroke="{shade(hair, 1.2)}" stroke-width="4" opacity="0.6"/>')
    elif style == "buzz":
        out.append(f'<path d="M180 192 C178 120 334 120 332 192 C320 156 290 140 256 140 C222 140 192 156 180 192 Z" fill="{hair}"/>')
    # brows
    bc = p.get("brow", shade(hair, 0.9) if style != "bob" or not p.get("silver") else "#8a8178")
    out.append(f'<path d="M206 186 C216 178 232 178 242 184" fill="none" stroke="{bc}" stroke-width="7" stroke-linecap="round"/>'
               f'<path d="M270 184 C280 178 296 178 306 186" fill="none" stroke="{bc}" stroke-width="7" stroke-linecap="round"/>')
    # eyes
    ec = "#2b2320"
    out.append(f'<ellipse cx="225" cy="212" rx="8" ry="10" fill="{ec}"/><ellipse cx="287" cy="212" rx="8" ry="10" fill="{ec}"/>'
               f'<circle cx="228" cy="208" r="2.6" fill="#ffffff"/><circle cx="290" cy="208" r="2.6" fill="#ffffff"/>')
    if p.get("lashes"):
        out.append(f'<path d="M212 206 L208 200 M298 200 L302 206" stroke="{ec}" stroke-width="4" stroke-linecap="round"/>')
    # nose
    out.append(f'<path d="M256 222 C252 240 248 252 252 258 C256 262 264 260 268 256" fill="none" stroke="{shade(skin, 0.7)}" stroke-width="5" stroke-linecap="round"/>')
    # cheeks
    if p.get("blush"):
        out.append(f'<ellipse cx="208" cy="250" rx="16" ry="9" fill="{p["blush"]}" opacity="0.35"/>'
                   f'<ellipse cx="304" cy="250" rx="16" ry="9" fill="{p["blush"]}" opacity="0.35"/>')
    # mouth
    lip = p.get("lip", shade(skin, 0.62))
    if p["mouth"] == "grin":
        out.append(f'<path d="M222 274 C236 306 276 306 290 274 Z" fill="#5a2a2a"/>'
                   f'<path d="M226 276 L286 276 C284 284 228 284 226 276 Z" fill="#ffffff"/>')
    elif p["mouth"] == "smile":
        out.append(f'<path d="M228 278 C242 294 270 294 284 278" fill="none" stroke="{lip}" stroke-width="7" stroke-linecap="round"/>')
    elif p["mouth"] == "closed_lips":
        out.append(f'<path d="M230 278 C244 292 268 292 282 278 C268 284 244 284 230 278 Z" fill="{lip}" stroke="{lip}" stroke-width="5" stroke-linejoin="round"/>')
    # glasses
    g = p.get("glasses")
    if g:
        gc = p.get("glasses_color", "#222222")
        if g == "round":
            out.append(f'<circle cx="225" cy="212" r="25" fill="#ffffff" fill-opacity="0.12" stroke="{gc}" stroke-width="6"/>'
                       f'<circle cx="287" cy="212" r="25" fill="#ffffff" fill-opacity="0.12" stroke="{gc}" stroke-width="6"/>'
                       f'<path d="M250 210 C254 204 258 204 262 210" fill="none" stroke="{gc}" stroke-width="5"/>'
                       f'<path d="M200 208 L182 204 M312 208 L330 204" stroke="{gc}" stroke-width="5"/>')
        else:
            out.append(f'<rect x="196" y="194" width="54" height="36" rx="6" fill="#ffffff" fill-opacity="0.12" stroke="{gc}" stroke-width="6"/>'
                       f'<rect x="262" y="194" width="54" height="36" rx="6" fill="#ffffff" fill-opacity="0.12" stroke="{gc}" stroke-width="6"/>'
                       f'<path d="M250 206 L262 206" stroke="{gc}" stroke-width="5"/>'
                       f'<path d="M196 204 L182 202 M316 204 L330 202" stroke="{gc}" stroke-width="5"/>')
    out.append('</g>')
    return svg("".join(out), defs)


PEOPLE = {
    # friendly coworker who speaks up
    "dana_okafor": dict(
        bg="#f2b880", bg_extra=['<circle cx="430" cy="90" r="120" fill="#f6c99a"/>'],
        skin="#7b4a2f", hair="afro", hair_color="#1d1512", top="sweater", sweater="#2f7f7a",
        lanyard="#d9573b", mouth="grin", glasses="round", glasses_color="#3a2418", blush="#c0504a"),
    # manager: polished studio headshot
    "morgan_manager": dict(
        bg_grad=("#d6dbe1", "#8d97a3"), skin="#f0c8a8", hair="sidepart", hair_color="#5a3c26",
        top="suit", jacket="#22324d", shirt="#e8eef6", tie="#7a1f2b", mouth="closed_lips", lip="#b9776b",
        pin="#c9a64a"),
    # host: Director, Office of Photonic Conduct
    "pat_reynolds": dict(
        bg_grad=("#e9e4f2", "#a99cc4"), skin="#e2b48e", hair="bob", hair_color="#c9c4bc", silver=True,
        top="blouse", jacket="#3b3f6b", shirt="#f3efe6", mouth="smile", lip="#a85a5a", glasses="rect",
        glasses_color="#5b4a3a", earring="#e8e2d4", necklace="#e8e2d4", lashes=True, blush="#d0806e"),
}


# ---------------------------------------------------------------- emblems
def tile(bg, icon, fg="#ffffff", ring=None):
    r = f'<rect width="{W}" height="{W}" fill="{bg}"/>'
    if ring:
        r += f'<circle cx="256" cy="256" r="214" fill="none" stroke="{ring}" stroke-width="10"/>'
    return svg(r + f'<g fill="none" stroke="{fg}" stroke-width="26" stroke-linecap="round" stroke-linejoin="round">{icon}</g>')


ICONS = {
    # simple white line icons centered on 256,256 within ~300 px
    "chart": '<path d="M136 376 L376 376"/><path d="M170 376 L170 300 M234 376 L234 240 M298 376 L298 270 M362 376 L362 170"/>'
             '<path d="M150 250 L230 180 L290 214 L370 132" stroke-width="18"/>',
    "gear": '<circle cx="256" cy="256" r="58"/>' + "".join(
        f'<path d="M256 256 m0 -96 l0 -34" transform="rotate({a} 256 256)" stroke-width="34"/>' for a in range(0, 360, 45))
        + '<circle cx="256" cy="256" r="104"/>',
    "wrench": '<path d="M176 336 L296 216"/><path d="M296 216 C276 170 310 128 356 134 L326 166 L340 198 L372 210 L402 180 C410 226 366 260 320 240" />'
              '<circle cx="164" cy="348" r="22"/>',
    "suit": '<path d="M200 180 C200 120 312 120 312 180 L312 220 L200 220 Z"/><rect x="214" y="168" width="84" height="32" rx="14" fill="#ffffff" stroke="none" opacity="0.9"/>'
            '<path d="M170 400 C170 300 200 250 256 250 C312 250 342 300 342 400"/>',
    "layout": '<path d="M130 330 L210 330 C260 330 260 200 310 200 L382 200"/><path d="M130 200 L382 200" opacity="0.45"/>'
              '<rect x="120" y="140" width="272" height="250" stroke-width="16"/>',
    "scope": '<rect x="118" y="146" width="276" height="200" rx="16"/><path d="M146 246 C172 190 192 190 214 246 C236 302 258 302 280 246 C302 190 324 190 346 246 L370 246" stroke-width="18"/>'
             '<path d="M200 390 L312 390"/>',
    "rf": '<path d="M256 380 L256 230"/><path d="M200 380 L312 380"/><circle cx="256" cy="210" r="16" fill="#ffffff"/>'
          '<path d="M206 160 C180 186 180 234 206 260 M306 160 C332 186 332 234 306 260" stroke-width="18"/>'
          '<path d="M170 124 C126 168 126 252 170 296 M342 124 C386 168 386 252 342 296" stroke-width="18"/>',
    "book": '<path d="M256 180 C220 150 170 146 126 156 L126 360 C170 350 220 354 256 384 C292 354 342 350 386 360 L386 156 C342 146 292 150 256 180 Z"/><path d="M256 180 L256 384"/>',
    "mesh": '<path d="M140 360 L256 140 L372 360 Z"/><path d="M198 250 L314 250 L256 360 Z"/><path d="M140 360 L372 360"/>',
    "megaphone": '<path d="M150 220 L230 220 L340 150 L340 362 L230 292 L150 292 Z"/><path d="M196 292 L216 372"/><path d="M380 220 C396 240 396 272 380 292" stroke-width="18"/>',
    "screen": '<rect x="126" y="140" width="260" height="180" rx="10"/><path d="M256 320 L256 380 M200 384 L312 384"/><path d="M170 270 L220 220 L260 250 L330 186" stroke-width="18"/>',
    "laser": '<rect x="110" y="214" width="120" height="84" rx="10"/><path d="M230 256 L400 256" stroke-width="16"/><path d="M372 222 L404 256 L372 290" stroke-width="16"/>'
             '<circle cx="150" cy="256" r="10" fill="#ffffff"/>',
    "mzm": '<path d="M110 256 L170 256 C200 256 200 186 236 186 L276 186 C312 186 312 256 342 256 L402 256" stroke-width="18"/>'
           '<path d="M170 256 C200 256 200 326 236 326 L276 326 C312 326 312 256 342 256" stroke-width="18"/>'
           '<path d="M226 150 L286 150 M226 362 L286 362" stroke-width="12"/>',
    "magnifier": '<circle cx="232" cy="230" r="86"/><path d="M296 296 L382 382" stroke-width="34"/><path d="M194 230 L222 258 L274 204" stroke-width="18"/>',
    "badge": '<path d="M256 120 L370 170 L370 260 C370 330 316 376 256 396 C196 376 142 330 142 260 L142 170 Z"/><path d="M204 260 L242 298 L314 220"/>',
    "box": '<path d="M136 196 L256 140 L376 196 L376 330 L256 386 L136 330 Z"/><path d="M136 196 L256 252 L376 196 M256 252 L256 386"/>',
    "flask": '<path d="M216 130 L296 130 M228 130 L228 220 L148 364 C140 380 150 392 168 392 L344 392 C362 392 372 380 364 364 L284 220 L284 130"/><path d="M188 312 L324 312" stroke-width="16"/>',
    "trophy": '<path d="M186 130 L326 130 L326 210 C326 260 290 290 256 290 C222 290 186 260 186 210 Z"/><path d="M186 160 L140 160 C140 210 160 230 190 236 M326 160 L372 160 C372 210 352 230 322 236"/>'
              '<path d="M256 290 L256 340 M200 386 L312 386 L300 340 L212 340 Z"/>',
    "lifebuoy": '<circle cx="256" cy="256" r="124"/><circle cx="256" cy="256" r="56"/><path d="M168 168 L216 216 M344 168 L296 216 M168 344 L216 296 M344 344 L296 296" stroke-width="22"/>',
    "ring": '<circle cx="256" cy="226" r="92"/><path d="M110 344 L402 344"/><path d="M110 344 L150 344" stroke-width="30"/>',
    "people": '<circle cx="256" cy="186" r="44"/><path d="M180 342 C180 278 214 250 256 250 C298 250 332 278 332 342"/>'
              '<circle cx="152" cy="216" r="32" stroke-width="20"/><path d="M98 340 C98 296 120 274 152 274" stroke-width="20"/>'
              '<circle cx="360" cy="216" r="32" stroke-width="20"/><path d="M414 340 C414 296 392 274 360 274" stroke-width="20"/>',
    "wafer": '<path d="M180 136 A136 136 0 1 0 332 136 Z"/><path d="M190 210 L322 210 M170 262 L342 262 M190 314 L322 314 M226 170 L226 360 M286 170 L286 360" stroke-width="10" opacity="0.7"/>',
}

CHANNELS = {
    # name: (filename, bg color, icon)
    "#figures": ("ch_figures", "#5b5fc7", "chart"),
    "#equipment": ("ch_equipment", "#4b5563", "wrench"),
    "#lab-ops": ("ch_lab_ops", "#0f766e", "gear"),
    "#cleanroom": ("ch_cleanroom", "#c2410c", "suit"),
    "#layout-review": ("ch_layout_review", "#7c3aed", "layout"),
    "#test-bench": ("ch_test_bench", "#15803d", "scope"),
    "#test-bench-2": ("ch_test_bench_2", "#4d7c0f", "scope"),
    "#rf-test": ("ch_rf_test", "#b91c1c", "rf"),
    "#paper-club": ("ch_paper_club", "#92400e", "book"),
    "#simulation": ("ch_simulation", "#1d4ed8", "mesh"),
    "#all-hands": ("ch_all_hands", "#be185d", "megaphone"),
    "#demo-prep": ("ch_demo_prep", "#0369a1", "screen"),
    "#laser-integration": ("ch_laser_integration", "#9f1239", "laser"),
    "#modulator-team": ("ch_modulator_team", "#6d28d9", "mzm"),
    "#outgoing-inspection": ("ch_outgoing_inspection", "#334155", "magnifier"),
    "#product-qual": ("ch_product_qual", "#047857", "badge"),
    "#purchasing": ("ch_purchasing", "#a16207", "box"),
    "#group-lab": ("ch_group_lab", "#0e7490", "flask"),
    "#integration-wins": ("ch_integration_wins", "#b45309", "trophy"),
    "#lab-help": ("ch_lab_help", "#dc2626", "lifebuoy"),
    "#ring-characterization": ("ch_ring_characterization", "#1e40af", "ring"),
}


def company():
    # Five Sigma Photonics: a true-scale Gaussian (computed, not sketched) with a dashed line at +5 sigma
    # and one amber event beyond it. At +5 sigma the density is 3.7e-6 of the peak, so the curve
    # itself is flat there; the dot is the "outside five sigma" outlier.
    import math
    x0, x1, base, peak = 48, 464, 384, 120
    lo, hi = -3.3, 6.2                       # plotted range in sigma
    px = lambda z: x0 + (z - lo) / (hi - lo) * (x1 - x0)
    pts = [(px(lo + i * (hi - lo) / 200), base - (base - peak) * math.exp(-0.5 * (lo + i * (hi - lo) / 200) ** 2)) for i in range(201)]
    line = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    area = line + f" L{x1} {base} L{x0} {base} Z"
    x5 = px(5.0)
    body = ('<rect width="512" height="512" fill="#0b2545"/>'
            f'<path d="{area}" fill="#e7ecef" fill-opacity="0.16"/>'
            f'<path d="{line}" fill="none" stroke="#e7ecef" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M{x5:.1f} 104 L{x5:.1f} {base}" stroke="#f2a541" stroke-width="12" stroke-dasharray="24 16" stroke-linecap="round"/>'
            f'<circle cx="{px(5.75):.1f}" cy="{base - 40}" r="20" fill="#f2a541"/>')
    return svg(body)


def office():
    # Office of Photonic Conduct: a seal with a balance whose beam is a light beam
    ticks = "".join(f'<path d="M256 64 L256 84" transform="rotate({a} 256 256)"/>' for a in range(0, 360, 15))
    body = ('<rect width="512" height="512" fill="#7f1d1d"/>'
            '<circle cx="256" cy="256" r="206" fill="#f5efe1"/>'
            f'<g stroke="#7f1d1d" stroke-width="6">{ticks}</g>'
            '<circle cx="256" cy="256" r="160" fill="#7f1d1d"/>'
            '<g fill="none" stroke="#f2c94c" stroke-width="14" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M256 150 L256 350 M206 352 L306 352"/>'
            '<path d="M146 186 L366 186" stroke-width="16"/>'
            '<path d="M170 192 L136 270 L204 270 Z M342 192 L308 270 L376 270 Z" stroke-width="10"/>'
            '<path d="M128 272 C140 300 196 300 212 272 M300 272 C312 300 368 300 384 272" stroke-width="12"/>'
            '</g><circle cx="256" cy="150" r="16" fill="#f5efe1"/>')
    return svg(body)


def partner_lab():
    # fictional university lab crest: shield, lens, and a star
    body = ('<rect width="512" height="512" fill="#e9e1cf"/>'
            '<path d="M256 74 L396 124 L396 250 C396 350 330 410 256 440 C182 410 116 350 116 250 L116 124 Z" fill="#14325c" stroke="#b8902f" stroke-width="12"/>'
            '<path d="M116 210 L396 210" stroke="#b8902f" stroke-width="10"/>'
            '<path d="M256 112 L266 142 L298 142 L272 160 L282 190 L256 172 L230 190 L240 160 L214 142 L246 142 Z" fill="#e9c46a"/>'
            '<ellipse cx="256" cy="306" rx="34" ry="78" fill="#cfe3f2" stroke="#e9c46a" stroke-width="8"/>'
            '<path d="M146 280 L230 280 M146 332 L230 332 M282 280 L366 306 M282 332 L366 306" stroke="#f2c94c" stroke-width="8" stroke-linecap="round"/>')
    return svg(body)


def vendor_laser():
    # fictional laser vendor "Lumenfold Lasers": a beam folded by two mirrors
    body = ('<rect width="512" height="512" fill="#0f4c5c"/>'
            '<path d="M96 170 L380 170 L132 342 L416 342" fill="none" stroke="#ffb703" stroke-width="26" stroke-linejoin="round" stroke-linecap="round"/>'
            '<path d="M373 109 L409 223" stroke="#e6f1f3" stroke-width="22" stroke-linecap="round"/>'
            '<path d="M103 289 L139 403" stroke="#e6f1f3" stroke-width="22" stroke-linecap="round"/>'
            '<rect x="64" y="146" width="44" height="48" fill="#e6f1f3"/>')
    return svg(body)


def vendor_wafer():
    # fictional crystal wafer supplier "Northcut Crystal": wafer with a primary flat and a trigonal line motif
    body = ('<rect width="512" height="512" fill="#2b2d42"/>'
            '<defs><clipPath id="wf"><path d="M188 410 A160 160 0 1 1 324 410 Z"/></clipPath></defs>'
            '<path d="M188 410 A160 160 0 1 1 324 410 Z" fill="#efe3b8"/>'
            '<g clip-path="url(#wf)" stroke="#c9a85c" stroke-width="7">'
            + "".join(f'<path d="M-200 {y} L800 {y}" transform="rotate({a} 256 266)"/>' for a in (0, 60, 120) for y in range(146, 400, 56))
            + '</g><path d="M188 410 L324 410" stroke="#8d6b2f" stroke-width="10" stroke-linecap="round"/>')
    return svg(body)


def meeting():
    return tile("#2563eb", ICONS["people"])


def build(outdir, only=None):
    jobs = {}
    for k, p in PEOPLE.items():
        jobs[k] = portrait(p)
    for name, (fn, bg, icon) in CHANNELS.items():
        jobs[fn] = tile(bg, ICONS[icon])
    jobs["org_five_sigma"] = company()
    jobs["org_office_photonic_conduct"] = office()
    jobs["org_partner_lab"] = partner_lab()
    jobs["vendor_lumenfold_lasers"] = vendor_laser()
    jobs["vendor_northcut_crystal"] = vendor_wafer()
    for fn, s in jobs.items():
        if only and fn != only:
            continue
        png = cairosvg.svg2png(bytestring=s.encode(), output_width=256, output_height=256)
        im = Image.open(io.BytesIO(png)).convert("RGB")
        im.save(f"{outdir}/{fn}.jpg", quality=90, optimize=True, progressive=True)
    return list(jobs)


if __name__ == "__main__":
    print("\n".join(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
