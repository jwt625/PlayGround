"""Build mpo_connector_and_adapter.blend: MPO plug (12/16 fibre, male/female), MPO adapter (receptacle), 4-port adapter plate, plug with 3 mm trunk cable stub.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/interconnect/build_mpo_connector_and_adapter.py -- assets/components/interconnect
Four ASSET_ collections in one file (a family): ASSET_mpo_plug, ASSET_mpo_adapter, ASSET_mpo_adapter_plate4 (4 adapters, linked duplicates), ASSET_mpo_plug_pair_demo is NOT built (mate in the scene).
Axes (ASSET_SPEC): Z up, plug mating face at y = 0 looking -Y, body and cable toward +Y; key on top (+Z).
Adapter: centre partition at y = 0 (the two ferrule faces meet there), openings at y = -17 (front) and y = +17 (rear).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import ic_common as I
import ic_optics as OPT
from ic_common import MB

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "mpo_connector_and_adapter"

# ------------------------------------------------------------------ dimensions (mm) with provenance, see the JSON
PLUG_W, PLUG_H = 12.4, 8.0          # sleeve outer (level C: 12.4 x 8.3 in fiber_connectors, ~12.5 x 7.6 in vendor text; 8.0 chosen mid-range)
HOUSING_W, HOUSING_H = 11.2, 7.0    # inner housing (estimate)
FACE_Y = 2.0                        # housing front plane: ferrule protrudes 2.0 mm (estimate)
KEY_W, KEY_H = 3.0, 0.7             # key rib on top (estimate)
SLEEVE_Y0, SLEEVE_Y1 = 9.0, 30.0
BOOT_Y0, BOOT_Y1 = 30.0, 70.0
CABLE_Y1 = 220.0
CABLE_R = 1.5                       # 3.0 mm round trunk cable (12F: common 3.0 mm OD; fiber_cables_and_trays uses 3.0 mm cords)
AD_W, AD_H, AD_L = 15.0, 10.0, 34.0  # adapter outer (vendor footprint 0.59 x 0.39 in = 15.0 x 9.9 mm) and length (estimate)
AD_IN_W, AD_IN_H = 12.7, 8.3
PLATE_W, PLATE_H, PLATE_T = 106.0, 33.6, 2.0   # HD MTP/MPO adapter plate class (nexconec listing 106 x 33.60 mm; thickness estimate)
PORT_PITCH = 24.0                   # estimate (4 x 24 = 96 < 106)
FLANGE_Y = -8.0                     # plate plane in adapter coordinates (front protrudes 9 mm, rear 25 mm)

C.reset()
M = I.Mats()
coll_all = bpy.context.scene.collection


def rect(w, h):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def mk(builder, name, mat, smooth=35.0):
    return builder.build(name, mat, smooth_deg=smooth)


def plug_asset():
    coll, root = C.new_asset("mpo_plug", accuracy="C")
    mat_h = M["plastic_green"]
    nm = "mpo_plug_"
    # inner housing: front plate with a ferrule window + solid body
    b = MB()
    b.ring_xz(rect(HOUSING_W, HOUSING_H), rect(7.6, 3.6), FACE_Y, FACE_Y + 2.0)
    b.box((-HOUSING_W / 2, HOUSING_W / 2), (FACE_Y + 2.0, SLEEVE_Y0 + 6.0), (-HOUSING_H / 2, HOUSING_H / 2), bev=0.4, seg=1)
    b.box((-KEY_W / 2, KEY_W / 2), (FACE_Y, SLEEVE_Y0 + 1.0), (HOUSING_H / 2 - 0.1, HOUSING_H / 2 + KEY_H), bev=0.2, seg=1)
    I.place(mk(b, nm + "housing", mat_h), coll, root)
    # push-pull sleeve (moves with p_sleeve_pull), grip ridges and latch recesses are children
    s = MB()
    s.box((-PLUG_W / 2, PLUG_W / 2), (SLEEVE_Y0, SLEEVE_Y1), (-PLUG_H / 2, PLUG_H / 2), bev=0.6, seg=2)
    sl = I.place(mk(s, nm + "sleeve", mat_h), coll, root)
    r = MB()
    for k in range(7):
        y = SLEEVE_Y0 + 8.0 + k * 1.8
        r.box((-4.2, 4.2), (y, y + 0.9), (PLUG_H / 2 - 0.02, PLUG_H / 2 + 0.25), bev=0.1, seg=1)
        r.box((-4.2, 4.2), (y, y + 0.9), (-PLUG_H / 2 - 0.25, -PLUG_H / 2 + 0.02), bev=0.1, seg=1)
    rg = I.place(mk(r, nm + "sleeve_ridges", mat_h), coll, sl)
    d = MB()
    for sx in (1, -1):
        xa, xb = sorted((sx * PLUG_W / 2 - 0.04, sx * PLUG_W / 2 + 0.04))
        d.box((xa, xb), (SLEEVE_Y0 + 1.0, SLEEVE_Y0 + 5.0), (-2.2, 2.2), bev=0.0, seg=1)
    I.place(mk(d, nm + "latch_recess", M["plastic_black"], 0), coll, sl)
    # crimp shell + boot (superellipse loft) + 3 mm cable
    c = MB()
    c.box((-4.9, 4.9), (SLEEVE_Y1, BOOT_Y0 + 6.0), (-3.2, 3.2), bev=0.6, seg=2)
    I.place(mk(c, nm + "crimp_shell", mat_h), coll, root)
    bt = MB()
    bt.loft_y([(BOOT_Y0 + 4.0, 9.0, 5.8, 3.0), (BOOT_Y0 + 12.0, 7.6, 5.2, 2.6), (BOOT_Y0 + 24.0, 5.0, 4.2, 2.2), (BOOT_Y1, 3.4, 3.4, 2.0)], npts=24)
    I.place(mk(bt, nm + "boot", mat_h), coll, root)
    cb = MB()
    cb.cyl((0, (BOOT_Y1 - 2.0 + CABLE_Y1) / 2, 0), CABLE_R, CABLE_Y1 - BOOT_Y1 + 2.0, axis="Y", segs=14)
    I.place(mk(cb, nm + "cable", M["jacket_yellow"]), coll, root)
    # ferrule variants
    variants = [("mpo12_female", 12, False), ("mpo12_male", 12, True), ("mpo16_female", 16, False)]
    for vn, nf, pins in variants:
        vc = C.sub_collection(coll, "VARIANT_" + vn)
        for o in OPT.mt_ferrule(M, nfib=nf, rows=1, apc=True, pins=pins, name=nm + vn):
            I.place(o, vc, root)
        if vn != "mpo12_female":
            I.hide_collection(vc, True)
    # hooks
    I.empty("HOOK_mate", coll, root, (0, 0, 0), kind="ARROWS", size=0.01).name = "HOOK_mate"
    I.empty("HOOK_key", coll, root, (0, 5, HOUSING_H / 2 + KEY_H), kind="ARROWS", size=0.005)
    I.empty("HOOK_grip", coll, root, (0, 20, PLUG_H / 2), kind="ARROWS", size=0.01)
    I.empty("HOOK_cable_end", coll, root, (0, CABLE_Y1, 0), kind="ARROWS", size=0.01)
    # push-pull driver: root["p_sleeve_pull"] (mm, 0..8) slides the sleeve (and ridges, recesses) toward the cable
    root["p_sleeve_pull"] = 0.0
    root.id_properties_ui("p_sleeve_pull").update(min=0.0, max=8.0, soft_max=8.0)
    dr = sl.driver_add("location", 1).driver
    dr.type = "SCRIPTED"
    v = dr.variables.new()
    v.name = "p"
    v.type = "SINGLE_PROP"
    v.targets[0].id = root
    v.targets[0].data_path = '["p_sleeve_pull"]'
    dr.expression = "p * 0.001"
    return coll, root


def adapter_geometry(coll, root, nm, ox=0.0, plate=False):
    """Adapter hollow body (+ flange ears, clips, centre partition). Returns objects placed under root at x offset ox (mm)."""
    mat = M["plastic_green"]
    b = MB()
    y0, y1 = -AD_L / 2, AD_L / 2
    hw, hh = AD_W / 2, AD_H / 2
    iw, ih = AD_IN_W / 2, AD_IN_H / 2
    b.box((-hw, -iw), (y0, y1), (-hh, hh), bev=0.3, seg=1)
    b.box((iw, hw), (y0, y1), (-hh, hh), bev=0.3, seg=1)
    b.box((-iw, iw), (y0, y1), (-hh, -ih), bev=0.3, seg=1)
    b.box((-iw, -1.7), (y0, y1), (ih, hh), bev=0.3, seg=1)     # top wall with the key slot in the middle
    b.box((1.7, iw), (y0, y1), (ih, hh), bev=0.3, seg=1)
    b.box((-1.7, 1.7), (y0, y1), (hh - 0.5, hh), bev=0.2, seg=1)  # thin slot floor (key slot)
    b.ring_xz(rect(AD_IN_W + 0.02, AD_IN_H + 0.02), rect(7.8, 3.8), -1.0, 1.0)  # centre partition with the ferrule window
    # side clips (latch springs) at both ends and flange ears
    for sx in (1, -1):
        for yc in (-12.0, 12.0):
            b.box((sx * hw - (0.0 if sx > 0 else 0.5), sx * hw + (0.5 if sx > 0 else 0.0)), (yc - 4.0, yc + 4.0), (-2.5, 2.5), bev=0.2, seg=1)
        b.box(tuple(sorted((sx * hw, sx * (hw + 3.2)))), (-1.2, 1.2), (-2.0, 2.0), bev=0.2, seg=1)
    ob = I.place(mk(b, nm + "body", mat), coll, root)
    return ob


def adapter_asset():
    coll, root = C.new_asset("mpo_adapter", accuracy="C")
    ob = adapter_geometry(coll, root, "mpo_adapter_")
    # the generic builder puts flange ears at y = 0 here (loose adapter); plate version offsets them
    I.empty("HOOK_port_front", coll, root, (0, -AD_L / 2, 0), kind="ARROWS", size=0.01)
    I.empty("HOOK_port_rear", coll, root, (0, AD_L / 2, 0), kind="ARROWS", size=0.01)
    I.empty("HOOK_mate_plane", coll, root, (0, 0, 0), kind="ARROWS", size=0.01)
    return coll, root


def plate_asset(adapter_body):
    coll, root = C.new_asset("mpo_adapter_plate4", accuracy="C")
    pm = M["galv_steel"]
    xs = [(k - 1.5) * PORT_PITCH for k in range(4)]
    cw, ch = AD_W + 0.3, AD_H + 0.3
    b = MB()
    ytop = (FLANGE_Y - PLATE_T / 2, FLANGE_Y + PLATE_T / 2)
    b.box((-PLATE_W / 2, PLATE_W / 2), ytop, (ch / 2, PLATE_H / 2), bev=0.2, seg=1)
    b.box((-PLATE_W / 2, PLATE_W / 2), ytop, (-PLATE_H / 2, -ch / 2), bev=0.2, seg=1)
    edges = [-PLATE_W / 2] + sum([[x - cw / 2, x + cw / 2] for x in xs], []) + [PLATE_W / 2]
    for i in range(0, len(edges), 2):
        b.box((edges[i], edges[i + 1]), ytop, (-ch / 2, ch / 2), bev=0.2, seg=1)
    I.place(mk(b, "mpo_adapter_plate4_plate", pm), coll, root)
    # four adapters: linked duplicates of one body that has its flange ears at the plate plane
    pb = MB()
    mat = M["plastic_green"]
    hw, hh, iw, ih = AD_W / 2, AD_H / 2, AD_IN_W / 2, AD_IN_H / 2
    y0, y1 = -AD_L / 2, AD_L / 2
    pb.box((-hw, -iw), (y0, y1), (-hh, hh), bev=0.3, seg=1)
    pb.box((iw, hw), (y0, y1), (-hh, hh), bev=0.3, seg=1)
    pb.box((-iw, iw), (y0, y1), (-hh, -ih), bev=0.3, seg=1)
    pb.box((-iw, -1.7), (y0, y1), (ih, hh), bev=0.3, seg=1)
    pb.box((1.7, iw), (y0, y1), (ih, hh), bev=0.3, seg=1)
    pb.box((-1.7, 1.7), (y0, y1), (hh - 0.5, hh), bev=0.2, seg=1)
    pb.ring_xz(rect(AD_IN_W + 0.02, AD_IN_H + 0.02), rect(7.8, 3.8), -1.0, 1.0)
    for sx in (1, -1):
        for yc in (-12.0, 12.0):
            pb.box((sx * hw - (0.0 if sx > 0 else 0.5), sx * hw + (0.5 if sx > 0 else 0.0)), (yc - 4.0, yc + 4.0), (-2.5, 2.5), bev=0.2, seg=1)
    pb.box((-hw - 0.5, hw + 0.5), (FLANGE_Y - 1.0, FLANGE_Y + 1.0), (-hh - 0.5, hh + 0.5), bev=0.2, seg=1)
    src = mk(pb, "mpo_adapter_plate4_adapter_0", mat)
    I.place(src, coll, root, loc_mm=(xs[0], 0, 0))
    for k in range(1, 4):
        I.link_dup(src, coll, root, (xs[k], 0, 0), name="mpo_adapter_plate4_adapter_%d" % k)
    for k in range(4):
        I.empty("HOOK_port_%d" % (k + 1), coll, root, (xs[k], -AD_L / 2, 0), kind="ARROWS", size=0.01)
        I.empty("HOOK_mate_plane_%d" % (k + 1), coll, root, (xs[k], 0, 0), kind="ARROWS", size=0.01)
    return coll, root


pc, pr = plug_asset()
ac, ar = adapter_asset()
lc, lr = plate_asset(None)
for r in (pr, ar, lr):
    r.location = (0, 0, 0)
# all three roots sit at the origin (overlapping in the file; the assembler appends one ASSET_ at a time)

meta = {
    "description": "MPO (multi-fibre push-on) family: plug (MPO-12 female/male pins, MPO-16 female; 8 degree APC ferrule face with fibre holes, guide pin holes/pins, key rib, push-pull sleeve with grip ridges and latch recesses, crimp shell, boot, 3.0 mm yellow trunk cable stub), single adapter (receptacle: hollow body with key slot, centre partition with ferrule window, side clips, flange ears), 4-port adapter plate (106 x 33.6 mm).",
    "sources": [
        {"what": "IEC 61754-7 (MPO family interface): rectangular MT ferrule normally 6.4 x 2.5 mm, two 0.7 mm guide pins (hole pitch 4.6 mm), up to 12 fibres between the pin holes; IEC 61754-7-1:2014 table lists guide pin spacing 4.597-4.603 mm and fibre hole diameter 0.699-0.701 (guide-pin hole) values (read via search summary; the sample PDF text was not extractable)",
         "url": "https://cdn.standards.iteh.ai/samples/18585/fbc4fa2047f7460c9f387a7525313309/IEC-61754-7-1-2014.pdf", "accessed": "2026-10-02"},
        {"what": "IEC 61754-7 Edition 3.0 (2008-03) preview listing", "url": "https://elstandard.se/documents/preview/429801", "accessed": "2026-10-02"},
        {"what": "MT ferrule fibre pitch 0.25 mm; MTP/MPO 12/16/24-fibre connectors share roughly the same housing size (about 12.5 mm wide x 7.6 mm tall in one vendor text)", "url": "https://www.bonelinks.com/a-beginners-guide-mtp-connector/ and https://www.fsgnetworks.com/products/mt-ferrule/ (search summaries)", "accessed": "2026-10-02"},
        {"what": "MPO adapter standard footprint 0.59 x 0.39 in (15.0 x 9.9 mm), vendor product page listing", "url": "https://www.fiberopticcableshop.com/fampommrkey.html (search summary)", "accessed": "2026-10-02"},
        {"what": "HD MTP/MPO adapter plate class 106 x 33.60 x 24 mm (W x H x D)", "url": "https://nexconec.com/hd-mpo-adapter-plates.html (search summary)", "accessed": "2026-10-02"},
        {"what": "MT ferrule geometry reused from the library: scripts/assets/interconnect/ic_optics.mt_ferrule (fiber_connectors asset, US Conec white paper / product page values)", "path": "assets/components/interconnect/fiber_connectors.json", "accessed": "2026-10-02"},
        {"what": "TIA-604-5 (FOCIS 5) and the IEC 61754-7 drawings themselves were NOT read (paywalled): housing, sleeve, boot, adapter lengths are estimates", "accessed": "2026-10-02"},
    ],
    "dimension_table": [
        dict(item="MT ferrule end face", value="6.4 x 2.5 mm, length 8.0", unit="mm", source="IEC 61754-7 (search summaries); length from fiber_connectors (not verified)", accuracy="A face / B length"),
        dict(item="guide pins / holes", value="0.7 dia, pitch 4.6", unit="mm", source="IEC 61754-7-1:2014 (4.597-4.603)", accuracy="A"),
        dict(item="fibre pitch, count", value="0.25, 12 (or 16 for the MPO-16 variant)", unit="mm", source="MT ferrule public specs", accuracy="A"),
        dict(item="APC end-face angle", value=8, unit="deg", source="IEC 61755 / industry standard", accuracy="A"),
        dict(item="plug outer (sleeve) W x H", value="12.4 x 8.0", unit="mm", source="12.4 x 8.3 (fiber_connectors, estimate) and about 12.5 x 7.6 (vendor text); 8.0 chosen", accuracy="C, range height 7.6-8.3"),
        dict(item="inner housing, ferrule protrusion, key rib", value="11.2 x 7.0; 2.0; 3.0 wide x 0.7", unit="mm", source="estimate", accuracy="C"),
        dict(item="sleeve y range, boot length, cable stub", value="9-30; 30-70 (boot), cable to y = 220", unit="mm", source="estimate (fiber_connectors uses MPO boot 32)", accuracy="C"),
        dict(item="trunk cable OD", value=3.0, unit="mm", source="common 12F MPO trunk jacket, same as the 3.0 mm cords of fiber_cables_and_trays", accuracy="B"),
        dict(item="adapter outer W x H, length", value="15.0 x 10.0; 34", unit="mm", source="vendor footprint 15.0 x 9.9; length estimate", accuracy="B width / C length"),
        dict(item="adapter plate", value="106 x 33.6 x 2.0, 4 ports at 24 mm pitch", unit="mm", source="plate W x H class from vendor listing; thickness and pitch estimates", accuracy="C"),
    ],
    "origin": "ASSET_mpo_plug: root at the centre of the mating face (ferrule face plane y = 0), body and cable toward +Y, key on top. ASSET_mpo_adapter: root at the centre partition (the plane where two mated ferrule faces meet). ASSET_mpo_adapter_plate4: root at the mate plane of the port row centre; plate plane at y = -8 (front of the plate is toward -Y; the adapter front openings protrude 9 mm in front of the plate, 25 mm behind).",
    "mating": "Plug into the adapter front: yaw the plug by pi about Z (its body then points -Y) and put the plug root on HOOK_mate_plane (face to face with the centre partition: the plug face meets y = 0, its housing front stops 2 mm short). Plug from the rear: no rotation. Insertion travel is along Y; the key rib is aligned with the adapter key slot only when the plug is not rolled (roll the plug by 0.1-0.3 rad about Y for a visible mismatch).",
    "hooks": {"HOOK_mate": "plug: mating face centre", "HOOK_key": "plug: top of the key rib", "HOOK_grip": "plug: top of the sleeve ridges", "HOOK_cable_end": "plug: end of the 3.0 mm cable stub (attach a curve or chain here)",
              "HOOK_port_front / HOOK_port_rear": "adapter: opening centres", "HOOK_mate_plane": "adapter: centre partition", "HOOK_port_1..4 / HOOK_mate_plane_1..4": "plate: opening centres / mate planes (x = -36, -12, 12, 36 mm)"},
    "custom_properties": {"p_sleeve_pull": "ASSET_mpo_plug root: 0..8 mm, driver slides the push-pull sleeve toward the cable"},
    "variants": {"VARIANT_mpo12_female": "default", "VARIANT_mpo12_male": "hidden: two guide pins protrude 3.5 mm", "VARIANT_mpo16_female": "hidden: 16 fibre holes"},
    "simplifications": ["no spring, MT clip, guide pin clip, label, or dust cap", "housing is boxes with a window plate: no internal parts, latch ramps are black rectangles", "key position is fixed (top); no Type A / Type B polarity flip modelled",
                        "ribbon cable not built (round 3.0 mm trunk stub only)", "ferrule is the shared library MT ferrule (boolean-cut holes, fibres glow cyan through the fiber_core material)", "adapter has no dust shutter and no internal alignment features other than the centre partition and the key slot"],
    "intended_usage": "S6 intro: patch cord with an MPO plug jams crooked into an MPO receptacle on an adapter plate (macro, scale-up x8 by the assembler). Any later fibre-plug close-up.",
}
meta["material_slots"] = sorted(m.name for m in bpy.data.materials if m.users)
blend = os.path.join(OUT, ASSET + ".blend")
prev = os.path.join(OUT, "previews")
meta["triangles_unique_meshes"] = {k: I.unique_tris(c) for k, c in (("mpo_plug", pc), ("mpo_adapter", ac), ("mpo_adapter_plate4", lc))}
C.finish(ASSET, blend, pc, meta, preview_dir=None)
print("TRIS", meta["triangles_unique_meshes"])
if "--no-preview" not in sys.argv:
    I.preview_views(pc, prev, ASSET + "_plug", [("three_quarter", (0, 25, 0), (-0.9, -0.7, 0.55), 160), ("face_closeup", (0, 0, 0), (0, -1, 0.15), 40, 120),
                                               ("top", (0, 45, 0), (0, -0.05, 1), 150)])
    for _o in bpy.context.scene.objects:
        _o.hide_render = False
    I.preview_views(ac, prev, ASSET + "_adapter", [("three_quarter", (0, 0, 0), (-0.9, -0.8, 0.6), 80)])
    for _o in bpy.context.scene.objects:
        _o.hide_render = False
    I.preview_views(lc, prev, ASSET + "_plate4", [("three_quarter", (0, 0, 0), (-0.8, -0.9, 0.5), 150), ("front", (0, 0, 0), (0, -1, 0.1), 160)])
