"""Hall layout (JSON for the scene agent) + bird flap vent asset + hall preview renders (workstream R, DevLog-007).

World frame = crude film hall (DevLog-004): aisle along +Y, x = 0 aisle centre, rack fronts at x = +-1.1 facing the
aisle, floor z = 0, end wall with the door at y = 13.0 (street beyond). Left row x < 0 (racks unrotated: rack +X front
faces +X), right row x > 0 (racks rotated 180 deg about Z).
Run: scripts/bslot.sh -b --python scripts/assets/datacenter/build_hall_layout.py -- assets/components/datacenter
Needs rack_gb300_nvl72.blend, rack_bird_q3400.blend (+ .json) in the output dir for the preview (layout JSON alone does not).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import rk_lib as K  # noqa: E402
from rk_lib import MB, Part  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

OUT, KV = K.parse_args()
PREV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(OUT)), "..", "..", "outputs", "assets", "datacenter"))
m = lambda x: x * 0.001

FACE_X = 1.1                    # rack front plane |x| (crude film v0.1)
RACK_D = 1.068
N_PER_ROW = 15
Y_START = 1.0
END_WALL_Y = 13.0               # crude film v0.1 end wall with door (x = 0, 1.4 m wide, 2.5 m tall)
CEIL = 4.2
TYPES = {
    "gb300_nvl72": dict(asset="assets/components/datacenter/rack_gb300_nvl72.blend", glb="assets/components/datacenter/rack_gb300_nvl72.glb",
                        collection="ASSET_rack_gb300_nvl72", width=0.6, depth=1.068, height=2.236),
    "bird_q3400": dict(asset="assets/components/datacenter/rack_bird_q3400.blend", glb="assets/components/datacenter/rack_bird_q3400.glb",
                       collection="ASSET_rack_bird_q3400", width=0.8, depth=1.068, height=2.236),
}


def is_bird(i):
    return i % 3 == 1          # C B C | C B C | ... -> 5 of 15 per row = 1/3


def rack_matrix(side, x, y):
    rz = 0.0 if side == "L" else math.pi
    return Matrix.Translation((x, y, 0)) @ Matrix.Rotation(rz, 4, "Z"), rz


bird_meta = json.load(open(os.path.join(OUT, "rack_bird_q3400.json")))
cells_local = bird_meta["cells"]["list"]
pad_local = next(h for h in bird_meta["hooks"] if h["name"] == "HOOK_landing_pad")["location"]
perch_local = [h["location"] for h in bird_meta["hooks"] if h["name"].startswith("HOOK_perch_")]

HERO_RACK, HERO_SLOT = "L07", "HOOK_bird_1_0_1"   # hero's front slot (alias HOOK_cell_1_1_1), left empty: hero hops out forward in S2
racks, cells_world, pads, perches = [], [], [], []
for side in ("L", "R"):
    y = Y_START
    sx = -1 if side == "L" else 1
    for i in range(N_PER_ROW):
        t = "bird_q3400" if is_bird(i) else "gb300_nvl72"
        w = TYPES[t]["width"]
        yc = y + w / 2
        x = sx * (FACE_X + RACK_D / 2)
        M, rz = rack_matrix(side, x, yc)
        rid = "%s%02d" % (side, i)
        racks.append(dict(id=rid, side=side, index=i, type=t, location=[round(x, 4), round(yc, 4), 0.0],
                          rotation_z_deg=round(math.degrees(rz), 1), front_normal=[float(-sx), 0.0, 0.0],
                          y_span=[round(y, 4), round(y + w, 4)]))
        if t == "bird_q3400":
            for c in cells_local:
                p = M @ Vector(c["feet_origin_m"])
                hc = M @ Vector(c["head_center_m"])
                occ = "hero_empty" if (rid == HERO_RACK and c["bird_hook"] == HERO_SLOT) else "crowd"
                cells_world.append(dict(rack=rid, hook=c["hook"], bird_hook=c["bird_hook"], compat_alias=c["compat_alias"],
                                        occupant=occ, tier=c["tier"], unit=c["unit"], col=c["col"], fill_index=c["fill_index"],
                                        slot_label=c["slot_label"], feet_origin=[round(v, 4) for v in p],
                                        head_center_neutral=[round(v, 4) for v in hc], bird_yaw_deg=round(math.degrees(rz), 1),
                                        head_out_own_face_m=c["head_out_own_face_m"], neck_extension_m=c["neck_extension_m"]))
            p = M @ Vector(pad_local)
            pads.append(dict(rack=rid, location=[round(v, 4) for v in p], facing_yaw_deg=round(math.degrees(rz), 1)))
            for k, pl in enumerate(perch_local):
                q = M @ Vector(pl)
                perches.append(dict(rack=rid, hook="HOOK_perch_%d" % k, location=[round(v, 4) for v in q],
                                    facing_yaw_deg=round(math.degrees(rz), 1)))
        y += w

n_bird = sum(1 for r in racks if r["type"] == "bird_q3400")
primary = max([p for p in pads if p["rack"].startswith("L")], key=lambda p: p["location"][1])
ENTR = dict(type="bird_flap_vent", asset="assets/components/datacenter/bird_flap_vent.blend",
            location=[0.0, END_WALL_Y - 0.06, 3.25], opening_m=[0.70, 0.50], inward_normal=[0.0, -1.0, 0.0],
            rotation_z_deg=-90.0, hinge="flap object bird_flap_vent_flap: rotate about its local Y (top hinge); positive = swings into the hall",
            note="in the end wall above the existing door (door x = 0, 1.4 m wide, 2.5 m tall in crude v0.1)")
# ------------------------------------------------------------------ overhead infrastructure (design, C): per row a power busway
# with tap-off boxes over every GB300 rack (the 3 upper power whips of each GB300 end at rack-local (-0.28, 0.185..0.209,
# 2.376) = tap-off box bottom), a yellow fibre raceway with a drop to the rear of every bird-rack top, a galvanised ladder
# tray for copper/power over the hot-aisle side, threaded hanger rods to the ceiling.
GB_WHIP_LOCAL = [(-0.28, 0.185 + 0.012 * k, 2.376) for k in range(3)]
OV = dict(busway=dict(local_x=-0.28, section_m=[0.16, 0.14], z=[2.476, 2.616]),
          raceway=dict(local_x=0.034, width=0.30, depth=0.10, z=[2.95, 3.05], drop_local_x=-0.20, drop_bottom_z=2.30),
          ladder=dict(local_x=-0.62, width=0.45, rail_h=0.10, z=3.30), y=[Y_START - 0.1, Y_START + 10.1], ceiling=CEIL)


def row_x(side, lx):
    sx = -1 if side == "L" else 1
    return sx * (FACE_X + RACK_D / 2 - lx)


overhead = dict(asset="assets/components/datacenter/overhead_infrastructure.blend", glb="assets/components/datacenter/overhead_infrastructure.glb",
                frame="world (hall) coordinates, same as this file", runs=[], tap_offs=[], raceway_drops=[])
for side in ("L", "R"):
    overhead["runs"].append(dict(type="power_busway", side=side, x=round(row_x(side, OV["busway"]["local_x"]), 4), y=OV["y"],
                                 z=OV["busway"]["z"], section_m=OV["busway"]["section_m"]))
    overhead["runs"].append(dict(type="fibre_raceway_yellow", side=side, x=round(row_x(side, OV["raceway"]["local_x"]), 4), y=OV["y"],
                                 z=OV["raceway"]["z"], width=OV["raceway"]["width"]))
    overhead["runs"].append(dict(type="ladder_tray_copper_power", side=side, x=round(row_x(side, OV["ladder"]["local_x"]), 4), y=OV["y"],
                                 z=[OV["ladder"]["z"], OV["ladder"]["z"] + OV["ladder"]["rail_h"]], width=OV["ladder"]["width"]))
for r in racks:
    M, _ = rack_matrix(r["side"], *r["location"][:2])
    if r["type"] == "gb300_nvl72":
        ends = [M @ Vector(p) for p in GB_WHIP_LOCAL]
        c = sum(ends, Vector()) / 3
        overhead["tap_offs"].append(dict(rack=r["id"], box_bottom_center=[round(c.x, 4), round(c.y, 4), 2.376],
                                         whip_ends=[[round(v, 4) for v in e] for e in ends]))
        for side_name, ly in (("left", -0.258), ("right", 0.258)):     # v4: front fibre bundles of the GB300 rise into these
            d = M @ Vector((0.10, (0.20 if ly > 0 else -0.20), OV["raceway"]["drop_bottom_z"]))
            overhead["raceway_drops"].append(dict(rack=r["id"], kind="gb300_front_%s" % side_name, bottom=[round(v, 4) for v in d]))
    else:
        d = M @ Vector((OV["raceway"]["drop_local_x"], 0.0, OV["raceway"]["drop_bottom_z"]))
        overhead["raceway_drops"].append(dict(rack=r["id"], kind="bird_rack_rear", bottom=[round(v, 4) for v in d]))

layout = dict(
    schema="pigeon_pluggable hall_layout v1", date="2026-10-10", units="m", author="workstream R (DevLog-007)",
    frame="aisle along +Y, x = 0 aisle centre, rack fronts at x = +-1.1 facing the aisle, floor z = 0 (crude film v0.1 hall). Rack origin = floor centre of the footprint; rack local +X = front. Left row unrotated, right row rotated 180 deg about Z.",
    hall=dict(cold_aisle_width=2 * FACE_X, rack_face_x=FACE_X, rack_rear_x=round(FACE_X + RACK_D, 3), end_wall_y=END_WALL_Y,
              ceiling_z=CEIL, rows_y_span=[Y_START, round(racks[N_PER_ROW - 1]["y_span"][1], 3)],
              note="hall walls/ceiling/door are built by the scene agent; these values match crude v0.1"),
    rack_types=TYPES,
    pattern="per row index i = 0..14: bird_q3400 if i % 3 == 1 else gb300_nvl72 (C B C repeating); same indices on both rows so bird racks face each other across the aisle",
    counts=dict(racks=len(racks), bird_racks=n_bird, compute_racks=len(racks) - n_bird, bird_fraction=round(n_bird / len(racks), 3),
                cells=len(cells_world), bird_slots=len(cells_world), crowd_birds=sum(c["occupant"] == "crowd" for c in cells_world),
                hero_empty_slots=1, per_rack=len(cells_local), per_tier={t: sum(c["tier"] == t for c in cells_world) for t in range(3)}),
    racks=racks,
    entrance=ENTR,
    hero_cell=dict(asset="assets/components/datacenter/hero_cell_mating.blend", glb="assets/components/datacenter/hero_cell_mating.glb",
                   rack=HERO_RACK, slot=HERO_SLOT, compat_alias="HOOK_cell_1_1_1",
                   location=next(c["feet_origin"] for c in cells_world if c["occupant"] == "hero_empty"),
                   rotation_z_deg=next(c["bird_yaw_deg"] for c in cells_world if c["occupant"] == "hero_empty"),
                   note="detailed mating cell for the S2 cutaway; asset origin = this slot's feet origin; hide collection CUTAWAY_hero_cell and the L07 rack (or its unit-1 chassis) in front of the camera; hooks HOOK_hero_lip, HOOK_hero_host_pads, HOOK_hero_cutaway_cam"),
    overhead=overhead,
    landing_pad_primary=dict(rack=primary["rack"], location=primary["location"], facing_yaw_deg=primary["facing_yaw_deg"],
                             note="rack-top pad of the left-row bird rack nearest the entrance"),
    landing_pads=pads,
    perch_spots=perches,
    cells=dict(note="v2 tiered caged farm (owner P2): one entry per bird slot (24 per bird rack: 3 chassis x (3 front + 2 middle + 3 back)). feet_origin = bird origin (30 mm behind its own tier face); bird yaw = rack rotation (beak to the aisle); tier 0 front/bottom row, 1 middle (recessed 80 mm, offset half a column), 2 back/top row (recessed 160 mm). hook = HOOK_cell_* compat alias where one exists (tier 0 -> HOOK_cell_<u>_1_<c>, tier 2 -> HOOK_cell_<u>_0_<c>), else HOOK_bird_<u>_<tier>_<c>; bird_hook always the v2 name. Fully filled from the top: per rack fill_index 0 = highest slot; every slot is filled (occupant crowd) except the hero's front slot L07 HOOK_bird_1_0_1 (occupant hero_empty, the hero hops out forward in S2). neck_extension_m = suggested forward neck stretch so the head centre is 0.05 m in front of the tier-0 face (0 / 0.046 / 0.126 m); 0 is also collision-free.",
               list=cells_world),
)
os.makedirs(OUT, exist_ok=True)
K.write_json(os.path.join(OUT, "hall_layout.json"), layout)
print("LAYOUT racks", len(racks), "bird", n_bird, "cells", len(cells_world), "primary pad", primary)

# ------------------------------------------------------------------ bird flap vent asset (local: opening in YZ, +X = into the hall)
K.reset()
L = K.palette()
AID = "bird_flap_vent"
coll, root = K.new_asset(AID, accuracy="C")
OW, OH = 700.0, 500.0
fr = MB()
for (y0, y1, z0, z1) in ((-OW / 2 - 60, OW / 2 + 60, OH / 2, OH / 2 + 60), (-OW / 2 - 60, OW / 2 + 60, -OH / 2 - 60, -OH / 2),
                         (-OW / 2 - 60, -OW / 2, -OH / 2, OH / 2), (OW / 2, OW / 2 + 60, -OH / 2, OH / 2)):
    fr.box2((m(-120), m(y0), m(z0)), (m(20), m(y1), m(z1)), 0)
fr.box2((m(-160), m(-OW / 2 - 60), m(-OH / 2 - 75)), (m(-120), m(OW / 2 + 60), m(-OH / 2 - 60)), 1)    # outside landing ledge
gl = MB()
gl.box2((m(20), m(-OW / 2 - 50), m(OH / 2 + 50)), (m(24), m(OW / 2 + 50), m(OH / 2 + 56)), 0)        # inside LED strip
gl.box2((m(20), m(-OW / 2 - 50), m(-OH / 2 - 56)), (m(24), m(OW / 2 + 50), m(-OH / 2 - 50)), 0)
K.instantiate([Part("frame", fr, [L["dark_steel"], L["steel"]], bevel=0.002), Part("led", gl, [L["led_c"]])], coll, root, AID)
hinge = K.empty(AID + "_flap", coll, root, (m(-50), 0, m(OH / 2)))
hinge["note"] = "top hinge; rotate about local Y (radians); positive swings the bottom edge toward +X (into the hall)"
fl = MB()
n = 8
for k in range(n):                                                         # strip-curtain style flap (8 vertical strips)
    y0 = -OW / 2 + k * OW / n
    fl.box2((m(-6), m(y0 + 2), m(-OH)), (m(6), m(y0 + OW / n - 2), m(0)), 0)
fl.box2((m(-14), m(-OW / 2), m(-30)), (m(14), m(OW / 2), m(0)), 1)
smoke = K.M("flap_smoked_pc", (0.05, 0.07, 0.08), 0.0, 0.15, None, 0.0, 0.6)
K.instantiate([Part("strips", fl, [smoke, L["dark_steel"]])], coll, hinge, AID + "_flap")
K.hook("entrance_center_inside", coll, root, (m(30), 0, 0), note="+X = into the hall")
bpy.context.view_layer.update()
bb = K.bbox(coll)
K.save_blend(os.path.join(OUT, AID + ".blend"))
K.export_glb(coll, os.path.join(OUT, AID + ".glb"))
K.write_json(os.path.join(OUT, AID + ".json"), dict(
    asset_id=AID, date="2026-10-10", units="m", size_m=[round(bb[1][i] - bb[0][i], 4) for i in range(3)],
    frame="origin = centre of the opening on the inner wall face; opening in the YZ plane; +X = into the hall; Z up",
    opening_m=[OW / 1000, OH / 1000], parts=["frame (steel, 60 mm border, outside landing ledge)", "LED strips inside (cyan)",
                                             "bird_flap_vent_flap: top-hinged 8-strip smoked curtain; rotate about local Y"],
    accuracy="C (design; no reference)", hooks=K.hooks_of(coll)))

# ------------------------------------------------------------------ overhead infrastructure asset (world coordinates)
K.reset()
L = K.palette()
OID = "overhead_infrastructure"
ocoll, oroot = K.new_asset(OID, accuracy="C")
bw = MB()      # 0 busway housing, 1 tap-off box, 2 amber LED, 3 dark joints
rw = MB()      # yellow raceway
ld = MB()      # 0 galvanised ladder, 1 cable bundles
rods = MB()
y0, y1 = OV["y"]
for side in ("L", "R"):
    sx = -1 if side == "L" else 1
    xb = row_x(side, OV["busway"]["local_x"])
    zb0, zb1 = OV["busway"]["z"]
    bw.box2((xb - 0.08, y0, zb0), (xb + 0.08, y1, zb1), 0)
    for k in range(int((y1 - y0) / 2.0) + 1):
        yy = y0 + k * 2.0
        bw.box2((xb - 0.085, yy - 0.02, zb0 - 0.005), (xb + 0.085, yy + 0.02, zb1 + 0.005), 3)
        rods.cyl((xb, yy, (zb1 + CEIL) / 2), 0.006, CEIL - zb1, "z", 8, 0)
    xr0 = row_x(side, OV["raceway"]["local_x"])
    w2 = OV["raceway"]["width"] / 2
    z0, z1 = OV["raceway"]["z"]
    rw.box2((xr0 - w2, y0, z0), (xr0 + w2, y1, z0 + 0.006), 0)                     # raceway floor
    for e in (-1, 1):
        rw.box2((xr0 + e * w2 - 0.003, y0, z0), (xr0 + e * w2 + 0.003, y1, z1), 0)    # side walls
    for k in range(int((y1 - y0) / 1.5) + 1):
        yy = y0 + k * 1.5
        for e in (-1, 1):
            rods.cyl((xr0 + e * (w2 + 0.02), yy, (z1 + CEIL) / 2), 0.005, CEIL - z0, "z", 8, 0)
        rods.box2((xr0 - w2 - 0.03, yy - 0.02, z0 - 0.03), (xr0 + w2 + 0.03, yy + 0.02, z0), 0)      # trapeze bar
    xl = row_x(side, OV["ladder"]["local_x"])
    lw2 = OV["ladder"]["width"] / 2
    zl = OV["ladder"]["z"]
    for e in (-1, 1):
        ld.box2((xl + e * lw2 - 0.004, y0, zl), (xl + e * lw2 + 0.004, y1, zl + OV["ladder"]["rail_h"]), 0)
    n = int((y1 - y0) / 0.3)
    for k in range(n + 1):
        yy = y0 + k * 0.3
        ld.box2((xl - lw2, yy - 0.012, zl), (xl + lw2, yy + 0.012, zl + 0.02), 0)
    for j, dx in enumerate((-0.11, 0.0, 0.12)):
        ld.cyl((xl + dx, (y0 + y1) / 2, zl + 0.02 + 0.035), 0.035 if j != 1 else 0.045, y1 - y0, "y", 12, 1)
    for k in range(int((y1 - y0) / 1.5) + 1):
        yy = y0 + k * 1.5
        for e in (-1, 1):
            rods.cyl((xl + e * (lw2 + 0.02), yy, (zl + CEIL) / 2), 0.005, CEIL - zl, "z", 8, 0)
for t in overhead["tap_offs"]:
    c = t["box_bottom_center"]
    bw.box2((c[0] - 0.09, c[1] - 0.07, c[2]), (c[0] + 0.09, c[1] + 0.07, OV["busway"]["z"][0]), 1)
    bw.box2((c[0] - 0.02, c[1] - 0.072, c[2] + 0.07), (c[0] + 0.02, c[1] - 0.07, c[2] + 0.08), 2)
for d in overhead["raceway_drops"]:
    b = d["bottom"]
    side = "L" if b[0] < 0 else "R"
    xr0 = row_x(side, OV["raceway"]["local_x"])
    edge = xr0 + (OV["raceway"]["width"] / 2 if abs(b[0]) > abs(xr0) else -OV["raceway"]["width"] / 2) * (1 if b[0] > 0 else -1)
    z0, z1 = OV["raceway"]["z"]
    hw_ = 0.035 if d.get("kind", "").startswith("gb300") else 0.05                                           # GB300 front drops 70 mm
    rw.box2((min(edge, b[0]) - hw_, b[1] - hw_, z0 + 0.02), (max(edge, b[0]) + hw_, b[1] + hw_, z1), 0)        # spill-out
    rw.box2((b[0] - hw_, b[1] - hw_, b[2]), (b[0] + hw_, b[1] + hw_, z0 + 0.02), 0)                            # drop
galv = K.M("galvanised", (0.55, 0.56, 0.57), 0.5, 0.55)
K.instantiate([Part("busway", bw, [K.M("busway_alu", (0.42, 0.43, 0.45), 0.5, 0.5), L["light_grey"], L["led_a"], L["dark_steel"]]),
               Part("fibre_raceway", rw, [L["cable_yellow"]]),
               Part("ladder_tray", ld, [galv, L["cable_black"]]),
               Part("hanger_rods", rods, [galv])], ocoll, oroot, OID)
bpy.context.view_layer.update()
obb = K.bbox(ocoll)
K.save_blend(os.path.join(OUT, OID + ".blend"))
K.export_glb(ocoll, os.path.join(OUT, OID + ".glb"))
K.write_json(os.path.join(OUT, OID + ".json"), dict(
    asset_id=OID, date="2026-10-10", units="m", frame="world (hall) coordinates of hall_layout.json; origin = hall origin",
    size_m=[round(obb[1][i] - obb[0][i], 4) for i in range(3)], triangles=K.count_tris(ocoll),
    parts=["power busway per row over the rack rear with tap-off boxes above every GB300 (upper power whips end at the box bottom)",
           "yellow fibre raceway per row with spill-out + drop to the rear of every bird-rack top (behind the landing deck) and two front drops per GB300 (v4: the front fibre bundles end inside them)",
           "galvanised ladder tray over the hot-aisle side with 3 black cable bundles", "threaded hanger rods to the ceiling (z = 4.2)"],
    accuracy="C (design; common data-hall practice, no specific reference)", runs=overhead["runs"]))

# ------------------------------------------------------------------ hall preview (racks linked as collection instances)
if int(KV.get("previews", 1)):
    os.makedirs(PREV, exist_ok=True)
    scn = bpy.context.scene
    hall = bpy.data.collections.new("PREVIEW_hall")
    scn.collection.children.link(hall)
    with bpy.data.libraries.load(os.path.join(OUT, "bird_flap_vent.blend"), link=True) as (src, dst):
        dst.collections = ["ASSET_bird_flap_vent"]
    ve = bpy.data.objects.new("PREVIEW_vent", None)
    ve.instance_type = "COLLECTION"
    ve.instance_collection = dst.collections[0]
    ve.location = ENTR["location"]
    ve.rotation_euler = (0, 0, math.radians(-90))
    hall.objects.link(ve)
    srcs = {}
    for t, d in TYPES.items():
        path = os.path.join(OUT, os.path.basename(d["asset"]))
        with bpy.data.libraries.load(path, link=True) as (src, dst):
            dst.collections = [d["collection"]]
        srcs[t] = dst.collections[0]
    for r in racks:
        e = bpy.data.objects.new("PREVIEW_" + r["id"], None)
        e.instance_type = "COLLECTION"
        e.instance_collection = srcs[r["type"]]
        e.location = r["location"]
        e.rotation_euler = (0, 0, math.radians(r["rotation_z_deg"]))
        hall.objects.link(e)
    bp = MB()
    for c in cells_world:
        if c["occupant"] == "crowd":
            K.bird_proxy(bp, c["feet_origin"], math.radians(c["bird_yaw_deg"]), neck_ext=c["neck_extension_m"])
    K.bird_proxy(bp, primary["location"], math.radians(primary["facing_yaw_deg"]))
    K.obj_from_mb("PREVIEW_birds", bp, K.bird_mats(), hall, smooth=True)
    # floor tiles (cold aisle perforated), walls with door + vent opening
    fl = MB()
    for ix in range(-5, 5):
        for iy in range(-1, 23):
            fl.box((ix * 0.6 + 0.3, iy * 0.6 + 0.3, -0.01), (0.596, 0.596, 0.02), 1 if abs(ix * 0.6 + 0.3) < 1.1 else 0)
    wl = MB()
    for (x0, x1, z0, z1) in ((-3.1, -0.7, 0, CEIL), (0.7, 3.1, 0, CEIL), (-0.7, 0.7, 2.5, 3.0), (-0.7, -0.35, 3.0, CEIL),
                             (0.35, 0.7, 3.0, CEIL), (-0.35, 0.35, 3.5, CEIL)):
        wl.box2((x0, END_WALL_Y, z0), (x1, END_WALL_Y + 0.12, z1), 0)
    wl.box2((-3.2, -0.8, 0), (-3.1, END_WALL_Y, CEIL), 0)
    wl.box2((3.1, -0.8, 0), (3.2, END_WALL_Y, CEIL), 0)
    K.obj_from_mb("PREVIEW_floor_tiles", fl, [K.principled("PREVIEW_tile", (0.16, 0.165, 0.17), 0.2, 0.6),
                                              K.principled("PREVIEW_tile_perf", (0.10, 0.105, 0.11), 0.3, 0.5)], hall)
    K.obj_from_mb("PREVIEW_walls", wl, [K.principled("PREVIEW_wall", (0.05, 0.05, 0.055), 0.0, 0.8)], hall)
    K.preview_setup(res=(1280, 720), samples=8, world=0.12, exposure=-0.2)
    for k in range(6):
        K.light("aisle%d" % k, "AREA", (0, 1.5 + k * 2.0, 3.9), 120, 1.2, (0, 1.5 + k * 2.0, 0))
    for sx in (-1, 1):
        for k in range(3):
            K.light("hot%d%d" % (sx, k), "AREA", (sx * 2.6, 2.5 + k * 4, 3.6), 60, 1.0, (sx * 2.6, 2.5 + k * 4, 0))
    V = lambda n: os.path.join(PREV, "hall_layout_%s.png" % n)
    scn.render.resolution_x, scn.render.resolution_y = 600, 1000
    ocoll.hide_render = True
    K.render_view(V("top"), (0, 6.3, 20), (0, 6.3, 0), ortho=14.0)
    ocoll.hide_render = False
    scn.render.resolution_x, scn.render.resolution_y = 960, 540
    K.render_view(V("aisle_from_entrance"), (0.0, 12.75, 3.1), (0, 5.0, 1.0), lens=24)
    K.render_view(V("aisle_eye"), (0.25, 0.3, 1.6), (0, 8.0, 1.2), lens=28)
    K.render_view(V("fiber_drops"), (0.45, 2.4, 2.55), (-1.1, 4.4, 2.3), lens=28)
    K.render_view(V("gb300_front_cabling"), (0.6, 2.2, 1.5), (-1.08, 3.4, 1.25), lens=30)
    K.render_view(V("aisle_bird_racks"), (0.75, 4.6, 1.55), (-1.1, 6.2, 1.25), lens=30)
    K.render_view(V("landing_pad"), (-0.2, 9.3, 3.2), tuple(primary["location"]), lens=30)
    K.render_view(V("overhead_whips"), (-0.5, 3.0, 3.1), (-1.9, 4.75, 2.45), lens=28)
