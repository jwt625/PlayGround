"""Build fiber_spaghetti_generator.blend: seeded pile of patch cords spilling out of a pulled-out tray (S6).

Run: Blender -b --python scripts/assets/interconnect/build_fiber_spaghetti_generator.py -- assets/components/interconnect
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import ic_common as I
import fiber_spaghetti as FS

C = I.C
OUT = C.argv_after_dashes()[0]
ASSET = "fiber_spaghetti_generator"
HERE = os.path.dirname(os.path.abspath(__file__))

C.reset()
M = I.Mats()
coll, root = C.new_asset(ASSET, accuracy="C")
I.add_prop(root, "p_squirm", 0.0, 0.0, 1.0, "squirm amplitude 0..1 (x 0.06 m noise displacement, weighted toward the free ends)")
I.add_prop(root, "p_squirm_speed", 1.0, 0.0, 10.0, "noise time scale (noise units per second)")
I.add_prop(root, "p_cord_diameter_mm", 2.0, 1.0, 50.0, "documentation: baked fibre radius = 1.0 mm (2.0 mm jacket); change the modifier input 'Fiber Radius' (m) to rescale")

t0 = time.time()
hi = C.sub_collection(coll, "LOD_high")
lo = C.sub_collection(coll, "LOD_low")
pile_hi = FS.build_pile("fiber_spaghetti_generator_pile_9072", count=9072, seed=44, length_range=(1.0, 3.0), spill_radius=1.2, slack=1.0,
                        fiber_radius=0.001, profile_res=4, n_pts=32, parent=root, collection=hi)
pile_lo = FS.build_pile("fiber_spaghetti_generator_pile_low_600", count=600, seed=44, length_range=(1.0, 3.0), spill_radius=1.2, slack=1.0,
                        fiber_radius=0.0035, profile_res=3, n_pts=32, parent=root, collection=lo)
print("generation s", time.time() - t0)
# squirm drivers on both piles
for ob in (pile_hi, pile_lo):
    ng = ob.modifiers["fiber_gn"].node_group
    for sock, expr in (("Squirm Amplitude", "p*0.06"), ("Squirm Speed", "p")):
        ident = FS._identifier(ng, sock)
        fc = ob.driver_add('modifiers["fiber_gn"]["%s"]' % ident)
        d = fc.driver
        d.type = "SCRIPTED"
        v = d.variables.new()
        v.name = "p"
        v.type = "SINGLE_PROP"
        v.targets[0].id = root
        v.targets[0].data_path = '["p_squirm"]' if sock == "Squirm Amplitude" else '["p_squirm_speed"]'
        d.expression = expr
tris_hi = FS.evaluated_tris(pile_hi)
tris_lo = FS.evaluated_tris(pile_lo)
I.hide_collection(lo, True)

# tray (simple sheet-metal drawer) so the spill has an origin
mb = I.MB()
tz = 0.35 * 1000
mb.box((-240, 240), (0, 620), (tz - 25, tz - 21), bev=0.5, seg=1)
for s in (-1, 1):
    mb.box((236, 240) if s > 0 else (-240, -236), (0, 620), (tz - 25, tz + 15), bev=0.5, seg=1)
mb.box((-240, 240), (616, 620), (tz - 25, tz + 15), bev=0.5, seg=1)
mb.box((-240, 240), (-4, 0), (tz - 25, tz - 5), bev=0.5, seg=1)    # low front lip
tray = mb.build("fiber_spaghetti_generator_tray", M["galv_steel"], smooth_deg=35)
I.place(tray, coll, root)
mb = I.MB()
for s in (-1, 1):
    mb.box((s * 255 - 6, s * 255 + 6), (-40, 640), (tz - 40, tz - 25), bev=0.8, seg=1)
mb.box((-120, 120), (-20, -10), (tz - 10, tz + 5), bev=1.0, seg=1)  # pull handle bar
rails = mb.build("fiber_spaghetti_generator_slide_rails", M["steel"], smooth_deg=35)
I.place(rails, coll, root)
C.hook("tray_lip", coll, root, loc=(0, 0, tz * 0.001))
C.hook("pile_center", coll, root, loc=(0, -0.66, 0.0))
C.hook("thread_through", coll, root, loc=(0.0, -0.25, 0.12))

# embed the generator source in the blend
for fn in ("fiber_spaghetti.py",):
    t = bpy.data.texts.new(fn)
    t.write(open(os.path.join(HERE, fn)).read())
meta = {
    "description": "Seeded Geometry Nodes fibre pile: strands are stored as edge polylines (fiber_color and squirm_w attributes), NG_fiber_spaghetti turns them into round tubes at evaluation time and adds noise squirm.",
    "parameters": {
        "count": "build_pile(count=...), default 9072 (hypothetical NVL576-style fibres per rack, own estimate, 200G per fibre; scripts/..., DevLog-000 Sec 6.2 fiber_bundle) ; low variant 600",
        "palette": "FS.DEFAULT_PALETTE list of RGB (yellow-heavy), per-strand colour index from the seeded RNG",
        "length_range": "(1.0, 3.0) m per strand (own estimate)",
        "spill_radius": "1.2 m landing radius around (0, -0.66)", "slack": "0 straight .. 2 very loopy (heading noise sigma 0.35 + 0.9 slack)",
        "seed": "44 (numpy default_rng); identical seed gives identical piles",
        "modifier_inputs": "Fiber Radius (m), Profile Resolution (3-16), Squirm Amplitude (m), Squirm Speed, Squirm Noise Scale, Material",
    },
    "usage": "Open the blend, run FS.build_pile(...) from the embedded text 'fiber_spaghetti.py' (or import scripts/assets/interconnect/fiber_spaghetti.py) with your own count/seed/palette/length/spill/slack; parent the result under the asset root.",
    "squirm_hook": "root[p_squirm] 0..1 drives modifier Squirm Amplitude (x 0.06 m) and root[p_squirm_speed] drives the noise time speed; noise follows scene time, so animation is deterministic per frame; displacement is weighted by distance along the strand (free ends move most).",
    "dimension_table": [
        dict(item="patch-cord jacket diameter (default)", value=2.0, unit="mm", source="task spec: 2.0 and 3.0 mm SM patch cords", accuracy="B"),
        dict(item="pile strands (high/low)", value="9072 / 600", unit="count", source="own estimate 9,072 fibres per rack (DevLog-000 Sec 6.2; ctx-summary 20260612_fiber_bundle)", accuracy="C"),
        dict(item="tray size 480 x 620 x 40", value=480, unit="mm", source="estimate (19-inch drawer)", accuracy="C"),
    ],
    "evaluated_triangles": {"LOD_high 9072 strands profile 4": tris_hi, "LOD_low 600 strands profile 3 radius 3.5 mm": tris_lo},
    "hooks": {"HOOK_tray_lip": "tray front lip centre", "HOOK_pile_center": "pile landing centre on the floor", "HOOK_thread_through": "suggested point where a fibre threads through Gary's hole"},
    "custom_properties": {"p_squirm": "0..1", "p_squirm_speed": "0..10", "p_cord_diameter_mm": "documentation"},
    "origin": "ROOT on the floor under the tray lip; tray body at +Y, spill toward -Y, floor z = 0.",
    "simplifications": ["strands interpenetrate (no collision); pile layering is a height-field heuristic", "cords have no connectors (open ends)", "strand start points are hidden inside the tray volume"],
    "material_slots": sorted(m.name for m in bpy.data.materials if m.users),
    "intended_usage": "S6 fibre mess, scaled up by the assembler; LOD_low for distant shots.",
}
blend = os.path.join(OUT, ASSET + ".blend")
C.finish(ASSET, blend, coll, meta, preview_dir=None)
prev = os.path.join(OUT, "previews")
I.preview_views(coll, prev, ASSET, [("three_quarter", (0, -500, 150), (0.8, -0.9, 0.55), 2800, 35), ("top", (0, -0.4 * 1000, 150), (0, -0.02, 1), 2600, 35)])
bpy.context.scene.frame_set(1)
root["p_squirm"] = 1.0
bpy.context.view_layer.update()
I.preview_views(coll, prev, ASSET, [("closeup_squirm", (0, -600, 80), (0.5, -1, 0.5), 700, 40)])
root["p_squirm"] = 0.0
lo.hide_render = False
hi.hide_render = True
I.preview_views(coll, prev, ASSET, [("low_variant_three_quarter", (0, -500, 150), (0.8, -0.9, 0.55), 2800, 35)])
