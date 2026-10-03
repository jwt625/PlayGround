"""wafer_300mm_siph: 300 mm silicon-photonics wafer with a reticle grid (26 x 33 mm fields), 7 x 9 mm dies as
individual objects (per-die colour via object colour), scribe test structures and alignment marks.
Run: Blender -b --python scripts/assets/fab_test/build_wafer_300mm_siph.py -- assets/components/fab_test
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, wafer_lib as W  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("wafer_300mm_siph", accuracy="B")
A.src("SEMI M1 wafer geometry (see wafer_300mm)", "https://en.wikipedia.org/wiki/Wafer_(electronics)", "wafer disc")
A.src("Reticle field 26 x 33 mm = maximum scanner field (ASML/Nikon/Canon DUV/EUV scanners, common knowledge)",
      "n/a (standard stepper/scanner full-field size)", "field grid")
A.src("Project PIC die size about 7 x 9 mm (film storyboard decision, DevLog-001/003)", "DevLog/DevLog-001-story-scenes-assets-proposal.md",
      "die size")
A.dim("reticle field", "26 x 33", "mm", "standard full scanner field", "A")
A.dim("die size", "7.0 x 9.0", "mm", "project PIC, 'about 7 x 9 mm'", "B")
A.dim("die pitch", "7.1 x 9.1", "mm", "7 x 9 die + 100 um scribe (estimate)", "C")
A.dim("dies per field", "3 x 3 (21.3 x 27.3 mm) + test-structure bands 4.7 mm (X) and 5.7 mm (Y)", "-", "layout choice to fit the 26 x 33 field", "C")
A.dim("edge exclusion for full dies", 3, "mm", "typical 3 mm", "B")
A.dim("film stack thickness (exaggerated)", 0.020, "mm", "real BEOL about 10 um; exaggerated for render stability", "C")
A.dim("pad size / pitch", "0.16 / 0.57", "mm", "generic", "C")
silicon = A.mat("silicon")
W.disc(A, "wafer_300mm_siph_disc", silicon)
A.text("wafer_300mm_siph_lasermark", "NC-SIPH-001", 2.0, (0, -139, W.T_WAFER + 0.01), "steel_dark")
dies = W.die_layout()
fields = W.field_list(dies)
film = W.die_material(A)
proto = W.build_die_proto(A, "wafer_300mm_siph_die_proto", "full", [film, A.mat("gold")])
proto.hide_render = proto.hide_viewport = True
objs = W.place_dies(A, proto, dies, "wafer_300mm_siph")
W.pcm_and_marks(A, dies, ["copper", "gold"])
A.hook("wafer_center_top", (0, 0, W.T_WAFER), A.root, size=0.05)
A.hook("wafer_notch", (0, -W.R_WAFER, W.T_WAFER / 2), A.root, size=0.02)
A.hook("wafer_bottom_center", (0, 0, 0), A.root, size=0.03)
A.prop("die_count", len(dies), "number of die objects (read-only)", unit="")
A.prop("p_pass_fraction_note", 0.1, "documentation only: storyboard gag = exactly 1 in 10 dies green (use die_map_tools.assign_pass_fail)", 0.0, 1.0)
diemap = [{"name": o.name, "row": d["row"], "col": d["col"], "x_mm": round(d["x"], 4), "y_mm": round(d["y"], 4),
           "field": [d["fi"], d["fj"]], "probe_order": d["order"]} for o, d in zip(objs, dies)]
with open(os.path.join(OUT, "wafer_300mm_siph_diemap.json"), "w") as f:
    json.dump({"die_size_mm": [W.DIE_X, W.DIE_Y], "pitch_mm": [W.PITCH_X, W.PITCH_Y], "field_mm": [W.FIELD_X, W.FIELD_Y],
               "n_dies": len(dies), "n_fields_with_dies": len(fields), "notch": "-Y", "dies": diemap}, f, indent=1)
A.preview_fit = 1.15
A.preview_shadows = False
meta = {
    "description": "300 mm silicon-photonics wafer: %d full dies (7 x 9 mm) in %d reticle fields; every die is its own object." % (len(dies), len(fields)),
    "origin": "bottom centre (z = 0 back surface); notch at -Y",
    "scene_usage": "S5 wafer-level test: parent root to HOOK_wafer_slot of the probe station; colour the die map as the probe steps.",
    "die_recolor_howto": {
        "mechanism": "All dies are linked duplicates (one shared mesh) of wafer_300mm_siph_die_proto. Their shared material MAT_fab_test_die_state reads the OBJECT colour: "
                     "base colour = obj.color[0:3], emission strength = obj.color[3] * 3 (alpha = glow 0..1). Set obj.color per die; no material edits needed.",
        "states": {"idle": [0.10, 0.14, 0.26, 0.0], "probing_glow": [1.0, 1.0, 1.0, 1.0], "pass": [0.10, 0.90, 0.25, 0.8], "fail": [0.80, 0.08, 0.08, 0.6]},
        "per_die_custom_props": ["die_row", "die_col", "die_x_mm", "die_y_mm", "die_probe_order (serpentine, row by row from -Y)", "die_field"],
        "pass_fail_gag": "scripts/assets/fab_test/die_map_tools.py: assign_pass_fail(root) makes exactly round(N/10) dies green (spread by a seeded shuffle); "
                         "assign_pass_fail_by_order() makes every 10th die in probe order green. Keyframe obj.color (4 values) at the probe times.",
        "diemap_json": "assets/components/fab_test/wafer_300mm_siph_diemap.json (die names, rows/cols, centres in wafer coordinates)"},
    "n_dies": len(dies), "n_fields_with_dies": len(fields),
    "simplifications": ["Edge partial dies omitted (full dies only, 3 mm edge exclusion)", "Die film 20 um (exaggerated)", "Ring/grating features are boxes, not lithographic layouts", "Scribe lanes are 100 um (true scale) and test structures live in the 4.7/5.7 mm field bands"]}
A.finish(OUT, meta, views=("three_quarter", "top"),
         closeups=[("dies_close", (-8, -48, 14), (-8, -18, 0.8), 80), ("field_close", (-15, -45, 36), (-1, -12, 0.8), 55), ("notch_edge", (0, -190, 28), (0, -149, 0.4), 90)])
