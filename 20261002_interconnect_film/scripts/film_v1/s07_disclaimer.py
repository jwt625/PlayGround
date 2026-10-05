"""S7: ad-style disclaimer card (2 s, 60 frames): black screen, white disclaimer text, small-font list of the actual sources.

v1.2: card text matches the spoken disclaimer (scripts/audio/narration.json "disclaimer") plus the parody line; laid out over the
whole 4:5 frame (large disclaimer block at the top, sources small at the bottom); complete from frame 1 (no fade; transition T6
sweeps it in); debug VO line removed; timecode only with the draft HUD (asm.timecode, FILM_HUD).
Run: FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s07_disclaimer.py -- scenes/v1/s07_disclaimer.blend
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import asm  # noqa: E402
import blender_lib as L  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "scenes/v1/s07_disclaimer.blend"
asm.new_scene(preset="standard", frames=60, world=None)
scn = bpy.context.scene
wd = bpy.data.worlds.new("black")
wd.use_nodes = True
wd.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
scn.world = wd
L.V(L.box("black", (0, 6, 2), (14, 0.1, 18), (0, 0, 0), emit=100), 0, 0, 2)
asm.shot(0.0, 2.0, (0, 0, 2), (0, 6, 2))

# scene-local overlay layers (frame is about x -2..2, y -2.5..2.5 in overlay layout units); lines are wrapped by hand
L.OVL["DISC_H"] = dict(size=0.40, color=(1.0, 0.86, 0.25), loc=(-1.75, 2.15), ax="LEFT", ay="TOP", shadow=False, font="Arial Black")
L.OVL["DISC_B"] = dict(size=0.26, color=(1.0, 1.0, 1.0), loc=(-1.75, 1.45), ax="LEFT", ay="TOP", shadow=False, font="Arial Bold")
L.OVL["SRC_B"] = dict(size=0.07, color=(0.78, 0.78, 0.78), loc=(-1.75, -2.30), ax="LEFT", ay="BOTTOM", shadow=False, font="Arial Bold")
L.WRAP["DISC_H"] = L.WRAP["DISC_B"] = 200
L.WRAP["SRC_B"] = 110
L.ovt("DISC_H", "DISCLAIMER:", 0, 0, 2.0)
L.ovt("DISC_B", "Gary is fictional.\nResults not typical.\nFigures vary by standard,\nvendor and mood.\n"
                "Parody; not affiliated\nwith any company.\nVoid where copper\nis cheaper.", 0, 0, 2.0)
src = ("SOURCES\n"
       "IEEE 802.3bj-2014, 802.3ck, 802.3dj task-force documents: copper reach objectives\n"
       "SemiAnalysis, 'Nvidia's Optical Boogeyman': NVL72 copper backplane\n"
       "Arista and Ciena slides, OFC 2026; Broadcom 800G slides, 2023: power per bit\n"
       "Cheng et al., Opt. Express 33, 24190 (2025): pluggable vs NPO vs CPO\n"
       "Broadcom TH5-51.2T Bailly CPO deck; NVIDIA Developer Blog, Scaling AI Factories with CPO\n"
       "Corning SMF-28 spec PI-1424; own hypothetical NVL576-style fiber count\n"
       "Package layout after NVIDIA's public Rubin Ultra slide; probe station after FormFactor CM300 photo\n"
       "Dimensions: OSFP and QSFP-DD MSAs; OCP HGX and OAM specs; OIF ELSFP-01.0; FormFactor CM300xi planning guide\n"
       "Eye, BER, ring-spectrum, heat-wave and decay animations: illustrative models, not measurements\n"
       "Style reference: Zack D. Films")
L.ovt("SRC_B", src, 0, 0, 2.0)
asm.timecode(7, seconds=2)
asm.finalize(OUT, frames=60)
