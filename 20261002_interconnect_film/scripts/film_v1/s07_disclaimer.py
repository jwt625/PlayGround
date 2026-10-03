"""S7: ad-style disclaimer card (2 s, 60 frames): black screen, white disclaimer text, small-font list of the actual sources."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import asm
import blender_lib as L

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "scenes/v1/s07_disclaimer.blend"
asm.new_scene(preset="standard", frames=60, world=None)
scn = bpy.context.scene
wd = bpy.data.worlds.new("black")
wd.use_nodes = True
wd.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
scn.world = wd
L.V(L.box("black", (0, 6, 2), (14, 0.1, 18), (0, 0, 0), emit=100), 0, 0, 2)
asm.shot(0.0, 2.0, (0, 0, 2), (0, 6, 2))
L.ovt("DISC", "DISCLAIMER:\nGary is fictional.\nResults not typical.\nFigures vary by standard,\nvendor and test condition.\n"
              "Parody; not affiliated\nwith any company.", 0, 0, 2.0)
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
L.ovt("SRC", src, 0, 0, 2.0)
L.ovt("FX", "[VO ~9 words/s: Gary is fictional. Results not typical. Figures vary by standard, vendor and mood. "
            "Void where copper is cheaper.]", 0, 0, 2.0)
for s in range(2):
    L.ovt("TC", "S7  0:%02d" % s, 0, s, s + 1)
asm.finalize(OUT, frames=60)
