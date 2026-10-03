"""Smoke test of the assembly framework: Gary + Manager + rack pair + daylight rig, one shot, stills at several frames."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm
OUT = sys.argv[sys.argv.index("--") + 1]
asm.new_scene(preset="standard")
asm.rig("daylight")
g = asm.append("characters/gary", actions=True)
asm.place(g.root, (0, 0, 0), yaw=0)
asm.play(g, "idle", 0.0, hold=False, repeat=3)
asm.play(g, "topple_back", 1.5)
asm.key_prop(g.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(g.root, "p_hole_1_radius", 1.5, 1.0, "CONSTANT")
m = asm.append("characters/manager", actions=True)
asm.place(m.root, (2.5, 0, 0), yaw=-1.57)
asm.play(m, "aim_gun", 0.5)
asm.key_prop(m.root, "p_anger", 0.0, 0.0)
asm.key_prop(m.root, "p_anger", 2.0, 1.0)
r = asm.append("datacenter/rack_pair_for_cable_gag")
asm.place(r.root, (0, 4, 0))
asm.shot(0.0, 3.0, (1.2, -5.0, 1.5), (1.2, 0, 1.0), lens=28)
asm.finalize(os.path.join(OUT, "test_asm.blend"))
