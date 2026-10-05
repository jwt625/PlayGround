"""Scene integration test: asm scene, append gary, schedule v2 actions through apply()/walk(), render a few frames.
Blender -b --python m2_scene_test.py -- <out_dir>
"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "scripts", "film_v1"))
sys.path.insert(0, HERE)
import bpy
import asm
import motion_v2 as M2
import m2_render as R
out = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out, exist_ok=True)
scn = asm.new_scene(frames=300, world=None)
gary = asm.append("characters/gary")
asm.place(gary.root, (0, 0, 0), yaw=0)
M2.apply(gary, "idle_breathe", 0.0, hold=True, repeat=4)                       # base idle, held
t_end = M2.walk(gary, 1.0, [(0, 0, 0), (0, -3.0, 0)])                          # walk 3 m forward (-y)
st = M2.apply(gary, "turn_left_90", t_end, hold=False)                         # turn, root yaw handoff
M2.apply(gary, "shrug_upper", t_end + 1.5, layer="gesture", hold=False)       # upper layer over the idle
M2.apply(gary, "shot_hit_fall", 7.0)
print("T_END", t_end, "turn strip", st.frame_start, st.frame_end, "yaw key", gary.root.rotation_euler[2])
# lighting + floor for the preview
cam = R.setup_scene()
for o in list(bpy.data.objects):
    pass
scn.render.resolution_x, scn.render.resolution_y = 540, 675
scn.frame_end = 300
for t, name in ((0.5, "idle"), (2.0, "walk_a"), (3.0, "walk_b"), (t_end + 0.4, "turn_a"), (t_end + 1.2, "turn_end"), (t_end + 2.2, "shrug_layer"), (7.4, "fall_a"), (9.5, "fall_end")):
    f = asm.F(t)
    scn.frame_set(f)
    bpy.context.view_layer.update()
    r = gary.root
    R.aim(cam, (r.location.x + 2.2, r.location.y - 3.0, 1.2), (r.location.x, r.location.y, 0.9), 40)
    scn.render.filepath = os.path.join(out, "scene_%s.png" % name)
    print("FRAME", name, f, "dead_eyed=%.2f shock=%.2f" % (r["p_expr_dead_eyed"], r["p_expr_shock"]), tuple(round(x, 2) for x in r.location), round(r.rotation_euler[2], 2))
    bpy.ops.render.render(write_still=True)
