"""Foot-slide and floor-penetration measurement on the evaluated boot-sole meshes (needs the character meshes visible)."""
import math

import bpy
import numpy as np


def measure_gait(root, arm, act, n_frames, speed, ground_tol=0.006, travel=(0.0, -1.0), cycles=2):
    """Play `cycles` cycles of a looping in-place gait while the root travels at `speed` (m/s, K-scaled).
    Returns dict(slide_mm_per_cycle (mean over feet), slide_mm_per_cycle_by_foot, worst_stance_drift_mm, min_z_mm, max_z_stance...)."""
    scn = bpy.context.scene
    dg = None
    soles = {sn: bpy.data.objects[root["asset_id"] + "_sole_" + sn] for sn in ("L", "R")}
    arm.animation_data_create().action = act
    tr = np.array([travel[0], travel[1], 0.0])
    pos = {sn: [] for sn in soles}
    zmin = 0.0
    total = n_frames * cycles
    for f in range(total + 1):
        scn.frame_set(f % n_frames if f % n_frames or f == 0 else n_frames)  # frame n == frame 0 of next cycle
        root.location = tuple(tr * speed * f / 30.0)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        for sn, o in soles.items():
            e = o.evaluated_get(dg)
            me = e.to_mesh()
            n = len(me.vertices)
            co = np.empty(n * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            M = np.array(o.matrix_world)
            w = co @ M[:3, :3].T + M[:3, 3]
            pos[sn].append(w)
            e.to_mesh_clear()
    res = {}
    for sn in soles:
        P = np.stack(pos[sn])           # (T, V, 3)
        zmin = min(zmin, float(P[..., 2].min()))
        contact = P[..., 2] < ground_tol
        slide = 0.0
        worst = 0.0
        drift = {}
        for t in range(total):
            both = contact[t] & contact[t + 1]
            if both.any():
                d = np.linalg.norm(P[t + 1, both, :2] - P[t, both, :2], axis=1)
                slide += float(np.median(d))
        # drift of a contact run: displacement of the same vertex from its first contact in the run
        run_start = {}
        for t in range(total + 1):
            idx = np.where(contact[t])[0]
            for v in list(run_start):
                if not contact[t, v]:
                    del run_start[v]
            for v in idx:
                if v not in run_start:
                    run_start[v] = P[t, v, :2].copy()
                else:
                    worst = max(worst, float(np.linalg.norm(P[t, v, :2] - run_start[v])))
        res[sn] = dict(slide_m_per_cycle=slide / cycles, worst_vertex_drift_m=worst)
    out = dict(
        slide_mm_per_cycle_L=round(res["L"]["slide_m_per_cycle"] * 1000, 1), slide_mm_per_cycle_R=round(res["R"]["slide_m_per_cycle"] * 1000, 1),
        slide_mm_per_cycle_mean=round((res["L"]["slide_m_per_cycle"] + res["R"]["slide_m_per_cycle"]) * 500, 1),
        worst_contact_vertex_drift_mm=round(max(res["L"]["worst_vertex_drift_m"], res["R"]["worst_vertex_drift_m"]) * 1000, 1),
        sole_min_z_mm=round(zmin * 1000, 1), root_speed_mps=speed, cycles=cycles, ground_tol_mm=ground_tol * 1000)
    root.location = (0, 0, 0)
    return out
