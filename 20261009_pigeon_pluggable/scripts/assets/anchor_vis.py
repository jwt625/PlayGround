"""Pick a label anchor that the camera can actually see (ray test), shared by the still and the animation renderers."""
from mathutils import Vector


def sample_points(objs, deps, per_obj=240):
    pts = []
    for o in objs:
        ev = o.evaluated_get(deps)
        me = ev.to_mesh()
        vs = me.vertices
        step = max(1, len(vs) // per_obj)
        mw = ev.matrix_world
        pts += [mw @ vs[i].co for i in range(0, len(vs), step)]
        ev.to_mesh_clear()
    return pts


def visible_anchor(scene, cam, objs, transparent=(), per_obj=240, max_tests=300):
    """Return (point, visible) for the sampled vertex closest to the parts' centroid that is not occluded.
    Objects whose names are in `transparent` (x-rayed skin, glass) are skipped by the ray."""
    deps = bpy_context_depsgraph()
    names = {o.name for o in objs}
    pts = sample_points(objs, deps, per_obj)
    if not pts:
        return None, False
    c = sum(pts, Vector()) / len(pts)
    cam_loc = cam.matrix_world.translation
    pts.sort(key=lambda p: (p - c).length)
    for p in pts[:max_tests]:
        d = p - cam_loc
        dist = d.length
        d.normalize()
        origin = cam_loc.copy()
        travelled = 0.0
        for _ in range(10):
            hit, loc, nrm, idx, ob, mtx = scene.ray_cast(deps, origin, d, distance=dist - travelled + 2e-5)
            if not hit:
                return p, True
            if ob.original.name in names:
                return p, True
            if ob.original.name in transparent:
                step = (loc - origin).length + 2e-6
                origin = loc + d * 2e-6
                travelled += step
                continue
            break
    return c, False


def bpy_context_depsgraph():
    import bpy
    return bpy.context.evaluated_depsgraph_get()
