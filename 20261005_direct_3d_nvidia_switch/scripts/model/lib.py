"""Modeling helpers (run inside Blender). World frame = tray frame (config/world.yaml): meters; X tray width
(0 midway between the outer side walls), Y from the front panel face (Y = 0) toward the rear, Z out of the open
top (Z = 0 at the crossbar top face). Gravity and the package have their own frames in config/frames.json:
build parts in the local frame and call apply_frame(objs, "package").

Conventions:
- Every group module scripts/model/<group>/build.py defines build(P: dict, coll) and reads its parameters
  from config/model/<group>.toml (millimeters unless a key says otherwise). No hard-coded dimensions in code.
- Object names: "<group>.<part>" (e.g. "case.tray_wall_near"). The evaluator attributes errors per object, so
  split parts at the level you want feedback on. Names starting with "_" are helpers and ignored by eval.
- Labels: separate thin objects named "<group>.label_<name>" with an image texture from data/textures/.
"""

from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
MM = 1e-3


def mm(*v):
    return tuple(x * MM for x in v) if len(v) > 1 else v[0] * MM


def group_collection(group: str) -> bpy.types.Collection:
    name = f"G_{group}"
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def clear_collection(coll: bpy.types.Collection) -> None:
    for o in list(coll.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)


def _obj_from_bm(name: str, bm: bmesh.types.BMesh, coll, mat=None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def box(name, size_mm, center_mm, coll, mat=None, bevel_mm: float = 0.0, segments: int = 2) -> bpy.types.Object:
    """Axis-aligned box, size and center in mm. Optional bevel (applied)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(mm(*size_mm)), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(mm(*center_mm)), verts=bm.verts)
    ob = _obj_from_bm(name, bm, coll, mat)
    if bevel_mm > 0:
        bevel(ob, bevel_mm, segments)
    return ob


def bevel(ob, width_mm: float, segments: int = 2, angle_deg: float = 30.0) -> None:
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width = width_mm * MM
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle_deg)
    apply_modifiers(ob)


def apply_modifiers(ob) -> None:
    dg = bpy.context.evaluated_depsgraph_get()
    oe = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(oe)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def boolean(target, cutter, op: str = "DIFFERENCE", remove_cutter: bool = True) -> None:
    m = target.modifiers.new("bool", "BOOLEAN")
    m.operation = op
    m.object = cutter
    m.solver = "EXACT"
    apply_modifiers(target)
    if remove_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)


def open_box(name, outer_mm, wall_mm, floor_mm, center_xy_mm, z0_mm, coll, mat=None, bevel_mm=0.0,
             open_sides: tuple[str, ...] = ()):
    """Open-top box (tray): outer (L, W, H) in mm, walls of thickness wall_mm, floor thickness floor_mm,
    bottom at z0_mm. open_sides may contain "+x", "-x", "+y", "-y" to remove that wall entirely."""
    L, W, H = outer_mm
    cx, cy = center_xy_mm
    ob = box(name, (L, W, H), (cx, cy, z0_mm + H / 2), coll, mat)
    il = L - 2 * wall_mm + (wall_mm + 1 if "+x" in open_sides else 0) + (wall_mm + 1 if "-x" in open_sides else 0)
    iw = W - 2 * wall_mm + (wall_mm + 1 if "+y" in open_sides else 0) + (wall_mm + 1 if "-y" in open_sides else 0)
    sx = ((wall_mm + 1) / 2 if "+x" in open_sides else 0) - ((wall_mm + 1) / 2 if "-x" in open_sides else 0)
    sy = ((wall_mm + 1) / 2 if "+y" in open_sides else 0) - ((wall_mm + 1) / 2 if "-y" in open_sides else 0)
    cut = box("_cut", (il, iw, H), (cx + sx, cy + sy, z0_mm + floor_mm + H / 2 + 0.5), coll)
    boolean(ob, cut)
    if bevel_mm > 0:
        bevel(ob, bevel_mm, 2)
    return ob


def cylinder(name, r_mm, h_mm, center_mm, coll, axis: str = "z", mat=None, verts: int = 48, r2_mm=None):
    """Cylinder (or cone frustum with r2_mm) centered at center_mm along axis x/y/z."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r_mm * MM,
                          radius2=(r2_mm if r2_mm is not None else r_mm) * MM, depth=h_mm * MM)
    rot = {"z": Matrix.Identity(3), "x": Matrix.Rotation(math.pi / 2, 3, "Y"),
           "y": Matrix.Rotation(-math.pi / 2, 3, "X")}[axis]
    bmesh.ops.rotate(bm, verts=bm.verts, matrix=rot)
    bmesh.ops.translate(bm, vec=Vector(mm(*center_mm)), verts=bm.verts)
    return _obj_from_bm(name, bm, coll, mat)


def tube_path(name, pts_mm, r_mm, coll, mat=None, resolution: int = 8, bevel_res: int = 4):
    """Wire/cable: smooth curve through points (mm) with circular cross-section radius r_mm."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = resolution
    cu.bevel_depth = r_mm * MM
    cu.bevel_resolution = bevel_res
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts_mm) - 1)
    for bp, p in zip(sp.bezier_points, pts_mm):
        bp.co = Vector(mm(*p))
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def place(ob, loc_mm=(0, 0, 0), rot_deg=(0, 0, 0)) -> None:
    """Apply a rigid transform to the mesh data (keeps object transforms identity)."""
    M = Matrix.Translation(Vector(mm(*loc_mm))) @ Matrix.Rotation(math.radians(rot_deg[2]), 4, "Z") @ \
        Matrix.Rotation(math.radians(rot_deg[1]), 4, "Y") @ Matrix.Rotation(math.radians(rot_deg[0]), 4, "X")
    if ob.type == "MESH":
        ob.data.transform(M)
    else:
        ob.matrix_world = M @ ob.matrix_world


# ---------- materials ----------

def mat_pbr(name, color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, transmission=0.0, ior=1.45,
            alpha=1.0, emission=None) -> bpy.types.Material:
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["IOR"].default_value = ior
    b.inputs["Alpha"].default_value = alpha
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = 1.0
    if alpha < 1.0 or transmission > 0:
        m.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    m.diffuse_color = (*color, 1.0)
    return m


def mat_image(name, image_path: str | Path, roughness=0.6) -> bpy.types.Material:
    """Material with an image texture on Base Color (labels, printed art). Path relative to project root."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    p = Path(image_path)
    if not p.is_absolute():
        p = ROOT / p
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = roughness
    # Photo textures already contain the real highlights; no added specular, so under the calibrated rig
    # (uniform white world, strength 1) the surface renders at the photo's own values.
    b.inputs["Specular IOR Level"].default_value = 0.0
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(p), check_existing=True)
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    return m


def label_quad(name, corners_mm, coll, mat, offset_mm: float = 0.05):
    """Flat label from 4 world corners (mm) in order: top-left, top-right, bottom-right, bottom-left of the
    printed image; UVs map the full image. offset_mm pushes it along its normal to avoid z-fighting."""
    c = [Vector(mm(*p)) for p in corners_mm]
    n = (c[3] - c[0]).cross(c[1] - c[0]).normalized()  # (down) x (right): toward the viewer of the print
    c = [p + n * offset_mm * MM for p in c]
    bm = bmesh.new()
    order = [0, 3, 2, 1]  # TL, BL, BR, TR: counter-clockwise seen from the front -> face normal = n
    vs = [bm.verts.new(c[i]) for i in order]
    f = bm.faces.new(vs)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop, t in zip(f.loops, [(0, 1), (0, 0), (1, 0), (1, 1)]):
        loop[uv].uv = t
    return _obj_from_bm(name, bm, coll, mat)


def load_params(group: str) -> dict:
    """Parameters from config/model/<group>.toml (TOML: Blender's Python 3.11 has tomllib; no PyYAML)."""
    import tomllib
    p = ROOT / "config" / "model" / f"{group}.toml"
    return tomllib.loads(p.read_text()) if p.exists() else {}


def frame_matrix(name: str) -> Matrix:
    """4x4 (Blender units, meters) mapping local frame coordinates to world; config/frames.json."""
    import json
    fr = json.loads((ROOT / "config" / "frames.json").read_text())["frames"][name]
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = fr["R"][i][j]
        M[i][3] = fr["t_mm"][i] * MM
    return M


def apply_frame(objs, name: str) -> None:
    """Move objects built in a local frame (mm-based coordinates via mm()) into the world (tray) frame."""
    M = frame_matrix(name)
    for ob in objs:
        if ob.type == "MESH":
            ob.data.transform(M)
        else:
            ob.matrix_world = M @ ob.matrix_world
