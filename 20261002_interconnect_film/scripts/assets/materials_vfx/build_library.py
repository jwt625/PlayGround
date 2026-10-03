"""Merge all materials_vfx node groups, materials, worlds and effect/rig collections into one library blend and verify it.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_library.py -- assets/components/materials_vfx
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bpy
import common as C

OUT = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx")
LIB = os.path.join(OUT, "materials_vfx_library.blend")
SRC = ["materials_clay", "materials_pbr_hardware", "shader_heat_wave", "shader_bullet_hole_note", "shader_egg_fry", "vfx_particles_and_effects", "lighting_and_world"]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
summary = {}
for name in SRC:
    path = os.path.join(OUT, name + ".blend")
    before = (set(bpy.data.node_groups.keys()), set(bpy.data.materials.keys()), set(bpy.data.collections.keys()), set(bpy.data.worlds.keys()))
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.worlds = [w for w in src.worlds if w not in bpy.data.worlds]
        dst.collections = [c for c in src.collections if c not in bpy.data.collections]
        dst.node_groups = [n for n in src.node_groups if n not in bpy.data.node_groups]
        dst.materials = [m for m in src.materials if m not in bpy.data.materials]
    # collections that were appended are not linked to the scene: keep them via fake user and exclude the demo spheres
    summary[name] = {"node_groups": sorted(set(bpy.data.node_groups.keys()) - before[0]), "materials": len(set(bpy.data.materials.keys()) - before[1]),
                     "collections": sorted(set(bpy.data.collections.keys()) - before[2]), "worlds": sorted(set(bpy.data.worlds.keys()) - before[3])}
for c in bpy.data.collections:
    c.use_fake_user = True
for ng in bpy.data.node_groups:
    ng.use_fake_user = True
for m in bpy.data.materials:
    m.use_fake_user = True
for w in bpy.data.worlds:
    w.use_fake_user = True
# drop demo-sphere collections of the material libraries (ASSET_materials_*, ASSET_shader_*) to keep the file lean
for c in list(bpy.data.collections):
    if c.name.startswith(("ASSET_materials_", "ASSET_shader_")):
        for o in list(c.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(c)
bpy.ops.file.pack_all() if False else None
bpy.ops.wm.save_as_mainfile(filepath=LIB)
size = os.path.getsize(LIB)
# missing-file check
missing = [i.name for i in bpy.data.images if i.filepath and not i.packed_file and i.source == "FILE"]
meta = {"category": "materials_vfx", "asset_id": "materials_vfx_library",
        "description": "One library blend with every node group, material, world and effect/rig collection of materials_vfx (append by name).",
        "bytes": size, "missing_files": missing, "contents": summary,
        "totals": {"node_groups": len(bpy.data.node_groups), "materials": len(bpy.data.materials), "collections": len(bpy.data.collections), "worlds": len(bpy.data.worlds)},
        "append_example": "with bpy.data.libraries.load(LIB, link=False) as (s, d): d.node_groups = ['NG_clay']; d.materials = ['MAT_vfx_clay_skin_light']; d.collections = ['ASSET_fx_muzzle_flash']"}
C.write_meta(os.path.splitext(LIB)[0] + ".json", meta)
print("LIB", size, meta["totals"], "missing", missing)
