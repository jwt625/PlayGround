"""Build render_presets: demo blend (representative test scene configured with the presets) + benchmark of render time per frame at 1080x1350.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_render_presets.py -- assets/components/materials_vfx
Scenes: clay character, hardware board, daylight rig + cartoon sky, smoke puff + muzzle flash + sparks effects, glow emission, comp group.
"""
import json
import os
import platform
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bpy
import common as C
import vfx_lib as V
import render_presets as RP

OUT = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx")
SCR = os.environ.get("VFX_SCRATCH", "/tmp")
PREV = os.path.join(OUT, "previews")
BLEND = os.path.join(OUT, "render_presets.blend")
LIGHT = os.path.join(OUT, "lighting_and_world.blend")
CLAY = os.path.join(OUT, "materials_clay.blend")
PBR = os.path.join(OUT, "materials_pbr_hardware.blend")
FX = os.path.join(OUT, "vfx_particles_and_effects.blend")


def build_scene():
    C.reset()
    scn = bpy.context.scene
    clay = ["MAT_vfx_clay_shirt_blue", "MAT_vfx_clay_skin_light", "MAT_vfx_clay_hardhat_orange", "MAT_vfx_clay_eye_white", "MAT_vfx_clay_trousers_navy",
            "MAT_vfx_clay_gray_mid", "MAT_vfx_clay_suit_gray", "MAT_vfx_clay_skin_pink", "MAT_vfx_clay_tie_red", "MAT_vfx_clay_wood_stock"]
    V.append_library(CLAY, materials=clay)
    V.append_library(PBR, materials=["MAT_vfx_soldermask_green", "MAT_vfx_nickel_plated", "MAT_vfx_anodized_black", "MAT_vfx_gold", "MAT_vfx_led_green",
                                     "MAT_vfx_silicon", "MAT_vfx_cable_yellow", "MAT_vfx_glass"])
    d = V.append_library(LIGHT, collections=["ASSET_light_daylight"], worlds=["WORLD_cartoon_sky"], node_groups=["NG_comp_post"])
    scn.collection.children.link(bpy.data.collections["ASSET_light_daylight"])
    scn.world = bpy.data.worlds["WORLD_cartoon_sky"]
    V.append_library(FX, collections=["ASSET_fx_smoke_puff", "ASSET_fx_muzzle_flash", "ASSET_fx_sparks_burst", "ASSET_fx_impact_stars"])
    for n in ("ASSET_fx_smoke_puff", "ASSET_fx_muzzle_flash", "ASSET_fx_sparks_burst", "ASSET_fx_impact_stars"):
        scn.collection.children.link(bpy.data.collections[n])
    M = lambda n: bpy.data.materials[n]
    r = bpy.data.objects.new("BENCH_subject", None)
    scn.collection.objects.link(r)

    def prim(kind, loc, size, mat, rot=(0, 0, 0)):
        if kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=48, ring_count=24, location=loc)
        elif kind == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1, depth=1, location=loc)
        o = bpy.context.active_object
        o.scale = size
        o.rotation_euler = rot
        if kind != "cube":
            V.smooth(o)
        o.data.materials.append(mat)
        o.parent = r
        return o
    # two clay characters (Gary + Manager look-alikes), shotgun, hardware board, floor
    prim("sphere", (0, 0, 0.55), (0.3, 0.24, 0.42), M("MAT_vfx_clay_shirt_blue"))
    prim("sphere", (0, 0, 1.08), (0.2, 0.2, 0.2), M("MAT_vfx_clay_skin_light"))
    prim("sphere", (0, 0, 1.2), (0.22, 0.22, 0.12), M("MAT_vfx_clay_hardhat_orange"))
    for sx in (-1, 1):
        prim("sphere", (sx * 0.075, -0.17, 1.1), (0.05, 0.05, 0.05), M("MAT_vfx_clay_eye_white"))
        prim("cyl", (sx * 0.1, 0, 0.17), (0.075, 0.075, 0.36), M("MAT_vfx_clay_trousers_navy"))
        prim("cyl", (sx * 0.36, 0, 0.62), (0.055, 0.055, 0.42), M("MAT_vfx_clay_skin_light"))
    prim("sphere", (1.6, 0.3, 0.7), (0.36, 0.27, 0.52), M("MAT_vfx_clay_suit_gray"))
    prim("sphere", (1.6, 0.3, 1.33), (0.23, 0.23, 0.23), M("MAT_vfx_clay_skin_pink"))
    prim("cyl", (1.25, -0.4, 1.0), (0.04, 0.04, 0.9), M("MAT_vfx_clay_gray_mid"), rot=(1.5708, 0, 0.3))
    prim("cube", (-1.1, -0.3, 0.03), (0.5, 0.35, 0.03), M("MAT_vfx_soldermask_green"))
    prim("cube", (-1.12, -0.3, 0.08), (0.2, 0.2, 0.07), M("MAT_vfx_nickel_plated"))
    prim("cube", (-0.95, -0.2, 0.1), (0.1, 0.12, 0.1), M("MAT_vfx_anodized_black"))
    prim("sphere", (-1.25, -0.45, 0.085), (0.05, 0.05, 0.05), M("MAT_vfx_gold"))
    prim("sphere", (-0.9, -0.5, 0.2), (0.12, 0.12, 0.12), M("MAT_vfx_glass"))
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.data.materials.append(M("MAT_vfx_clay_gray_mid"))
    # effects placed: smoke at Manager's ears, muzzle flash near the gun, sparks and impact stars near Gary
    for n, loc, t in (("ROOT_fx_smoke_puff", (1.6, 0.3, 1.5), 1.0), ("ROOT_fx_muzzle_flash", (1.25, -0.9, 1.0), 0.04),
                      ("ROOT_fx_sparks_burst", (-1.1, -0.3, 0.2), 0.25), ("ROOT_fx_impact_stars", (0.0, -0.3, 1.1), 0.15)):
        o = bpy.data.objects[n]
        o.location = loc
        o["p_t"] = t
    cam, tg = RP.camera_rig(scn, "BENCH_CAM", lens=28.0, loc=(0.4, -5.2, 1.6), target_loc=(0.3, 0.0, 0.9))
    return scn


def timed_render(scn, label, n=3):
    times = []
    path = os.path.join(SCR, "bench_%s.png" % label)
    scn.render.filepath = path
    for i in range(n):
        scn.frame_set(1 + i)
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(time.perf_counter() - t0)
    return times


def main():
    scn = build_scene()
    results = {}
    cfgs = [("draft_nocomp", "draft", None), ("standard_nocomp", "standard", None), ("standard_comp", "standard", "cartoon"),
            ("hero_nocomp", "hero", None), ("hero_comp", "hero", "cartoon")]
    for label, preset, comp in cfgs:
        RP.apply_render_preset(scn, preset, 1080, 1350)
        scn.render.resolution_percentage = 100          # measure the full 1080x1350 for every preset
        RP.set_output_png(scn, os.path.join(SCR, "bench_"))
        if comp:
            RP.setup_comp(scn, comp)
        else:
            scn.use_nodes = False
        ts = timed_render(scn, label)
        results[label] = {"preset": preset, "comp": comp, "samples": RP.PRESETS[preset]["samples"], "raytracing": RP.PRESETS[preset]["raytracing"],
                          "first_frame_s": round(ts[0], 2), "next_frames_s": [round(x, 2) for x in ts[1:]],
                          "mean_warm_s": round(sum(ts[1:]) / len(ts[1:]), 2)}
        print("BENCH", label, results[label])
        if label == "standard_comp":
            shutil.copy(os.path.join(SCR, "bench_%s.png" % label), os.path.join(SCR, "bench_standard_comp_full.png"))
    # save demo blend configured with the 'standard' preset + comp (draft preset available through RP)
    RP.apply_render_preset(scn, "standard", 1080, 1350)
    RP.setup_comp(scn, "cartoon")
    scn.frame_start, scn.frame_end = 1, 300
    RP.set_output_png(scn, "//render_presets_demo_")
    scn.frame_set(1)
    C.save(BLEND)
    meta = {
        "category": "materials_vfx", "asset_id": "render_presets",
        "description": "Python module scripts/assets/materials_vfx/render_presets.py (copy at assets/components/materials_vfx/render_presets.py) plus a demo .blend with a representative scene.",
        "module_api": ["apply_render_preset(scn, 'draft'|'standard'|'hero')", "set_output_png(scn, prefix)", "set_output_h264(scn, path)", "camera_rig(scn, name, lens, loc, target_loc)",
                       "add_shake(obj, amp_loc, amp_rot, freq, frame_range)", "whip_pan(target, f0, f1, to_loc)", "hard_cut(scn, [(frame, camera)])",
                       "speed_ramp(obj, data_path, keys)", "attach_dof(cam, focus_obj, fstop)", "setup_comp(scn, preset)"],
        "presets": RP.PRESETS,
        "benchmark": {
            "machine": "Apple M4 Pro, 24 GB, Blender 4.2.3 EEVEE Next (Metal), headless (-b), %s" % platform.platform(),
            "scene": "BENCH: 2 clay characters (spheres/cylinders with NG_clay), PBR hardware board (nickel lid, mask, gold, glass), daylight rig (sun + 2 area lights) + cartoon sky world, "
                     "smoke puff + muzzle flash + sparks + impact stars effects, floor; 1080x1350, PNG output to scratch",
            "results_seconds_per_frame": results,
            "note": "first frame includes shader compilation; mean_warm_s is the per-frame cost for an animation (frames 2-3). Wall time of bpy.ops.render.render incl. compositing and PNG write.",
        },
        "usage": ["Call apply_render_preset on each scene; add the compositor via setup_comp (append NG_comp_post from lighting_and_world.blend)."],
    }
    C.write_meta(os.path.splitext(BLEND)[0] + ".json", meta)
    shutil.copy(os.path.join(HERE, "render_presets.py"), os.path.join(OUT, "render_presets.py"))
    # preview: the standard+comp frame downscaled to 900 wide
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(SCR, "bench_standard_comp_full.png"), "-vf", "scale=540:-1",
                    os.path.join(PREV, "render_presets_bench_frame.png")], check=False)
    print("DONE", json.dumps(results, indent=1))


main()
