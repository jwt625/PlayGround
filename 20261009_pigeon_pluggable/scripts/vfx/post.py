"""Render look (workstream E, DevLog-013): EEVEE Next presets and the compositor (bloom + vignette + film grain, optional
chromatic aberration) for Blender 4.2. The Glare node's BLOOM type (new in 4.2) replaces EEVEE Legacy bloom.
"""
import bpy

PRESETS = {
    #          TAA samples, ray resolution scale, volume tile, volume samples, shadows ray/step
    "draft": dict(samples=12, ray_res="4", vol_tile="16", vol_samples=24, shadow_rays=1, shadow_steps=4),
    "review": dict(samples=32, ray_res="2", vol_tile="8", vol_samples=48, shadow_rays=1, shadow_steps=6),
    "final": dict(samples=64, ray_res="1", vol_tile="4", vol_samples=64, shadow_rays=2, shadow_steps=8),
}


def _set(obj, attr, val):
    try:
        setattr(obj, attr, val)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def setup_eevee(scene, preset="draft", raytracing=True, volumetrics=True, vol_start=0.1, vol_end=80.0, motion_blur=False,
                view_transform=None, look=None, exposure=None, gamma=None):
    """EEVEE Next settings for the night sets. raytracing=True gives screen-space reflections for wet ground / puddles.
    view_transform/look/exposure are only changed when given (S owns the film-wide colour management; the previews use
    view_transform='AgX', look='AgX - Medium High Contrast')."""
    p = PRESETS[preset]
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    e = scene.eevee
    e.taa_render_samples = p["samples"]
    _set(e, "use_shadows", True)
    _set(e, "shadow_ray_count", p["shadow_rays"])
    _set(e, "shadow_step_count", p["shadow_steps"])
    _set(e, "use_raytracing", raytracing)
    _set(e, "ray_tracing_method", "SCREEN")
    ro = getattr(e, "ray_tracing_options", None)
    if ro is not None:
        _set(ro, "resolution_scale", p["ray_res"])
        _set(ro, "use_denoise", True)
    _set(e, "use_gtao", True)
    _set(e, "gtao_distance", 0.6)
    _set(e, "volumetric_tile_size", p["vol_tile"])
    _set(e, "volumetric_samples", p["vol_samples"])
    _set(e, "volumetric_start", vol_start)
    _set(e, "volumetric_end", vol_end)
    _set(e, "use_volumetric_shadows", volumetrics and preset != "draft")
    scene.render.use_motion_blur = motion_blur
    if view_transform:
        scene.view_settings.view_transform = view_transform
    if look:
        _set(scene.view_settings, "look", look)
    if exposure is not None:
        scene.view_settings.exposure = exposure
    if gamma is not None:
        scene.view_settings.gamma = gamma
    return scene


def setup_compositor(scene, bloom=True, bloom_threshold=1.0, bloom_size=7, bloom_mix=0.0, streaks=0.0, vignette=0.28,
                     vignette_radius=0.92, grain=0.035, chroma=0.0):
    """Replace the compositor tree: Render Layers -> Glare BLOOM -> (optional Glare STREAKS mixed by `streaks`) ->
    vignette (blurred ellipse multiply, strength 0..1) -> film grain (per-pixel random NOISE texture, overlay, 0..0.1)
    -> optional lens dispersion (chroma, ~0.005-0.02) -> Composite + Viewer. Returns {name: node}."""
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    N = {}

    def node(kind, name, x, y=0):
        n = nt.nodes.new(kind)
        n.name = n.label = name
        n.location = (x, y)
        N[name] = n
        return n

    rl = node("CompositorNodeRLayers", "render_layers", -800)
    cur = rl.outputs["Image"]
    if bloom:
        g = node("CompositorNodeGlare", "bloom", -550)
        if not _set(g, "glare_type", "BLOOM"):
            g.glare_type = "FOG_GLOW"
        g.quality = "HIGH"
        g.threshold = bloom_threshold
        g.size = bloom_size
        g.mix = bloom_mix
        nt.links.new(cur, g.inputs["Image"])
        cur = g.outputs["Image"]
    if streaks > 0:
        s = node("CompositorNodeGlare", "streaks", -400, -200)
        s.glare_type = "STREAKS"
        s.quality = "MEDIUM"
        s.threshold = max(bloom_threshold * 2.0, 2.0)
        s.streaks = 4
        s.angle_offset = 0.785
        s.fade = 0.85
        s.mix = 1.0
        nt.links.new(cur, s.inputs["Image"])
        m = node("CompositorNodeMixRGB", "streak_add", -250)
        m.blend_type = "ADD"
        m.inputs["Fac"].default_value = streaks
        nt.links.new(cur, m.inputs[1])
        nt.links.new(s.outputs["Image"], m.inputs[2])
        cur = m.outputs["Image"]
    if vignette > 0:
        el = node("CompositorNodeEllipseMask", "vignette_mask", -400, -350)
        el.width, el.height = vignette_radius, vignette_radius
        bl = node("CompositorNodeBlur", "vignette_blur", -250, -350)
        bl.filter_type = "FAST_GAUSS"
        bl.use_relative = True
        bl.factor_x = bl.factor_y = 30.0
        bl.size_x = bl.size_y = 200
        nt.links.new(el.outputs["Mask"], bl.inputs["Image"])
        mv = node("CompositorNodeMixRGB", "vignette", -100)
        mv.blend_type = "MULTIPLY"
        mv.inputs["Fac"].default_value = vignette
        nt.links.new(cur, mv.inputs[1])
        nt.links.new(bl.outputs["Image"], mv.inputs[2])
        cur = mv.outputs["Image"]
    if grain > 0:
        tex = bpy.data.textures.get("VFX_grain") or bpy.data.textures.new("VFX_grain", "NOISE")
        tx = node("CompositorNodeTexture", "grain_tex", -100, -350)
        tx.texture = tex
        mg = node("CompositorNodeMixRGB", "grain", 50)
        mg.blend_type = "OVERLAY"
        mg.inputs["Fac"].default_value = grain
        nt.links.new(cur, mg.inputs[1])
        nt.links.new(tx.outputs["Color"], mg.inputs[2])
        cur = mg.outputs["Image"]
    if chroma > 0:
        ld = node("CompositorNodeLensdist", "chroma", 200)
        ld.use_fit = True
        ld.inputs["Dispersion"].default_value = chroma
        nt.links.new(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    cp = node("CompositorNodeComposite", "composite", 400)
    nt.links.new(cur, cp.inputs["Image"])
    vw = node("CompositorNodeViewer", "viewer", 400, -200)
    nt.links.new(cur, vw.inputs["Image"])
    return N


def key_compositor(scene, node_name, input_name, frame, value):
    """Key a compositor value per shot (e.g. vignette Fac, grain Fac)."""
    s = scene.node_tree.nodes[node_name].inputs[input_name]
    s.default_value = value
    s.keyframe_insert("default_value", frame=frame)
