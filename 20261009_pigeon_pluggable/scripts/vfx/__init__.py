"""VFX library for the pigeon-pluggable film (workstream E, DevLog-013). Blender 4.2 EEVEE Next.

Usage inside Blender (scripts/ on sys.path):
    import sys; sys.path.insert(0, "<repo>/scripts")
    import vfx
    holo = vfx.hologram_material("VFX_tree_holo")
    b = vfx.make_beam("link_0", p_tx_eye, p_rx_eye, radius=0.004, color=vfx.TX_RED); vfx.key_beam(b, 120, 180)
    vfx.setup_eevee(scene, "review"); vfx.setup_compositor(scene)
API reference: DevLog/DevLog-013-environment-vfx.md section 5.
"""
from .atmosphere import haze_cards, light_cone, night_sky_world, sky_dome, world_haze
from .materials import (BEAM_NOTE, HOLO_CYAN, RX_CYAN, TX_RED, beam_material, eye_glow_material, hologram_material,
                        key_beam, key_beam_endpoints, key_eye_power, key_input, key_obj_alpha, make_beam, neon_material,
                        puddle_material, set_beam, set_fast_reflections, wet_ground_material)
from .post import key_compositor, setup_compositor, setup_eevee
from . import beams, datafx, hud  # noqa: E402  (F4 HUD, F6 beam animation, F7 high-speed data look; DevLog-013 section 8)
from .beams import beam_hops, beam_sweep, beam_track, mimo_links, street_beam_choreo
from .datafx import apply_data_look, data_beam_material, data_emitter_material, key_blink, make_data_beam
from .hud import hud_overlay

__all__ = [
    "BEAM_NOTE", "HOLO_CYAN", "RX_CYAN", "TX_RED",
    "hologram_material", "beam_material", "make_beam", "set_beam", "key_beam", "key_beam_endpoints", "neon_material",
    "wet_ground_material", "puddle_material", "set_fast_reflections", "eye_glow_material", "key_eye_power", "key_obj_alpha", "key_input",
    "night_sky_world", "sky_dome", "world_haze", "haze_cards", "light_cone",
    "setup_eevee", "setup_compositor", "key_compositor",
    "hud_overlay", "beam_track", "beam_sweep", "beam_hops", "mimo_links", "street_beam_choreo", "data_emitter_material", "data_beam_material", "make_data_beam",
    "apply_data_look", "key_blink",
]
