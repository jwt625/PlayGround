"""Provisional film v4 SFX events (63.0 s) until S writes the final scripts/audio/scene_events.json.

Run (project root): uv run --no-project python scripts/audio/make_provisional_events_v4.py [--slot-start 47.9]
  [--v3 outputs/audio/scene_events_film_v3_164119.json]
Writes outputs/audio/scene_events_v4_provisional.json (scene_events schema).

Built from (DevLog-014 section 14):
1. the v3 events snapshot with event_overrides_v3.json applied (door 44.23 and two satellite pings dropped);
2. the v3 hatchery assembly events (names starting "assembly_") removed;
3. every v3 event at or after the end of the v3 slot (SLOT_V3_END = 49.70: flashback_out, S6, S7) shifted +1.0 s
   (W4: assembly slot +1.0 s, later scenes shift +1.0 s);
4. the v4 assembly events from assets/components/assembly/assembly_hatchery.json ("events", build frames), mapped as
   film t = slot_start + build_frame / SPEED / 30 (slot 92 film frames = 144 build frames at SPEED 1.5652);
5. the S6 -> S7 scooter zoom transition (W7), peaking on the 61.0 s cut.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, OUT_AUDIO, PROJ, rel  # noqa: E402

SPEED, FPS, SLOT_V3_END, SHIFT = 1.5652, 30.0, 49.70, 1.0
ASM = os.path.join(PROJ, "assets", "components", "assembly", "assembly_hatchery.json")

# (event key in assembly_hatchery.json or a build frame, kind, name, extra); bonds handled separately
ASM_CUES = [
    ("S2_iso_place", "place_click", "asm_isolator_placed", {"params": {"f": 4200}}),
    ("S2_lens2_place", "place_click", "asm_lens2_placed", {"params": {"f": 4700}}),
    ("S2_tool_leaves_lens", "align_stage", "asm_tool_leaves_lens", {"dur": 0.25, "gain": 0.8}),
    ("S2_dfb_on", "dfb_on", "asm_dfb_on", {"gain": 0.6}),
    ("S2_uv_on", "uv_cure", "asm_uv_cure", {"dur": 0.3}),
    ("S4_press_down", "pneumatic", "asm_press_down", {"dur": 0.35}),
    ("S4_lid_closed", "lid_press", "asm_lid_closed", {"gain": 1.2}),
    ("S4_platen_up", "pneumatic", "asm_press_up", {"dur": 0.4}),
    ("S5_heads_in", "coupler_in", "asm_couplers_in", {"dur": 0.35}),
    ("S5_eyes_on", "eye_glow_on", "asm_eyes_on", {"dur": 0.45}),
    ("S5_eye_diagram_on", "hud_tick", "asm_eye_diagram_on", {"params": {"f": 1900}}),
    ("S5_pass", "pass_chime", "asm_pass", {}),
    ("S5_eye_diagram_off", "hud_tick", "asm_eye_diagram_off", {"params": {"f": 1500}}),
    ("S5_heads_out", "coupler_out", "asm_couplers_out", {"dur": 0.35}),
    ("S5_hop1", "hop", "asm_hop", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-start", type=float, default=47.9)
    ap.add_argument("--v3", default=os.path.join(OUT_AUDIO, "scene_events_film_v3_164119.json"))
    ap.add_argument("--overrides", default=os.path.join(HERE, "event_overrides_v3.json"))
    ap.add_argument("--out", default=os.path.join(OUT_AUDIO, "scene_events_v4_provisional.json"))
    args = ap.parse_args()
    ev = json.load(open(args.v3))["events"]
    ov = json.load(open(args.overrides))
    for d in ov.get("drop", []):
        ev = [e for e in ev if not (e.get("name") == d["name"] and abs(e["t"] - d["t"]) <= d.get("tol", 0.02))]
    n_asm = sum(1 for e in ev if e.get("name", "").startswith("assembly_"))
    ev = [e for e in ev if not e.get("name", "").startswith("assembly_")]
    n_shift = 0
    for e in ev:
        if e["t"] >= SLOT_V3_END:
            e["t"] = round(e["t"] + SHIFT, 3)
            n_shift += 1
    A = json.load(open(ASM))["events"]
    f2t = lambda f: round(args.slot_start + f / SPEED / FPS, 3)  # noqa: E731
    new = []
    for key, kind, name, extra in ASM_CUES:
        new.append(dict({"t": f2t(A[key]), "name": name, "kind": kind}, **extra))
    bonds = list(range(A["S3_bond_first"], 144, A["S3_bond_period"]))[::3]      # every 3rd bond: texture, not a machine gun
    for i, f in enumerate(bonds):
        new.append({"t": f2t(f), "name": "asm_wire_bond_%d" % i, "kind": "wire_bonder", "gain": 0.6})
    new.append({"t": 59.95, "name": "scooter_zoom_to_card", "kind": "scooter_zoom", "dur": 1.1, "params": {"peak": 0.95}})
    ev = sorted(ev + new, key=lambda e: e["t"])
    with open(args.out, "w") as f:
        f.write('{\n "_doc": "PROVISIONAL film v4 events (63.0 s) by A: v3 snapshot + overrides, v3 assembly removed, +1.0 s after %.2f, v4 assembly from %s at slot %.3f s; replaced by S\'s scene_events.json when it lands.",\n "events": [\n'
                % (SLOT_V3_END, rel(ASM), args.slot_start))
        f.write(",\n".join("  " + json.dumps(e) for e in ev) + "\n ]\n}\n")
    print("v3 assembly events removed %d, shifted +%.1f s: %d, new assembly %d (bonds %d), total %d -> %s"
          % (n_asm, SHIFT, n_shift, len(new) - 1, len(bonds), len(ev), rel(args.out)))
    for e in new:
        print("  %6.3f %-12s %s" % (e["t"], e["kind"], e["name"]))


if __name__ == "__main__":
    main()
