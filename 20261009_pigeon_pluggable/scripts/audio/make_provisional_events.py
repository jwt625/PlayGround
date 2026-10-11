"""Provisional SFX cue list (until the scene build writes scripts/audio/scene_events.json).

Run (project root): uv run --no-project python scripts/audio/make_provisional_events.py
Writes scripts/audio/sfx_events_provisional.json in the scene_events schema. Timing: DevLog-006 rev 3 storyboard beats,
snapped to the narration (P6: VO is the master clock): an anchor is either a film time (float) or "<line id>+dt"
/ "<line id>.end+dt" resolved from scripts/audio/narration.json. Positions are not known here, so pan is a guess.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, rel  # noqa: E402

NARR = {L["id"]: L for L in json.load(open(os.path.join(HERE, "narration.json")))["lines"]}


def T(a):
    if isinstance(a, (int, float)):
        return float(a)
    ref, _, dt = a.partition("+")
    if ref.endswith(".end"):
        return NARR[ref[:-4]]["t1"] + float(dt or 0)
    return NARR[ref]["t0"] + float(dt or 0)


# (anchor, kind, name, extra)
CUES = [
    # S1 street, tree, bird flap, landing pad (0-10)
    (0.0, "street_amb", "s1 night street bed", {"dur": 6.2, "params": {"fade": 0.8}}),
    (0.0, "hologram", "s1 tree hologram", {"dur": 4.8}),
    (0.15, "takeoff", "s1 hero takes off beside the trunk", {"pan": -0.2}),
    (1.0, "flight", "s1 hero rises along the tree", {"dur": 3.6, "pan": -0.1}),
    (0.6, "footstep", "s1 subject walks to the door", {"pan": 0.4, "gain": 0.7}),
    (1.15, "footstep", "s1 subject step", {"pan": 0.4, "gain": 0.7}),
    (1.7, "footstep", "s1 subject step", {"pan": 0.45, "gain": 0.7}),
    (2.25, "footstep", "s1 subject step", {"pan": 0.45, "gain": 0.6}),
    (2.9, "door", "s1 subject enters the datacenter door", {"pan": 0.5, "gain": 0.6}),
    ("s1c+0.2", "hud_tick", "s1 CYBER PIGEON node pulse", {"params": {"f": 1800}}),
    ("s1c+0.9", "flight", "s1 turn to the facade", {"dur": 1.3}),
    ("s1c+1.6", "whoosh", "s1 camera follows through the flap", {"dur": 0.8}),
    ("s1c+2.0", "flap_vent", "s1 bird flap swings", {}),
    ("s1c+2.0", "hall_amb", "hall bed S1-S2", {"dur": 16.0, "params": {"fade": 0.8}}),
    ("s1c+2.6", "wing_flap", "s1 glide down", {}),
    ("s1c+2.95", "wing_flap", "s1 glide down", {"gain": 0.8}),
    ("s1c+3.3", "landing_thump", "s1 lands on the pad", {}),
    ("s1c+3.35", "pad_charge_hum", "s1 pad charging", {"dur": 1.1}),
    ("s1c+3.6", "head_servo", "s1 neighbour glance", {"pan": -0.5}),
    ("s1c+3.9", "head_servo", "s1 neighbour glance", {"pan": 0.5}),
    ("s1d+0.0", "cell_dock", "s1 hero backs into its pigeonhole", {}),
    ("s1d+0.5", "head_servo", "s1 heads track the subject", {"pan": -0.6}),
    ("s1d+0.65", "head_servo", "s1 heads track", {"pan": 0.6}),
    ("s1d+0.8", "head_servo", "s1 heads track", {"pan": -0.2}),
    ("s1d+0.95", "head_servo", "s1 heads track", {"pan": 0.3}),
    ("s1d+1.1", "head_servo", "s1 heads track", {"pan": -0.7}),
    ("s1d+0.2", "footstep", "s1 subject in aisle", {"gain": 0.5}),
    ("s1d+0.75", "footstep", "s1 subject in aisle", {"gain": 0.5}),
    ("s1d+1.3", "footstep", "s1 subject in aisle", {"gain": 0.5}),
    # S2 tail leash, beak latch, eyes, feet (10-20)
    (10.15, "leash_pull", "s2 subject pulls the tail leash", {}),
    (10.55, "beak_open", "s2 beak latch releases", {}),
    (10.75, "cell_undock", "s2 hero slides out", {}),
    (11.85, "whoosh", "s2 turned upright", {"dur": 0.5, "gain": 0.6}),
    ("s2b+0.2", "head_servo", "s2 presents right eye", {"pan": 0.3}),
    ("s2b.end+0.05", "beak_latch", "s2 beak snaps", {}),
    ("s2c+0.0", "head_servo", "s2 presents left eye", {"pan": -0.3}),
    ("s2c+1.4", "whoosh", "s2 tilt nose-up to the feet", {"dur": 0.6, "gain": 0.6}),
    ("s2e+0.0", "cell_dock", "s2 plugged back in (cutaway)", {"gain": 0.8}),
    ("s2e+0.08", "contact_gnd", "s2 claws GND mate first", {}),
    ("s2e+0.45", "contact_pwr", "s2 palms power mate", {}),
    ("s2e+0.8", "contact_sig", "s2 toe pads signal mate", {}),
    ("s2e+1.05", "eye_glow_on", "s2 eyes light", {}),
    ("s2e.end+0.15", "beak_latch", "s2 beak bites the cage lip", {}),
    # S3 right eye transmit (20-30)
    ("s3a+0.0", "head_servo", "s3 hero turns its right eye to the aisle", {}),
    ("s3a.end+0.0", "fill_whoosh", "s3 push into the TX eye (red fill)", {"dur": 0.9, "params": {"color": "red"}}),
    ("s3b+0.0", "pic_hum", "s3 TX macro bed", {"dur": 6.2}),
    ("s3b+0.75", "laser_on", "s3 DFB laser 1", {"pan": -0.3}),
    ("s3b+1.0", "laser_on", "s3 DFB laser 2", {"pan": 0.3, "gain": 0.8}),
    ("s3c+0.0", "pam4_chatter", "s3 8 MZMs blink PAM4", {"dur": 2.3}),
    ("s3c+0.4", "pd_flash", "s3 monitor PD wink", {"pan": -0.4}),
    ("s3c+1.3", "pd_flash", "s3 monitor PD wink", {"pan": 0.4}),
    ("s3d+0.0", "beam_launch", "s3 8 lenslets launch 8 beams", {}),
    ("s3e+0.0", "fill_whoosh", "s3 zoom out of the TX eye", {"dur": 0.8, "params": {"color": "red", "peak": 0.3}, "gain": 0.8}),
    ("s3e.end+0.1", "link_blip", "s3 beams cross the aisle", {"pan": 0.5}),
    ("s3a+0.0", "hall_amb", "hall bed S3 head (ducked by the fill)", {"dur": 1.9, "gain": 0.8, "params": {"fade": 0.4}}),
    ("s3e+0.2", "hall_amb", "hall bed S3 tail / S4 head", {"dur": 3.6, "gain": 0.8, "params": {"fade": 0.4}}),
    # S4 left eye receive (30-40)
    (30.1, "head_servo", "s4 hero turns: left eye to the aisle", {"pan": -0.2}),
    ("s4a+0.7", "link_blip", "s4 partner beams into the RX eye", {"pan": -0.4}),
    ("s4a.end+0.0", "fill_whoosh", "s4 cyan fill into the RX eye", {"dur": 0.9, "params": {"color": "cyan"}}),
    ("s4b+0.0", "tia_hum", "s4 RX macro bed", {"dur": 4.6}),
    ("s4b+0.0", "pd_flash", "s4 PD 1", {"params": {"f": 2200}}),
    ("s4b+0.12", "pd_flash", "s4 PD 2", {"params": {"f": 2350}}),
    ("s4b+0.24", "pd_flash", "s4 PD 3", {"params": {"f": 2500}}),
    ("s4b+0.36", "pd_flash", "s4 PD 4", {"params": {"f": 2650}}),
    ("s4b+0.9", "pd_flash", "s4 PD 5", {"params": {"f": 2800}}),
    ("s4b+1.02", "pd_flash", "s4 PD 6", {"params": {"f": 2950}}),
    ("s4b+1.14", "pd_flash", "s4 PD 7", {"params": {"f": 3100}}),
    ("s4b+1.26", "pd_flash", "s4 PD 8", {"params": {"f": 3250}}),
    ("s4c+0.4", "i2c_stream", "s4 TIA output rushes to the DSP", {"dur": 1.6, "gain": 0.8}),
    ("s4c.end-0.3", "whoosh", "s4 POV pull out into the hall", {"dur": 0.7}),
    ("s4c.end-0.2", "hall_amb", "hall bed S4 POV / S5", {"dur": 11.7, "params": {"fade": 0.6}}),
    ("s4d+0.0", "hud_lock", "s4 subject boxed and tracked", {}),
    # S5 exploded view and hatchery (40-50)
    (40.15, "whip", "s5 pull out, whip back to the hero", {}),
    (40.9, "explode", "s5 parts separate", {}),
    ("s5a+1.15", "hud_tick", "s5 stamp NARROWCOM", {"params": {"f": 2000}}),
    ("s5a+1.75", "hud_tick", "s5 stamp MERVALL", {"params": {"f": 2200}}),
    ("s5a.end+0.1", "hud_tick", "s5 stamp MINLINEAR", {"params": {"f": 2400}}),
    ("s5b+0.3", "pd_flash", "s5 crystal tick", {"gain": 0.6, "params": {"f": 3400}}),
    ("s5b+0.8", "pd_flash", "s5 crystal tick", {"gain": 0.6, "params": {"f": 3400}}),
    ("s5b+1.3", "pd_flash", "s5 crystal tick", {"gain": 0.6, "params": {"f": 3400}}),
    ("s5b+1.9", "i2c_stream", "s5 I2C telemetry stream", {"dur": 1.8}),
    ("s5b.end+0.0", "snap_back", "s5 parts snap back", {}),
    ("s5c+0.1", "hud_tick", "s5 leg band HATCHED BY OUTNOLIGHT", {"params": {"f": 1600}}),
    ("s5c+0.0", "conveyor", "s5 hatchery line fades in", {"dur": 1.6}),
    ("s5c+0.3", "robot_servo", "s5 robot places the DSP", {"pan": -0.4}),
    ("s5c+0.6", "assembly_clunk", "s5 shell closed", {"pan": 0.2}),
    ("s5c+0.9", "robot_servo", "s5 robot places optics", {"pan": 0.4, "params": {"f0": 380, "f1": 600}}),
    ("s5c+1.15", "eye_glow_on", "s5 new bird eyes light", {"dur": 0.45, "gain": 0.7}),
    ("s5c.end+0.0", "wing_flap", "s5 bird hops off", {}),
    # S6 street (50-60)
    (50.2, "door", "s6 subject walks out the door", {}),
    (50.2, "street_amb", "s6 night street bed", {"dur": 9.8, "params": {"fade": 0.8}}),
    (50.6, "flight", "s6 hero + two birds follow", {"dur": 1.6, "pan": 0.2}),
    (50.7, "footstep", "s6 subject", {"pan": 0.1}),
    (51.25, "footstep", "s6 subject", {"pan": 0.1}),
    (51.8, "footstep", "s6 subject", {"pan": 0.1, "gain": 0.8}),
    (52.35, "footstep", "s6 subject", {"pan": 0.1, "gain": 0.7}),
    ("s6b+0.0", "head_servo", "s6 perched heads track", {"pan": -0.6}),
    ("s6b+0.2", "head_servo", "s6 perched heads track", {"pan": 0.5}),
    ("s6b+0.4", "link_blip", "s6 eye to eye", {"pan": -0.4}),
    ("s6b+0.9", "link_blip", "s6 bird to bird", {"pan": 0.4}),
    ("s6b+1.3", "head_servo", "s6 pair trades glances", {"pan": 0.2}),
    ("s6b.end+0.3", "link_blip", "s6 link", {"pan": 0.0}),
    ("s6c+0.15", "sat_ping", "s6 rooftop bird uplinks to the satellite", {}),
    ("s6d+0.0", "cctv_whir", "s6 CCTV camera pans", {"pan": 0.5}),
    ("s6d+0.6", "link_blip", "s6 CCTV links to a bird", {"pan": 0.4}),
    ("s6d+1.1", "cctv_whir", "s6 second CCTV", {"pan": -0.5, "gain": 0.8}),
    ("s6d+1.5", "link_blip", "s6 CCTV link", {"pan": -0.4}),
    (58.85, "head_servo", "s6 heads turn to camera", {"pan": -0.5}),
    (58.95, "head_servo", "s6 heads turn to camera", {"pan": 0.5}),
    (59.1, "head_servo", "s6 heads turn to camera", {"pan": 0.0}),
    (59.5, "blink", "s6 hero blinks its TX eye (TX disable)", {}),
    # S7 disclaimer
    (59.97, "disclaimer_whoosh", "s7 hard cut to the card", {}),
]


def main():
    ev = []
    for a, kind, name, extra in CUES:
        if isinstance(a, str) and "-" in a and "+" not in a:   # "<id>.end-0.3"
            ref, dt = a.split("-")
            t = T(ref) - float(dt)
        else:
            t = T(a)
        e = {"t": round(t, 3), "name": name, "kind": kind}
        e.update(extra)
        ev.append(e)
    ev.sort(key=lambda e: e["t"])
    out = os.path.join(HERE, "sfx_events_provisional.json")
    with open(out, "w") as f:
        f.write('{\n "_doc": "Provisional SFX cues by A (DevLog-014) in the scene_events.json schema; replaced by S\'s scene_events.json when present.",\n "events": [\n')
        f.write(",\n".join("  " + json.dumps(e) for e in ev) + "\n ]\n}\n")
    print("%d events -> %s" % (len(ev), rel(out)))


if __name__ == "__main__":
    main()
