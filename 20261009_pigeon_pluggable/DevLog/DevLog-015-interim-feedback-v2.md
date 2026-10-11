# DevLog-015: Interim feedback during v2 - orchestration

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | Dispatched; all items fold into film v2 |
| Coordinator | Claude (max 5 concurrent subagents; Blender cap 3 via `scripts/bslot.sh`; fast render tier, full film <= 10 min) |

## 1. Owner feedback (verbatim, 2026-10-10)

```text
interrim feedback:
- bird head turning should be modeled better, e.g., neck should deform accordingly instead of just head turn and leaving gaps between head and neck
- circulators should be labeled
- the VO needs to be improved, have a subagent read and sample my writing styles from my blog and use similar style. Do not repeat same info like left eyes send right eyes receive got repeated twice.
- for 0:38 scene of pigeon PoV, add fancy HUD display of various battlefield intel style info like from games like battlefield or MGS.
- exploded view should call out more parody companies
- outdoor scene at ~0:55 should have beams from their eyes sweeping around the scene as well as beams jumping among each other. Satellite should also be a starlink array, and the beam toggle to follow one a bit and jump to the next one coming in the FoV and repeat few times.
- shots near the send eyes should have all emitters fast blinking randomly, same for the zoom in shot of inside the sender and receiver, the beams should be fast moving and blinking to show high speed data.
- opening scene should show actual picture of different kinds of birds with a simple model, and fade in the actual scene instead of on top of or inside the scene, contrast is horrible.

Have subagents work on the improvements/fixes in parallel.
```

## 2. Routing

| # | Item | Owner | Notes |
|---|---|---|---|
| F1 | Neck deforms with head turns (no head/neck gaps) for hero and crowd proxies | C | Armature/neck chain skinned to the neck meshes (hero neck meshes from H; built at rig time, H files untouched) |
| F2 | Label the micro isolators in the TX path (the owner calls them circulators; the modeled part is a Faraday isolator stack: polarizer / garnet / polarizer / half-wave plate between the two Si lenses) | S | Label as "MICRO ISOLATOR (FARADAY)" in the S3 macro and exploded labels |
| F3 | VO rewrite in the owner's writing style (sampled from his blog), no repeated information | W (new) then A | W writes narration rev 3; A re-voices, Whisper check, re-times; S re-aligns |
| F4 | S4 POV HUD: battlefield / MGS-style intel overlay | E builds `scripts/vfx/hud.py`; S integrates | Parody/generic only; no game logos or fonts copied |
| F5 | Exploded view: more parody company call-outs | S | Parody names mapped to research-verified roles (Section 3) |
| F6 | S6 ~0:55: eye beams sweeping the scene and hopping between birds; satellite becomes a LEO constellation train (Starlink-like, generic, no logos); beam tracks one satellite, hands off to the next entering the view, several times | E (constellation asset + beam VFX), S (staging) | Beam note stays on screen |
| F7 | High-speed data look: TX-eye shots have all emitters blinking fast and randomly; inside TX/RX macro shots beams are fast-moving pulse trains, blinking | E (animated beam/emitter materials), S (apply) | |
| F8 | Opening: a clean, separate intro with real bird photos (licensed) and simple models of the matching form factors, then fade into the street scene (not overlaid in the scene) | Tree agent (new), S (fade) | Photos only with recorded licences (Wikimedia Commons CC0/CC BY/CC BY-SA/PD); credits on the S7 card |

## 3. Parody names for exploded-view call-outs (labels only; roles from `research/form_factors_and_companies.md` and `research/hero_800g_osfp_dr8.md`)

| Part | Parody | Real role basis |
|---|---|---|
| DSP | NARROWCOM / MERVALL / MINLINEAR / CREDON'T | Broadcom, Marvell, MaxLinear, Credo PAM4 DSPs |
| CW DFB lasers | DIMENTUM / INCOHERENT | Lumentum, Coherent CW lasers |
| TIA / drivers | SEMTREK / MACOMBO | Semtech FiberEdge TIAs, MACOM drivers/TIAs |
| SiPh PIC foundry | TOWERING SEMI / GLOBALFOUNDLINGS | Tower Semiconductor, GlobalFoundries Fotonix |
| InP substrate (lasers, PDs) | SUMITOMOO / AXED | Sumitomo Electric, AXT InP substrates |
| Module maker (the hatchery) | OUTNOLIGHT / OPTOELINK | Innolight, Eoptolink |
| Contract assembly | FABRINOT | Fabrinet |
| PIC marking (existing) | ROCKDOVE PHOTONICS | parody |

## 4. Status

| Owner | Status |
|---|---|
| S | items F2, F5, F6/F7 staging, F8 fade, integration |
| C | F1 |
| E | F4, F6 asset/VFX, F7 VFX |
| W | F3 script delivered (DevLog-016, narration_rev3_draft.json); coordinator picks: 'Target acquired. Hi, Gary.' and 'Smile. The cameras are in on it.' |
| Tree | F8 intro |
| A | voicing rev 3 |

## 5. Progress log

- 2026-10-10: interim feedback recorded and dispatched.
- 2026-10-10: W delivered narration rev 3 in the owner's blog style (style guide + script; right/left stated once). Coordinator picked two alternatives; A voicing.
- 2026-10-10: E delivered F4 (intel HUD), F6 (generic LEO train + beam_track/sweep/hops), F7 (PAM4 data emitters and scrolling data beams). Coordinator checked previews; forwarded to S.
- 2026-10-10: C delivered F1 neck deformation (5-bone neck armature, seam gap ~2.5 mm vs 9-50 mm before; checked the old/new sheet). Forwarded to S with the 0.5x neck-extension guidance.
- 2026-10-10: Tree agent delivered F8 intro (14 Commons photos with licences in references/birds/INDEX.md, 15 simple form-factor models, separate high-contrast card wall, 4.5 s move ending on the hero card, xfade 4.5-5.3 s). Coordinator checked stills; S told to render it as its own pass and cross-fade, remove the street hologram tree, and add credits (S7 compact line + outputs/CREDITS_<date>.md).
- 2026-10-10: A voiced rev 3 (narration.json rev 3, vo_20261010_rev3.wav, -16 LUFS; Whisper: same-sound spellings only; TTS spelling fixes for Incoming, T-I-A, Mach Zender). Word anchors forwarded to S. A waits for film_v2 to mix/mux.
- 2026-10-10 14:45: S was cut off by an API session limit mid-edit (after a 12:21 v2 render that already includes intro, rev 3 captions, HUD, Gary, isolator label, parody call-outs, street beams). Coordinator reviewed the 12:21 sheets and resumed S (finish fixes; S5 labels inside the frame, S6 satellites larger, hatchery clearer; cleanup of 810 MB leftover frames).
