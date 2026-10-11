# DevLog-000: Pigeon pluggable conspiracy short - kickoff, decisions, research index

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Status | Mapping narrowed, research cached, storyboard v1 in DevLog-001 for review; no Blender work started |
| Scope | Short 1 of a series: one pluggable (800G OSFP DR8) modeled as a cyberpunk surveillance pigeon; introduce its components with zoom-ins and exploded views, narrated |
| Sibling project | `../20261002_interconnect_film/` (style, pipeline, characters, hardware assets reused) |
| Authors | Claude, at Wentao Jiang's request |

Path convention: paths are relative to this folder. Public side project: no startup or day-job material.

## 1. Instructions (verbatim from Wentao, 2026-10-08/09)

Message 1:
```text
I actually wants a totally different storyline but similar topic and style: I want to dig into the pluggable transceiver supplychain ecossytem and components evolution/requirements but as a conspiracy meme by modeling the pluggable as a pigeon, and it is observing the surroundings. Propose some high level mapping and ask for my feedback.
```

Message 2:
```text
good, one eye for Tx and one eye for Rx. Two feets are the Tx electrical contacts and Rx contacts. Could make multiple pigeons do FSO between their eyes towards the end, e.g. a three nodes link with their Tx eyes pointing at each others Rx eyes forming a ring.
You should have more detailed component mapping for the internals of the pigeon, e.g. theres also slow control electronics, prism and thin film filter MUX/DEMUX for VCSEL based transceivers. The retina should be actual optoelectronic components like VCSELs or EMLs or SiPho + CW DFB deeper in its head or body.
I like the ancestor. Try coming up with a bio/genetic tree from actual birds, and swap actual pluggable tech evolution/tech tree into it. e.g. qsfp-dd could be hummingbord, and XPO should be some big birds. Could open with this and zoom in onto one.
Let's make the first short just focus on one type of pigeon / pluggable, and the film will be just introducing its components, with various zoom in shots / explosion view of the interior, with proper voice overs.
Q&A:
1. creepy surveillance drone pigeon, cyberpunk style.
2. yeah have a dummy victim human character that is being monitored and has the same model as gary.
3. same length and similar scene durations. Try keeping the camera more fluid instead of using the scope for transitioning
4. parody names that are close to actual companies (narrowcom (broadcom), mervall (marvell), optoelink (eoptolink), outnolight (innolight) etc.), make sure the narrator pronounces them explicitly correctly, e.g. out-no-light
```

Message 3:
```text
now narrow down the mappings and assumptions, and come up with more detailed scenes.
```

Message 4:
```text
1. C
2. trait-matched. Can simplify the tree if not enough pluggable variants
3. names the components when explaining.
4. new project folder.
Cache important research results and comclusions, as well as assets that might be useful later.
Once you are done, give me a high level summary of the scenes and zoom-ins to review.
```

## 2. Decisions

| # | Decision | Source |
|---|---|---|
| D1 | Hero: 800G OSFP DR8 (8x100G PAM4, MPO-16 APC, 500 m, 1311 nm), silicon photonics variant with CW DFB laser deep in the body | Wentao, option C |
| D2 | One eye TX, one eye RX; left foot = TX electrical contacts, right foot = RX electrical contacts | Wentao |
| D3 | Retina = the real optoelectronic parts (SiPh Mach-Zehnder modulators fed by CW DFB lasers deeper in the head; PIN photodiode array + TIA on RX) | Wentao |
| D4 | Bird tree trait-matched (not strict lineage), simplified to about a dozen leaves; opens the film, then zooms onto the hero | Wentao |
| D5 | Narrator names components while explaining (unlike the interconnect film's labels-only rule) | Wentao |
| D6 | Creepy cyberpunk surveillance-drone pigeon look | Wentao |
| D7 | Monitored victim "the Subject" on the Gary model | Wentao |
| D8 | 60 s + 2 s disclaimer, 10 s scenes, fluid continuous camera (no scope transitions) | Wentao |
| D9 | Parody company names close to the real ones; narrator `say` spellings verified by Whisper round trip | Wentao |
| D10 | Finale: three pigeons in a free-space-optics ring (each TX eye into the next RX eye) | Wentao |
| D11 | VCSEL + prism + thin-film-filter MUX/DEMUX anatomy reserved for a later short (VCSEL species); not in the DR8 hero | Claude (follows D1) |
| D12 | External-laser-source storylines are out of scope for this series | Claude |

## 3. Parody names and pronunciation

Roles from `research/form_factors_and_companies.md`. `say` = text sent to TTS.

| On screen | Real | Role used in film | say |
|---|---|---|---|
| NARROWCOM | Broadcom | DSP brain, lasers | "Narrow-com" |
| MERVALL | Marvell | DSP brain | "Mer-vall" |
| OUTNOLIGHT | Innolight | module maker (the hatchery) | "Out-no-light" |
| OPTOELINK | Eoptolink | module maker | "Op-toe-link" |
| INCOHERENT | Coherent | CW laser, VCSELs | "In-co-here-ent" |
| DIMENTUM | Lumentum | EML/CW lasers | "Dim-en-tum" |
| MINLINEAR | MaxLinear | DSP | "Min-linear" |
| CREDON'T | Credo | AEC (cuckoo), DSP | "Cred-don't" |
| FABRINOT | Fabrinet | contract manufacturer | "Fab-ri-not" |

## 4. Research cache (`research/`)

| File | Content |
|---|---|
| `research/hero_800g_osfp_dr8.md` | Hero internals, OSFP MSA Rev 5.2 mechanicals (60 pads, mating order, orientation), DSP/laser/PD/TIA parts, locked typical choices |
| `research/alt_hero_400g_qsfp112_fr4.md` | 400G QSFP112 FR4 internals (TFF Z-block, EML/DML), QSFP112 MSA pins and mating order; for a later short |
| `research/birds.md` | Phylogeny (Jarvis 2014 / Prum 2015 / Stiller 2024), pigeon vision (violet not UV, oil droplets), pecten, RFC 1149 and the 2001 Bergen test, safe and unsafe claims |
| `research/form_factors_and_companies.md` | Form-factor timeline 1x9 to XPO, XPO facts, company roles, export controls and EML shortage |
| `references/INDEX.md` | 11 files, 54 MB, SHA-256 + license per file: OSFP MSA Rev 5.22, QSFP112 MSA Rev 2.1.2, CMIS 5.4; FS and Approved Networks 800G DR8, Naddod 400G FR4, Coherent Z-block datasheets; Stiller 2024 Fig. 1 (CC BY 4.0), Prum 2015 (reference only), Commons pigeon side view and feet (CC BY-SA). Blocked: Marvell Spica Gen2 brief, Cisco PDF (403). No licensed DR8 teardown photo or oil-droplet micrograph found. Binaries git-ignored |

Own references outside this folder (not copied): Innolight 800G OSFP PSM8 and 400G OSFP DR4+ teardown photos in the blog post `$BLOG/_posts/2025-09-15-weekly-OFS-64.md` (`$BLOG` = `../../jwt625.github.io`, images under `assets/images/2025/20241215_20250915/20250913_03*.jpg`); 3DGS capture of an opened Innolight pluggable in `~/Documents/3DGS/innolight/`; teardown write-up https://outside5sigma.com/teardown/100G-200G-400G-800G/.

## 5. Reuse from the interconnect film (link, do not copy)

`../20261002_interconnect_film/assets/components/`: `characters/gary_v2.blend` (Subject), `characters/motion_v2/` (132 actions), `interconnect/osfp_module.blend`, `osfp_cage_and_host_connector.blend`, `mpo_connector_and_adapter.blend`, `fiber_connectors.blend`, `photonics/pic_die.blend`, `grating_coupler_closeup.blend`, `datacenter/datahall_environment.blend`, `tray_nvlink_switch.blend`. Pipeline: `scripts/audio/make_vo.py` + `narration.json` (Kokoro TTS), `scripts/audio/make_sfx.py`, `scripts/film_v1/bslot.sh` (max 3 Blender processes), `asm.py` caption/overlay helpers.

## 6. TODO

- [x] Folder, DevLog-000, research cache
- [x] Hero DR8 research cached (`research/hero_800g_osfp_dr8.md`)
- [x] Reference downloads indexed (`references/INDEX.md`)
- [x] DevLog-001 storyboard v1 written and audited (fresh context, 15 findings applied)
- [ ] Wentao reviews DevLog-001 storyboard v1
- [ ] Pigeon rig (cyber pigeon: base mesh, feather-fin shell, articulated head/neck/eyes/feet, exploded-view parts with hooks)
- [ ] Bird-tree hologram generator (data-driven from a YAML of leaves)
- [ ] Narration with pronunciation check (Whisper round trip)
- [ ] Crude v0 build (blocking, camera path, captions) for timing review

## 7. Progress log

- 2026-10-08: concept proposals in chat (pigeon mapping v1, v2).
- 2026-10-09: three research agents (hero FR4 candidate, birds, form factors and companies) returned; Wentao chose hero C (800G OSFP DR8), trait-matched tree, narrated component names, new folder. Folder created; research cached; DR8 research and reference download agents launched.
- 2026-10-09: DR8 research cached; TX = right eye/foot (MSA orientation). DevLog-001 mapping + storyboard v1 written; fresh-context audit (15 findings) applied.
