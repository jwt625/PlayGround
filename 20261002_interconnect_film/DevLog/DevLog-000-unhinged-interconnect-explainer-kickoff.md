# DevLog-000: Unhinged optical-interconnect explainer film (Blender) - kickoff

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Status | Research in progress; story/scene proposal not yet written; no Blender work started |
| Scope | Comedic, exaggerated Zach D Films style explainer on optical interconnects; Blender animation; reusable component library |
| Authors | Claude, at Wentao Jiang's request |

Path convention: all paths in this file are relative to this project folder (`20261002_interconnect_film/`). `$BLOG` = `../../jwt625.github.io` (sibling repo of PlayGround). Items under the home directory (agent history, ChatGPT export) cannot be expressed relatively and are written as `~/...`.

Privacy note: this is a public-side-project folder. Startup-workspace and day-job material is deliberately not referenced here (see the MRM section for what that means in practice).

## 1. Instructions (verbatim from Wentao, 2026-10-02)

```text
Let's discuss a zach d film style explanation video but more unhinged (similar to his mt fuji bayblade). I am currently thinking about the topic of optical interconnects and how is the electrical interface pushed closer and closer to the logic/ASIC, and the problem each step solve should have dramatic human scale consequence (e.g. near-closed eye diagram appearing on scope got seen by manager and the manager blast the operator/technician with a shotgun). It should be obviously exaggerated and comical.
We have plenty of resources in the playground folder, in my blog (jwt625.github.io, on electrical packaging, various ieee 802.3 posts, writeup on SiPho and CPO etc.), and in separate repos (link architecture demo with eye diagram traces), local codex convo log/cache (there is definitely one on MRM crosstalk, maybe it has corresponding playground project or separate project, if not then it is in webUI chatGPT export for sure).
You should be using blender for the actual amination. Reconstruct various components as needed (transceivers, advanced packaging with PIC and EIC and interposer etc. based on public and my local images), and organize them properly for future similar video work use. Let's work in playground for now and spin them off later.
Read and research, and propose more detailed stories and scenes first, as well as ask any clarification questions, as well as list of assets to download or DIY.
```

Follow-up (same day): "create directory as well as md devlog inside to include my instructions and initial research results and references. Use relative paths."

Interpretation of mode: discuss + research + propose first. No animation or asset-building until the proposal is reviewed.

## 2. Deliverables of the discussion phase

- [x] Project folder + this DevLog
- [ ] Style bible for the target format (pending web research return; Section 5)
- [ ] Story spine: the "electrical interface moves toward the ASIC" chain, one problem + one human-scale catastrophe per step
- [ ] Detailed scene list (per-scene: technical claim, number + source, gag, visual assets needed)
- [ ] Clarification questions for Wentao (few, numbered)
- [ ] Asset list: download vs DIY-in-Blender, with licenses
- [ ] Reusable Blender library layout proposal (Section 7)
- [ ] Fresh-context audit of the technical claims in the script (record corrections here)

## 3. Technical spine: how he frames the chain (from the blog)

Source: `$BLOG/_posts/2026-08-20-silicon-photonics-and-co-packaged-optics.md` (drafted Dec 2025, revised 2026-08-22).

- "The pluggable optical module sits at the front panel, while high-speed copper traces connect it to the switch ASIC. Those traces can be tens of centimeters long ..." and "good luck getting all the up-to-date S parameters from a dozen different vendors."
- "Co-packaged optics moves the optical engines right next to the switch ASIC ... The long electrical path becomes a very short package-level connection."
- Figure caption (Cheng et al.): "The electrical reach shrinks as optical conversion moves from the faceplate toward the ASIC."
- Recap: electrical signaling gets harder and more energy-intensive with distance and data rate; fiber has very low propagation loss; optics pays most of its energy at the two conversion endpoints; "now moving closer and closer to the chip."
- Benefits he lists: less drive/equalization power, more lasers/wavelengths/fibers around the ASIC, higher I/O density, simpler retiming/DSP, more assembly in semiconductor flows. Costs: yield, repair, laser reliability.
- Joke recap already in the post: "Copper bad bad, but copper is cheap."

Candidate step ladder (draft, to be refined in the proposal; ordering is by distance between electrical conversion point and ASIC):

1. Long copper backplane / DAC (passive)
2. Active copper (AEC/ACC) and retimed DSP pluggable at the faceplate
3. LPO / LRO (remove or lighten the DSP)
4. NPO (near-package optics, module on substrate/HDI interposer next to the package)
5. CPO (optical engine in the package, on organic substrate)
6. Optical I/O chiplet / 2.5D-3D integration (EIC on PIC, hybrid bond, TSV; COUPE, Photonic Fabric, NVIDIA ISSCC 2026 DWDM link)

## 4. Facts and numbers harvested from the blog (with provenance)

Provenance key: Own = stated by Wentao without citation; Vendor slide = photo of a third-party slide (read the slide before quoting); AI-gen = figure carries a Gemini watermark, treat as unsourced; Sim = his own GPT-generated simulation, not trustworthy as fact. All numbers below are as printed in the blog; none are independently verified yet (verification pass is a TODO, Section 8).

| Beat | Number | Source (post file under `$BLOG/_posts/`, citation) |
|---|---|---|
| Compute outruns I/O | FLOPS 3x/2yr; Ethernet switch capacity 2x/2yr; interconnect speed 1.4x/2yr; Ethernet lane speed 1.4x/2yr; AI parameters 400x/2yr (as printed on figure). Separately compute 60,000x vs I/O 30x over 20 yr (IBM) | `2026-08-20-silicon-photonics-and-co-packaged-optics.md`, Cheng et al., Opt. Express 2025, doi 10.1364/OE.555476; `2024-12-29-weekly-OFS-27.md`, Knickerbocker 2024, arXiv 2412.06570 |
| Copper dies fast | RF cable near 25 GHz loses about half its power per ~1 m (~3 dB/m); tens of cm of PCB + package + connector between pluggable and ASIC | CPO post; Own, no citation, he flags strong dependence on cable/connectors |
| Package electrical path (400G/lane) | PHY, CoWoS bridges, microbumps, Ajinomoto substrate, BGA, vias, connectors; CPC channel vs NPC/NPO channel, TP0/TP5 test points; no dB values on slide | CPO post; Sakai, IEEE 802.3 400GPL, May 2026 (sakai_400GPL_01a_2605) |
| 400G connector | CPC connector >100 GHz bandwidth enabling PAM4 at 448G; note reads "30 dB loss at 100 GHz" (check against slide); OSFP fixture can marginally support PAM6 at 175 GBd | `2026-06-22-weekly-OFS-104.md` (TEF 2025 panel slide); "10 GHz/8 months" connector progress rate is flagged by him as unverified |
| NVLink lane ladder | 50G / 100G / 200G per lane; 4x50G = 600 GB/s, 2x100G = 990 GB/s, 2x200G = 1.8 TB/s | `OFC2026-notes`/`2026-04-04-OFC2026.md`, IMG_3963 (NVIDIA slide; OCR garbled, re-read) |
| Rates | 53.125 Gb/s/lane = 26.5625 GBd PAM4; 400G DR4: 8x50G electrical to 4x100G optical; 100G PAM4 ~ 50 GBd | `2026-06-13-100G-200G-400G-800G.md` (Broadcom DR4 datasheet); CPO post (Own for 50 GBd) |
| Eye diagrams (real) | 128 Gb/s NRZ at Vpp 0.8: SNR 5.2, ER 3.8 dB. 192 Gb/s PAM4 at Vpp 1.6: TDECQ 2.5 dB, ER 3.5 dB | CPO post, figure `nrz-pam4-eye-diagrams.webp`, Blum (Intel) |
| Reach / power ladder | Passive Cu 2 m 0 pJ/b; active Cu 5 m 3-4; RF-microwave 10-20 m 3-4; SR8 VCSEL 20-50 m 1-2; DR8 InP/TFLN 2 km 3-4; DR8 SiPh 2 km 4-6; FR4/LR4 2 km 4-6; CL 10-20 km 15; ZR 20-1000 km 20-30 | `2026-04-04-OFC2026.md` IMG_4060, Bechtolsheim (Arista) slide |
| Scale-up power | Copper twinax ~4 pJ/bit; LPO/LRO ~10; scale-up optics target ~5; ~15 in figure (column ambiguous) | `2026-04-04-OFC2026.md` IMG_4077, Meta slide (re-read) |
| Retiming power | 1.6T retimed optics 25 W; 1.6T linear optics 10 W; active copper 2.5 W (4 m); passive copper 0 W (1 m); "Retimer DSP power consumption costs $Bs" | `2026-04-04-OFC2026.md` IMG_4150, Ciena slide |
| LPO then CPO | Broadcom pJ/bit vs year 2024-2029 for DSP gens, LRO, LPO, CPO, VCSEL NPO, AEC, ACC, DAC ("Broadcom estimates"); points not in OCR | `2026-04-04-OFC2026.md` IMG_4082 |
| XPO compromise | OSFP 32 per 1U; XPO: 75% fewer switch racks, 44-50% less floor space, up to 6.4 Pb/s per rack; first results pre-FEC BER 1e-8 (LPO), 1e-10 (retimed); his take "compromise ... sunk cost" | `2026-04-04-OFC2026.md` IMG_4061-4068, Arista slides; opinion Own |
| Intel OCI | 8 fibers x 8 wavelengths = 64 lanes x 32 Gb/s = 4 Tb/s bidirectional, 5 pJ/bit; "~10x smaller, ~40x the bandwidth" of a 100G pluggable | CPO post + `2024-12-15-weekly-OFS-25.md`, Optica OPN 2024 (Wills); the 10x/40x ratio is his arithmetic |
| Ranovus/MediaTek | 8x8x100G = 6.4 Tb/s per engine, 4 pJ/bit incl. laser; 8 engines = 51.2T | CPO post, OFC 2024 PDF |
| Energy ballparks | Optical link today ~10-20 pJ/bit; CPO target 1-5; off-chip electrical 5-20; on-chip 100-600 fJ/bit; modulator device 10-50 fJ/bit; with driver 0.5-3 pJ/bit; ring heater 5-20 mW static | CPO post; AI-gen figures, Miller 2017 (doi 10.1364/JOSAB.34.000A01) only for order-of-magnitude. Unsourced |
| Energy identity | P = E_bit x R; 1 Tb/s at 5 pJ/bit = 5 W | CPO post, Own derivation (safe) |
| Latency | Pluggable ~100-300 ns fixed + ~6 ns/m; CPO ~10-30 ns; DAC ~10-30 ns + ~4 ns/m | CPO post; AI-gen, "rough curves", unsourced |
| NVIDIA DWDM link | 32 Gb/s/lambda, 8x32 = 256 Gb/s/fiber; TX+RX lane 0.013 mm^2; 2.59 pJ/b; 7 nm EIC 3D-stacked on 65 nm PIC | CPO post, Song et al., ISSCC 2026, doi 10.1109/ISSCC49663.2026.11409081; OCR table scrambled, verify |
| Beachfront / 3D | Marvell Photonic Fabric 16 Tb/s (Gen1), 64 Tb/s (Gen2), 3.6x shoreline vs CPO chiplets; Marvell COUPE: N3P EIC on 65 nm PIC, hybrid bonding, TSV through PIC, C4 to organic substrate, 16 lambda/fiber = 1 Tbps/fiber; Lumentum VCSEL >1.5 Tbps/mm, <2.5 pJ/bit | `2026-04-04-OFC2026.md` IMG_4055/4056/4091; vendor slides |
| GPU interconnect target | 30 m reach, <5 pJ/bit excl. PHY, 1 Tbps/mm; copper-to-optics ~5:1 | OFS-104; slide author not named |
| Why not interposer yet | HBM4 ~2 TB/s; CPO optical engines barely under 1 TB/s; current CPO engines on organic substrate | CPO post; Own |
| Fiber is easy, ends are hard | SMF-28 up to 0.18 dB/km at 1550 nm; connector 0.4 dB "so good until it adds up" | CPO post (Corning PI-1424); `2026-07-15-weekly-OFS-107.md` (Stone 2026, 400GPL) |
| Coupling | Corning GlassBridge <1.5 dB/facet TE, <0.3 dB PDL; good edge coupler 0.4-0.6 dB; detachable glass coupler ~1 dB at 95% yield; optical backplane IL target <2.5 dB | `2026-04-04-OFC2026.md` IMG_4118; OFS-114; OFS-117 (ECTC 2026); OFS-96 (OCI-MSA, no link) |
| MRM pain | "Most of [Ayar Labs'] power ... is the heaters tuning the microrings" (hearsay at OFC); thermal tuning figure assumes ~55 C baseline | `2025-06-08-weekly-OFS-50.md`; `2026-08-31-weekly-OFS-114.md` (Yoon 2026, arXiv 2608.24637) |
| MRM crosstalk (Sim) | 70 GHz FWHM rings, 100 GBd NRZ; spacing 150/200/400 GHz; -6 and -10 dBm; with/without FFE; conclusion 150 GHz hurts, interleave. He wrote that he did not read the code and BER is "probably still not trustworthy". Do not use BERs as facts | OFS-114 and figure `$BLOG/assets/images/2026/20240917_20260829/20260822_014812_0.jpg` |
| Reliability | 200 FIT/node: MTBF 50 h at 100k nodes, 5 h at 1M nodes; Broadcom TH5-Bailly: >4M CW lasers shipped since 2016, <0.1 FIT, 85C/85%RH 1000 h, -40..85 C 500 cycles | `2026-04-04-OFC2026.md` IMG_4149 (Google), IMG_4085 (Broadcom) |
| Cost | Copper ~$0.05/Gbps, optics ~$0.5/Gbps; CPO "$0.10-0.25/Gb/s" | OFS-114 (Lu 2026, OCP SROI); `2026-08-23-weekly-OFS-113.md` (Lee 2025, IEDM) |
| CPO yield chain | known good die -> known good OE -> known good CPO -> known good CPO switch | CPO post; Own |

Blog-wide gaps (from the mining pass): no per-inch / Nyquist insertion-loss budget derivation; no retimer/DSP-internals post; no 802.3dj standalone post (the 802.3 material is cited slides inside the CPO post and OFS-104/107/117, and the cache `../20260521_ieee802.3dj/`); no microbump/C4 pitch or electrical beachfront Tbps/mm numbers; no interposer cross-section photos of his own. The raw tweet JSON under `$BLOG/_posts/scraping/` was not mined.

## 5. Zach D Films style reference

Research date 2026-10-02. Method limit: YouTube pages could not be fetched and no video was watched; style traits come from secondary text (portfolio of a contractor, wiki/TV Tropes search snippets, one low-reliability blog). Tags: [S] stated by a secondary source, [D] design recommendation (not verified as the channel's practice). Local history search found no prior conversation about this channel or Blender explainer work.

Correction: the channel is "Zack D. Films" (@zackdfilms), not "Zach D". The Mt Fuji video is a vertical YouTube Short, "Spinning Mt Fuji Like A Beyblade" (https://www.youtube.com/shorts/JE3LSo1Rp54; oEmbed confirms title/uploader, 113x200 portrait). TikTok version: https://www.tiktok.com/@zackdfilms92/video/7642701134880181517 (premise per snippet: spin Mt Fuji with giant steel cables and thousands of underground anchors, with a feasibility breakdown). Confirmed format is 30-60 s Shorts; a 5-8 min video is an extension of the style (no verified long-form examples found).

Sources: https://supersets.chrisreillyportfolio.com/zackdfilms.php (production workflow), https://zackdfilms.fandom.com/wiki/Zack_D._Films, https://tvtropes.org/pmwiki/pmwiki.php/WebAnimation/ZackDFilms, https://celebtechbuzz.com/zack-d-films/ (low reliability), animator breakdowns https://www.youtube.com/watch?v=UKeft_JNgEw and https://www.youtube.com/watch?v=GtIkLLgDFpQ (titles only).

### 5.1 Style bible (draft)

| # | Trait | Tag / source |
|---|---|---|
| 1 | 30-60 s, vertical, hook in first 1-2 s | [S] portfolio, blog |
| 2 | Odd opening question -> visual reveal -> fast explanation -> final twist | [S] secondary, unchecked against videos |
| 3 | Absurd "what if" scale premise treated with engineering seriousness (Fuji: cables + anchors, feasibility breakdown) | [S] TikTok title/snippet, wiki snippet |
| 4 | Stylized clay-like 3D, exaggerated realism, cinematic camera, slightly unsettling | [S] blog descriptors |
| 5 | Narration fast, crisp, confident young male voice; likely processed human, not AI (unconfirmed) | [S] snippets |
| 6 | Dark-humor "nightmare fuel" edutainment mixed with real facts | [S] wiki snippet |
| 7 | SFX (screams, grunts) carry gags; animated captions synced to narration | [S] TV Tropes snippet |
| 8 | Music | not verified |
| 9 | Stack: Blender, Cascadeur (physics-assisted character animation), EmberGen (particles/fire/smoke), After Effects, BlenderKit assets | [S] portfolio |
| 10 | Template-driven: base 3D file, comp/edit templates, modular prefabs, custom Python for file organization, Flamenco render management | [S] portfolio |
| 11 | Many uploads/day, batch-researched topics | [S] blog only |
| 12 | One consistent character rig and look across gags | [D] |
| 13 | Per-gag beat: calm technical statement -> escalation -> hard cut to physical human-scale consequence | [D] |
| 14 | Chapter cards per integration level to carry a 5-8 min runtime | [D] |
| 15 | SFX sting on every punchline; hard audio cut before reveal; captions ~2-4 words/line | [D] |

Comparable channels: not verified. Next step for real style fidelity: with Wentao's OK, fetch the Fuji Short locally (yt-dlp, only that segment, rate-limited) into `references/` and analyze frames; or Wentao gives 2-3 reference videos with timestamps.

### 5.2 Blender production notes

- Current stable: Blender 5.2 LTS (released 2026-07-14; 5.2.2 on 2026-09-15; supported to July 2028): https://www.blender.org/releases/5-2/. 5.x requires Apple Silicon on macOS 13+: https://www.blender.org/download/requirements/. 5.2 adds online asset libraries, per-asset import-method preference, 2x faster instancing-heavy EEVEE, Cycles texture cache (.tx); Geometry Nodes Python API changes require re-saving asset files: https://developer.blender.org/docs/release_notes/5.2/. Installed locally: 4.2.3.
- Cycles Metal GPU render + denoise on Apple Silicon (macOS 13+); spills to system memory when GPU memory is full: https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html. MetalRT is automatic on M3+, off by default on M1/M2: https://developer.blender.org/docs/release_notes/4.0/cycles/. Quantitative M-series Cycles-vs-EEVEE timings: not verified; run a 10-frame test (device medians at https://opendata.blender.org/).
- Recommended pipeline [D]: stylized clay/toon look in EEVEE for speed; Cycles + denoise only for glossy close-ups (cage, scope screen). Headless `blender -b file.blend -P script.py -- args` (flags: https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html). `lib/` one .blend per category as asset libraries (https://docs.blender.org/manual/en/latest/files/asset_libraries/index.html); `shots/shot_NN.blend` link collections (https://docs.blender.org/manual/en/latest/files/linked_libraries/index.html; overrides for posed characters); generate shots from a spec file; render image sequences, not video; Flamenco optional.
- Eye diagram on the scope [D]: NumPy waveform with lossy channel + noise -> fold 2 UI -> 2-D histogram -> heat colormap; sweep loss/noise per frame so the eye closes; write PNG/EXR sequence; emissive image-sequence texture on the scope screen with bloom and phosphor/scanline shader. Geometry Nodes 3D eye fly-through only if wanted.
- Assembly: PNG sequence -> ffmpeg (stitch, loudness); delivery H.264 yuv420p + AAC + `+faststart`, fixed fps. Local ffmpeg lacks drawtext (Section 6.2), so captions come from Blender/Pillow/canvas.
- TTS vs own voice: not researched; the channel voice is likely a processed human recording, so the closest match is Wentao's own voice with compression/EQ and a fast read.

### 5.3 Public asset sources and licensing

- Poly Haven: all CC0 (https://polyhaven.com/license); electronics models not confirmed. BlenderKit: Royalty Free or CC0, RF forbids reselling the assets themselves, login in Blender (official license page not read). Sketchfab: CC BY needs credit + change notice, CC0 none; only models with a Download button; each model's license must be checked (https://sketchfab.com/developers/download-api/guidelines). GrabCAD community: private/non-commercial by default, needs poster permission for monetized video. TraceParts: free registration, commercial terms unclear.
- NVIDIA marks: written permission required for logo, video use needs approval (https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/). Recommendation: generic models, fictional brands, no real logos. Not legal advice.

| Category | Candidates | Action |
|---|---|---|
| OSFP / QSFP-DD / SFP | OSFP MSA spec PDFs only, no STEP (Rev 5.22 2025-08-14; OSFP-XD Rev 1.11 2025-11-23): https://www.osfpmsa.org/specification.html; QSFP-DD hardware spec Rev 5.1: http://www.qsfp-dd.com/wp-content/uploads/2020/08/QSFP-DD-Hardware-rev5.1.pdf; TE QSFP-DD cage/connector STEP via TraceParts (registration); Wikimedia Commons SFP photos (license per file, e.g. QSFP-40G-SR4 is CC BY-SA 3.0) | Manual download of spec PDFs for real dimensions; DIY the models |
| DAC / AEC | Amphenol OSFP STP via site/TraceParts (login) | DIY twinax (simple, better for stylized look) |
| NVIDIA / Broadcom CPO | NVIDIA press, Broadcom Bailly deck https://docs.broadcom.com/doc/th5-51.2t-bailly-cpo; no 3D models found | Reference only; DIY |
| TSMC COUPE, Ayar Labs, Lightmatter | Tom's Hardware coverage; https://ayarlabs.com/media-archive/ ; https://lightmatter.co/products/m1000/ | Reference only (no reuse terms read) |
| 2.5D packaging | Wikimedia Commons AMD Fiji interposer photo (license per file page); https://anysilicon.com/cowos-package/ | Draw own layered cross-section |
| Racks / servers | Sketchfab CC BY data-center server rack (27.9k tris): https://sketchfab.com/3d-models/data-center-server-rack-01b49d9d1f694ba5b41e4b2b0e10d16e ; DGX H100 model (license not read) | Credit author in description; NVL72 model not found |
| Oscilloscopes | Sketchfab Tektronix models (licenses not read); no Keysight sampling scope found | DIY generic sampling scope; eye texture from our model |
| MPO / fiber arrays | US Conec catalog (dimensions): https://www.usconec.com/media/2bsp1emu/us-conec-product-catalog.pdf ; no STEP link | Procedural |

### 5.4 Technical timeline (verified-by-reading unless noted; vendor claims flagged)

"Derived" = arithmetic by the research agent. Compare the same loss definition when putting numbers on screen: 802.3ck 28 dB is at 26.56 GHz; 802.3dj 40 dB is die-to-die at 53.125 GHz.

| Claim | Number(s) | Source | Confidence |
|---|---|---|---|
| 802.3bj 100G (4x25G NRZ) backplane and copper reach | IL <= 35 dB at 12.9 GHz; twinax >= 5 m | https://www.ieee802.org/3/minutes/nov12/1112_bj_close_report.pdf | High |
| 802.3cd 50G/lane KR | <= 30 dB at 13.28 GHz | search snippets, e.g. https://www.ieee802.org/3/cd/public/Sept16/li_3cd_01b_0916.pdf | Medium (not read) |
| 802.3ck 100G/lane objectives | Backplane IL <= 28 dB at 26.56 GHz; twinax >= 2 m | https://www.ieee802.org/3/100GEL/P802_3ck_Objectives_2018mar.pdf | High |
| 802.3ck C2M channel | End-to-end IL target 16 dB (frequency not on slide text, presumably 26.56 GHz) | https://www.ieee802.org/3/ck/public/19_03/sun_3ck_04b_0319.pdf | Medium |
| 802.3dj 200G/lane backplane | Die-to-die IL <= 40 dB at 53.125 GHz | https://grouper.ieee.org/groups/802/3/dj/projdoc/KeyMotions_3dj_240314.pdf | High |
| 802.3dj 200G/lane passive copper | Twinax >= 1 m objective (ck: 2 m) | Keysight/Samtec/Cisco abstract (PDF fetch blocked) https://www.keysight.com/us/en/assets/3122-1507/application-notes/Validation-of-Achieving-200-Gbs-Signaling-per-Electrical-Lane-Over-1-meter-of-Passive-Twinaxial-Copper-Cable.pdf | Medium |
| 802.3dj completion | Target July 2026; secondary page reports draft D2.4; final approval not confirmed as of 2026-10 | https://www.ieee802.org/3/ | Low |
| PCB loss at 112G Nyquist | 200 mm trace: Megtron 6 18.5 dB vs Megtron 8 14.2 dB (vendor blog, simulated) | https://www.nextpcb.com/blog/megtron-6-vs-megtron-8-pcb | Low |
| PCB loss near 56 GHz | roadmap ~1.1 dB/in; best measured ~1.4 dB/in room temp | https://www.signalintegrityjournal.com/articles/3159-next-generation-pcb-loss-analysis (snippet) | Low-medium |
| Pluggable vs CPO electrical channel at 200G/lane | up to 22 dB vs ~4 dB; NVIDIA says "64x" (blog) / 63x (press) signal integrity | https://developer.nvidia.com/blog/scaling-ai-factories-with-co-packaged-optics-for-better-power-efficiency | Medium (vendor) |
| Pluggable vs CPO power per interface (NVIDIA) | ~30 W vs as low as 9 W; 3.5x power efficiency; Quantum-X 144x800G; SN6810 128x800G; SN6800 512x800G | same NVIDIA blog | Medium (vendor) |
| Broadcom 2023 slides | 14 W/800G pluggable vs 5.5 W/800G CPO (derived ~17.5 vs ~6.9 pJ/bit); Bailly 51.2T 5.5 W/800G optical; Humboldt 25.6T measured 6.4 W/800G | https://docs.broadcom.com/doc/th5-51.2t-bailly-cpo | High that slides say it; vendor claim |
| Bailly 70% optical-interconnect power cut | ~5.5 W per 800G | https://opticalconnectionsnews.com/2024/03/ofc-2024-new-51-2t-cpo-switch-delivers-70-power-reduction/ ; https://www.broadcom.com/company/news/product-releases/61946 (snippets) | Medium |
| Davisson 102.4T (Tomahawk 6) | >70% optics power reduction vs pluggables; 3.5x lower interconnect power; 16 optical engines, 200G/channel; announced 2025-10-08. Ignore the "below 0.1 nJ/bit" figure (unreliable) | https://convergedigest.com/broadcom-unveils-102-4-tbps-davisson-cpo-switch-for-ai-clusters/ | Medium |
| LPO module power | 800G LPO DR8 8.5 W max, ~50% below DSP-based 800G; 800G 2xFR4 LPO 7 W | FS press release (financialcontent.com mirror 2025-08-23); LPO MSA https://www.businesswire.com/news/home/20240321794629/en/Twelve-Industry-Leaders-Collaborate-to-Define-Specifications-for-Linear-Pluggable-Optics | Low-medium |
| Why copper inside NVL72 | 5,000 NVLink copper cables, 2 miles total; Jensen Huang: transceivers + retimers alone would cost 20 kW | https://newsletter.semianalysis.com/p/nvidias-optical-boogeyman-nvl72-infiniband | Medium (CEO statement) |
| SerDes power growth in a 51.2T box | Cisco 2021 slide: ASIC SerDes power 25x vs 2010, core 8x, optics 26x (garbled extraction) | https://grouper.ieee.org/groups/802/3/B400G/public/21_02/chopra_b400g_01_210208.pdf | Medium-low |
| Tomahawk 5 lanes | 512 SerDes at 100G PAM4 | https://www.nextplatform.com/2022/08/16/like-a-drumbeat-broadcom-doubles-ethernet-bandwidth-with-tomahawk-5/ (snippet) | Medium |

Gag-accuracy notes: the story is not "optics replace copper everywhere"; short scale-up links stay copper (NVL72) because the host-to-module channel budget at 200G/lane is the pain, not the fiber. The documented passive-DAC reach shrink is 5 m (bj) -> 2 m (ck) -> >= 1 m (dj).

Could not verify: any hands-on detail of Zack D. Films videos (music, caption styling, cadence, chapters, long form, AI voice); comparable channels; Cycles-vs-EEVEE timings on M-series with 5.2; BlenderKit official license text; reuse terms of vendor press images; individual Sketchfab model licenses; whether OSFP/MPO/Molex STEP files are downloadable; final approval of 802.3dj; SerDes share of total 51.2T ASIC power (likely ISSCC 2025 Tomahawk 5 paper, inaccessible); NPO pJ/bit figures; real-deployment DAC reach at 200G/lane; TTS options.

## 6. Existing local resources

### 6.1 Blog images usable as 3D reconstruction references

All under `$BLOG/assets/images/`. "Own" = his photos/figures. "Third-party" = credit required, check licensing before any redistribution; used as reference for DIY modeling only.

| Path | Shows | Origin |
|---|---|---|
| `2025/20251219_CPO/pluggable-npo-cpo-comparison.webp` | Cross-section schematics: pluggable, NPO (HDI interposer), CPO (interposer substrate). Best single reference for the "move the interface closer" chain | Third-party: Cheng et al., Opt. Express 33, 24190 (2025), Fig. 2 |
| `2025/20251219_CPO/ieee-400g-package-electrical-path.jpg` | Package cross-section: CPC channel, bump pads, build-up/core, BGA, PCB, NPC/NPO channel | Third-party: Sakai, IEEE 802.3 400GPL, May 2026 |
| `2025/20251219_CPO/intel-100g-transceiver-annotated.webp` | Annotated Intel 100G CWDM4 module, ~18 mm long | Third-party: Blum (Intel) labels |
| `2025/20251219_CPO/intel-100g-{laser-source,modulator,receiver,optical-mux}-teardown.webp`, `laser-integration-teardown.webp`, `cisco-finisar-400g-dr4.webp` | Microscope photos of laser/monitor-PD coupons, modulator+driver, TIA+PD, echelle mux | Own |
| `2025/20251219_CPO/photon-electron-hardware.webp` | Intel 2017 CWDM transceiver | Credited photographer (@sokol_cc) |
| `2025/20250117_20250127_long_thread/20250122_043517_*.jpg`, `20250127_000458_*.jpg`, `20250117_085006_*.jpg` | Innolight 200G FR4 EML TOSA; Intel 100G internals | Own |
| `2025/20250425_20250425_long_thread/20250425_044131_*.jpg`, `20250425_183530_*.jpg` | Cisco Finisar 400G DR4 interior | Own |
| `2025/20250913_20250913_long_thread/20250913_030604_*.jpg`, `_030659_*`, `_031929_*` | Innolight 400G OSFP DR4+ and 800G OSFP PSM8 (uncaptioned) | Own |
| `2025/20251219_CPO/intel-oci-chip-pencil.webp`, `intel-oci-eic-pic-fiber.webp` | Intel OCI chiplet vs pencil eraser; EIC + PIC + fiber array | Third-party: Intel via Optica OPN 2024 |
| `2025/20251219_CPO/ranovus-optical-engines-on-asic.webp`, `ranovus-external-internal-laser.webp`, `nvidia-cpo-photonic-switch.webp` | Ranovus engines around MediaTek ASIC; NVIDIA CPO switch | Third-party (Ranovus/MediaTek OFC 2024; NVIDIA Hot Chips 2025) |
| `2025/20251219_CPO/nrz-pam4-eye-diagrams.webp` | Real 128G NRZ and 192G PAM4 eyes | Third-party: Blum (Intel) |
| `2025/20251219_CPO/sota-DWDM-nvidia.png` | NVIDIA ISSCC 2026 link comparison | Third-party: Song et al. |
| `2025/20251219_CPO/{optical-link-energy-breakdown,interconnect-energy-vs-distance,latency-vs-distance}.webp` | Chalkboard-style summaries | AI-gen; style reference only, not data |
| `2025/20251219_CPO/ieee-total-serdes-shipments.jpg`, `ieee-fiber-channel-loss-budget.jpg`, `ieee-200g-vcsel-60m-om4.jpg` | IEEE 802.3 slides | Third-party |
| `2026/OFC2026/IMG_3963.JPG` | NVLink copper vs optics 50/100/200G | Third-party slide photo |
| `2026/OFC2026/IMG_4055.JPG`, `IMG_4056.JPG` | Marvell COUPE cross-section; Photonic Fabric | Third-party slide photos |
| `2026/OFC2026/IMG_4060.JPG`, `IMG_4067.JPG` | Arista reach/pJ/failure table; XPO BER | Third-party slide photos |
| `2026/OFC2026/IMG_4077.JPG`, `IMG_4082.JPG`, `IMG_4084.JPG` | Meta scale-up pJ/bit; Broadcom power progression; Broadcom CPO assembly flow | Third-party slide photos |
| `2026/OFC2026/IMG_4147.JPG`, `IMG_4150.JPG`, `IMG_4173.JPG` | NVIDIA ISSCC link; Ciena 1.6T power; MRM vs EAM vs microLED vs VCSEL | Third-party slide photos |
| `2026/20240917_20260829/20260822_014812_0.jpg`, `20260822_054713_0.jpg` | 4x4 MRM crosstalk eye panel (150/200/400 GHz) and a second panel | Own (GPT-5.6 simulation) |
| `2026/20250223_20260617/20260607_155409_*.jpg`, `20260605_013324_0.jpg` | IBM 2011 CPO; copper SI from Mike Peng Li's book | Third-party |
| `2026/20250223_20260617/20260613_174615_0.jpg`, `20260613_175341_0.jpg` | LPO vs CPC power; CPC connector S-parameters | Third-party (TEF panel slides) |
| `2024/20240702_20241207/20241205_072610_0.jpg`, `20241205_084830_0.jpg` | Intel OCI at OFC24 | Third-party |
| `2025/SiPho_basics/image38.png`, `image52.png` | Intel CPO; TSMC CPO | Third-party |

Caveat: contents of the teardown and OFC2026 images were taken from post captions/OCR, not opened. The 100G-200G-400G-800G post mentions Gaussian-splatted transceiver scans but no files were found; his own splats of an Innolight pluggable and an Intel 100G CWDM module exist in `~/Documents/3DGS/` (Section 6.2) and are a stronger reference than photos for DIY transceiver modeling.

### 6.2 Other local material (survey 2026-10-02; agent-reported, spot-check before relying)

Material from private workspaces (startup, day-job, diary repos) was surveyed but is deliberately not listed in this public-side file.

**Eye-diagram / link model (most relevant code asset): `../20260523_serdes/`**
- Plain HTML/JS, no dependencies. Live: https://jwt625.github.io/optical-dsp-link/. Design notes: `../20260523_serdes/DevLog-000-Optical-DSP-Link-Architecture.md`, `DevLog-001-*.md`, `DevLog-002-Interactive-Eye-Diagram-Panel.md`.
- `../20260523_serdes/eye_diagram_lab.html` is the usable eye model (settings ~lines 276-300; waveform builder `generateFrame()` ~lines 376-455; `foldEye()` folds to a 2-UI heatmap with persistence). Chain: NRZ/PAM4 symbols from PRBS/random -> optional 3-tap Tx FIR -> smoothstep edge shaping (width from driver bandwidth + loss) + Gaussian and sinusoidal boundary jitter -> one-pole driver LPF -> one-pole channel LPF set by `loss` -> reflection echo (`reflection`, `reflectionDelay` UI) -> AWGN -> optional 3-tap Rx FFE + CDR early/late sampling. 128 samples/UI. No DFE.
- Knobs: loss 0..1, reflection -0.45..0.45, noise 0..0.5, jitter 0..0.18 UI, txPre/txPost, driverBw, rxPre/rxPost, persistence 0.75..0.995. Presets: clean, lossy (backplane look), preemphasis, reflection, jitter, pam4. Story mapping idea: retimed = clean regenerated edges (low noise/jitter); LPO = higher loss, no DSP.
- Limits: behavioral/synthetic by its own README ("visual teaching over standards accuracy"); channel is one-pole, not real S-parameters. `optical_dsp_link_architecture.html` ISI is a crude 2-tap model; its eye is redrawn, not from a stored waveform.
- No export exists. Cheapest route to Blender: port `generateFrame`/`foldEye` to NumPy, write per-scenario density PNG/EXR plus 1-D waveform arrays; 2-D accumulator (192 x 150 bins) can be an animated image texture or displacement map. Alternative: headless Playwright capture of the canvas.
- Serializer pages give a ready mux-tree gag (112G example: 16 bits at 7 GHz to 1 stream, UI 8.93 ps).

**Literature / standards caches (text only; PDFs archived on the NAS, which was not mounted at survey time)**
- `../20260320_OFC/output/full_metadata/ofc_full_metadata.csv` (1003 rows); extracted text in `../20260320_OFC/extracted_text/` (677 papers; `paper_text_index.json`). Best-fit papers for the ladder (text local): W1D.7 Co-Design of Electronic and Photonic Systems for Future LPO, NPO, and CPO; M4B.2 256 Gb/s DWDM optical I/O in 3D-stacked EIC/PIC; M2B.1 200G LPO design challenges; Th1C.3 400G/lane linear-drive optics; W2A.43 LPO for 102.4T switch; M4B.6 AI interconnect scale-up; M4B.5 16-ch SiPh engine PAM6; Th3C.2 / Th1D.3 glass waveguide substrate / fan-out for CPO; Th2A.14 90 GHz Si MRM (224G PAM4); Th3C.4 CPO towards photonics chiplets; M1B.1 8-lambda DWDM NRZ CPO link with QD-SOA; Th4A.6 TFLN wafer-level CPO engine. Metadata only (not extracted): M4B.1 (optical I/O chiplets), Tu3I.5 (Beyond CPO), Tu3I.6-8 (packaging/interconnects), W1B.1.
- `../20260614_ISSCC_2026/cache/isscc_2026_relevant_papers.json` (36 picks, metadata only): 32 Gb/s/lambda 256 Gb/s/fiber DWDM 3D-stacked 7 nm EIC / 65 nm PIC; 3.19 pJ/b electro-optical router for photonic interposers; 280 mW 112G PAM4 transceiver 5 nm; UCIe die-to-die; Forum 2 "Electrical and Optical Links Towards 400G+".
- `../20260521_ieee802.3dj/` (README.md, `IEEE-802p3dj-guide.md`, `ieee802_3dj_browser/`, `ieee802_3_ai_map/ieee8023-ai-relationships.svg` and `.png`): standards talking points, not visuals.
- `../20260531_OCP_ISSCC/`: nothing relevant.

**Visual / gag source material**
- `../20260612_fiber_bundle/actual_packed_cable_bundle_cross_sections.png` and `.svg`, `generate_cable_bundle_svg.py`: 9072 fibers per rack, 8F up to 3456F cables (~135 mm equivalent diameter trunk). Fiber-spaghetti gag.
- `../20251117_GDSJam/gdsjam/public/previews/PIC_example_20251213.png`, `PIC_component_showcase_20251214.png`: PIC layout previews (duplicated across gdsjam-* subfolders).
- `../20260915_fruit_fly_nn/assets/generated/hardware-v2/*.glb` + `manifest.json`: 9 procedural optical-bench prefabs (fc-apc-plug, fc-bulkhead, sma-plug, optical-amplifier, splitter-19, tiptilt-collimator, phase-cassette, phase-driver, fly-console), generated by `scripts/generate_hardware_assets.py` (trimesh); Z-up. Same project: `flybody-articulated.glb` (23 MB articulated fly; possible cast member).
- `../20260327_multipole_cow/outputs/cow_reconstruction_l0.obj` ... `l24.obj` (cow mesh; possible mascot).
- `../memes/` stills; `../20260823_photonics_lineage/generated/` (possible cameo only); `../20261001_eo_modulator_atlas/` (modulator facts, peripheral).

**3D / capture sources of real hardware (outside the repo)**
- Gaussian splats and photo sets in `~/Documents/3DGS/`: `innolight/export.ply` (143 MB) with 82 photos in `innolight/images/` and `innolight/heic/` (opened Innolight pluggable: PCB, DSP/driver with thermal pads, fiber stubs, edge connector; best real reference for the pluggable scene); `intel100G/intel100G_CWDM.ply` with `intel100G/HEIC/`. Splat PLYs are not natively importable in Blender (needs an add-on or conversion). Also blueFors, coherent_laser, wirebonder, CRT_display.
- `~/Documents/blender/untitled*.blend|.stl` (Dec 2025): contents not inspected; strings suggest default camera/light + one mesh.
- A proven headless Blender recipe exists from an earlier private project: `Blender -b --python script.py -- ...`, Cycles with Metal and CPU fallback, run on this machine with Blender 4.2.3 (script location withheld here; ask Wentao or copy the pattern into `scripts/` when needed).
- Not found anywhere: 3D models of backplane, retimed pluggable, LPO, NPO, CPO, optical I/O chiplet, ASIC, interposer, EIC/PIC; no .fbx; no teardown images named transceiver/cpo/osfp/qsfp/cowos/emib in the repos searched (the teardown photos are in the blog, Section 6.1).

**Prior video/meme pipelines (conventions to reuse)**
- `../20260130_TFLN_stop_drinking_meme/`: remake of a source video; `voiceline.md` line-by-line table; `generate_final_video.py` (ffmpeg drawtext captions: Times New Roman Bold, #ecef8a, dark shadow, uppercase); hand-annotated caption boxes (`final_bboxes.json`, `interactive_bbox.py`). Will not run as-is on the current ffmpeg (no drawtext).
- `../20260925_GTA4_meme/`: Node (draw.js shared by browser editor and `render.mjs` using @napi-rs/canvas piping RGBA to ffmpeg libx264 crf 19 + AAC); 1440x900, 30 fps, 54 s; cutouts with slow drift/zoom, fade through black, invented satirical captions (not quotes), `CAST.md` casting rationale, sourced references. Best template for a card/cutout-driven comedic cut that can host Blender clips. Soundtrack ripped via yt-dlp (rights not granted; do not reuse for the film).
- `../20260714_frontier_lab_meme/`: Pillow compositing, image only.
- Voice: no narration/TTS pipeline found; only Whisper (`/opt/homebrew/bin/whisper`) and macOS `say` (novelty voices exist). No music/SFX library.

**Environment (2026-10-02)**
- Blender 4.2.3 at `/Applications/Blender.app` (not a brew cask; default config only).
- ffmpeg 8.1.2 with libx264/libx265/videotoolbox/prores but NO drawtext/subtitles filter (no freetype/libass): captions must be rendered in Blender, Pillow, or canvas, or install a freetype-enabled ffmpeg.
- Apple M4 Pro, 12 cores, 16-core GPU, 24 GB RAM. Disk: 13 GiB free of 460 GiB (97% full); PlayGround alone is 39 GB. Renders and caches need a disk plan (NAS was not mounted).
- Default `python3` on PATH is a stray 3.14 venv python; use `uv` (at `~/.local/bin/uv`) with a pinned 3.12/3.13 for any scripts. Node 23.7.0. Also present: yt-dlp, iMovie, OBS. No DaVinci, Final Cut, Audacity, Inkscape.

**Gaps identified**
- All cast hardware must be built (procedural GLB style or from the Innolight/Intel captures and the blog photos as references).
- Eye-diagram export tooling does not exist; model has no DFE and no real backplane S-parameters.
- OFC/IEEE figures and PDFs are not local (NAS unmounted); several key packaging talks not extracted.
- No TTS/music/SFX tooling; ffmpeg cannot burn captions; low disk.

### 6.3 MRM crosstalk: where it lives

- Original discussion: ChatGPT web UI export, conversation titled "Fast Resonance Modulation Plot" (created 2026-08-17 local evening; last updated 2026-09-05; model gpt-5-6-thinking; attachment: Intel paper "An 800 Gbps Fiber Silicon Photonic Microring-Based DWDM Transceiver in an Open-Cavity Package"). Export location: `~/Downloads/chatGPT-data-export-20260928-*/conversations-010.json` (not copied here).
- It is a microring WDM link study, not a thermal-crosstalk study: 200 vs 400 GHz grid spacing, odd/even interleaving (Intel), 70 GHz FWHM rings at 100 GBd NRZ, spacings 150/200/400 GHz, TIA noise, 9-tap FFE trained on the desired channel only, BER methodology iterated several times.
- Findings as stated in the conversation (unverified, partly superseded inside the conversation itself): RX Lorentzian leakage about 5.16% / 2.97% / 0.76% at 150 / 200 / 400 GHz; 400 GHz gives ~6 dB less isolated leakage than 200 GHz; a one-pole 50 GHz electrical filter attenuates carrier beats only ~10 / 12.3 / 18.1 dB at 150 / 200 / 400 GHz, leaving the 200-vs-400 conclusion open. Final BER table was explicitly provisional with non-monotonic quirks.
- Public write-up with the figure: `$BLOG/_posts/2026-08-31-weekly-OFS-114.md`.
- The Codex-built derived simulation project lives in a separate private workspace; its path is intentionally not recorded in this public-side folder. If the video needs that data, ask Wentao to export what he is comfortable publishing (eye panel PNG and the 4x4 panel already public in the blog).
- No project folder for MRM crosstalk exists under PlayGround (confirmed by the local survey, 2026-10-02).
- Other ChatGPT conversations touching rings incidentally: "Microring Modulator Bandwidth" (2026-05-22), "MRR control energy estimation" (2026-05-25; neighboring-ring thermal crosstalk as a runtime tracking cost), "MRM Operating Temperature Mentioned" (2026-08-27).
- Video-related prior art in his history: GTA4 loading-screen meme (2026-09-26/27, project `../20260925_GTA4_meme/`), frontier-lab meme (`../20260714_frontier_lab_meme/`), milkshake meme video (2026-04-06), ChatGPT "Photonics Meme Examples" (2026-06-12). No Blender explainer work found.

## 7. Blender library layout (draft, to be confirmed in the proposal)

Goal: reusable across future videos, spin-off-able to its own repo.

```text
20261002_interconnect_film/
  DevLog/
  README.md                  (after proposal is approved)
  config/                    (YAML: scene numbers, units, palette; no logic)
  references/                (raw + md extracts with metadata header: url, date, SHA-256)
  assets/                    (one .blend per component, linked into scenes)
    components/              (transceiver_osfp, switch_asic_package, pic, eic, interposer, substrate, fiber_array, ...)
    props/                   (scope, rack, cubicle, human rig, shotgun-free gag props, etc.)
    materials/
  scenes/                    (one .blend per scene, linking assets/)
  scripts/                   (bpy generators, headless render, eye-diagram data export)
  data/                      (eye/channel arrays exported from the 20260523_serdes model)
  outputs/                   (one clean final per milestone; no temp versions)
```

Rules to carry: scripted/headless bpy where possible; components built as linked library assets with real dimensions (MSA mechanical drawings where available); eye diagrams driven from model data, not hand-drawn; logs with timestamps in filenames; resumable renders skip existing frames.

## 8. TODO / open items

- [x] Receive and record the web research (Section 5)
- [x] Receive and record the local assets/environment survey (Section 6.2)
- [ ] Decide disk plan (13 GiB free) and ffmpeg caption path before any render work
- [ ] Verify the Section 4 numbers that will be used in jokes/narration against primary sources; record corrections
- [ ] Write the story/scene proposal (new DevLog-001) and list clarification questions
- [ ] Fresh-context audit of the proposal; record audit corrections

## 9. Progress log

- 2026-10-02: Folder and DevLog created. Four research subagents launched (blog corpus, local assets/environment, MRM conversation search, web research). Blog corpus and MRM search reports received and recorded above.
- 2026-10-02: Local assets/environment survey received and recorded in Section 6.2 (private-workspace material intentionally excluded). Web research (style bible, Blender notes, public assets, technical table) received and recorded in Section 5. All four research agents complete; proposal drafted in `DevLog-001-story-scenes-assets-proposal.md`.
- 2026-10-02: Scope changed after Wentao's reviews (see DevLog-001 Section 1): LPO and thermal runaway cut, six 10 s scenes, 4:5, plain-English narration, NARROWCOM parody logo. DevLog-001 supersedes the chapter list in Section 3 and the Ch5-Ch7 plan here; Sections 4-7 here remain the research record.
