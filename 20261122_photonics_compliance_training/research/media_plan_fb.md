# Media plan: FB-1550 (feedback course), photo lane

Date: 2026-11-22. Owner: photo lane FB. Files: `course/media/fb/*.jpg`, registry `course/media/credits_fb.js`.

Sources: Wikimedia Commons (license and author read per file from the file page `extmetadata` via the Commons API) and the NASA Image and Video Library (public domain, NASA photographer credit). No generated images. All JPEGs are re-encoded (EXIF and GPS stripped), long side at most 1920 px (Commons now serves only standard thumbnail widths; 1920 is the largest standard step below 2400; three small originals were fetched at 960 or 1280 px), under 700 KB.

People rule check: no image shows a recognizable face in a role that could read as Gary or anyone at fault. People appear only in `cleanroom_fiber_team` (hoods, masks, safety glasses, named NASA technicians, neutral role), `cleanroom_robot_cell` (masked, side view, neutral), `trade_fair_hall` (distant crowd, generic non-photonics fair). Captions below keep them neutral: never caption these three as Gary, Morgan, Alex or a culprit.

Coordinates: all `x, y, w, h` are fractions of the delivered image (0..1, origin top-left). For `spot` targets, `x, y` is the box center and `w, h` the full box size (left edge = x - w/2). If the engine expects top-left, subtract w/2 and h/2. Every point was placed on a 10 x 10 grid overlay of the delivered file and checked with drawn markers.

## Inventory (26 images)

| File | Size | Use | Module |
|---|---|---|---|
| cleanroom_fiber_team.jpg | 1920x1080 | hero / photo | 0 |
| cleanroom_robot_cell.jpg | 1920x1080 | photo | 0 |
| meeting_room_empty.jpg | 1920x1080 | split / photo | 0 |
| fiber_apc_patchcord.jpg | 1920x1080 | photo | 1 |
| fiber_stripped_singlemode.jpg | 960x640 | split / check image | 1 |
| tisa_laser_table.jpg | 1920x1080 | hero | 1 |
| spot_patch_panel.jpg | 1920x1440 | spot | 1 |
| fp_sodium_rings.jpg | 1920x1080 | hero / photo | 2 |
| mirror_mounts.jpg | 1916x1078 | photo | 2 |
| fp_scanning_interferometer.jpg | 1920x1080 | split | 2 |
| hotspot_optics_table.jpg | 1920x1080 | hotspots | 2 |
| hotspot_littrow_lasers.jpg | 1920x1080 | hotspots | 3 |
| laser_diode_needle.jpg | 1920x1080 | photo | 3 |
| laser_diode_to_can.jpg | 1920x1440 | split | 3 |
| trade_fair_hall.jpg | 1920x1080 | photo / check image (demo scenario) | 3 |
| isolator_on_table.jpg | 1920x1080 | hero | 4 |
| isolator_free_space.jpg | 1920x1080 | photo | 4 |
| card_isolator.jpg, card_fp_interferometer.jpg, card_beam_dump.jpg, card_iris.jpg | 1200x900 each | cards (one set, same photographer, same studio style) | 4 |
| microdisk_sem.jpg | 960x720 | split | 5 |
| siph_wafer_300mm.jpg | 1920x1080 | hero | 5 |
| scope_flat_trace.jpg | 1164x655 | split / check image | 6 |
| power_meter_head.jpg | 1920x1080 | photo | 6 |
| lab_bench_cables.jpg | 1920x1080 | hero | 6 |

## Module 0: Welcome and Course Navigation

1. `cleanroom_fiber_team.jpg` as the image of the title slide, layout `hero`.
   - Placement: the existing `slide` with layout `title` (first step); or a new `photo` step right after it.
   - Caption: "Our people work with light every day. Light, in turn, works with our people. This course keeps that relationship professional."
   - Ken Burns: `kb: [0.62, 0.45, 1.0, 0.60, 0.42, 1.12]` (slow push toward the gloved hand and the fiber bundle).
2. `cleanroom_robot_cell.jpg` as `photo`.
   - Placement: after "Course Enforcement Features".
   - Caption: "Automation does not exempt anyone from this training. The robot has completed it. It scored 100 percent."
3. `meeting_room_empty.jpg` as `slide` layout `split` or `photo`.
   - Placement: after "Key Terms Used in This Course", before the practice question.
   - Caption: "Room 1550-B is reserved for your annual Return Loss Conversation. Please arrive with your reflections."

## Module 1: Recognizing Unsolicited Feedback

1. `tisa_laser_table.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Ripple", before "What Is Unsolicited Feedback?".
   - Caption: "Lasers live among surfaces. Every surface is a potential correspondent."
   - Note: this is a titanium-sapphire laser with a green pump; the course is about diode lasers. Use only as mood, do not caption it as a DFB laser.
2. `fiber_apc_patchcord.jpg` as `photo`.
   - Placement: after "Scenario 2: The Flat Cleave".
   - Caption: "Per the uploader, both ends of this patch cord are angle polished (APC). Green is the conventional color code for APC. The angle is the point."
   - Physics point: an angled (8 degree) polish sends the end-face Fresnel reflection outside the core's acceptance cone; a flat end sends it straight back into the fiber.
3. `fiber_stripped_singlemode.jpg` as `check` image (`image: { src }`) on the first knowledge check, or `split` on "Return Loss Budgets and Placement".
   - Caption (if used as photo): "A single-mode fiber, stripped. The end you cannot see from here is also an interface."
4. `spot_patch_panel.jpg` as `spot` (the course's spot-the-hazard scene). Placement: after the photo of item 2, before "Scenario 3: Isolator Placement".
   - Title: "Spot the Unterminated Ends"
   - Prompt: "This distribution frame passed its last audit. Find five places where a fiber ends in air."
   - Context, for the content lane: real photo of a fiber-to-the-home distribution point in France, uploaded as an example of neglect. The connectors are SC type; most are the green or teal color used for angle-polished connectors. Do not claim any visible connector is flat polished: that cannot be seen in the photo.
   - Targets (center x, y; box w, h):

| # | x | y | w | h | label | why (physics) |
|---|---|---|---|---|---|---|
| 1 | 0.325 | 0.800 | 0.080 | 0.110 | Unmated connector, end face toward you | An open connector is a glass-to-air interface. A flat polished silica end reflects about 3.4 percent (about -14.7 dB, Fresnel with n about 1.44) straight back down the fiber; an 8 degree angled end steers the reflection out of the core. Either way, an uncapped end face collects dust, and dust is a reflector nobody budgeted. |
| 2 | 0.215 | 0.520 | 0.050 | 0.170 | Hanging teal connector | Unmated and uncapped. If this fiber is live, its end is the far mirror of a cavity that includes everything back to the transmitter. |
| 3 | 0.345 | 0.195 | 0.090 | 0.170 | Two connectors parked in mid-air | A connector that is not mated is an interface waiting to become a cavity. Park unused ends in a capped parking slot, not in the cable spaghetti. |
| 4 | 0.265 | 0.435 | 0.050 | 0.120 | Green connector, no dust cap | The angle polish only helps while the end face is clean and undamaged. Uncapped ends get scratched and dirty, and a scratch or particle scatters light back into the core. |
| 5 | 0.205 | 0.675 | 0.170 | 0.120 | Cluster of loose connectors | Several more unmated ends. Each one is its own interface. Every interface reflects. |

   - missWhy: "Not that spot. There are more loose ends in this frame than the audit asks for; find the five marked by the auditor. Look for connectors hanging free, not plugged into an adapter."
   - Note: more unmated connectors are visible than the five targets (left edge). The missWhy above covers clicks on them.

## Module 2: Every Interface Is a Cavity

1. `fp_sodium_rings.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Wiggles", before "The Parasitic Etalon, in Two Formulas".
   - Caption: "Two parallel reflecting surfaces, sodium light, and rings. This is what a cavity looks like when it is invited."
   - Physics: Fabry-Perot fringes of the sodium D line; each ring is a constant round-trip phase.
2. `hotspot_optics_table.jpg` as `hotspots`, style `numbered`.
   - Placement: after "Interactive: Parasitic Fabry-Perot Ripple", before "Cavities You Did Not Design".
   - Title: "Every Interface on This Table"
   - Prompt: "This is a working table in a cold-atom laboratory. Open each marker to review the reflectors in plain sight."
   - Spots:

| # | x | y | label | body (physics point) |
|---|---|---|---|---|
| 1 | 0.51 | 0.43 | Large round optic | Whatever this optic is meant to transmit, each of its surfaces also reflects. An uncoated glass surface returns about 4 percent at normal incidence (Fresnel, n about 1.5); an anti-reflection coating reduces that, it does not remove it. |
| 2 | 0.79 | 0.73 | Large glass cube | A cube has parallel faces. Two parallel faces are an etalon, and at normal incidence the cube also sends a reflection straight back toward the source. A small tilt sends that reflection somewhere harmless. |
| 3 | 0.445 | 0.695 | Rotation mount with angle scale | Rotation mounts usually carry a wave plate or polarizer: a thin plate with two parallel faces. A plate of thickness L and index n is a low-finesse Fabry-Perot with free spectral range c / (2 n L). |
| 4 | 0.14 | 0.715 | Small glass cube | Smaller does not mean exempt. Every pair of parallel faces is a cavity, and a table holds dozens of them. |
| 5 | 0.82 | 0.65 | Kinematic mount | The two adjustment knobs set the angle of the optic. Angle is your main control over where a reflection goes: aim it into a beam block, not back into the laser. |
| 6 | 0.69 | 0.72 | Ring-mounted lens | Curved surfaces reflect too. A lens surface near a focus can return light almost exactly along the incoming axis. |

3. `mirror_mounts.jpg` as `photo`.
   - Placement: after "Scenario 2: The Accidental Laser".
   - Caption: "Mirrors are the reflectors we schedule. All other reflectors are walk-ins."
4. `fp_scanning_interferometer.jpg` as `slide` layout `split` or `check` image.
   - Placement: beside the check question on ripple period (0.1 nm at 1550 nm).
   - Caption: "A scanning Fabry-Perot interferometer: a cavity built on purpose, with a known length, so its ripple can be trusted. Yours did not come with a datasheet."

## Module 3: Mode Hop Awareness

1. `laser_diode_needle.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Thermometer Ticket", before "Why Lasers Hop".
   - Caption: "A tunable diode laser chip in the eye of a needle (NASA JPL). Small enough to lose, sensitive enough to notice your fiber."
2. `hotspot_littrow_lasers.jpg` as `hotspots`, style `callout`.
   - Placement: after "Interactive: Mode Hop Map", before "Mode Hop Fine Print".
   - Title: "Anatomy of a Mode Hop Risk"
   - Prompt: "Per the photographer, the two blue units in front are diode lasers with Littrow resonators, part of a setup for cooling and trapping rubidium atoms. Open each marker."
   - Spots:

| # | x | y | label | body (physics point) |
|---|---|---|---|---|
| 1 | 0.33 | 0.66 | Diode laser with Littrow resonator | Inside, a grating at the Littrow angle sends its first diffraction order back into the diode. This is strong feedback by design, with consent: the external cavity and the grating choose the lasing mode. |
| 2 | 0.65 | 0.66 | Second diode laser | Mode-hop-free tuning of a grating-tuned external-cavity laser needs the grating angle and the cavity length to change together. Turn only one, and the laser jumps by one external-cavity mode spacing. |
| 3 | 0.70 | 0.53 | Rotation mount with angle scale | Downstream optics reflect too. Light returned from them re-enters the external cavity and adds a third mirror; this is why external-cavity laser setups commonly place an optical isolator directly after the laser. |
| 4 | 0.25 | 0.59 | Kinematic mirror mount | A steering mirror. Its angle decides whether a stray reflection walks off or lines up with the laser again. |
| 5 | 0.30 | 0.53 | Lens in a lens tube | A lens near a focus is an efficient retro-reflector if its surface is square to the beam. |
| 6 | 0.85 | 0.16 | Small video monitor | A video monitor updates a few tens of times per second. A hop that lasts microseconds, or a hop and a hop back, will not appear on it. See Chapter 6. |

3. `trade_fair_hall.jpg` as `photo` or as the `check` image for the incident scenario where the laser mode-hops during a customer demo and Gary hides it.
   - Placement: next to the content lane's demo scenario (kicker "What would you do?").
   - Caption: "Customer demonstrations are high-visibility events. So are mode hops."
   - Note: generic fair (MESAP 2012, Croatia), distant visitors; do not caption any visitor.
4. `laser_diode_to_can.jpg` as `split` or check image.
   - Placement: beside "Scenario 1: The Datasheet" or the check on mode-hop-free tuning range.
   - Caption: "A TO-can laser diode with its cap removed. The laser chip is at the front; the photodiode that monitors its power sits at the back (per the photographer). The monitor photodiode reports power, not wavelength."

## Module 4: Reciprocity Is Not Optional

1. `isolator_on_table.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Diode", before "Policy 4.1: Definition of an Isolator".
   - Caption: "A real optical isolator. It contains a magnet. That is not a coincidence."
   - Physics: Faraday rotation in a magnetized material breaks reciprocity; the polarizers turn that rotation into direction-dependent loss.
2. `cards` step, style `flip`, mode `choose`, using the four card images (all by Geek3, same studio style).
   - Placement: after "Isolation Facts and Myths", before "Scenario 1: The Contrast Ratio".
   - Title: "Which of These Is an Isolator?"
   - Prompt: "Choose the component that passes light in one direction and blocks the return path."
   - Cards:

| image | label | correct | why |
|---|---|---|---|
| card_isolator.jpg | Faraday isolator | true | Correct. A Faraday isolator uses a static magnetic field along the beam to rotate polarization in the same sense for both directions; between two polarizers that makes forward and backward transmission different. A magnet is how a passive linear device gets permission to be non-reciprocal. |
| card_fp_interferometer.jpg | Fabry-Perot interferometer | false | Not quite. A Fabry-Perot is linear, static and non-magnetic, so it is reciprocal: transmission is the same in both directions. It is also two mirrors, which makes it a source of feedback, not a cure for it. |
| card_beam_dump.jpg | Beam dump | false | Not quite. A beam dump absorbs whatever arrives, from either direction. It is the right place to send a reflection you have already steered away, but it does not decide direction. |
| card_iris.jpg | Iris aperture | false | Not quite. An iris clips the beam the same way in both directions. It can block a misaligned reflection that walked off axis, but light that returns along the same path passes the same hole. |

   - correctSay: "The one with the magnet."
3. `isolator_free_space.jpg` as `photo`.
   - Placement: after "Scenario 3: The Straight Arrow".
   - Caption: "The arrow on an isolator is a commitment. Install accordingly."

## Module 5: Backscatter in High-Q Resonators

1. `siph_wafer_300mm.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Split Personality", before "Why Resonances Split".
   - Caption: "A 300 mm silicon photonics wafer. Every die on it is a collection of interfaces, each with an opinion about direction."
2. `microdisk_sem.jpg` as `slide` layout `split`.
   - Placement: with "Why Resonances Split" (replace the host layout image) or after "Interactive: Backscatter Doublet".
   - Caption: "A silica microdisk resonator, about 40 micrometres across, on its pedestal (electron micrograph). Light circulates at the rim in both directions. Any roughness on that rim couples one direction into the other."
   - Physics: sidewall roughness couples the clockwise and counter-clockwise modes; at high Q the coupling exceeds the linewidth and the resonance splits into a doublet.
3. Third full-screen candidate: none sourced with a license that fits (see Gaps). Reuse `fp_scanning_interferometer.jpg` is not appropriate here. Recommend a Blender render of a microring on a chip, or `cleanroom_fiber_team.jpg` reused at a different Ken Burns crop for "Scenario 1: Scrap the Wafer" only if the content lane wants a person-free crop (crop to the instrument at left: `kb: [0.25, 0.55, 1.6, 0.25, 0.55, 1.7]`, which excludes both technicians).

## Module 6: Your Instruments Are Averaging

1. `lab_bench_cables.jpg` as `photo` (hero).
   - Placement: after "Cold Open: The Steady Number", before "Every Instrument Integrates".
   - Caption: "Somewhere on this table, an instrument is averaging. It will not tell you which one."
2. `scope_flat_trace.jpg` as `slide` layout `split`, or `check` image.
   - Placement: with "Every Instrument Integrates".
   - Caption: "The original file is titled 'Oscilloscope with noise'. At this setting, the trace is calm. Calm is a display setting."
   - Physics: the bandwidth and timebase of the measurement decide which fluctuations are visible; a slow detector or long average reports the mean of a hopping laser.
3. `power_meter_head.jpg` as `photo`.
   - Placement: after "Scenario 1: The Calm Wavemeter".
   - Caption: "A thermal or photodiode power head integrates. It reports the average power of a laser that may be hopping between two modes many times per second."
   - Caution for the content lane: the photo does not show whether this head is thermal or photodiode; keep the caption generic as written.

## Gaps (no suitable real photo found; candidates for Blender or code-drawn coverage)

- Microring or racetrack resonator micrograph on a photonic chip (only a microdisk SEM and 3D renders were found on Commons with usable licenses). Module 5 is one full-screen image short.
- Optical spectrum analyzer screen showing a laser mode hop or a side-mode; no free-licensed OSA screen photo found. Code-drawn trace recommended (the course already draws traces).
- Butterfly-packaged DFB laser module, fiber-pigtailed; none found.
- Flat-cleaved versus angle-cleaved fiber end face micrograph; none found.
- Grating coupler or flip-chip bonded laser on a silicon photonics chip (Module 1 cold open subject); none found.
- Lab at night with a running unattended measurement; no free-licensed interior night lab photo found.
- Photonics conference exhibition booth (customer demo); the trade fair photo is a generic regional fair, not photonics.
- Wavemeter (optical) front panel; none found.

## Rate limiting log

- Commons API returned HTTP 429 three times (2026-11-22 about 15:00, 15:12 and 15:29 local), likely together with the other photo lane's traffic from the same address. Each time all requests stopped for at least 5 to 10 minutes and the spacing was raised (1.2 s, then 3 s, 5 s, 10 s). Downloads used standard Commons thumbnail widths (1920 px); a non-standard 2400 px request returned HTTP 400 ("Use thumbnail sizes listed on https://w.wiki/GHai").
