# Media plan: Ferroelectric Workplace Conduct (photos)

Date: 2026-11-22. Photo lane for `courses/ferroelectric.js`. All images are real photographs (no generated images), registered in `course/media/credits_fe.js` (alt, credit, license, source page, kind `photo`, w, h). Licenses and authors were read from each file's Wikimedia Commons `extmetadata` (or the NASA Image Library record), not guessed.

- 31 images in `media/fe/`, 5 in `media/shared/` (shared images are generic office imagery; the Feedback course may use them too).
- JPEG, long side at most 2400 px, every file under 700,000 bytes, EXIF removed.
- Crops: 16:9 for `photo` and `hero`; portrait images (`bends_sem_inspection`, `bends_layout_review`, `pyro_furnaces`, `pyro_cryostat`) are meant for `split`; card images are cropped to about 4:3. Hotspot and spot images are the full frame, and their coordinates are normalized to the delivered file.
- CC BY-SA and CC BY images were cropped. If the references page has room, add "cropped" after the credit.
- People: nobody identifiable is cast as Gary or anyone at fault. People appear only in hoods and masks, from behind, as gloved hands, or far away. Captions never name or blame the people in a photo.
- Captions are suggestions in the course voice. Facts in them are kept to things that are true in general or already stated in the course text.

## Welcome and Course Navigation

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/welcome_litho_bay.jpg` | `slide` layout `hero` (image on the existing title slide) | (title slide itself) | Real lithography bay under yellow light, people far away: sets the "this is a real workplace" tone | Welcome to the workplace. The crystal axes are already here. |
| `shared/conference_room_modern.jpg` | `slide` layout `split` (on the host slide) | "A message from the Office of Photonic Conduct" | Empty corporate meeting room, which is where messages like this one come from | The Office of Photonic Conduct is always available. This room is booked until further notice. |
| `fe/welcome_fiber_test_floor.jpg` | `photo`, `kb: [0.5, 0.5, 1.0, 0.6, 0.45, 1.15]` | "Course enforcement rules" | Real photonics assembly and test floor, everyone hooded and masked | Photonics is a team activity. Orientation is everyone's responsibility. |
| `fe/welcome_cleanroom_corridor.jpg` | `photo` | "Meet the people in this course" | A long cleanroom aisle with one distant gowned figure: lonely, deadpan | Somewhere down this corridor, a chip is being loaded. We hope it is being loaded correctly. |
| `shared/open_plan_office.jpg` | `slide` layout `hero` | (on "Our mantra") | Empty open-plan office for the corporate frame | Our mantra applies at every desk, including the empty ones. |

## Crystal Orientation Awareness

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/orient_ln_wafer_zcut.jpg` | `photo`, `kb: [0.5, 0.55, 1.0, 0.45, 0.8, 1.35]` (push toward the flat) | "Let's take a closer look" | A real lithium niobate wafer, hand-labeled z-cut, with its flat at the bottom | A z-cut lithium niobate wafer. Note the flat. The supplier specification states which crystal direction the flat marks. Read it before you need it. |
| `fe/orient_wafer_flat.jpg` | `slide` layout `split` | "Four orientation hazards" (as a new host slide, or on the existing closer-look slide) | Wafer full of dies with a clear flat: the flat is the only built-in reference that survives until dicing | Every die on this wafer knows where the flat is. After dicing, none of them will. |
| `fe/orient_microscope_operator.jpg` | `check` `image` | the incident check "What would you do?" (flat at three o'clock) | Hooded operator seen from behind at a microscope: an anonymous stand-in for the person deciding | (no caption; image beside the question) |
| `fe/orient_hotspot_mask_aligners.jpg` | `hotspots`, style `numbered` | "What went wrong, technically" (incident: the flat at three o'clock, which takes place in a lithography bay) | Real lithography bay with two mask aligners, a video monitor and inspection microscopes | Prompt: Tour the lithography bay. Open every point to continue. |

### Hotspots: `fe/orient_hotspot_mask_aligners.jpg` (1920 x 1285)

The Commons description says: "Two different mask aligner (left: MA-6), (right: MJB-3) under inactinic light". The labels below claim only that.

| # | x | y | Label | Body (the handling point) |
|---|---|---|---|---|
| 1 | 0.27 | 0.40 | Alignment microscope | The aligner's microscope lines up marks on the mask with marks on the wafer. It aligns to the pattern you already made, not to the crystal. If layer one went down 90 degrees from the flat, every later layer will line up perfectly with the mistake. |
| 2 | 0.30 | 0.47 | Mask holder and wafer stage | A round wafer is placed with its flat against a reference. That is the only moment the crystal axes enter the process. A diced piece has no flat. Load it by its die mark, or do not load it. |
| 3 | 0.60 | 0.33 | Second mask aligner | Two aligners, two loading habits. Before you move a job from one tool to the other, confirm which edge each one treats as down. |
| 4 | 0.50 | 0.24 | Video monitor | A camera image can be rotated or mirrored relative to the stage. Check it against a known mark before you use it to judge a direction. |
| 5 | 0.80 | 0.24 | Inspection microscopes | Inspect after exposure and before etch. Check that the waveguides run in the intended direction relative to the flat. Resist can be stripped and redone. An etch cannot. |
| 6 | 0.88 | 0.06 | Yellow light | The room lighting leaves out the short wavelengths that expose photoresist. It does not leave out orientation errors. |

## Sources of Orientation Misconduct (new chapter)

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `shared/cube_farm.jpg` | `slide` layout `hero` | (on "Orientation misconduct can come from anywhere") | Endless cubicles: "anywhere" made literal | Orientation misconduct can come from anywhere. Including here. |
| Tool card set (below) | `cards`, style `grid`, mode `explore` | "Category 1 vignette: The prealigner" | The user wants real tools blamed for losing orientation. Explore mode: every tool has a real way to rotate your wafer, and none of them knows the crystal axes | Prompt: Meet the tools. Open each card to learn how it could rotate your wafer without telling you. |
| `shared/cubicle_corridor.jpg` | `photo` | "Category 8 vignette: You get used to it" | Long, beige, quiet corridor for "hostile crystal environment" | A hostile crystal environment rarely looks hostile. It looks like this. |
| `shared/meeting_room_table.jpg` | `photo` | "Category 5 vignette: Standard orientation" | Empty meeting table where conventions get assumed | Standard orientation is whatever was assumed in this room. Write it down before you leave it. |

### Cards: tool set (all real photos, similar framing at about 4:3)

| Card image | Label | Body (true handling point) |
|---|---|---|
| `fe/card_tool_spin_coater.jpg` | The spin coater | A wafer can sit on a vacuum chuck in any rotation. The spinner does not care, and it does not record where the flat was. |
| `fe/card_tool_mask_aligner.jpg` | The mask aligner | The aligner registers the mask to marks on the wafer, or to the flat if you place it against the reference. A piece with no flat and no mark can go on in any of four rotations. |
| `fe/card_tool_wafer_prober.jpg` | The wafer prober | A prober squares the wafer to the die pattern it can see. It knows the street grid, not the crystal axes. |
| `fe/card_tool_wafer_cassette.jpg` | The wafer cassette | A cassette holds wafers in slots. Whether their flats line up depends on whoever loaded it. |
| `fe/card_packaged_chip.jpg` | The package | A die can be attached and bonded into a package in more than one rotation. After the lid goes on, the only record is the bonding diagram. |

If the content lane wants `mode: 'choose'` instead, the question could be "Which of these tools knows the crystal axes of your wafer?". That needs one more card ("None of them. You do.") with no photo, or a text-only card, because every photo card is honestly wrong.

## Respecting Boundaries: Mode Conversion in Bends

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/bends_fiber_coils.jpg` | `slide` layout `hero` (on "Let's take a closer look") or `photo` after the cold open | "Cold open: The gentle bend" | Real coiled fiber glowing in loops: bends, literally | Bends are a normal part of working life. On x-cut, every bend is also a change of crystal direction. |
| `fe/bends_sem_inspection.jpg` | `slide` layout `split` | "What the literature reports" | Real SEM inspection of a nanostructure array, yellow room, gloved hands only | The microscope shows the geometry. The crystal direction is not in the image. It is in your layout, relative to the flat. |
| `fe/bends_layout_review.jpg` | `check` `image` | on "Gary's x-cut ring shows TE to TM conversion..." | Hooded figure from behind, at a microscope with a layout on screen | (no caption) |
| (gap) | `video` keyframe | incident "The rods at twelve o'clock" | No free photo of a polarization-maintaining fiber end face with stress rods was found | Code-drawn end face or Blender render |

## Photorefractive Conduct

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/pr_green_laser.jpg` | `photo`, `kb: [0.5, 0.5, 1.0, 0.7, 0.5, 1.3]` | "Cold open: The record Q" | Real 532 nm beam through cage optics | Shorter wavelengths are generally more photorefractive in lithium niobate. Green is not a neutral color. |
| `fe/pr_ln_crystal_mount.jpg` | `slide` layout `split` | "Let's take a closer look" | Real lithium niobate crystal in a mount (NIST photo; the crystal is the small clear piece in the white holder) | A real lithium niobate crystal. It looks calm. It keeps a record of the light you send through it. |
| `fe/pr_optics_lab.jpg` | `photo` | "Scenario 2: Protective cladding" | Busy real laser lab at night, lights on the table | Every optical table is a workplace. Power levels are workplace conduct. |

## Pyroelectric Awareness

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/pyro_hot_plate.jpg` | `photo` | "Cold open: The hotplate" | Real, well-used lab hot plate surface (cropped so the brand plate is out of frame) | A hot plate. It has no idea what pyroelectricity is. You do. |
| `fe/pyro_furnaces.jpg` | `slide` layout `split` | (on "Let's take a closer look") | Real furnace stack: industrial-scale temperature change | Every thermal step is a temperature change. Every temperature change is charge. |
| `fe/pyro_hotspot_resist_bay.jpg` | `hotspots`, style `callout` | "Where temperature change happens at work" | Real photoresist bay with spin coaters, wash bottles and safety signs | Prompt: Inspect the resist bay. Open every point to continue. |
| Temperature card set (below) | `cards`, style `flip`, mode `explore` | "Scenario 3: The neighbor" | Four real places where a lithium niobate chip changes temperature | Prompt: Turn over each card to see where your crystal changes temperature. |
| `fe/pyro_burnt_board.jpg` | `photo` | incident chat "The thread: Amplifier failure on fixture 2" | Real burnt component on a circuit board, the "after" image | Representative image of a failed component. Not the fixture 2 amplifier, which declined to be photographed. |
| `fe/pyro_esd_label.jpg` | `check` `image` | the incident check "You are the bystander" (hot plug) | Real caution label for static-sensitive devices | (no caption) |
| `fe/pyro_cracked_die.jpg` | `photo` or `check` `image` | "Why is a fast hotplate step riskier for z-cut lithium niobate than for silicon?" | Real micrograph of a cracked die | This die is silicon, and it still cracked. Lithium niobate is more brittle. Ramp accordingly. |

### Hotspots: `fe/pyro_hotspot_resist_bay.jpg` (1920 x 1285)

Commons description: "Photoresist spin coating machines under inactinic light". Only clearly visible items are labeled. The silver box beside the open spinner and the two cabinets at the far left and right are not labeled, because their function could not be confirmed from the photo.

| # | x | y | Label | Body (the handling point) |
|---|---|---|---|---|
| 1 | 0.53 | 0.47 | Spin coater, lid open | Spin coating happens at room temperature. The bake after it does not. On z-cut, every ramp up and every cooldown builds surface charge. Use the slow-ramp recipe and give the wafer a conductive discharge path. |
| 2 | 0.87 | 0.78 | Spin coater controller | Spin speed and acceleration are programmed and repeatable. Hold your bake ramps to the same standard. |
| 3 | 0.67 | 0.53 | Wash bottles | Do not rinse or quench a hot lithium niobate wafer. A fast temperature drop builds charge just like a fast rise, and thermal shock can crack the substrate. |
| 4 | 0.69 | 0.16 | Eye protection sign | Safety glasses protect you from solvents. Your devices need their own protection from charge. Both are required. |
| 5 | 0.58 | 0.335 | Electrical hazard sign | This sign warns about the tool's voltage. A heated z-cut wafer makes its own voltage and comes with no sign. Treat it as if it had one. |
| 6 | 0.055 | 0.52 | Fire extinguisher | Charge from a heated wafer is handled with grounding and slow ramps, not an extinguisher. Know where it is anyway. |

### Cards: temperature set

| Card image | Label | Body |
|---|---|---|
| `fe/pyro_hot_plate.jpg` | Hot plates and bakes | Resist bakes and cooldowns. The fastest temperature change in most process flows. |
| `fe/pyro_furnaces.jpg` | Furnaces and anneals | Slow ramps in, slow ramps out. "Slow" is a number. Ask for it. |
| `fe/bias_wire_bonder.jpg` | Bonding | Many bonding processes heat the stage. Packaging is a thermal step too. |
| `fe/pyro_cryostat.jpg` | Cryostats | Cooling to 4 kelvin is a temperature change of nearly 300 kelvin. (Matches the existing "Cryostats" tab.) |

## Bias Point Retention

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/bias_scope.jpg` | `photo`, `kb: [0.45, 0.4, 1.0, 0.45, 0.35, 1.25]` | "Cold open: Quadrature on Monday" | Real oscilloscope with live traces | The trace on Friday is a measurement. The trace on Monday is a different measurement. |
| `fe/bias_wire_bonder.jpg` | `photo` | "Bias conduct in the literature" | Real manual wire bonder: where modulators get packaged and biased | Packaging fixes the electrodes in place. It does not fix the bias point. |
| `fe/bias_spot_bench.jpg` | `spot` | "Scenario 2: The overnight hold" | Real cluttered measurement bench with several visible conduct problems | Prompt: A measurement bench, Monday morning. Find the five hazards. (Do not call it Gary's: it is a real, credited person's own bench.) |
| `fe/card_tool_wafer_prober.jpg` (reuse) | `check` `image` | incident check "What would you do?" (the chip that was turned around, RF probe station) | A real chuck with a wafer on it | (no caption) |

### Spot: `fe/bias_spot_bench.jpg` (1920 x 1455)

Only hazards that are actually visible are marked. Caveat: brand names are legible but small (Windows splash screen, IBM keyboard, Tektronix, Fluke, BenQ, Mastercraft). Blur them if they look too prominent at full size.

| # | x | y | w | h | Label | why |
|---|---|---|---|---|---|---|
| 1 | 0.735 | 0.61 | 0.07 | 0.18 | Soldering tool lying on the notes, out of its stand | An unmanaged heat source at a measurement bench. Heating a lithium niobate chip shifts its bias through thermo-optic and pyroelectric effects. It is also scorching the records. |
| 2 | 0.78 | 0.48 | 0.05 | 0.14 | Open drink can | No food or drink at the test bench. A spill on a bias connection is an unplanned bias change. |
| 3 | 0.458 | 0.92 | 0.07 | 0.06 | Bare board held only by clip leads | The device under test is loose on an ordinary desk surface. Any tug on a lead changes the contact and the DC path. A bias point that moves when someone walks past is not a bias point. |
| 4 | 0.775 | 0.865 | 0.13 | 0.13 | Loose sticky notes | Bias voltages, times, and drift readings go into the lab notebook with date and time. Sticky notes are not records. |
| 5 | 0.58 | 0.82 | 0.22 | 0.18 | Tangle of test leads | Crossing, unlabeled leads make it impossible to say which supply drives which electrode. Route them, label them, record them. |

`missWhy`: "That is not one of the hazards we are looking for. It may still be a hazard. Please keep looking."

## Simulation Integrity

| Image | Step type | Place after | Why | Caption |
|---|---|---|---|---|
| `fe/sim_screen_from_behind.jpg` | `photo` | "Cold open: Q equals seven times ten to the thirteen" | Gowned engineer from behind, staring at a screen (real microscope console; the closest unbranded "engineer at screen" photo found) | Results do not review themselves. |
| `fe/sim_wafer_dies.jpg` | `slide` layout `hero` (on "Let's take a closer look") | (closer-look slide) | Real patterned wafer close-up: the fabricated device, which has no symmetry planes you did not build | The fabricated device has no symmetry planes you did not pay for. |
| `shared/meeting_room_table.jpg` | `photo` | "Simulation integrity hazards" | Design review room before "Scenario 1: The quarter model" | Design reviews are where quarter models become full-size decisions. |
| `fe/sim_lab_notebook.jpg` | `slide` layout `split` | "Scenario 3: The template frame" | Real gloved hands writing in a lab notebook | Record the crystal frame, the rotation convention, the mesh, and the boundary conditions. In ink. |

## Reuse

`shared/meeting_room_table.jpg` is placed twice (Sources chapter and Simulation Integrity). `fe/card_tool_wafer_prober.jpg` doubles as a check image. `fe/pyro_hot_plate.jpg`, `fe/pyro_furnaces.jpg` and `fe/bias_wire_bonder.jpg` are both full-screen images and cards. Drop any reuse that looks repetitive.

## Sourcing notes

- Thumbnails were fetched at the standard 1920 px width, not 2400. Commons serves standard thumbnail sizes, and 1920 px is enough for full-bleed 16:9.
- Commons returned HTTP 429 five times: twice for original-file downloads, twice for keyword search, once for a category-prefix listing. Each time the batch stopped. After that only category browsing and cached thumbnail URLs were used, never originals. The final sourcing stopped at the fifth 429.

## Gaps (no suitable free real photo found; for the Blender lane or a code-drawn graphic)

| Needed | Why missing |
|---|---|
| Dicing saw | No Commons category found by browsing. Keyword search was stopped after rate limiting. |
| Gel-pak or die tray with dies | One NASA image showed a chip in a gel box, but it is credited "courtesy of University of Delaware", so its public-domain status is unclear and it was excluded. |
| Stepper or prealigner wafer stage close-up | Only whole-tool views (the two mask aligners and the yellow-room bays). |
| Manual probe station vacuum chuck without prominent branding | The probe station photos on Commons carry large vendor logos. The wafer prober (covers off) is used instead. |
| Computer screen running a generic simulation UI | Only branded FEA screenshots were found. A code-drawn mock solver window is recommended. |
| Lithium niobate wafer on a hot plate, sparks or discharge | None exist with a free license. Blender. |
| Square chip in tweezers with no mark, or a chip loaded 90 or 180 degrees off | Blender (renders `render/tfln_die_fixture_*deg.jpg` already exist). |
| PM fiber end face with stress rods (incident "The rods at twelve o'clock") | Code-drawn. |
| A spot-the-hazard scene in a lithium niobate fab | The real spot scene is an electronics measurement bench, so it suits Bias Point Retention. A fab-bench version would need Blender. |
