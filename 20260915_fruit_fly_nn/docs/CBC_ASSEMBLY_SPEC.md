# CBC scene assembly specification — equipment, placement, ports and cable schedule

## 0. Authority and scope

This is the implementation specification for the next hardware-scene revision. It replaces the approximate placement and shared-output wiring suggestions in the earlier review. The coding agent implements it; this document contains no implementation code.

The assembly is a **generic illustrative laboratory system**, not a qualified hardware BOM. Geometry and wiring must nevertheless be consistent. Use the cached vendor CAD as packaging reference, not as evidence that a particular commercial EDFA, phase modulator or motor supports the invented control/power interface below. In particular, the cached LN65S-FC is FC/PC; it is not the proposed generic PM/APC cassette.

**Deliverable:** one supported bench, 19 complete channel chains, a vertical hex aperture, a wired console with more knobs, and the fly directly in front operating reachable controls. Actual phase, amplitude, pointing, focus and neural activity retain their existing data sources. Staged foreleg motion is permitted.

**Important migration:** the old `wiring-plan.json` has shared logical outputs and 215 edges. It is not the finished physical cable schedule. Implement the explicit distributors and **241 internal cable runs** specified here, plus one external power-entry cord. Do not retain overlapping plugs on the old shared endpoints.

## 1. Coordinates and placement conventions

- All numbers in this document are **mechanical display millimeters**, before one uniform scene-scale conversion. They do not change solver wavelength, pitch or target distance.
- World +X is right, +Y is up, +Z is optical propagation. An operator at the front looks toward -Z. The aperture faces +Z. Table top is Y=0.
- Table top spans X=-1600…1600 and Z=-1850…300. Thickness 60 mm extends below Y=0. Put support legs below the table; do not suspend it in space.
- Equipment location tables give **footprint center X/Z and supporting-surface Y**, not the mesh origin. Place the asset's lowest foot on that surface. Existing case assets have feet reaching local Z=-4 mm, so their converted origin needs a +4 mm lift above the surface.
- General enclosure orientation **A**: asset +X→world +X, asset +Y→world -Z, asset +Z→world +Y. Thus lids face up; the current assets' front ports on local -Y face world +Z. This is a -90° rotation about X before placement.
- Orientation **B**: A followed by 180° world-Y yaw. Lids still face up; the current splitter's local +Y output bank now faces world +Z, and its single input faces -Z.
- Orientation **C**: A followed by +90° world-Y yaw. The phase cassette's input pigtail faces +Z, its output pigtail -Z, and its RF socket +X. This avoids pointing a pigtail straight into the adjacent driver.
- New generic boxes use the world-aligned convention directly: local right +X, local up +Y, front +Z. No additional import rotation.
- The aperture mount is a special assembly: base rests horizontally, vertical pedestal supports a collimator emitting +Z. **Do not apply A to the current mount and accept its resulting upward emission. Rebuild/reorient its optical child assembly independently of its base.**
- Fly asset: native Y-up, forward +X. Align its forward vector with world -Z. Place from stance/contact anchors, not from the target fly's heading convention.

## 2. Counted equipment and component list

Counts below distinguish assemblies from their children. Do not count an integrated motor twice in the scene totals.

### 2.1 Shared equipment

| Stable ID | Count | Component / envelope W×D×H mm | Required visible detail | Asset action |
|---|---:|---|---|---|
| TABLE | 1 | Optical table, 3200×2150×60 top | Horizontal top, edge thickness, four legs; hole pattern optional | New |
| TRAY-R1…R4 | 4 | Channel tray, 1580×350×8 | Raised underside supports, mounting strips; clear front/rear cable openings | New |
| PLATE-CH01…19 | 19 | Removable channel baseplate, 260×320×4 | Four fasteners; channel label; feet/stand-offs | New; children of trays |
| RISER-PH01…19 | 19 | Phase-cassette spacer, 35×90×20 | Raises pigtail exits above rear electrical trough; two mounting screws | New; children of channel plates |
| RACK-L | 1 | Left shared-instrument stand, 560×700 | Two independent shelves, top surfaces Y=20 and Y=180 | New |
| RACK-R | 1 | Right electronics stand, 640×760 | Shelf surfaces Y=20, 180, 340; cable exits on sides/rear | New |
| SEED | 1 | Generic seed laser instrument, 200×140×60 | Enclosed box, optical OUT, DC IN, status/power indicator | Replace old cube |
| SPLIT | 1 | Custom PM 1×19 splitter, 210×120×32 body | Single input, 19 outputs, mounting feet, channel labels | Reuse splitter-19; orientation B |
| PSU | 1 | Generic bench supply, 240×220×100 | External power inlet, one DC trunk output, display/switch | New; no unsupported voltage/power ratings |
| PDU-M | 1 | Main DC distribution box, 240×100×45 | One input, six separate labeled outputs | New |
| PDU-S | 1 | Shared-device DC distribution box, 120×80×35 | One input, three outputs | New |
| PDU-R1…R4 | 4 | Row DC distribution boxes, 200×80×35 | One input, ten outputs each; R4 has two capped spares | New |
| IO | 1 | Phase/gain command chassis, 560×180×90 | One console link, 19 phase-command outputs, 19 gain-command outputs, DC input | New |
| MC | 1 | 57-axis schematic motor-controller chassis, 320×180×90 | One console link, 19 combined channel outputs, DC input | New; each channel harness carries tip/tilt/focus services |
| CON | 1 | Control console, 520×300 footprint | Sloped top, 43 knobs, 19 selector buttons, status display, three rear connectors | New version; old five-knob panel is not final |
| OP-PLATFORM | 1 | Fly operator platform, 560×330 | Supported flat stance surface, unobstructed front of console | New |
| FLY-CTRL | 1 | Enlarged articulated controlling fly | Foreleg motion, four supporting legs, wings at rest | Reuse existing articulated fly; separate target fly unchanged |
| AP-FRAME | 1 | Aperture frame, 400×400 outer plate/frame | Vertical support, two feet/pedestals, rear cable combs | New |
| AP-LEDGE01…19 | 19 | Aperture mount ledges, 45×50 | Horizontal feet contact for each mount, braced to vertical frame | Integrated support children of AP-FRAME |

### 2.2 Per-channel equipment, repeated exactly 19 times

| ID pattern | Count | Component | Envelope / construction | Asset action |
|---|---:|---|---|---|
| PH-CHnn | 19 | EO phase cassette | Body 80×30×16; PM pigtails forward/rearward after rotation; RF socket toward +X | Reuse phase-cassette, orientation C |
| DR-CHnn | 19 | Electrical phase driver | Body 105×80×38; top fins extend to 45; RF front, power/command rear | Reuse phase-driver, A |
| AMP-CHnn | 19 | Generic optical amplifier | Body 160×130×55; fins to 62; optical IN/OUT front, DC/command rear | Reuse optical-amplifier, A |
| COL-CHnn | 19 | Tip/tilt collimator assembly | Maximum transverse envelope 45×45; optical depth ≤70; front +Z | Rework current mount; keep recognizable collimator and lens |
| MT-CHnn / MY-CHnn | 19+19 | Tip and tilt motors | Two distinct actuator bodies on each COL, clear of neighboring cells | Integrated children of COL, not extra free-standing boxes |
| FOC-CHnn | 19 | Focus stage, including one motor | Coaxial lens carriage within COL depth allowance; travel indicated by moving carriage | New, child of COL; no extra optical patch cable |
| JB-CHnn | 19 | Rear aperture motor breakout | 30×22×12; one combined input and three separate local motor outputs | New, frame-mounted behind each COL |

The 19 focus motors plus 38 tip/tilt motors give **57 visible motor/actuator subassemblies**. MC is one schematic chassis housing their driver electronics; do not create 57 unexplained driver boxes.

### 2.3 Connector and cable inventory

| Item | Exact installed count | Count basis |
|---|---:|---|
| Optical cable runs | 58 | 1 seed→splitter + 19 splitter→cassette + 19 cassette→amplifier + 19 amplifier→collimator |
| Fixed cassette pigtails | 38 | Two per PH; these are part of the 58 runs, not additional patch cords |
| Two-ended removable optical patch cords | 20 | Seed→splitter plus 19 amplifier→collimator |
| Male FC/APC optical plugs | 78 | 40 on the 20 patch cords; 38 at free ends of cassette pigtails |
| Installed FC/APC equipment receptacles | 78 | SEED 1 + SPLIT 20 + AMP 38 + COL 19 |
| Additional standalone FC mating sleeves | 0 | Do not add another sleeve on top of a modeled equipment socket |
| RF coax runs / SMA plugs / SMA sockets | 19 / 38 / 38 | One DR→PH cable per channel, two male ends and two equipment sockets |
| Low-level command cables | 40 | Two console links + 19 IO→DR + 19 IO→AMP |
| Internal DC cables | 48 | Seven trunks + 38 channel feeds + three shared-device feeds |
| Motor cables | 76 | 19 MC→JB multicore trunks + 57 JB→actuator tails |
| External supply-entry cord | 1 | Scene boundary power inlet→PSU; separate from 48 internal DC cables |
| Unused power sockets with caps | 2 | PDU-R4 outputs 09 and 10 |

Each command/DC/motor cable has two terminated ends; model **80 command plugs, 96 internal-DC plugs, and 152 motor-harness plugs**. These are separate connector families, not FC plugs. The external supply cord uses its own inlet/plug representation. Housing-integrated receptacles are owned by equipment assets; plugs belong to cable assemblies.

## 3. Shared equipment locations and stacking

Locations are (X, support Y, Z), in mm. Bottom feet touch the specified support surface. Envelopes exclude cable leads, which have reserved lanes below.

| Item | Location | Orientation and stacking |
|---|---|---|
| RACK-L | (-1220,0,-1380) | Footprint X=-1500…-940, Z=-1730…-1030; shelves at Y=20 and 180 |
| SEED | (-1330,20,-1490) | Front optical OUT +Z; occupies X=-1430…-1230, Z=-1560…-1420 |
| SPLIT | (-1100,20,-1320) | B: 19 outputs +Z, input -Z; occupies X≈-1205…-995, Z≈-1387…-1253 including sockets |
| PSU | (-1220,180,-1490) | Front display +Z, external/DC ports -Z; 120 mm or more clear above lower instruments |
| PDU-S | (-1350,20,-1150) | Output panel +Z; accessible on lower shelf front-left |
| RACK-R | (1200,0,-1400) | Footprint X=880…1520, Z=-1780…-1020 |
| PDU-M | (1200,20,-1500) | Outputs +Z; trunk input -Z |
| IO | (1200,180,-1440) | 38 command outputs +Z; DC and console link -Z; leave ≥80 mm top clearance |
| MC | (1200,340,-1360) | 19 motor outputs +Z; DC/console link -Z; visible top vents |
| PDU-Rr | (1010,20,Zr) | One beside each channel row, centered at that row's Zr; output face toward -X row electrical trough |
| CON | (-1220,20,-250) | Front toward +Z, rear connector panel -Z; sloped top specified in §8 |
| OP-PLATFORM | (-1220,0,85) | Top Y=20; Z=-80…250; separate support directly in front of console |
| FLY-CTRL | panel-relative, see §8 | Centerline X=-1220; faces -Z; stance on platform; fit foreleg reach before finalizing body root |
| AP-FRAME | (0,0,100) | Frame center at (0,300,100); outer plate X±200, Y100…500; pedestals bridge to table |

**Stacking rules:** only RACK-L/R provide vertical stacking; modules sit on independent shelves, never directly on another instrument's lid. The 19 channel plates are a single horizontal layer. Do not stack amplifier/driver/cassette inside a channel cell. Keep top fins exposed. Shelf posts must avoid port/boot clearance volumes.

## 4. Exact 19-channel placement and aperture mapping

Four electronics rows have centers Zr=-1490,-1100,-710,-320. Five column centers Xc=-640,-320,0,320,640. Row 4 column 5 is intentionally empty; do not populate a twentieth channel.

Each tray spans X=-790…790, with depth 350 centered on Zr. Tray top Y=20. Each baseplate top Y=24. Apply asset foot offset above that plane.

| Channel | Electronics row / col | Cell center (Xc,Zr) | Aperture emission center (X,Y,Z) |
|---|---|---|---|
| CH01 | R1/C1 | (-640,-1490) | (-65,412.583,100) |
| CH02 | R1/C2 | (-320,-1490) | (0,412.583,100) |
| CH03 | R1/C3 | (0,-1490) | (65,412.583,100) |
| CH04 | R1/C4 | (320,-1490) | (-97.5,356.292,100) |
| CH05 | R1/C5 | (640,-1490) | (-32.5,356.292,100) |
| CH06 | R2/C1 | (-640,-1100) | (32.5,356.292,100) |
| CH07 | R2/C2 | (-320,-1100) | (97.5,356.292,100) |
| CH08 | R2/C3 | (0,-1100) | (-130,300,100) |
| CH09 | R2/C4 | (320,-1100) | (-65,300,100) |
| CH10 | R2/C5 | (640,-1100) | (0,300,100) |
| CH11 | R3/C1 | (-640,-710) | (65,300,100) |
| CH12 | R3/C2 | (-320,-710) | (130,300,100) |
| CH13 | R3/C3 | (0,-710) | (-97.5,243.708,100) |
| CH14 | R3/C4 | (320,-710) | (-32.5,243.708,100) |
| CH15 | R3/C5 | (640,-710) | (32.5,243.708,100) |
| CH16 | R4/C1 | (-640,-320) | (97.5,243.708,100) |
| CH17 | R4/C2 | (-320,-320) | (-65,187.417,100) |
| CH18 | R4/C3 | (0,-320) | (0,187.417,100) |
| CH19 | R4/C4 | (320,-320) | (65,187.417,100) |

The aperture values are the canonical `channel-layout.json` transverse coordinates ×65 mm, then translated to Y=300 mm. This gives the required 3+4+5+4+3, not the electronics grid. Use the canonical file for full precision; the table is rounded to 0.001 mm for review. The 65 mm is mechanical display pitch only.

### 4.1 Inside every electronics cell

| Component | Center relative to cell (ΔX,ΔZ) | Support | Orientation |
|---|---|---|---|
| PH-CHnn | (-95,-85) | 20 mm riser on baseplate: foot plane Y=44; origin lifted 4 mm | C; input pigtail +Z, output -Z; RF +X |
| DR-CHnn | (+65,-110) | Baseplate top Y=24; origin lifted 4 mm | A; RF +Z; power/command -Z |
| AMP-CHnn | (+30,+65) | Baseplate top Y=24; origin lifted 4 mm | A; two optical sockets +Z; power/command -Z |

**Packing check against existing manifest:** amplifier including sockets is approximately 160.4×141.5 mm in tabletop projection; driver 105.4×92.5; cassette has 114 mm end-to-end pigtail-exit span. In this placement:

- Amplifier body/socket projection fits about Xc-50.2…Xc+110.2 and Zr-4.5…Zr+137.0.
- Driver fits about Xc+12.3…Xc+117.7 and Zr-154.5…Zr-62.0. The 57.5 mm front/rear gap to the amplifier permits rear electrical leads to turn into side lanes instead of entering the driver case.
- Rotated cassette body is approximately Xc-110…Xc-80 and Zr-125…Zr-45. Pigtail exits are (Xc-95,Zr-28) and (Xc-95,Zr-142), facing +Z/-Z respectively. The forward pigtail can continue beside the amplifier with approximately 44.8 mm centerline-to-case clearance. Its RF socket lies at (Xc-72,Zr-85), facing the open gap to the driver.
- The rear pigtail's 20 mm straight lead ends at Zr-162, then turns into the left inter-column corridor. Its riser puts the pigtail at Y=56, 21 mm above the E-Rr electrical centerline at Y=35. Keep tray sidewalls/support hardware below that crossing or add a bridge notch. The front pigtail stays in the same left corridor; neither runs through the amplifier footprint.
- Bodies fit the 260×320 plate. The amplifier front optical plugs extend into the reserved front routing corridor; rear driver plugs extend into the rear trough. Adjacent plate centers are 320 mm apart, leaving 60 mm inter-column corridors. Trays are 390 mm apart, leaving 40 mm between tray outlines and 70 mm between plate outlines. Use tray-edge notches where a boot/lead enters a trough.

These footprint calculations replace the earlier conflicting placement; they are not a mesh-level collision simulation. The coding agent must still check the curved leads, connector clearances and shelf fasteners before accepting the cell.

### 4.2 Aperture assembly orientation and child locations

- Each COL emission center is from the table above; rest optical direction +Z. Rear optical socket lies 70 mm behind the emission plane, at the same X/Y, normal -Z. Its final position follows the moving optical assembly.
- Gimbal pivot lies 35 mm behind emission center. Tip rotates about world-local X; tilt about the nested local Y. Positive/negative command conventions must match the actual pointing-vector transform.
- Each mount sits on its own AP-LEDGE, with its foot-contact surface at that channel's optical-center Y minus 35 mm. The ledge is horizontal and braced to the vertical frame; it does not float at the channel height. Its final depth/fastener placement must accommodate the revised mount base while keeping the rear optical socket accessible.
- Tip motor occupies the left/lower quadrant; tilt motor the right/lower quadrant, within the 45×45 projected envelope. Do not protrude into the adjacent 65 mm cell.
- Focus carriage translates along the optical axis inside COL. Nominal mechanical travel for illustration is ±2 mm; map physical solver focus to display travel explicitly, not as a fabricated calibrated actuator law.
- JB is fixed to the rear frame at (aperture X, aperture Y-20, Z=-55); its combined input faces -Z. Three motor tails run toward their local motor sockets; leave one service loop for the moving carriage/gimbal.
- All moving electrical/fiber endpoints belong to their actual moving children. Fixed frame breakouts do not rotate with the optic.

### 4.3 Aperture service-loop packing

An 80 mm fiber loop does not fit independently inside a 65 mm transverse cell. Stagger the loops in depth rather than drawing 19 coplanar intersecting circles:

| Loop group | Channels | Initial loop-center Z |
|---|---|---|
| L1 | 01,04,07,10,13,16,19 | -80 mm |
| L2 | 02,05,08,11,14,17 | -180 mm |
| L3 | 03,06,09,12,15,18 | -280 mm |

Put loops primarily in XZ planes at each optic's Y, with their supported ends in separate comb slots. Adjust individual offsets if their incoming/outgoing leads intersect a neighbor; the table sets a deterministic starting allocation, not permission to ignore collisions. Lowest optical center Y≈187 mm keeps the loop plane above the channel equipment, whose tallest starting prefab reaches about Y=90 mm. Mount the combs on frame extensions; do not leave loops unsupported across a long air gap.

## 5. Port schedule

### 5.1 Existing asset port names and world orientation

Keep these names or provide an explicit alias table in the registry. The following offsets are mm after each specified orientation (C for PH; A for DR/AMP), relative to the converted asset origin, before foot lift/placement.

| Component / port | Offset (X,Y,Z) | Outward normal | Interface |
|---|---|---|---|
| PH.optical_in | (0,8,+57) | +Z | Fixed PM pigtail exit, **not a socket** |
| PH.optical_out | (0,8,-57) | -Z | Fixed PM pigtail exit |
| PH.rf | (+23,8,0) | +X | Female SMA |
| DR.rf_out | (0,19,+48) | +Z | Female SMA |
| DR.dc_power | (-25,19,-45) | -Z | DC receptacle |
| DR.command | (+25,19,-45) | -Z | Low-level phase-command receptacle |
| AMP.optical_in | (-42,27.5,+72) | +Z | FC/APC receptacle |
| AMP.optical_out | (+42,27.5,+72) | +Z | FC/APC receptacle |
| AMP.dc_power | (-25,27.5,-70) | -Z | DC receptacle |
| AMP.command | (+25,27.5,-70) | -Z | Gain-command receptacle |
| COL.fiber_in | Relative to emission: (0,0,-70) | -Z | FC/APC, moving endpoint |
| COL.emission | (0,0,0) | +Z | Free-space aperture, **no cable/plug** |

SPLIT uses B. Its input is (0,16,-67), normal -Z. Its CH01…10 outputs are at local asset X=-85.5,-66.5,…,+85.5 mm, height 10; CH11…19 repeat the first nine X positions at height 24. After B, world X signs reverse and all outputs face +Z at Z offset +67. Preserve the labels—do not renumber channels to match a camera view. These 19 sockets already exist in the splitter mesh.

### 5.2 New shared component ports

For the following boxes, offsets are relative to body footprint center and base plane; +Z is front. Provide actual recess/flange geometry at every occupied socket.

| Equipment | Port names / quantity | Placement and facing |
|---|---|---|
| SEED | optical_out (1); dc_power (1) | Optical (0,30,+70), +Z; DC (0,25,-70), -Z |
| PSU | ac_in (1); dc_out (1) | Rear (-60,35,-110) and (+60,35,-110), both -Z; physically distinct connector shapes |
| PDU-M | dc_in (1); R1,R2,R3,R4,SHARED,MOTOR (6) | Input rear (0,22,-50), -Z; outputs front X=-90,-54,-18,+18,+54,+90 at Y22,Z50, +Z |
| PDU-S | dc_in (1); SEED,CONSOLE,IO (3) | Input rear (0,17,-40), -Z; outputs front X=-35,0,+35 at Y17,Z40, +Z |
| PDU-Rr | dc_in (1); O01…O10 (10) | Local input rear (0,17,-40), -Z; outputs X=-81,-63,…,+81 at Y17,Z40, +Z; yaw chassis so output normal becomes world -X |
| IO | console (1), dc_power (1) | Rear X=-60,+60; Y40,Z=-90, -Z |
| IO | PH01…PH19 (19); GAIN01…GAIN19 (19) | Front X=-243…+243 at 27 mm pitch; PH row Y30, GAIN row Y65; Z=+90, +Z |
| MC | console (1), dc_power (1) | Rear X=-45,+45; Y40,Z=-90, -Z |
| MC | AX01…AX19 (19) | Front bank: AX01…10 X=-108…+108 at 24 mm pitch,Y30; AX11…19 X=-96…+96,Y65; Z90,+Z |
| CON | phase_bus (1), motor_bus (1), dc_power (1) | Rear connector panel, X offsets -160,0,+160; rear face normal -Z; center height ≥25 above support; see §8 |
| JB-CHnn | IN (1); TIP,TILT,FOCUS (3) | IN on rear -Z face; three smaller outputs on front +Z face at X=-10,0,+10; distinct keyed connector sizes; no socket overlap |
| MT/MY/FOC | drive (1 each) | Socket at stationary housing rear/side, normal toward JB; motor lead does not terminate on the lens glass |
| BOUNDARY | AC (1) | Rear-left table boundary, (-1550,40,-1810); external power boundary marker |

The separate PH/GAIN outputs are low-level command interfaces; DR.rf_out is the actual electrical drive lead to the phase cassette. Do not call IO→DR an optical connection. MC's combined harness carries power and control for all three actuators; do not add separate PSU cables directly to each motor.

## 6. Exact connection schedule

Every row below defines one cable or a fully enumerated repeated family. Cable IDs are stable. A cable must have exactly two physical terminals. Equipment sockets are never used twice.

### 6.1 Shared and distribution cables

| Cable ID | From | To | Type / routing |
|---|---|---|---|
| EXT-AC | BOUNDARY.AC | PSU.ac_in | One external power-entry cord, rear table edge |
| O-SEED | SEED.optical_out | SPLIT.input | PM FC/APC–FC/APC patch, left rack loop |
| C-CON-PH | CON.phase_bus | IO.console | Shielded command harness, rear-console→electrical spine→right rack |
| C-CON-MOT | CON.motor_bus | MC.console | Separate command harness, same tray with separate lane |
| P-MAIN | PSU.dc_out | PDU-M.dc_in | Main DC trunk along rear power tray |
| P-R1 | PDU-M.R1 | PDU-R1.dc_in | Row power feeder |
| P-R2 | PDU-M.R2 | PDU-R2.dc_in | Row power feeder |
| P-R3 | PDU-M.R3 | PDU-R3.dc_in | Row power feeder |
| P-R4 | PDU-M.R4 | PDU-R4.dc_in | Row power feeder |
| P-SHARED | PDU-M.SHARED | PDU-S.dc_in | Shared-instrument feeder |
| P-MOTOR | PDU-M.MOTOR | MC.dc_power | Motor-controller DC feeder |
| P-SEED | PDU-S.SEED | SEED.dc_power | Local DC |
| P-CON | PDU-S.CONSOLE | CON.dc_power | DC via left-rear console approach |
| P-IO | PDU-S.IO | IO.dc_power | DC via rear power tray |

### 6.2 Repeated channel cables (n=01…19, no exceptions)

| Cable family | From | To | Installed count / terminal details |
|---|---|---|---|
| O-IN-n | SPLIT.CHnn | PH-CHnn.optical_in | 19 fixed input-pigtail runs; FC male at splitter only, captive at cassette |
| O-MID-n | PH-CHnn.optical_out | AMP-CHnn.optical_in | 19 fixed output-pigtail runs; captive at cassette, FC male at amplifier |
| O-OUT-n | AMP-CHnn.optical_out | COL-CHnn.fiber_in | 19 two-ended FC/APC patch cords |
| RF-n | DR-CHnn.rf_out | PH-CHnn.rf | 19 short SMA coax runs; both terminals male SMA |
| C-PH-n | IO.PHnn | DR-CHnn.command | 19 phase-command leads |
| C-GAIN-n | IO.GAINnn | AMP-CHnn.command | 19 gain-command leads |
| P-DR-n | Row PDU port in next table | DR-CHnn.dc_power | 19 DC leads |
| P-AMP-n | Row PDU port in next table | AMP-CHnn.dc_power | 19 DC leads |
| M-TRUNK-n | MC.AXnn | JB-CHnn.IN | 19 combined three-axis harnesses |
| M-TIP-n | JB-CHnn.TIP | MT-CHnn.drive | 19 local motor tails |
| M-TILT-n | JB-CHnn.TILT | MY-CHnn.drive | 19 local motor tails |
| M-FOCUS-n | JB-CHnn.FOCUS | FOC-CHnn.drive | 19 local focus tails |

### 6.3 Per-channel distributor assignment — no shared sockets

| CH | IO phase / gain | MC output | Driver DC source | Amplifier DC source |
|---|---|---|---|---|
| 01 | PH01 / GAIN01 | AX01 | PDU-R1.O01 | PDU-R1.O02 |
| 02 | PH02 / GAIN02 | AX02 | PDU-R1.O03 | PDU-R1.O04 |
| 03 | PH03 / GAIN03 | AX03 | PDU-R1.O05 | PDU-R1.O06 |
| 04 | PH04 / GAIN04 | AX04 | PDU-R1.O07 | PDU-R1.O08 |
| 05 | PH05 / GAIN05 | AX05 | PDU-R1.O09 | PDU-R1.O10 |
| 06 | PH06 / GAIN06 | AX06 | PDU-R2.O01 | PDU-R2.O02 |
| 07 | PH07 / GAIN07 | AX07 | PDU-R2.O03 | PDU-R2.O04 |
| 08 | PH08 / GAIN08 | AX08 | PDU-R2.O05 | PDU-R2.O06 |
| 09 | PH09 / GAIN09 | AX09 | PDU-R2.O07 | PDU-R2.O08 |
| 10 | PH10 / GAIN10 | AX10 | PDU-R2.O09 | PDU-R2.O10 |
| 11 | PH11 / GAIN11 | AX11 | PDU-R3.O01 | PDU-R3.O02 |
| 12 | PH12 / GAIN12 | AX12 | PDU-R3.O03 | PDU-R3.O04 |
| 13 | PH13 / GAIN13 | AX13 | PDU-R3.O05 | PDU-R3.O06 |
| 14 | PH14 / GAIN14 | AX14 | PDU-R3.O07 | PDU-R3.O08 |
| 15 | PH15 / GAIN15 | AX15 | PDU-R3.O09 | PDU-R3.O10 |
| 16 | PH16 / GAIN16 | AX16 | PDU-R4.O01 | PDU-R4.O02 |
| 17 | PH17 / GAIN17 | AX17 | PDU-R4.O03 | PDU-R4.O04 |
| 18 | PH18 / GAIN18 | AX18 | PDU-R4.O05 | PDU-R4.O06 |
| 19 | PH19 / GAIN19 | AX19 | PDU-R4.O07 | PDU-R4.O08 |

PDU-R4.O09/O10 are capped and labeled SPARE. No phantom CH20. Internal fan-out occurs inside the PDU/IO/MC enclosures; never represent it by five plugs occupying a single external port.

## 7. Cable trays, routing lanes and spline geometry

### 7.1 Reserved routing infrastructure

| Route ID | Centerline / envelope, mm | Purpose |
|---|---|---|
| F-L | X=-850, Y=35, Z=-1710…-100; width 80 | Optical distribution spine from splitter to channel rows |
| E-R | X=850, Y=35, Z=-1710…-100; width 60 | Phase/gain command and row DC spine; separate internal dividers |
| P-REAR | Z=-1785, Y=35, X=-1450…1450; width 50 | Main/shared power trunks and rack-to-rack service |
| F-Rr | Z=Zr+172, Y=35, X=-850…790; width 30 | Front optical trough per row; port lead/drop joins this trough |
| E-Rr | Z=Zr-172, Y=35, X=-790…850; width 30 | Rear row command/DC trough; split power and command compartments |
| F-RETURN | X=-900, Y=80, Z=-1500…-90; width 35 | Amplifier output fibers returning toward aperture; physically separate from F-L |
| AP-FIBER | Z=-100, Y rises from 80 to 300, X=-900…-220 | Aperture fiber approach and rear comb; no crossing in front of lenses |
| AP-MOTOR | X=+240, Z=-80, Y=40…470 | Motor bundle approach on right aperture upright |
| CON-EXIT | Behind console Z≈-450, X≈-1220, Y≈35 | Three separated boot leads, then command toward rear/right; power toward PDU-S |

These are corridors, not one centerline for every cable. Add cable combs/dividers and deterministic lane offsets; a bundle may have a shared sleeve but its logical members retain their endpoints and channel IDs. At a crossing of optical and electrical trays, use a short raised bridge with ≥15 mm centerline separation and keep bends within radius limits.

### 7.2 Route per cable family

- **O-IN-n:** splitter output plug → 30 mm straight boot lead → F-L → row's F-Rr → nearest left inter-column gap → PH input pigtail. Leave a radius-30 mm minimum loop in side clearance; use the corrected cassette orientation in §4.1.
- **O-MID-n:** PH output pigtail → local service loop beside the cell → AMP input boot. Keep it outside the RF lead and amplifier chassis. Do not route over fins.
- **O-OUT-n:** AMP output boot → F-Rr → F-RETURN → AP-FIBER → fixed rear comb → moving COL rear connector via an Ω loop. Each endpoint remains behind the aperture plane.
- **RF-n:** driver front RF socket → short coax U-shaped lead → phase RF socket. Keep it within the channel plate's front/side clearance, away from optical loops. It does not join a long shared trunk.
- **C-PH/C-GAIN:** IO front → right-side rack cable comb → E-R → E-Rr → device rear. Preserve PH/GAIN identification at both ends.
- **P-DR/P-AMP:** row PDU's dedicated output → E-Rr power compartment → device rear. Do not snake every device back to PSU independently.
- **M-TRUNK:** MC front → right-side vertical cable ladder → AP-MOTOR → fixed JB.IN. Carry three actuator services inside one multicore jacket, then split only at JB.
- **M-TIP/TILT/FOCUS:** short tails from JB to each actual actuator housing. Provide a local flex loop only where relative motion occurs.

### 7.3 Spline/termination requirements

1. Obtain the **plug cable-exit** position and outward tangent after all parent transforms. A captive pigtail uses its cassette exit instead. Equipment mating plane is not the cable start.
2. Reserve a straight boot lead: optical 20–30 mm, SMA 12–20 mm, multicore 20–35 mm, sized to the modeled connector. Do not curve inside the boot.
3. Enter and leave tray corridors with cubic lead segments tangent to the terminal/straight lead. Use piecewise cubic or rounded centripetal interior splines with C1 continuity. Arrival tangent is opposite the destination's outward tangent.
4. Use at least two routed interior waypoints for a long run, plus entry/exit handles; never substitute a single universal midpoint arch. Obstacles and lane selection determine waypoints.
5. Optical Ω loop near COL: begin with ~80 mm loop diameter and sufficient free length for the configured display motion. Verify the loop remains behind the optic and clear of adjacent cells; loops can occupy staggered Z planes rather than the 65 mm transverse cell alone.
6. Initial visual minimum bend radii: optical 30 mm, RF 20 mm, command/DC 25 mm, motor multicore 30 mm. These are scene constraints, not manufacturer-certified installation limits. Check sampled curvature, not just waypoint distances.
7. Cable outside diameters: optical 2 mm, RF 3 mm, command 4 mm, branch DC 4 mm, DC trunk 8 mm, motor trunk 6 mm, motor tails 2.5 mm. Scale these with the same mechanical transform as the connectors.
8. Render tubes for near/selected cables; 6–8 radial segments is sufficient initially. Sample more densely near bends, less along straights. LOD can replace distant bundles with simpler geometry without deleting their registry edges.
9. Clearance check uses cable radius plus 2 mm against equipment/support bounds, excluding only the declared insertion corridor. Flag and fix intersecting routes. Transform world curve points back to the cable-parent frame.
10. Cache fixed routes; update only moving terminal leads/loops when actuators move. Highlighting does not alter geometry or routing.

## 8. Console detailed layout and fly contact arrangement

### 8.1 Panel geometry

CON footprint center is (-1220,-250) in X/Z. Sloped control surface center is Y=75. Define surface coordinates u=right, v=rearward, w=normal; u span ±260 mm, v span ±150 mm. Tilt the surface 15° so the rear edge is higher than the front. Front faces +Z. Use a wedge/pedestal beneath it; don't rotate a rectangular box through its support surface.

Rear connector panel is vertical, behind the sloped surface, and has three separate sockets at u=-160,0,+160. Keep rear boot leads outside the console and clear of its hinge/pedestal. The front edge has no exiting cables.

### 8.2 Exact controls and counts

| Control | Count | Surface placement / behavior |
|---|---:|---|
| CH01…CH10 phase/amplitude pairs | 20 knobs | Strip centers u=-225,-190,-155,-120,-85,-50,-15,+20,+55,+90; phase v=+100, amplitude v=+70 |
| CH11…CH19 phase/amplitude pairs | 18 knobs | Strip centers u=-207.5,-172.5,-137.5,-102.5,-67.5,-32.5,+2.5,+37.5,+72.5; phase v=+25, amplitude v=-5 |
| Selected phase/amplitude/tip/tilt/focus | 5 knobs | u=-80,-40,0,+40,+80; v=-110; 18 mm diameter; readable labels immediately above |
| Hex channel selector | 19 buttons | Center (u=180,v=+15); canonical 3+4+5+4+3 at 18 mm pitch; 7 mm button diameter |
| Selected-channel/status display | 1 | Center (u=180,v=+105); footprint 100×35 mm; channel ID, ownership mode and key values |
| Ownership/status lamps | 3 | Manual/controller/replay indicators alongside display; do not invent independent operating modes |

Small channel knobs: 12 mm diameter, 8 mm height, pointer and collar; strip labels and CH numbers above each pair. The large near-edge row is the fly's operating area. Phase/amplitude fine controls mirror the selected channel's pair; they do not add a new parameter. All 43 knobs indicate actual command state even if not being touched by the fly.

### 8.3 Fly placement and animation constraints

- Place FLY-CTRL on the centerline u=0 directly in front of the console, facing -Z. OP-PLATFORM top is Y=20. Start with supporting-foot centroid at X=-1220,Z=+30, then fit foreleg reach; body root depends on the asset's rig and must not be guessed from its bounding-box center.
- The nearest target row is v=-110. Use its actual transformed `touch_*` positions for foreleg targets. Bring the stance toward the panel until both central controls can be reached with flexed joints and body/head clearance. The fly's thorax must remain in front of the panel edge, not above its interior.
- Uniformly enlarge the whole rig enough for a ~160 mm-wide reachable near row; do not stretch individual legs. Verify the active foreleg chain's reach before locking scale. If outer controls remain out of reach, use a small supported lateral body shift instead of teleportation or straightening the leg beyond its limits.
- Four middle/hind feet have platform contact anchors. Animate the two forelegs using the actual articulated chain. Provide named targets for each of the five near knobs, left/right rest positions, and an optional near selector/reset target if one is added.
- Gesture sequence: lift from rest → reach above target → lower to rim/contact → follow a short tangent arc while knob indication tracks actual state → release → retract. Avoid passing straight through another knob. Initial durations: 0.2 s lift, 0.4 s reach, 0.15 s contact, 0.3–0.8 s turn, 0.4 s retract; blend/rate-limit to current playback speed.
- Use left leg for phase/amplitude, right for tip/tilt/focus initially. On a target beyond comfortable reach, shift stance visibly while maintaining support. Remote channel knobs may update electronically without a hand touching each one.
- Wings rest. Neural activity comes from controller state, not gesture timing. Hide/reposition neural overlay in console camera so contacts remain visible. Pause and reduced-motion preserve useful knob indications.

## 9. Required asset work and integration ownership

### 9.1 Registry migration (do not mix the old and new topologies)

| Existing registry item | New assembly ID / action |
|---|---|
| seed / splitter / supply / console | SEED / SPLIT / PSU / CON |
| CHnn-phase / CHnn-driver / CHnn-amplifier | PH-CHnn / DR-CHnn / AMP-CHnn |
| CHnn-mount | COL-CHnn with explicit MT/MY/FOC child actuator nodes |
| CHnn-focus placeholder with command/DC | Replace with FOC-CHnn inside COL; powered/controlled through JB.FOCUS motor tail |
| phase-command-junction shared output | Replace with IO and its 38 unique PH/GAIN sockets |
| motor-driver-junction shared outputs | Replace with MC, 19 AX outputs, 19 JB breakouts and 57 local motor leads |
| supply.dc_out used repeatedly | Replace with one PSU→PDU-M trunk and the distribution hierarchy in §6 |
| mount.motor_bus | Replace with explicit JB and actuator socket anchors; do not leave an extra dangling cable |

Publish the revised equipment/port/cable inventory as an implementation artifact and compare its counts to §10. Retire the superseded 215-edge physical rendering rather than drawing both networks.

### 9.2 Asset assignments

| Asset task | Count / output | Must contain |
|---|---|---|
| Reuse case prefabs | 19 PH, 19 DR, 19 AMP, 1 SPLIT | Correct A/B/C transforms, foot placement, existing named ports |
| New seed/supply/distribution/control cases | SEED, PSU, PDU-M, PDU-S, four PDU-R, IO, MC | All physical ports from §5; no box-center fallbacks |
| New console | 1 | 43 independent knob pivots, 19 selector meshes, status area, three sockets, foreleg targets |
| Revised collimator mount | 19 instances | Horizontal +Z optics, supported base, nested tip/tilt, two motors, focus carriage/motor, moving fiber anchor |
| Motor breakout | 19 instances | One trunk input and three tail outputs; fixed frame mount |
| Support hardware | TABLE, two racks, four trays, 19 plates, 19 phase risers, aperture frame with 19 mount ledges, operator platform | Contact surfaces and cable-clearance openings |
| Cable management | F-L, E-R, P-REAR, 8 row troughs, return spine, aperture combs, console exits | Divided lanes, supported bridges, clamps/breakouts |
| Electrical connector families | Command, DC branch/trunk, motor trunk/tail, power entry | Correctly distinct shapes, mating and cable-exit anchors |

Review-only work does not generate these assets. The coding agent may build/refine procedural geometry or use the cached references as appropriate. Do not copy vendor CAD into runtime assets without resolving redistribution rights.

## 10. Completion checks and exact evidence

1. **Inventory audit:** 19 channels; 19 PH/DR/AMP/COL/JB; 57 actuator children; 43 knobs; 19 selector buttons; exactly two capped spare row-PDU outputs. No CH20.
2. **Connection audit:** 58 optical + 19 RF + 40 command + 48 DC + 76 motor = 241 internal runs. One additional external cord. Every occupied equipment socket has exactly one mating plug. Pigtail exits and free-space apertures are typed separately.
3. **Placement audit:** top view includes footprint rectangles and named routing corridors; side view shows feet on supports, shelf clearances, horizontal lids, vertical aperture and forward-facing fly.
4. **Cell audit:** verify the corrected cassette orientation and lead corridors in §4.1; submit CH10 closeups showing both pigtails, RF cable, amplifier ports and rear electrical leads before repeating the layout.
5. **Aperture audit:** exact canonical channel identities and 3+4+5+4+3; rear socket/service-loop closeup for CH01, CH10, CH19; motion clip with no detachment or collision.
6. **Console audit:** front and side stills plus a continuous foreleg gesture clip; all knob labels legible; phase/amplitude mirrors synchronized; four supporting feet contact platform.
7. **Scientific audit:** display scaling/packing do not change solver outputs. Clearly register mechanical aperture to optical visualization; do not leave unexplained competing apertures or beams launching from unrelated positions.
8. **Review report:** link screenshots/video and state actual missing geometry/clearance failures. “Models loaded” or “cables counted” is insufficient. Reuse the running server after checking port ownership; this task does not authorize disturbing unrelated services.
