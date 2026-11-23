# RF probe references and modeling spec (2026-11-22)

Reference material for the probe-station renders (`course/tools/blender/rf_probe.py`, `shot_probe_station.py`).
The downloaded files in `dl/` are vendor-copyrighted and git-ignored. They are for measuring only: never ship
them in `course/`, never crop them into the course, and keep vendor names and logos off the rendered parts.

## Sources (all fetched 2026-11-22)

| Key | What | URL |
|---|---|---|
| GGB40 | GGB Picoprobe Model 40A datasheet (3/99), 4 pages, page 4 = mounting styles and dimensions (P, DP, EDP, LP, C, T, GR), page 2 = tip close-up photo | https://ggb.com/wp-content/uploads/2025/04/mod40a.pdf (also https://antena.fe.uni-lj.si/literatura/Razno/Konferenca%20midem%202015/cpw/mod40a.pdf) |
| GGB67 | GGB Model 67A datasheet: female 1.85 mm (V) connector, 2.4 mm compatible, same mounting styles | https://ggb.com/wp-content/uploads/2025/04/mod67a.pdf |
| GGB110 | GGB Model 110H datasheet: female 1.0 mm connector | https://ggb.com/wp-content/uploads/2025/04/mod110h.pdf |
| GGBimg | GGB product photos: P-style 40A body (black body, three-hole raised flange, threaded connector on a 45 degree face, coax leaving low at the front), tip close-up | https://ggb.com/wp-content/uploads/2025/04/40ap-1.webp, 40ap-2.webp, 40a-1.webp (product page https://ggb.com/products/model-40a/) |
| MPIT | MPI Dual TITAN probes datasheet QMS-C-AS-234-02 (03-2024), p. 9 body dimensions of T40S/T50S/T67S/T145S | https://atv-systems.de/files/theme/content/products/mpi/AST/Probes%20%26%20Accessories/mpi-dual-titan-probes-data-sheet.pdf |
| MPIP | MPI MicroPositioner selection guide AST-AS-0113-02 (02-2025), p. 4 MP60 with E/W and N/S RF arms and TITAN probe | https://www.mpi-corporation.com/wp-content/uploads/ASTPDF/MPI-MicroPositioner-Selection-Guide.pdf |
| MPIimg | MPI MP60 / MP80 product photos | https://www.mpi-corporation.com/wp-content/uploads/slider82/micropositioner-mp60-4.jpg, 4-mp60.jpg, micropositioner-mp80-6.jpg |
| FFR | FormFactor (Cascade) application note, Infinity probe mechanical layout rules: overtravel to skate 2:1 (50 um overtravel, 25 um skate), substrate extension typically 100 um, pads of one probe in a straight line at equal pitch | https://www.psirep.com/system/files/infinity_probe_mechanical_layout_rules_-_infinity-probe-mechanical-layout-rules-an.pdf |
| CON | 1.85 mm connector coupling thread M7 x 0.75 (mates with 2.4 mm); Amphenol SV and Southwest Microwave 1.85 mm catalogs | https://mpd.southwestmicrowave.com/docs/1.85-mm-Connectors.pdf (web search summary, drawing text not parsed) |
| KEY | Keysight connector care: 1.85 mm torque 56 N-cm | https://helpfiles.keysight.com/csg/d9300a/Help/Infiniium-UG/Content/Topics/Connector-Care/connectors_1_85.htm |
| PAP | Papers using GGB probes on TFLN devices (probe types, not geometry): arXiv:2201.10790 (GGB 50A), arXiv:2112.13660 (GGB 40A-GSG-100-DP) | https://arxiv.org/abs/2201.10790, https://arxiv.org/abs/2112.13660 |

Not reachable: everythingrf.com (Cloudflare challenge, stopped), FormFactor InfinityXT datasheet mirror (returned HTML), GGB old `40a.html` (404).

## Modeled probe (generic 67 GHz class GSG probe, GGB P-style body)

| Feature | Modeled value | Source |
|---|---|---|
| Overall length, flange back to tip | 35.3 mm (1.39") | GGB40 p4, P-style side view |
| Coax from body front to tip | 8.1 mm (0.32") | GGB40 p4, P-style top view |
| Tip drop below body bottom | 1.65 mm (about 0.065", read off the drawing at 230 px/in), final bend near the tip, landing angle about 15 degrees | GGB40 p4 side view (scaled from drawing, estimate) |
| Body height (bottom to flange top) | 9.3 mm (0.255" + 0.11") | GGB40 p4 |
| Mounting flange | 30.5 mm wide (1.2"), 2.8 mm thick (0.11"), top flush with body top, open underneath for the arm plate | GGB40 p4 |
| Mounting holes | 3 x 4.7 mm (0.185") at 10.16 mm (0.4") pitch, 3.9 mm (0.155") from the back edge; same 10.16 mm pitch on MPI (3 x 4.5 thru, counterbore 7.4 x 2.4) | GGB40 p4, MPIT p9 |
| Top-view taper | full width to 10 mm (0.40") from the back, then tapering to 12 mm (0.47") at the front face | GGB40 p4 top view (scaled, estimate) |
| Connector seat | pocket 0.40" to 0.73" from back, floor 3.3 mm above body bottom, 45 degree face rising to the top at 0.96"; connector axis normal to it, pointing up and back | GGB40 p4 ("connector pointing back at a 45 degree angle"), GGBimg |
| Connector | female 1.85 mm (V) jack, M7 x 0.75 coupling thread, about 5.5 mm of thread showing | GGB67, CON |
| Mating cable plug | 8 mm hex coupling nut, knurled grip, boot, 5.5 mm OD phase-stable cable with a smooth 60 to 80 mm bend radius | estimate (typical test cable proportions, not from a datasheet) |
| Coax line | 1.2 mm OD (0.047" class), metal outer conductor | estimate from GGB tip photo proportions (GGBimg 40a-1) |
| Tip | coax end ground to a cone (1.2 mm to about 0.45 mm wide flat end over 0.9 mm), three spring contacts 0.35 mm long, 30 um wide at the contact | GGB40 p2 photo (proportions estimated) |
| Footprint | GSG, 150 um pitch (G to S) | GGB40 ordering example 40A-GSG-150-P |
| Planarity | all three contacts touch the pads (within 2 um) | probe planarization practice |
| Skate marks | 25 um long, behind each contact (50 um overtravel at 2:1) | FFR |
| Body finish | black anodized, threaded connector bright stainless; no logos, no serial label text | GGBimg (logo removed on purpose) |

## Modeled positioner and arm (generic manual RF positioner, MP60 proportions)

| Feature | Modeled value | Source |
|---|---|---|
| Footprint | 89 x 60 mm, magnetic steel base plate | MPIP p4 |
| Height to body top / knob top | 88.8 / 105.8 mm | MPIP p4 |
| Travel X/Y/Z | 20 / 20 / 13 mm (not animated) | MPIP p4 |
| Controls | knurled Z knob on top, micrometer heads on X and Y | MPIimg |
| Arm reach | probe tip 67.6 mm in front of the positioner front face (137.6 - 70) | MPIP p4 E/W arm drawing |
| Probe tip below the positioner base | about 30 mm (platen above the chuck) | MPIP p4 drawing, scaled (estimate) |
| Arm | L-shaped black-anodized arm: vertical drop plate on the Z carriage, horizontal bar ending in a plate under the probe flange, three M4 cap screws | MPIP p4 (shape), screw size estimate |

## Layout on the die (fixes the old in-line layout where probes and fibers came from the same sides)

- Fibers: lensed fibers on the waveguide axis at the west (input) and east (output) facets, held from above by
  V-groove holders on arms that go out along the optical axis to fiber stages on the platen.
- RF: active section (|x| < 1.40 mm) ground / signal / ground = 300 / 40 / 300 um with 8 um gaps holding the
  arms; tapered at |x| = 1.40..1.58 mm to the feed geometry (signal 80 um, gaps 40 um, grounds 300 um) that
  bends (0.45 mm centreline radius) north at the input end and south at the output end and ends 0.95 mm inside
  the north and south edges. The last 150 um of each line are the GSG pads; contacts land 90 um from the line
  end at 150 um pitch (checked numerically: all six contacts on their conductor, 0.5 um above the gold). The input probe lands from the north,
  the termination (port 2) probe from the south. Pads sit well inside the edge, so the reference waveguide and
  the facets stay clear; the coax passes 0.3 mm or more above the die edge.
- Optical arms: each arm leaves its gap with an S-bend before the CPW bends; at each bend one arm passes under the
  CPW (oxide-clad crossing, not drawn). With pads off the optical axis one such crossing per end is
  topologically unavoidable for a push-pull GSG electrode (the arm in the inner gap is enclosed by the bend).
