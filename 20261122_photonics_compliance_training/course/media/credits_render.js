/* Blender renders for both courses (Blender 4.2 LTS, Cycles). Scenes are built from the scripts in course/tools/blender/; 'source' names the script and its arguments. 2026-11-22. */
window.LMS_MEDIA = Object.assign(window.LMS_MEDIA || {}, {
  "media/render/tfln_die_fixture_0deg.jpg": {
    "alt": "Rendered view of a thin-film lithium niobate Mach-Zehnder modulator die (6 mm square): input waveguide and splitter on the left, two arms that pass around the RF pads into a straight gold ground-signal-ground electrode, combiner and output on the right, with probe pad sets at both ends of the electrode, seated correctly in a black anodized die fixture: the die's gold +Z arrow points the same way as the fixture's engraved +Z arrow, and the waveguide ends line up with the fiber slots marked IN and OUT",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_die_fixture.py variant=0",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/tfln_die_fixture_90deg.jpg": {
    "alt": "The same modulator die in the same fixture, rotated 90 degrees: the square die still fits the pocket, but its +Z arrow points sideways, its polished input and output facets now face the closed north and south walls of the pocket, and the IN and OUT fiber slots line up with no waveguide",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_die_fixture.py variant=90",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/tfln_die_fixture_180deg.jpg": {
    "alt": "The same modulator die in the same fixture, rotated 180 degrees: the die's +Z arrow points opposite to the fixture's +Z arrow, the label is upside down, and the waveguide is 0.8 mm off the fiber slot line",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_die_fixture.py variant=180",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/tfln_probe_station_closeup.jpg": {
    "alt": "Rendered probe-station view: two ground-signal-ground microwave probes, each on a positioner arm with a coaxial connector and cable, come in at a shallow angle from left and right and land on the pad sets at the two ends of a lithium niobate modulator's electrode; lensed fibers held from above reach the input and output facets through the fixture's slots, and a microscope objective hangs above",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_probe_station.py",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/tfln_gel_tray_one_rotated.jpg": {
    "alt": "Twelve lithium niobate modulator dies (MZM-01 to MZM-12) placed by hand on the gel film of a die tray, all with their +Z arrows up and electrodes running left to right except one die in the middle row, which is rotated 90 degrees",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_gelpak.py",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/ln_wafer_carrier.jpg": {
    "alt": "A double-side-polished 100 mm lithium niobate wafer in an open black single-wafer carrier: the transparent, faintly yellow-tinted crystal shows the molded size mark of the carrier floor through it, a window reflection on its polished face, and its 0.5 mm edge catching the light along the 32.5 mm primary flat and the 18 mm secondary flat, 90 degrees apart",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_wafer.py scene=carrier",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/ln_wafer_hotplate.jpg": {
    "alt": "A single-side-polished lithium niobate device wafer, milky because of its lapped back, with a grid of gold modulator dies on its polished front, lying on the aluminium plate of a laboratory hot plate whose seven-segment display reads 115 degrees C",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_wafer.py scene=hotplate",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/pyro_discharge_comb_micrograph.jpg": {
    "alt": "Rendered illustration of an optical micrograph of interdigitated gold electrodes (4 micrometer fingers and gaps) on lithium niobate after a discharge: segments of three fingers are gone, their ends resolidified into small gold spheres, a fused gold bridge spans one gap, a small crater with radial cracks sits in the substrate, a thin discoloured film surrounds the site, and a nearby finger tip is notched; 20 micrometer scale bar",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_pyro_damage.py",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/butterfly_isolator_fc_bench.jpg": {
    "alt": "A 14-pin butterfly laser on a finned laser mount bolted into the 25 mm hole grid of a black optical breadboard, with D-sub cables to the mount. The yellow fiber pigtail runs to an inline isolator fusion-spliced into the line (heat-shrink splice sleeves on both sides) whose printed arrow points away from the laser, and ends in a capped FC/APC connector with a green boot next to a capped FC/PC patch cord with a blue boot",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_butterfly.py",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/butterfly_isolator_reversed_bench.jpg": {
    "alt": "The same butterfly laser bench, except the inline isolator is spliced in backwards: its printed arrow points back toward the laser",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_butterfly.py iso_reverse=1",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/fc_apc_vs_pc_ferrules.jpg": {
    "alt": "Two FC patch-cord connectors lying on an anti-static mat with dust caps off: zirconia ferrule, plug-body barrel with its key tab, knurled coupling nut, crimp sleeve and a short slotted boot tapering onto the yellow jacket. The rear one with a green boot has an 8 degree angle-polished ferrule tip (APC), the front one with a blue boot a flat, slightly domed tip (PC)",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_butterfly.py view=ferrules",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/pm_panda_endface_0deg.jpg": {
    "alt": "Rendered illustration of a fiber inspection scope view of a PANDA polarization-maintaining fiber end face: the two stress rods sit above and below the core, in line with the key marker at twelve o'clock; inspection-zone rings overlaid",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_panda_endface.py variant=0",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/pm_panda_endface_90deg.jpg": {
    "alt": "Rendered illustration of the same inspection scope view of a second PANDA connector whose fiber is rotated 90 degrees: the stress rods sit at three and nine o'clock while the key marker is still at twelve o'clock",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_panda_endface.py variant=90",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/sin_microring_lensed_fiber.jpg": {
    "alt": "Rendered illustration of an optical micrograph of a silicon nitride add-drop ring: a lensed fiber tip meets the chip edge on a bus waveguide with an inverse taper; the bus and a drop waveguide pass within a micrometer of a 100 micrometer radius ring, 2.5 micrometers wide; 50 micrometer scale bar",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_sin_ring.py",
    "kind": "render",
    "w": 1920,
    "h": 1080
  },
  "media/render/butterfly_isolator_reversed_closeup.jpg": {
    "alt": "Close view of the inline fiber isolator from the reversed bench: the printed arrow on its label points back toward the butterfly laser mount at the right, the splice sleeves sit on both sides",
    "credit": "Rendered in Blender for this course",
    "license": "CC BY 4.0, project asset",
    "source": "course/tools/blender/shot_butterfly.py iso_reverse=1 view=iso_close",
    "kind": "render",
    "w": 1920,
    "h": 1080
  }
});
