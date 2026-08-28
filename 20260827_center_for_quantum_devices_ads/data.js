export const equipment = [
  {
    id: "uvisel-ellipsometer", name: "UVISEL spectroscopic ellipsometer", category: "Metrology", location: "Tech M264",
    manufacturer: "ISA / Jobin Yvon", model: "UVISEL", tag: "66186", purchaseDate: "1999-09-22", page: 39,
    summary: "Phase-modulated ellipsometer for film thickness and optical-constant measurements.",
    condition: "Condition not stated in the audit.", image: "p039-02-x274.jpg"
  },
  {
    id: "varian-fts40", name: "FTIR spectrometer", category: "Optical test", location: "Tech M264",
    manufacturer: "Varian", model: "FTS 40", tag: "70964", purchaseDate: "2005-05-09", page: 39,
    summary: "Michelson-interferometer system for infrared transmission and absorption spectra.",
    condition: "Condition not stated; manuals and parts were present.", image: "p040-02-x277.jpg"
  },
  {
    id: "oxford-ionfab", name: "Ion-beam sputtering system", category: "Deposition", location: "Tech M264",
    manufacturer: "Oxford Instruments", model: "Ionfab 300+", tag: "70333", purchaseDate: "2004-08-31", page: 40,
    summary: "PVD system used for Y₂O₃ facet coatings on quantum cascade lasers.",
    condition: "Audit notes pumps require oil and chillers require water before operation.", image: "p040-01-x279.jpg"
  },
  {
    id: "veeco-vs7700", name: "Thermal evaporator — indium", category: "Deposition", location: "Tech M264",
    manufacturer: "Veeco", model: "VS7700", tag: "59282", purchaseDate: "1994-08-01", page: 41,
    summary: "Thermal evaporation system normally used for indium deposition.",
    condition: "Cryopump reported faulty and needing replacement.", image: "p041-01-x283.jpg"
  },
  {
    id: "cha-sec600rap", name: "Thermal evaporator — contact metal", category: "Deposition", location: "Tech M264",
    manufacturer: "CHA", model: "SEC-600RAP", tag: "62775", purchaseDate: "1994-12-13", page: 41,
    summary: "Thermal evaporation system normally used for contact-metal deposition.",
    condition: "Cryopump reported faulty and needing replacement.", image: "p041-03-x282.jpg"
  },
  {
    id: "legacy-mocvd", name: "Legacy MOCVD system", category: "Epitaxy", location: "Tech M264",
    manufacturer: "Not recorded", model: "Not recorded", tag: "54222", purchaseDate: "Not recorded", page: 41,
    summary: "MOCVD system documented as having been brought from France to Northwestern in 1991.",
    condition: "Operational state is not documented; specialist review is recommended before deinstallation.",
    noteType: "Catalog safety note", image: "p041-02-x284.jpg"
  },
  {
    id: "guyson-multiblast", name: "Grit blaster", category: "Fabrication", location: "Tech M264",
    manufacturer: "Guyson", model: "Multiblast", tag: "71272", purchaseDate: "2005-08-08", page: 42,
    summary: "Pressurized abrasive-blasting cabinet for cleaning and surface preparation.",
    condition: "Condition not stated in the audit.", image: "p042-01-x287.jpg"
  },
  {
    id: "bwtek-yag", name: "Nd:YAG laser", category: "Optical test", location: "Tech M277",
    manufacturer: "B&W Tek", model: "BWC-FL", tag: "59274", purchaseDate: "2005-07-18", page: 42,
    summary: "Benchtop solid-state laser source listed in the detector-testing room.",
    condition: "The source also assigns tag 59274 to the EMS life-test system; asset identity requires owner verification.", image: "p043-01-x291.jpg"
  },
  {
    id: "ems-life-test", name: "Laser / LED life-test system", category: "Electrical test", location: "Tech M277",
    manufacturer: "EMS", model: "Not recorded", tag: "59274", purchaseDate: "Not recorded", page: 43,
    summary: "Extended device logging at controlled current, voltage, power, and temperature.",
    condition: "The source also assigns tag 59274 to the B&W Tek laser; asset identity requires owner verification.", image: "p043-02-x292.jpg"
  },
  {
    id: "janis-cryostat", name: "Optical cryostat", category: "Cryogenic", location: "Tech M277",
    manufacturer: "Janis Research / Newport / Leybold", model: "STVP-100 / RS2000 / Minitop", tag: "70963, 70607, 76059, 70678, 56085", purchaseDate: "Mixed", page: 43,
    summary: "Grouped cryogenic test setup; the audit lists a Janis cryostat, Newport RS2000, blackbody source, Leybold Minitop pump, and one unidentified tagged component.",
    condition: "Condition not stated in the audit.", image: "p045-01-x304.jpg"
  },
  {
    id: "bede-d1-xrd", name: "High-resolution X-ray system", category: "Metrology", location: "Tech M277",
    manufacturer: "Bede", model: "D1", tag: "65253", purchaseDate: "1998-03-16", page: 44,
    summary: "High-resolution XRD and metrology platform for semiconductor films and crystals.",
    condition: "Condition not stated in the audit.", image: "p044-01-x296.jpg"
  },
  {
    id: "biorad-dl4600", name: "DLTS system", category: "Electrical test", location: "Tech M277",
    manufacturer: "Bio-Rad", model: "DL4600", tag: "50401", purchaseDate: "1991-10-15", page: 44,
    summary: "Deep-level transient spectroscopy system for electrically active defects in semiconductors.",
    condition: "Condition not stated in the audit.", image: "p044-02-x298.jpg"
  },
  {
    id: "wire-bonders", name: "Wire bonder pair", category: "Packaging", location: "Tech M277 / M266",
    manufacturer: "Not recorded", model: "Not recorded", tag: "80493, 75520", purchaseDate: "Not recorded", page: 45,
    summary: "Gold-wire bonding equipment for semiconductor device test packaging.",
    condition: "Audit says both were tentatively moved near the M266 entrance.", image: "p045-02-x302.jpg"
  },
  {
    id: "rf-test-lot", name: "RF and impedance instrument lot", category: "Electrical test", location: "Tech M277",
    manufacturer: "Thorlabs / Agilent / HP / SRS / Keithley / mixed", model: "40 GHz E–O converter; E4407B; 4192A; SR770; 707A; switching matrix", tag: "83471, 80491, 60627, 72884, 70657, 70601", purchaseDate: "Mixed", page: 45,
    summary: "Spectrum, impedance, switching-matrix, and electro-optic test instruments.",
    condition: "Several items were shelf-located; working condition was not verified.", image: "p046-01-x309.jpg"
  },
  {
    id: "hydrogen-purifier", name: "Hydrogen purifier", category: "Facilities", location: "Tech M266",
    manufacturer: "Johnson Matthey", model: "HP-50", tag: "65149", purchaseDate: "Not recorded", page: 46,
    summary: "Hydrogen purification equipment associated with laboratory process utilities.",
    condition: "Specialist utility isolation and EHS review are recommended before removal.",
    noteType: "Catalog safety note", image: "p046-02-x307.jpg"
  },
  {
    id: "hecd-laser", name: "He-Cd laser", category: "Optical test", location: "Tech M252",
    manufacturer: "Not recorded", model: "Not recorded", tag: "65736", purchaseDate: "Not recorded", page: 47,
    summary: "Helium-cadmium laser documented under the east-side table in room M252.",
    condition: "Condition not stated in the audit.", image: "p047-01-x313.jpg"
  },
  {
    id: "headway-spinner", name: "Photoresist spinner", category: "Lithography", location: "Tech M252",
    manufacturer: "Headway Research", model: "PWM32", tag: "66290", purchaseDate: "Not recorded", page: 48,
    summary: "Benchtop spinner installed at the M252 wet bench.",
    condition: "Condition not stated; wet-bench cabinet maintenance was noted.", image: "p048-01-x320.jpg"
  },
  {
    id: "mikron-m305", name: "Blackbody calibration source", category: "Optical test", location: "Tech M252",
    manufacturer: "Mikron", model: "M305", tag: "60623", purchaseDate: "Not recorded", page: 48,
    summary: "Calibration source near an associated Janis liquid-nitrogen dewar.",
    condition: "Condition not stated in the audit.", image: "p048-03-x322.jpg"
  },
  {
    id: "oxford-pecvd", name: "PECVD system", category: "Deposition", location: "Tech M252 / M260",
    manufacturer: "Oxford Instruments", model: "80+", tag: "65743, 84000", purchaseDate: "Not recorded", page: 48,
    summary: "Plasma-enhanced CVD platform for SiO₂ and Si₃N₄ deposition.",
    condition: "Checked 2026-04-13; scrubber water supply and air valve faults are documented.", image: "p048-02-x321.jpg"
  },
  {
    id: "set-fc150-m252", name: "Automated flip-chip bonder", category: "Packaging", location: "Tech M252",
    manufacturer: "SET-Smart", model: "FC150", tag: "74931", purchaseDate: "Not recorded", page: 49,
    summary: "Automated device bonder used for detector packaging.",
    condition: "Audit describes it as well-functioning.", image: "p049-01-x326.jpg"
  },
  {
    id: "suss-maba6", name: "Mask aligner", category: "Lithography", location: "Tech M252",
    manufacturer: "SÜSS MicroTec", model: "MA/BA6", tag: "73558", purchaseDate: "Not recorded", page: 49,
    summary: "Contact/proximity photolithography mask aligner.",
    condition: "Condition not stated in the audit.", image: "p049-02-x325.jpg"
  },
  {
    id: "vistec-ebl", name: "Electron-beam lithography system", category: "Lithography", location: "Tech M252 / M260",
    manufacturer: "Leica / Vistec", model: "LION LV-1", tag: "65960", purchaseDate: "Not recorded", page: 50,
    summary: "Electron-beam writer with associated 15 kVA UPS, chiller, pumps, parts, and manuals.",
    condition: "Vacuum leak prevents high vacuum; audit says repair is required.", image: "p050-01-x329.jpg"
  },
  {
    id: "newport-holography", name: "Holographic lithography system", category: "Lithography", location: "Tech M252",
    manufacturer: "Newport", model: "Not recorded", tag: "70966", purchaseDate: "Not recorded", page: 50,
    summary: "Optical lithography setup listed alongside the e-beam writer.",
    condition: "Condition not stated in the audit.", image: "p050-02-x330.jpg"
  },
  {
    id: "mattson-galaxy", name: "Mattson FTIR", category: "Optical test", location: "Tech M252",
    manufacturer: "Mattson", model: "Galaxy 3000", tag: "53018", purchaseDate: "Not recorded", page: 51,
    summary: "Infrared Fourier-transform spectrometer listed in the cleanroom inventory.",
    condition: "Condition not stated in the audit; no unambiguous equipment photograph was identified.", image: null
  },
  {
    id: "metallurgical-microscope", name: "Metallurgical microscope", category: "Metrology", location: "Tech M252",
    manufacturer: "Not recorded", model: "ML7530", tag: "66497", purchaseDate: "Not recorded", page: 51,
    summary: "Metallurgical microscope listed in the M252 cleanroom inventory.",
    condition: "Condition not stated in the audit; no unambiguous equipment photograph was identified.", image: null
  },
  {
    id: "m260-pecvd-support", name: "PECVD scrubber and support pumps", category: "Facilities", location: "Tech M260",
    manufacturer: "CDO / Leybold / mixed", model: "Scrubber / D40DCS", tag: "65999, 78770", purchaseDate: "Mixed", page: 51,
    summary: "Dedicated scrubber and vacuum-pump support equipment serving the M252 PECVD system.",
    condition: "Audit reports a scrubber water-supply problem and an air-operated valve requiring replacement.", image: "p051-03-x335.jpg"
  },
  {
    id: "m260-nitrogen-generator", name: "Nitrogen generator", category: "Facilities", location: "Tech M260",
    manufacturer: "Not recorded", model: "Not recorded", tag: "Not recorded", purchaseDate: "Not recorded", page: 52,
    summary: "Nitrogen-generation equipment documented with the PECVD support utilities.",
    condition: "Condition not stated in the audit.", image: "p052-02-x341.jpg"
  },
  {
    id: "ebl-ups", name: "15 kVA EBL UPS", category: "Electrical test", location: "Tech M260",
    manufacturer: "APC", model: "15 kVA, 208 V", tag: "76770", purchaseDate: "Not recorded", page: 52,
    summary: "Uninterruptible power supply associated with the M252 electron-beam lithography system.",
    condition: "Condition not stated in the audit.", image: "p052-02-x341.jpg"
  },

  /*
   * NOT FOR SALE — intentionally excluded from the exported inventory.
   * Preserve these records and their extracted assets for traceability, but do
   * not move them above this comment without explicit disposition approval.
   * The exclusion starts with the probe-station microscope and includes every
   * subsequent listing in the original catalog order.
   */
  /*
  {
    id: "karl-suss-prober", name: "Probe-station microscope", category: "Electrical test", location: "Cook 04008",
    manufacturer: "Karl Süss", model: "10577065", tag: "067236", purchaseDate: "2000-12-10", page: 54,
    summary: "Manual probe-station microscope for semiconductor wafers and substrates.",
    condition: "Condition not stated in the audit.", image: "p055-01-x356.jpg"
  },
  {
    id: "ks-7100ad", name: "Wafer dicing system", category: "Fabrication", location: "Cook 04008",
    manufacturer: "K&S", model: "7100AD", tag: "068086", purchaseDate: "2002-01-16", page: 55,
    summary: "Dicing saw for partitioning processed wafers into die.",
    condition: "Condition not stated in the audit.", image: "p055-02-x357.jpg"
  },
  {
    id: "logitech-pm2a", name: "Wafer polishing system", category: "Fabrication", location: "Cook 04008",
    manufacturer: "Logitech", model: "PM2A", tag: "66031", purchaseDate: "1992-03-12", page: 55,
    summary: "Wafer polishing and thinning setup with pump and workstation.",
    condition: "Condition not stated; consumables require separate EHS handling.", image: "p055-03-x358.jpg"
  },
  {
    id: "ibond5000", name: "Ball bonder", category: "Packaging", location: "Cook 04008",
    manufacturer: "Micro Point Pro", model: "iBond5000-Ball", tag: "88711", purchaseDate: "Not recorded", page: 56,
    summary: "Ball-bonding system for semiconductor device packaging.",
    condition: "Condition not stated in the audit.", image: "p056-01-x363.jpg"
  },
  {
    id: "leybold-pt-flex", name: "Cryopump pumping station", category: "Vacuum", location: "Cook 04008",
    manufacturer: "Oerlikon Leybold", model: "PT Flex 70", tag: "74779", purchaseDate: "2008-11-24", page: 56,
    summary: "Mobile pumping station for establishing high vacuum in cryopumps.",
    condition: "Condition not stated in the audit.", image: "p056-02-x362.jpg"
  },
  {
    id: "bruker-ifs66vs", name: "FTIR spectrometer", category: "Optical test", location: "Cook 04008",
    manufacturer: "Bruker", model: "IFS 66v/S", tag: "71865", purchaseDate: "2006-02-19", page: 56,
    summary: "Vacuum FTIR with configurable mid-IR and THz optical paths.",
    condition: "Pump-oil maintenance is noted; overall condition not verified.", image: "p060-03-x380.jpg"
  },
  {
    id: "seir-9705", name: "Infrared dewar", category: "Cryogenic", location: "Cook 04008",
    manufacturer: "SE-IR Corp.", model: "9705", tag: "72494", purchaseDate: "2006-10-20", page: 57,
    summary: "Detector dewar associated with infrared spectral measurements.",
    condition: "Condition not stated in the audit.", image: "p057-01-x366.jpg"
  },
  {
    id: "loomis-lsd100", name: "Wafer scriber / cleaver", category: "Fabrication", location: "Cook 04008",
    manufacturer: "Loomis", model: "LSD-100", tag: "73310", purchaseDate: "2007-07-11", page: 57,
    summary: "Precision scribing and cleaving machine for semiconductor samples.",
    condition: "Condition not stated in the audit.", image: "p057-02-x368.jpg"
  },
  {
    id: "electrophysics-pv320", name: "Infrared camera", category: "Optical test", location: "Cook 04008",
    manufacturer: "Electrophysics", model: "PV320L2", tag: "71043", purchaseDate: "2005-06-16", page: 58,
    summary: "Infrared camera system for device and optical characterization.",
    condition: "Condition not stated in the audit.", image: "p058-03-x372.jpg"
  },
  {
    id: "seir-camera", name: "SE-IR camera system", category: "Optical test", location: "Cook 04008",
    manufacturer: "SE-IR", model: "Not recorded", serial: "4757", tag: "75127", purchaseDate: "2009-05-07", page: 58,
    summary: "Infrared camera and test system.",
    condition: "Condition not stated in the audit.", image: "p058-01-x373.jpg"
  },
  {
    id: "keithley-2410", name: "SourceMeter", category: "Electrical test", location: "Cook 04008",
    manufacturer: "Keithley", model: "2410", tag: "72647", purchaseDate: "2006-12-28", page: 59,
    summary: "High-voltage source-measure unit for semiconductor characterization.",
    condition: "Condition not stated in the audit.", image: "p059-01-x376.jpg"
  },
  {
    id: "oxford-icp100", name: "ICP etching system", category: "Etch", location: "Cook 04012",
    manufacturer: "Oxford Instruments", model: "Plasmalab 100", tag: "73223", purchaseDate: "2007-06-04", page: 61,
    summary: "Inductively coupled plasma etcher for high-aspect-ratio semiconductor features.",
    condition: "Condition not stated; includes Ebara A70W pump.", image: "p061-01-x390.jpg"
  },
  {
    id: "solid-source-mbe", name: "Solid-source MBE reactor", category: "Epitaxy", location: "Cook 04012",
    manufacturer: "Not recorded", model: "Not recorded", tag: "Not recorded", purchaseDate: "Not recorded", page: 61,
    summary: "Molecular-beam epitaxy reactor used for type-II superlattice photodetector growth.",
    condition: "Operational state not documented; specialist decommissioning required.", image: "p061-02-x391.jpg"
  },
  {
    id: "philips-hrxrd", name: "Philips high-resolution XRD", category: "Metrology", location: "Cook 04012",
    manufacturer: "Philips / PANalytical / Spellman", model: "Not recorded", tag: "50589 system; 30983 PANalytical subsystem; 83022 Spellman generator", purchaseDate: "Mixed", page: 61,
    summary: "2 kW Cu Kα HR-XRD configuration with four-bounce Ge monochromator, precision goniometer, and scintillation detector.",
    condition: "X-ray generator reported faulty and requiring repair.", image: "p061-04-x388.jpg"
  },
  {
    id: "digital-nanoscope", name: "Atomic force microscope", category: "Metrology", location: "Cook 04014",
    manufacturer: "Digital Instruments / Bruker", model: "Nanoscope", tag: "65398", purchaseDate: "1998-06-07", page: 62,
    summary: "Scanning-probe microscope for nanoscale 3D surface topography.",
    condition: "Condition not stated in the audit.", image: "p062-01-x397.jpg"
  },
  {
    id: "nanometrics-ecv", name: "Electrochemical C–V profiler", category: "Metrology", location: "Cook 04014",
    manufacturer: "Nanometrics", model: "ECVPro", tag: "75659", purchaseDate: "2009-12-01", page: 62,
    summary: "Automated carrier-concentration depth profiling for epitaxial semiconductor layers.",
    condition: "Condition not stated in the audit.", image: "p062-03-x399.jpg"
  },
  {
    id: "zygo-newview", name: "Optical profilometer", category: "Metrology", location: "Cook 04014",
    manufacturer: "Zygo", model: "NewView 7300", tag: "75511", purchaseDate: "2009-10-23", page: 63,
    summary: "White-light optical profilometer for 3D wafer surface maps.",
    condition: "Condition not stated in the audit.", image: "p063-01-x403.jpg"
  },
  {
    id: "biorad-hl5500", name: "Hall measurement system", category: "Electrical test", location: "Cook 04014",
    manufacturer: "Bio-Rad", model: "HL5500M", tag: "50406", purchaseDate: "1991-10-15", page: 63,
    summary: "Van der Pauw resistivity, carrier-concentration, and mobility measurements.",
    condition: "Condition not stated in the audit.", image: "p063-02-x404.jpg"
  },
  {
    id: "oxford-inca-sem", name: "SEM / EDS system", category: "Metrology", location: "Cook 04014",
    manufacturer: "Oxford Instruments", model: "INCA 350 X3", tag: "74044", purchaseDate: "2008-04-14", page: 64,
    summary: "Scanning-electron microscopy / microanalysis system listed in the characterization lab.",
    condition: "Condition not stated in the audit.", image: "p064-01-x409.jpg"
  },
  {
    id: "uv-pl", name: "UV photoluminescence system", category: "Optical test", location: "Cook 04014",
    manufacturer: "CVI Melles Griot / Newport / EG&G / GAM Laser", model: "45-LRS-303 / RS1000 / EX5/200", tag: "74198, 59215, 60625, 69014", purchaseDate: "Mixed", page: 64,
    summary: "Photoluminescence setup configured for III-nitride materials.",
    condition: "Condition not stated in the audit.", image: "p064-03-x408.jpg"
  },
  {
    id: "ir-pl", name: "IR photoluminescence system", category: "Optical test", location: "Cook 04014",
    manufacturer: "Mixed", model: "Custom optical bench", tag: "Multiple", purchaseDate: "Mixed", page: 65,
    summary: "Custom optical setup configured for infrared semiconductor materials.",
    condition: "Subsystem-level inventory; working condition not verified.", image: "p065-02-x412.jpg"
  },
  {
    id: "cha-ebevap", name: "Electron-beam evaporator", category: "Deposition", location: "Cook 04016",
    manufacturer: "CHA", model: "SEC-600", tag: "00057827", purchaseDate: "1993-06-09", page: 66,
    summary: "Metal evaporator outfitted for Au, Pt, Ti, Ni, and AuGe deposition.",
    condition: "Condition not stated; source material and accessories require separate review.", image: "p066-01-x417.jpg"
  },
  {
    id: "aixtron-1225", name: "Aixtron MOCVD reactor", category: "Epitaxy", location: "Cook 04016",
    manufacturer: "Aixtron", model: "System 1225", tag: "00057196", purchaseDate: "1994-01-31", page: 67,
    summary: "Early horizontal-flow commercial MOCVD reactor used for III-nitride growth.",
    condition: "Operational state not documented; specialist decommissioning required.", image: "p068-02-x424.jpg"
  },
  {
    id: "emcore-discovery", name: "Emcore MOCVD reactor", category: "Epitaxy", location: "Cook 04074",
    manufacturer: "Emcore", model: "Discovery 125", tag: "00062539", purchaseDate: "1995-05-02", page: 69,
    summary: "Compound-semiconductor MOCVD reactor with seven associated backend units.",
    condition: "Electrically on at audit; pumps and computers shut down. Water-bath servicing noted.", image: "p069-02-x430.jpg"
  },
  {
    id: "eferel-mocvd", name: "Eferel MOCVD reactor", category: "Epitaxy", location: "Cook 04074",
    manufacturer: "Eferel", model: "Custom", tag: "00054151", purchaseDate: "1992-08-12", page: 69,
    summary: "Custom reactor used for Fe-doped InP regrowth for quantum cascade lasers.",
    condition: "Requires cooling-water oversight; specialist decommissioning required.", image: "p069-03-x431.jpg"
  },
  {
    id: "karl-suss-mjb3", name: "Mask aligner and spinner bay", category: "Lithography", location: "Cook 04074A",
    manufacturer: "Karl Süss / Headway", model: "MJB 3 / 1EC101DR48", tag: "52488, 52674", purchaseDate: "1992-08", page: 70,
    summary: "Class-1000 yellow-room mask aligner with adjacent spinner and hotplates.",
    condition: "Room hoods were certified 2026-04-23; tool function not independently verified.", image: "p070-04-x437.jpg"
  },
  {
    id: "gas-source-mbe", name: "Gas-source MBE reactor", category: "Epitaxy", location: "Cook 04078 / 04078A",
    manufacturer: "Intevac / Veeco / EPI", model: "MOMBE", tag: "51436", purchaseDate: "1992-03-16", page: 75,
    summary: "Multi-vendor gas-source molecular-beam epitaxy system used for quantum cascade lasers.",
    condition: "Chase pressure and mechanical-pump monitoring were active audit responsibilities.", image: "p075-02-x454.jpg"
  },
  {
    id: "set-fc150-4078", name: "High-precision die bonder", category: "Packaging", location: "Cook 04078",
    manufacturer: "SET-Smart", model: "FC150", tag: "74930", purchaseDate: "2009-02-10", page: 75,
    summary: "Flip-chip / die bonder for laser placement accuracy and planarity.",
    condition: "Condition not stated in the audit.", image: "p076-02-x459.jpg"
  },
  {
    id: "plasmatherm-ecr", name: "ECR dry-etch system", category: "Etch", location: "Cook 04078",
    manufacturer: "Plasma-Therm", model: "SLR-770ECR", tag: "00064441", purchaseDate: "1996-11-05", page: 76,
    summary: "Electron-cyclotron-resonance plasma etcher for semiconductor device processing.",
    condition: "Condition not stated; spare and replacement parts are documented nearby.", image: "p076-04-x460.jpg"
  },
  {
    id: "cdo-scrubber", name: "CDO toxic-gas scrubber", category: "Facilities", location: "Cook penthouse",
    manufacturer: "Not recorded", model: "CDO", tag: "Not recorded", purchaseDate: "Not recorded", page: 80,
    summary: "Controlled decomposition oxidation unit serving MBE and MOCVD reactor effluent.",
    condition: "Facility infrastructure; transfer/disposition requires Northwestern EHS and licensed contractors.", noteType: "Catalog safety note", image: "p080-05-x489.jpg"
  },
  {
    id: "nitrogen-generator", name: "Nitrogen generator", category: "Facilities", location: "Cook penthouse",
    manufacturer: "Not recorded", model: "Not recorded", tag: "Not recorded", purchaseDate: "Not recorded", page: 80,
    summary: "Central nitrogen supply equipment for CQD process systems.",
    condition: "Facility infrastructure; working condition not verified.", image: "p080-06-x491.jpg"
  },
  {
    id: "penthouse-air-compressor", name: "Air compressor", category: "Facilities", location: "Cook penthouse",
    manufacturer: "Atlas Copco", model: "Not recorded", tag: "Not recorded", purchaseDate: "Not recorded", page: 80,
    summary: "Central compressed-air supply for process valves and equipment chambers.",
    condition: "Facility infrastructure; working condition not verified.", image: "p080-07-x493.jpg"
  },
  {
    id: "hydrogen-generator", name: "Hydrogen generator", category: "Facilities", location: "Cook penthouse",
    manufacturer: "Proton OnSite", model: "Not recorded", specification: "200 L/day", tag: "Not recorded", purchaseDate: "Not recorded", page: 80,
    summary: "Central hydrogen generator supporting MBE and MOCVD reactors.",
    condition: "Hazardous facility infrastructure; institutional review and licensed removal required.", noteType: "Catalog safety note", image: "p080-04-x496.jpg"
  },
  {
    id: "di-water", name: "Deionized-water plant", category: "Facilities", location: "Cook penthouse",
    manufacturer: "Not recorded", model: "Not recorded", specification: "18 MΩ water system", tag: "Not recorded", purchaseDate: "Not recorded", page: 80,
    summary: "Mixed-bed ion exchange, filtration, and UV sterilization for cleanroom DI water.",
    condition: "Facility infrastructure; working condition not verified.", image: "p081-02-x500.jpg"
  },
  {
    id: "ln2-tank", name: "3,000-gallon liquid-nitrogen tank", category: "Facilities", location: "Exterior / Cook service lines",
    manufacturer: "Not recorded", model: "Not recorded", specification: "3,000 gallon", tag: "Not recorded", purchaseDate: "Not recorded", page: 84,
    summary: "Bulk LN₂ storage vessel and distribution infrastructure dedicated to CQD.",
    condition: "Fixed cryogenic infrastructure; institutional ownership and code-compliant disposition must be confirmed.", noteType: "Catalog safety note", image: "p084-01-x510.jpg"
  }
  */
];

export const categories = ["All", ...new Set(equipment.map((item) => item.category))];
