export interface PyramidItem {
  id: string;
  name: string;
  description: string;
  image: string;
  tier: 'additive' | 'local' | 'bigfab';
  // Absolute positioning coordinates (percentages)
  top: number;
  left: number;
  width: number;
  height: number;
  zIndex: number;
  entryOrder: number;
}

export const pyramidItems: PyramidItem[] = [
  // TIER 1: ADDITIVE CORE TECHNOLOGIES (Items 1-13)
  // Top-left wedge (0-50% width, 0-50% height)
  { id: 'photoresist', name: 'Photoresist Bottle', description: 'High-purity photoresist material for lithography', image: '/images/pyramid/photoresist.webp', tier: 'additive', top: 10, left: 35, width: 11, height: 11, zIndex: 20, entryOrder: 1 },
  { id: 'direct-write', name: 'Direct-Write Tool', description: 'Direct-write lithography or inkjet printing system', image: '/images/pyramid/direct-write-tool.webp', tier: 'additive', top: 5, left: 22, width: 11, height: 11, zIndex: 19, entryOrder: 2 },
  { id: 'lpbf', name: 'LPBF Metal Printer', description: 'Laser Powder Bed Fusion for metal parts', image: '/images/pyramid/lpbf-metal-printer.webp', tier: 'additive', top: 20, left: 15, width: 11, height: 11, zIndex: 18, entryOrder: 3 },
  { id: 'tpp', name: 'TPP System', description: 'Two-Photon Polymerization for micro/nano structures', image: '/images/pyramid/tpp-system.webp', tier: 'additive', top: 28, left: 25, width: 11, height: 11, zIndex: 17, entryOrder: 4 },
  { id: 'metal-powder', name: 'Metal Powder Container', description: 'Fine metal powder for LPBF printing', image: '/images/pyramid/metal-powder.webp', tier: 'additive', top: 25, left: 5, width: 11, height: 11, zIndex: 16, entryOrder: 5 },
  { id: 'resin-vat', name: 'Resin Vat', description: 'UV-curable resin for SLA/DLP printing', image: '/images/pyramid/resin-vat.webp', tier: 'additive', top: 35, left: 32, width: 11, height: 11, zIndex: 15, entryOrder: 6 },
  { id: 'di-water', name: 'DI Water Bottle', description: 'Deionized water for cleanroom use', image: '/images/pyramid/di-water.webp', tier: 'additive', top: 15, left: 45, width: 11, height: 11, zIndex: 14, entryOrder: 7 },
  { id: 'aerosol-jet', name: 'Aerosol Jet Printer', description: 'Aerosol jet printing for electronics', image: '/images/pyramid/aerosol-jet-printer.webp', tier: 'additive', top: 12, left: 10, width: 11, height: 11, zIndex: 13, entryOrder: 8 },
  { id: 'filament', name: 'Filament Spools', description: 'Conductive 3D printing filament', image: '/images/pyramid/filament-spools.webp', tier: 'additive', top: 38, left: 12, width: 11, height: 11, zIndex: 12, entryOrder: 9 },
  { id: 'conductive-ink', name: 'Conductive Ink', description: 'Silver/copper ink for printed electronics', image: '/images/pyramid/conductive-ink.webp', tier: 'additive', top: 30, left: 40, width: 11, height: 11, zIndex: 11, entryOrder: 10 },
  { id: 'solder-paste', name: 'Solder Paste Syringe', description: 'Solder paste for assembly', image: '/images/pyramid/solder-paste.webp', tier: 'additive', top: 42, left: 20, width: 11, height: 11, zIndex: 10, entryOrder: 11 },
  { id: 'silicon-boule', name: 'Silicon Boule', description: 'Raw silicon crystal before wafer slicing', image: '/images/pyramid/silicon-boule.webp', tier: 'additive', top: 5, left: 38, width: 11, height: 11, zIndex: 9, entryOrder: 12 },
  { id: 'smd', name: 'SMD Components', description: 'Surface-mount components', image: '/images/pyramid/smd-components.webp', tier: 'additive', top: 18, left: 2, width: 11, height: 11, zIndex: 8, entryOrder: 13 },

  // TIER 2: LOCAL & ACCESSIBLE (Items 14-26)
  // Top-right wedge (50-100% width, 0-50% height)
  { id: 'pcb-mill', name: 'Desktop PCB Mill', description: 'Benchtop CNC mill for circuit boards', image: '/images/pyramid/desktop-pcb-mill.webp', tier: 'local', top: 12, left: 60, width: 11, height: 11, zIndex: 20, entryOrder: 14 },
  { id: 'sem', name: 'Benchtop SEM', description: 'Desktop scanning electron microscope', image: '/images/pyramid/benchtop-sem.webp', tier: 'local', top: 18, left: 75, width: 11, height: 11, zIndex: 19, entryOrder: 15 },
  { id: 'oscilloscope', name: 'Oscilloscope', description: 'Benchtop oscilloscope', image: '/images/pyramid/oscilloscope.webp', tier: 'local', top: 28, left: 65, width: 11, height: 11, zIndex: 18, entryOrder: 16 },
  { id: 'reflow', name: 'Reflow Oven', description: 'Small reflow oven for PCB assembly', image: '/images/pyramid/reflow-oven.webp', tier: 'local', top: 20, left: 85, width: 11, height: 11, zIndex: 17, entryOrder: 17 },
  { id: 'multimeter', name: 'Multimeter', description: 'Digital multimeter', image: '/images/pyramid/multimeter.webp', tier: 'local', top: 35, left: 55, width: 11, height: 11, zIndex: 16, entryOrder: 18 },
  { id: 'pick-place', name: 'Pick-and-Place', description: 'Desktop pick-and-place machine', image: '/images/pyramid/pick-and-place.webp', tier: 'local', top: 30, left: 88, width: 11, height: 11, zIndex: 15, entryOrder: 19 },
  { id: 'function-gen', name: 'Function Generator', description: 'Signal generator for testing', image: '/images/pyramid/function-generator.webp', tier: 'local', top: 15, left: 55, width: 11, height: 11, zIndex: 14, entryOrder: 20 },
  { id: 'power-supply', name: 'Power Supply', description: 'Benchtop DC power supply', image: '/images/pyramid/power-supply.webp', tier: 'local', top: 10, left: 70, width: 11, height: 11, zIndex: 13, entryOrder: 21 },
  { id: 'logic-analyzer', name: 'Logic Analyzer', description: 'USB logic analyzer', image: '/images/pyramid/logic-analyzer.webp', tier: 'local', top: 38, left: 78, width: 11, height: 11, zIndex: 12, entryOrder: 22 },
  { id: 'soldering', name: 'Soldering Station', description: 'Quality soldering station', image: '/images/pyramid/soldering-station.webp', tier: 'local', top: 5, left: 82, width: 11, height: 11, zIndex: 11, entryOrder: 23 },
  { id: 'hot-air', name: 'Hot Air Rework', description: 'Hot air rework station', image: '/images/pyramid/hot-air-rework.webp', tier: 'local', top: 42, left: 62, width: 11, height: 11, zIndex: 10, entryOrder: 24 },
  { id: 'tweezers', name: 'Tweezers Set', description: 'Precision tweezers for SMD', image: '/images/pyramid/tweezers-set.webp', tier: 'local', top: 8, left: 52, width: 11, height: 11, zIndex: 9, entryOrder: 25 },
  { id: 'hand-tools', name: 'Hand Tools', description: 'Wire cutters, pliers, screwdrivers', image: '/images/pyramid/hand-tools-set.webp', tier: 'local', top: 40, left: 90, width: 11, height: 11, zIndex: 8, entryOrder: 26 },

  // TIER 3: BIG FAB (Items 27-38)
  // Bottom-center wedge (25-75% width, 50-100% height)
  { id: 'euv', name: 'ASML EUV Machine', description: 'Extreme ultraviolet lithography system', image: '/images/pyramid/asml-euv.webp', tier: 'bigfab', top: 55, left: 42, width: 11, height: 11, zIndex: 10, entryOrder: 27 },
  { id: 'cleanroom', name: 'Cleanroom Worker', description: 'Gatekept, inaccessible process', image: '/images/pyramid/cleanroom-worker.webp', tier: 'bigfab', top: 65, left: 32, width: 11, height: 11, zIndex: 9, entryOrder: 28 },
  { id: 'tsmc-fab', name: 'TSMC Fab Exterior', description: 'Modern semiconductor fab facility', image: '/images/pyramid/tsmc-fab-exterior.webp', tier: 'bigfab', top: 65, left: 54, width: 11, height: 11, zIndex: 9, entryOrder: 29 },
  { id: 'wafer-fab', name: 'Wafer Fab Interior', description: 'Traditional fab with massive equipment', image: '/images/pyramid/wafer-fab-interior.webp', tier: 'bigfab', top: 75, left: 45, width: 11, height: 11, zIndex: 8, entryOrder: 30 },
  { id: 'slm', name: 'SLM Machine', description: 'Selective Laser Melting system', image: '/images/pyramid/slm-machine.webp', tier: 'bigfab', top: 72, left: 35, width: 11, height: 11, zIndex: 7, entryOrder: 31 },
  { id: 'nanoscribe', name: 'Nanoscribe Tool', description: 'Commercial TPP system for 3D nanoprinting', image: '/images/pyramid/nanoscribe-tool.webp', tier: 'bigfab', top: 72, left: 55, width: 11, height: 11, zIndex: 7, entryOrder: 32 },
  { id: 'fdm-printer', name: '3D Printer (FDM)', description: 'Desktop FDM 3D printer', image: '/images/pyramid/fdm-3d-printer.webp', tier: 'bigfab', top: 58, left: 65, width: 11, height: 11, zIndex: 6, entryOrder: 33 },
  { id: 'laser-cutter', name: 'Laser Cutter', description: 'Desktop laser cutter', image: '/images/pyramid/laser-cutter.webp', tier: 'bigfab', top: 82, left: 40, width: 11, height: 11, zIndex: 5, entryOrder: 34 },
  { id: 'fume-extractor', name: 'Fume Extractor', description: 'Soldering fume extractor', image: '/images/pyramid/fume-extractor.webp', tier: 'bigfab', top: 82, left: 50, width: 11, height: 11, zIndex: 5, entryOrder: 35 },
  { id: 'storage', name: 'Component Storage', description: 'Organized component storage', image: '/images/pyramid/component-storage.webp', tier: 'bigfab', top: 88, left: 45, width: 11, height: 11, zIndex: 4, entryOrder: 36 },
  { id: 'copper-wire', name: 'Copper Wire Spool', description: 'Fine copper wire for bonding', image: '/images/pyramid/copper-wire-spool.webp', tier: 'bigfab', top: 60, left: 28, width: 11, height: 11, zIndex: 3, entryOrder: 37 },
  { id: 'gold-wire', name: 'Gold Wire Spool', description: 'Gold bonding wire for interconnects', image: '/images/pyramid/gold-wire-spool.webp', tier: 'bigfab', top: 85, left: 30, width: 11, height: 11, zIndex: 2, entryOrder: 38 },
];

export const tierLabels = {
  additive: 'Additive Core Technologies',
  local: 'Local & Accessible',
  bigfab: 'Big Fab (Minimize)',
};

export const tierDescriptions = {
  additive: 'Advanced materials and additive manufacturing equipment',
  local: 'Desktop tools and test equipment accessible to everyone',
  bigfab: 'Traditional centralized fab infrastructure - use sparingly',
};

