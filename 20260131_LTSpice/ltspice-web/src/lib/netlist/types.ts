/**
 * Netlist generation types
 */

import type { Component, Wire, Junction, Point } from '../schematic/types';

/** A net is a set of electrically connected points */
export interface Net {
	id: string;
	name: string;           // Node name (e.g., "0" for ground, "1", "2", or named like "in", "out")
	points: Point[];        // All points belonging to this net
	isGround: boolean;      // True if this net is connected to ground
}

/** Pin connection info */
export interface PinConnection {
	componentId: string;
	componentName: string;  // Instance name (e.g., "R1")
	pinName: string;        // Pin name (e.g., "1", "2", "+", "-", "B", "C", "E")
	position: Point;        // Absolute position in schematic
	netId: string | null;   // Which net this pin is connected to
}

/** Result of connectivity analysis */
export interface ConnectivityResult {
	nets: Net[];
	pinConnections: PinConnection[];
	errors: string[];       // Floating pins, unconnected components, etc.
}

/** SPICE component line */
export interface SpiceComponent {
	prefix: string;         // R, C, L, V, I, D, Q, M
	name: string;           // Full name (e.g., R1)
	nodes: string[];        // Node names in order
	value: string;          // Component value or model
	extra?: string;         // Additional parameters
}

/** Generated netlist */
export interface GeneratedNetlist {
	title: string;
	components: SpiceComponent[];
	directives: string[];   // .tran, .ac, etc.
	errors: string[];
	warnings: string[];
}

/** Component type to SPICE prefix mapping */
export const COMPONENT_PREFIX: Record<string, string> = {
	resistor: 'R',
	capacitor: 'C',
	inductor: 'L',
	voltage: 'V',
	current: 'I',
	diode: 'D',
	npn: 'Q',
	pnp: 'Q',
	nmos: 'M',
	pmos: 'M',
	ground: '',  // Ground is not a component, it defines node 0
};

/** Default component values */
export const DEFAULT_VALUES: Record<string, string> = {
	resistor: '1k',
	capacitor: '1u',
	inductor: '1m',
	voltage: 'DC 5',
	current: 'DC 1m',
	diode: 'D',
	npn: 'NPN',
	pnp: 'PNP',
	nmos: 'NMOS',
	pmos: 'PMOS',
};

/** Pin order for SPICE netlist (component type -> pin names in order) */
export const PIN_ORDER: Record<string, string[]> = {
	resistor: ['1', '2'],
	capacitor: ['1', '2'],
	inductor: ['1', '2'],
	voltage: ['+', '-'],
	current: ['+', '-'],
	diode: ['A', 'K'],       // Anode, Cathode (NGSpice: D1 anode cathode model)
	npn: ['C', 'B', 'E'],    // Collector, Base, Emitter
	pnp: ['C', 'B', 'E'],
	nmos: ['D', 'G', 'S'],   // Drain, Gate, Source (+ Bulk, often tied to Source)
	pmos: ['D', 'G', 'S'],
};

