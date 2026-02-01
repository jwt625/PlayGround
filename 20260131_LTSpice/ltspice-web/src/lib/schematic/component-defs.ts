/**
 * Component visual definitions for schematic rendering
 * All coordinates are in schematic units (grid = 10 units)
 * Origin (0,0) is the component center
 */

import type { ComponentType, Pin } from './types';

/** Drawing command types */
type DrawCmd = 
	| { type: 'M'; x: number; y: number }      // Move to
	| { type: 'L'; x: number; y: number }      // Line to
	| { type: 'A'; cx: number; cy: number; r: number; start: number; end: number }  // Arc
	| { type: 'C'; x: number; y: number; r: number }  // Circle
	| { type: 'R'; x: number; y: number; w: number; h: number }  // Rectangle
	| { type: 'T'; x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end' };  // Text

/** Component definition */
export interface ComponentDef {
	type: ComponentType;
	name: string;
	shortcut: string;  // Keyboard shortcut
	paths: DrawCmd[][];  // Multiple paths for different stroke styles
	pins: Omit<Pin, 'id'>[];  // Pin positions relative to origin
	labelOffset: { x: number; y: number };  // Where to draw the label
	valueOffset: { x: number; y: number };  // Where to draw the value
	width: number;   // Bounding box
	height: number;
}

// Resistor: zigzag pattern
const RESISTOR: ComponentDef = {
	type: 'resistor',
	name: 'Resistor',
	shortcut: 'r',
	paths: [[
		{ type: 'M', x: 0, y: -30 },
		{ type: 'L', x: 0, y: -20 },
		{ type: 'L', x: 8, y: -16 },
		{ type: 'L', x: -8, y: -8 },
		{ type: 'L', x: 8, y: 0 },
		{ type: 'L', x: -8, y: 8 },
		{ type: 'L', x: 8, y: 16 },
		{ type: 'L', x: 0, y: 20 },
		{ type: 'L', x: 0, y: 30 },
	]],
	pins: [
		{ x: 0, y: -30, name: '1' },
		{ x: 0, y: 30, name: '2' }
	],
	labelOffset: { x: 12, y: -10 },
	valueOffset: { x: 12, y: 5 },
	width: 20,
	height: 60
};

// Capacitor: two parallel lines
const CAPACITOR: ComponentDef = {
	type: 'capacitor',
	name: 'Capacitor',
	shortcut: 'c',
	paths: [[
		{ type: 'M', x: 0, y: -30 },
		{ type: 'L', x: 0, y: -5 },
		{ type: 'M', x: -12, y: -5 },
		{ type: 'L', x: 12, y: -5 },
		{ type: 'M', x: -12, y: 5 },
		{ type: 'L', x: 12, y: 5 },
		{ type: 'M', x: 0, y: 5 },
		{ type: 'L', x: 0, y: 30 },
	]],
	pins: [
		{ x: 0, y: -30, name: '1' },
		{ x: 0, y: 30, name: '2' }
	],
	labelOffset: { x: 15, y: -10 },
	valueOffset: { x: 15, y: 5 },
	width: 24,
	height: 60
};

// Inductor: coil pattern
const INDUCTOR: ComponentDef = {
	type: 'inductor',
	name: 'Inductor',
	shortcut: 'l',
	paths: [[
		{ type: 'M', x: 0, y: -30 },
		{ type: 'L', x: 0, y: -20 },
		{ type: 'A', cx: 0, cy: -15, r: 5, start: -90, end: 180 },
		{ type: 'A', cx: 0, cy: -5, r: 5, start: -90, end: 180 },
		{ type: 'A', cx: 0, cy: 5, r: 5, start: -90, end: 180 },
		{ type: 'A', cx: 0, cy: 15, r: 5, start: -90, end: 180 },
		{ type: 'M', x: 0, y: 20 },
		{ type: 'L', x: 0, y: 30 },
	]],
	pins: [
		{ x: 0, y: -30, name: '1' },
		{ x: 0, y: 30, name: '2' }
	],
	labelOffset: { x: 12, y: -10 },
	valueOffset: { x: 12, y: 5 },
	width: 16,
	height: 60
};

// Ground: standard ground symbol
const GROUND: ComponentDef = {
	type: 'ground',
	name: 'Ground',
	shortcut: 'g',
	paths: [[
		{ type: 'M', x: 0, y: -10 },
		{ type: 'L', x: 0, y: 0 },
		{ type: 'M', x: -15, y: 0 },
		{ type: 'L', x: 15, y: 0 },
		{ type: 'M', x: -10, y: 5 },
		{ type: 'L', x: 10, y: 5 },
		{ type: 'M', x: -5, y: 10 },
		{ type: 'L', x: 5, y: 10 },
	]],
	pins: [
		{ x: 0, y: -10, name: '0' }
	],
	labelOffset: { x: 0, y: 20 },
	valueOffset: { x: 0, y: 20 },
	width: 30,
	height: 20
};

// Voltage source: circle with + and -
const VOLTAGE: ComponentDef = {
	type: 'voltage',
	name: 'Voltage Source',
	shortcut: 'v',
	paths: [[
		{ type: 'M', x: 0, y: -40 },
		{ type: 'L', x: 0, y: -20 },
		{ type: 'C', x: 0, y: 0, r: 20 },
		{ type: 'M', x: 0, y: 20 },
		{ type: 'L', x: 0, y: 40 },
	], [
		// + sign (top)
		{ type: 'M', x: -5, y: -10 },
		{ type: 'L', x: 5, y: -10 },
		{ type: 'M', x: 0, y: -15 },
		{ type: 'L', x: 0, y: -5 },
		// - sign (bottom)
		{ type: 'M', x: -5, y: 10 },
		{ type: 'L', x: 5, y: 10 },
	]],
	pins: [
		{ x: 0, y: -40, name: '+' },
		{ x: 0, y: 40, name: '-' }
	],
	labelOffset: { x: 25, y: -10 },
	valueOffset: { x: 25, y: 5 },
	width: 40,
	height: 80
};

// Current source: circle with arrow
const CURRENT: ComponentDef = {
	type: 'current',
	name: 'Current Source',
	shortcut: 'i',
	paths: [[
		{ type: 'M', x: 0, y: -40 },
		{ type: 'L', x: 0, y: -20 },
		{ type: 'C', x: 0, y: 0, r: 20 },
		{ type: 'M', x: 0, y: 20 },
		{ type: 'L', x: 0, y: 40 },
	], [
		// Arrow pointing up
		{ type: 'M', x: 0, y: 12 },
		{ type: 'L', x: 0, y: -12 },
		{ type: 'M', x: -5, y: -7 },
		{ type: 'L', x: 0, y: -12 },
		{ type: 'L', x: 5, y: -7 },
	]],
	pins: [
		{ x: 0, y: -40, name: '+' },
		{ x: 0, y: 40, name: '-' }
	],
	labelOffset: { x: 25, y: -10 },
	valueOffset: { x: 25, y: 5 },
	width: 40,
	height: 80
};

// Diode: triangle with bar
const DIODE: ComponentDef = {
	type: 'diode',
	name: 'Diode',
	shortcut: 'd',
	paths: [[
		{ type: 'M', x: 0, y: -30 },
		{ type: 'L', x: 0, y: -10 },
		{ type: 'M', x: -12, y: -10 },
		{ type: 'L', x: 12, y: -10 },
		{ type: 'L', x: 0, y: 10 },
		{ type: 'L', x: -12, y: -10 },
		{ type: 'M', x: -12, y: 10 },
		{ type: 'L', x: 12, y: 10 },
		{ type: 'M', x: 0, y: 10 },
		{ type: 'L', x: 0, y: 30 },
	]],
	pins: [
		{ x: 0, y: -30, name: 'A' },
		{ x: 0, y: 30, name: 'K' }
	],
	labelOffset: { x: 15, y: -10 },
	valueOffset: { x: 15, y: 5 },
	width: 24,
	height: 60
};

// NPN transistor
const NPN: ComponentDef = {
	type: 'npn',
	name: 'NPN BJT',
	shortcut: 'q',
	paths: [[
		// Base line
		{ type: 'M', x: -30, y: 0 },
		{ type: 'L', x: -10, y: 0 },
		// Vertical bar
		{ type: 'M', x: -10, y: -15 },
		{ type: 'L', x: -10, y: 15 },
		// Collector
		{ type: 'M', x: -10, y: -8 },
		{ type: 'L', x: 10, y: -20 },
		{ type: 'L', x: 10, y: -30 },
		// Emitter with arrow
		{ type: 'M', x: -10, y: 8 },
		{ type: 'L', x: 10, y: 20 },
		{ type: 'L', x: 10, y: 30 },
	], [
		// Arrow on emitter
		{ type: 'M', x: 4, y: 12 },
		{ type: 'L', x: 10, y: 20 },
		{ type: 'L', x: 2, y: 18 },
	]],
	pins: [
		{ x: -30, y: 0, name: 'B' },
		{ x: 10, y: -30, name: 'C' },
		{ x: 10, y: 30, name: 'E' }
	],
	labelOffset: { x: 15, y: -25 },
	valueOffset: { x: 15, y: 25 },
	width: 40,
	height: 60
};

// PNP transistor (arrow reversed)
const PNP: ComponentDef = {
	type: 'pnp',
	name: 'PNP BJT',
	shortcut: 'Q',
	paths: [[
		{ type: 'M', x: -30, y: 0 },
		{ type: 'L', x: -10, y: 0 },
		{ type: 'M', x: -10, y: -15 },
		{ type: 'L', x: -10, y: 15 },
		{ type: 'M', x: -10, y: -8 },
		{ type: 'L', x: 10, y: -20 },
		{ type: 'L', x: 10, y: -30 },
		{ type: 'M', x: -10, y: 8 },
		{ type: 'L', x: 10, y: 20 },
		{ type: 'L', x: 10, y: 30 },
	], [
		// Arrow pointing toward base
		{ type: 'M', x: -4, y: 12 },
		{ type: 'L', x: -10, y: 8 },
		{ type: 'L', x: -2, y: 6 },
	]],
	pins: [
		{ x: -30, y: 0, name: 'B' },
		{ x: 10, y: -30, name: 'C' },
		{ x: 10, y: 30, name: 'E' }
	],
	labelOffset: { x: 15, y: -25 },
	valueOffset: { x: 15, y: 25 },
	width: 40,
	height: 60
};

// NMOS transistor
const NMOS: ComponentDef = {
	type: 'nmos',
	name: 'NMOS',
	shortcut: 'm',
	paths: [[
		// Gate
		{ type: 'M', x: -30, y: 0 },
		{ type: 'L', x: -15, y: 0 },
		{ type: 'M', x: -15, y: -15 },
		{ type: 'L', x: -15, y: 15 },
		// Channel
		{ type: 'M', x: -10, y: -15 },
		{ type: 'L', x: -10, y: 15 },
		// Drain
		{ type: 'M', x: -10, y: -10 },
		{ type: 'L', x: 10, y: -10 },
		{ type: 'L', x: 10, y: -30 },
		// Source
		{ type: 'M', x: -10, y: 10 },
		{ type: 'L', x: 10, y: 10 },
		{ type: 'L', x: 10, y: 30 },
		// Body connection
		{ type: 'M', x: -10, y: 0 },
		{ type: 'L', x: 10, y: 0 },
		{ type: 'L', x: 10, y: 10 },
	], [
		// Arrow on body
		{ type: 'M', x: 4, y: 0 },
		{ type: 'L', x: 10, y: 0 },
		{ type: 'M', x: 6, y: -3 },
		{ type: 'L', x: 10, y: 0 },
		{ type: 'L', x: 6, y: 3 },
	]],
	pins: [
		{ x: -30, y: 0, name: 'G' },
		{ x: 10, y: -30, name: 'D' },
		{ x: 10, y: 30, name: 'S' }
	],
	labelOffset: { x: 15, y: -25 },
	valueOffset: { x: 15, y: 25 },
	width: 40,
	height: 60
};

// PMOS transistor
const PMOS: ComponentDef = {
	type: 'pmos',
	name: 'PMOS',
	shortcut: 'M',
	paths: [[
		{ type: 'M', x: -30, y: 0 },
		{ type: 'L', x: -20, y: 0 },
		{ type: 'C', x: -17, y: 0, r: 3 },  // Circle on gate
		{ type: 'M', x: -14, y: -15 },
		{ type: 'L', x: -14, y: 15 },
		{ type: 'M', x: -10, y: -15 },
		{ type: 'L', x: -10, y: 15 },
		{ type: 'M', x: -10, y: -10 },
		{ type: 'L', x: 10, y: -10 },
		{ type: 'L', x: 10, y: -30 },
		{ type: 'M', x: -10, y: 10 },
		{ type: 'L', x: 10, y: 10 },
		{ type: 'L', x: 10, y: 30 },
		{ type: 'M', x: -10, y: 0 },
		{ type: 'L', x: 10, y: 0 },
		{ type: 'L', x: 10, y: 10 },
	], [
		// Arrow pointing out
		{ type: 'M', x: -4, y: 0 },
		{ type: 'L', x: -10, y: 0 },
		{ type: 'M', x: -6, y: -3 },
		{ type: 'L', x: -10, y: 0 },
		{ type: 'L', x: -6, y: 3 },
	]],
	pins: [
		{ x: -30, y: 0, name: 'G' },
		{ x: 10, y: -30, name: 'D' },
		{ x: 10, y: 30, name: 'S' }
	],
	labelOffset: { x: 15, y: -25 },
	valueOffset: { x: 15, y: 25 },
	width: 40,
	height: 60
};

// Export all component definitions
export const COMPONENT_DEFS: Record<ComponentType, ComponentDef> = {
	resistor: RESISTOR,
	capacitor: CAPACITOR,
	inductor: INDUCTOR,
	ground: GROUND,
	voltage: VOLTAGE,
	current: CURRENT,
	diode: DIODE,
	npn: NPN,
	pnp: PNP,
	nmos: NMOS,
	pmos: PMOS,
};

// Get component def by shortcut key
export function getComponentByShortcut(key: string): ComponentDef | null {
	const lowerKey = key.toLowerCase();
	for (const def of Object.values(COMPONENT_DEFS)) {
		if (def.shortcut === lowerKey) return def;
	}
	return null;
}

