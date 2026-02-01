/**
 * SPICE netlist generator
 * Converts schematic to SPICE netlist text
 */

import type { Schematic, Component } from '../schematic/types';
import type { GeneratedNetlist, SpiceComponent, PinConnection } from './types';
import { COMPONENT_PREFIX, DEFAULT_VALUES, PIN_ORDER } from './types';
import { analyzeConnectivity } from './connectivity';

/** Generate SPICE netlist from schematic */
export function generateNetlist(schematic: Schematic, title: string = 'Untitled Circuit'): GeneratedNetlist {
	const result: GeneratedNetlist = {
		title,
		components: [],
		directives: [],
		errors: [],
		warnings: []
	};

	// Analyze connectivity
	const connectivity = analyzeConnectivity(schematic);

	// Add connectivity errors/warnings
	for (const err of connectivity.errors) {
		if (err.startsWith('Warning:')) {
			result.warnings.push(err);
		} else {
			result.errors.push(err);
		}
	}

	// Build a map of component pins to net names
	const pinToNet = new Map<string, string>();
	for (const pc of connectivity.pinConnections) {
		const key = `${pc.componentId}:${pc.pinName}`;
		const net = connectivity.nets.find(n => n.id === pc.netId);
		pinToNet.set(key, net?.name || '?');
	}

	// Generate SPICE components
	for (const comp of schematic.components) {
		// Skip ground - it's not a component, it defines node 0
		if (comp.type === 'ground') continue;

		const spiceComp = componentToSpice(comp, pinToNet);
		if (spiceComp) {
			result.components.push(spiceComp);
		} else {
			result.errors.push(`Failed to generate SPICE for ${comp.attributes['InstName'] || comp.id}`);
		}
	}

	// Note: eecircuit-engine doesn't support .save directive, so we calculate
	// resistor/capacitor currents client-side from node voltages instead.
	// See current-calculator.ts for the implementation.

	// Add default simulation directive (.tran is always needed for transient analysis)
	result.directives.push('.tran 1u 10m');
	result.directives.push('.end');

	return result;
}

/** Convert a component to SPICE format */
function componentToSpice(comp: Component, pinToNet: Map<string, string>): SpiceComponent | null {
	const prefix = COMPONENT_PREFIX[comp.type];
	if (prefix === undefined || prefix === '') return null;
	
	const instName = comp.attributes['InstName'] || `${prefix}?`;
	const value = comp.attributes['Value'] || DEFAULT_VALUES[comp.type] || '';
	
	// Get node names in correct order
	const pinOrder = PIN_ORDER[comp.type];
	if (!pinOrder) return null;
	
	const nodes: string[] = [];
	for (const pinName of pinOrder) {
		const key = `${comp.id}:${pinName}`;
		const netName = pinToNet.get(key);
		if (netName) {
			nodes.push(netName);
		} else {
			nodes.push('?');
		}
	}
	
	return {
		prefix,
		name: instName,
		nodes,
		value
	};
}

/** Convert generated netlist to text */
export function netlistToText(netlist: GeneratedNetlist): string {
	const lines: string[] = [];
	
	// Title line (first line is always the title in SPICE)
	lines.push(`* ${netlist.title}`);
	lines.push('');
	
	// Add warnings as comments
	for (const warn of netlist.warnings) {
		lines.push(`* ${warn}`);
	}
	if (netlist.warnings.length > 0) lines.push('');
	
	// Add errors as comments
	for (const err of netlist.errors) {
		lines.push(`* ERROR: ${err}`);
	}
	if (netlist.errors.length > 0) lines.push('');
	
	// Components
	for (const comp of netlist.components) {
		const nodePart = comp.nodes.join(' ');
		let line = `${comp.name} ${nodePart} ${comp.value}`;
		if (comp.extra) {
			line += ` ${comp.extra}`;
		}
		lines.push(line);
	}
	
	if (netlist.components.length > 0) lines.push('');
	
	// Directives
	for (const dir of netlist.directives) {
		lines.push(dir);
	}
	
	return lines.join('\n');
}

/** Generate netlist text directly from schematic */
export function schematicToNetlist(schematic: Schematic, title?: string): string {
	const netlist = generateNetlist(schematic, title);
	return netlistToText(netlist);
}

/** Generate node labels for display on schematic */
export function generateNodeLabels(schematic: Schematic): import('../schematic/types').NodeLabel[] {
	const connectivity = analyzeConnectivity(schematic);
	const labels: import('../schematic/types').NodeLabel[] = [];

	for (const net of connectivity.nets) {
		if (net.points.length === 0) continue;

		// Find a good position for the label - prefer wire endpoints
		// Use the first point as a simple heuristic
		const pos = net.points[0];

		labels.push({
			name: net.name,
			x: pos.x,
			y: pos.y,
			isGround: net.isGround
		});
	}

	return labels;
}
