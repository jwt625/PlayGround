/**
 * Component renderer - draws components on canvas
 */

import type { Component, Rotation } from './types';
import { COMPONENT_DEFS } from './component-defs';

/** Colors for rendering */
const COLORS = {
	component: '#00ff00',      // Green for components (LTSpice style)
	componentSelected: '#ffff00',  // Yellow when selected
	pin: '#ff0000',            // Red for pins
	label: '#ffffff',          // White for labels
	value: '#00ffff',          // Cyan for values
};

/**
 * Render a component on the canvas
 */
export function renderComponent(
	ctx: CanvasRenderingContext2D,
	comp: Component,
	scale: number,
	isSelected: boolean = false,
	isGhost: boolean = false  // For placement preview
): void {
	const def = COMPONENT_DEFS[comp.type];
	if (!def) return;

	ctx.save();
	
	// Move to component position
	ctx.translate(comp.x, comp.y);
	
	// Apply rotation
	ctx.rotate((comp.rotation * Math.PI) / 180);
	
	// Apply mirror
	if (comp.mirror) {
		ctx.scale(-1, 1);
	}

	// Set styles
	const color = isSelected ? COLORS.componentSelected : COLORS.component;
	ctx.strokeStyle = isGhost ? `${color}80` : color;
	ctx.fillStyle = isGhost ? `${color}80` : color;
	ctx.lineWidth = 2 / scale;
	ctx.lineCap = 'round';
	ctx.lineJoin = 'round';

	// Draw all paths
	for (const path of def.paths) {
		ctx.beginPath();
		for (const cmd of path) {
			drawCommand(ctx, cmd);
		}
		ctx.stroke();
	}

	// Draw pins (small circles)
	ctx.fillStyle = isGhost ? `${COLORS.pin}80` : COLORS.pin;
	for (const pin of def.pins) {
		ctx.beginPath();
		ctx.arc(pin.x, pin.y, 3 / scale, 0, Math.PI * 2);
		ctx.fill();
	}

	// Draw labels (only if not ghost)
	if (!isGhost) {
		const fontSize = 12 / scale;
		ctx.font = `${fontSize}px monospace`;
		ctx.textBaseline = 'middle';

		// Counter-rotate labels to keep them readable
		// First undo mirror, then undo rotation
		ctx.save();
		if (comp.mirror) {
			ctx.scale(-1, 1);
		}
		ctx.rotate((-comp.rotation * Math.PI) / 180);

		// Calculate label position in world space (after component transform)
		// We need to transform the offset points through the component transform
		const rad = (comp.rotation * Math.PI) / 180;
		const cos = Math.cos(rad);
		const sin = Math.sin(rad);

		// Transform label offset through rotation (and mirror)
		let lx = def.labelOffset.x;
		let ly = def.labelOffset.y;
		if (comp.mirror) lx = -lx;
		const labelX = lx * cos - ly * sin;
		const labelY = lx * sin + ly * cos;

		let vx = def.valueOffset.x;
		let vy = def.valueOffset.y;
		if (comp.mirror) vx = -vx;
		const valueX = vx * cos - vy * sin;
		const valueY = vx * sin + vy * cos;

		// Instance name
		const instName = comp.attributes['InstName'] || '';
		if (instName) {
			ctx.fillStyle = COLORS.label;
			ctx.textAlign = 'start';
			ctx.fillText(instName, labelX, labelY);
		}

		// Value
		const value = comp.attributes['Value'] || '';
		if (value) {
			ctx.fillStyle = COLORS.value;
			ctx.textAlign = 'start';
			ctx.fillText(value, valueX, valueY);
		}

		ctx.restore();
	}

	ctx.restore();
}

/**
 * Execute a single draw command
 */
function drawCommand(
	ctx: CanvasRenderingContext2D,
	cmd: Record<string, unknown>
): void {
	switch (cmd.type) {
		case 'M':
			ctx.moveTo(cmd.x as number, cmd.y as number);
			break;
		case 'L':
			ctx.lineTo(cmd.x as number, cmd.y as number);
			break;
		case 'A': {
			ctx.arc(
				cmd.cx as number,
				cmd.cy as number,
				cmd.r as number,
				((cmd.start as number) * Math.PI) / 180,
				((cmd.end as number) * Math.PI) / 180
			);
			break;
		}
		case 'C': {
			// Circle - draw as separate path
			const x = cmd.x as number;
			const y = cmd.y as number;
			const r = cmd.r as number;
			ctx.moveTo(x + r, y);
			ctx.arc(x, y, r, 0, Math.PI * 2);
			break;
		}
		case 'R': {
			ctx.rect(
				cmd.x as number,
				cmd.y as number,
				cmd.w as number,
				cmd.h as number
			);
			break;
		}
		case 'T': {
			// Text is handled separately
			break;
		}
	}
}

/**
 * Check if a point is inside a component's bounding box
 */
export function hitTestComponent(
	comp: Component,
	px: number,
	py: number,
	tolerance: number = 5
): boolean {
	const def = COMPONENT_DEFS[comp.type];
	if (!def) return false;

	// Transform point to component local coordinates
	let lx = px - comp.x;
	let ly = py - comp.y;

	// Reverse rotation
	const rad = (-comp.rotation * Math.PI) / 180;
	const cos = Math.cos(rad);
	const sin = Math.sin(rad);
	const rx = lx * cos - ly * sin;
	const ry = lx * sin + ly * cos;

	// Reverse mirror
	const fx = comp.mirror ? -rx : rx;

	// Check bounding box
	const hw = def.width / 2 + tolerance;
	const hh = def.height / 2 + tolerance;
	return Math.abs(fx) <= hw && Math.abs(ry) <= hh;
}

/**
 * Get the next rotation value
 */
export function nextRotation(current: Rotation): Rotation {
	const rotations: Rotation[] = [0, 90, 180, 270];
	const idx = rotations.indexOf(current);
	return rotations[(idx + 1) % 4];
}

