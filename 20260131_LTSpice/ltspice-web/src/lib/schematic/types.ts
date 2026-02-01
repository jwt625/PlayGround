/**
 * Schematic editor types
 */

/** 2D point in schematic coordinates */
export interface Point {
	x: number;
	y: number;
}

/** View transform state */
export interface ViewTransform {
	offsetX: number;  // Pan offset in screen pixels
	offsetY: number;
	scale: number;    // Zoom level (1 = 100%)
}

/** Grid settings */
export interface GridSettings {
	size: number;      // Grid spacing in schematic units (default: 10)
	snapEnabled: boolean;
	visible: boolean;
}

/**
 * Editor modes - mapped from LTSpice F-keys to number keys
 * F2 -> 2: Component (handled by component shortcuts)
 * F3 -> 3: Wire
 * F4 -> 4: Net label (future)
 * F5 -> 5: Delete
 * F6 -> 6: Duplicate
 * F7 -> 7: Move
 * F8 -> 8: Drag (move with wires, future)
 */
export type EditorMode =
	| 'select'    // Default mode - click to select
	| 'wire'      // 3 or W: Draw wires
	| 'delete'    // 5: Delete on click
	| 'duplicate' // 6: Duplicate selected items
	| 'move'      // 7: Move selected items
	| 'place';    // Placing a component

/** Mouse/interaction state */
export interface InteractionState {
	mode: EditorMode;
	isDragging: boolean;
	dragStart: Point | null;
	mousePos: Point;           // Screen coordinates
	schematicPos: Point;       // Schematic coordinates
	selectedIds: Set<string>;
}

/** Wire routing direction preference */
export type WireDirection = 'horizontal-first' | 'vertical-first';

/** Wire drawing state */
export interface WireDrawState {
	isDrawing: boolean;
	startPoint: Point | null;
	currentPoint: Point | null;
	direction: WireDirection;
	segments: Wire[];  // Completed segments in current wire chain
}

/** Component rotation (degrees, clockwise) */
export type Rotation = 0 | 90 | 180 | 270;

/** Component type identifiers */
export type ComponentType = 
	| 'resistor' | 'capacitor' | 'inductor'
	| 'voltage' | 'current' | 'ground'
	| 'diode' | 'npn' | 'pnp' | 'nmos' | 'pmos';

/** Pin on a component */
export interface Pin {
	id: string;
	x: number;  // Relative to component origin
	y: number;
	name: string;
}

/** Component instance in schematic */
export interface Component {
	id: string;
	type: ComponentType;
	x: number;
	y: number;
	rotation: Rotation;
	mirror: boolean;
	attributes: Record<string, string>;  // InstName, Value, etc.
	pins: Pin[];
}

/** Wire segment */
export interface Wire {
	id: string;
	x1: number;
	y1: number;
	x2: number;
	y2: number;
}

/**
 * Explicit junction - where wires are intentionally connected
 * Wires crossing visually are NOT connected unless a junction exists
 */
export interface Junction {
	id: string;
	x: number;
	y: number;
}

/** Complete schematic state */
export interface Schematic {
	components: Component[];
	wires: Wire[];
	junctions: Junction[];  // Explicit wire-to-wire connections
}

/** Default values */
export const DEFAULT_GRID: GridSettings = {
	size: 10,
	snapEnabled: true,
	visible: true
};

export const DEFAULT_VIEW: ViewTransform = {
	offsetX: 0,
	offsetY: 0,
	scale: 1
};

export const DEFAULT_INTERACTION: InteractionState = {
	mode: 'select',
	isDragging: false,
	dragStart: null,
	mousePos: { x: 0, y: 0 },
	schematicPos: { x: 0, y: 0 },
	selectedIds: new Set()
};

export const DEFAULT_WIRE_DRAW: WireDrawState = {
	isDrawing: false,
	startPoint: null,
	currentPoint: null,
	direction: 'horizontal-first',
	segments: []
};

/** Keyboard shortcut mappings (LTSpice F-keys -> number keys) */
export const MODE_SHORTCUTS: Record<string, EditorMode> = {
	'3': 'wire',      // F3 -> Wire mode
	'5': 'delete',    // F5 -> Delete mode
	'6': 'duplicate', // F6 -> Duplicate mode
	'7': 'move',      // F7 -> Move mode
};

/** Get mode display name */
export function getModeName(mode: EditorMode): string {
	switch (mode) {
		case 'select': return 'Select';
		case 'wire': return 'Wire (3)';
		case 'delete': return 'Delete (5)';
		case 'duplicate': return 'Duplicate (6)';
		case 'move': return 'Move (7)';
		case 'place': return 'Place';
		default: return mode;
	}
}

