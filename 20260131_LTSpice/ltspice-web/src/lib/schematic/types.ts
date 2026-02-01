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

/** Mouse/interaction state */
export interface InteractionState {
	mode: 'select' | 'pan' | 'wire' | 'place';
	isDragging: boolean;
	dragStart: Point | null;
	mousePos: Point;           // Screen coordinates
	schematicPos: Point;       // Schematic coordinates
	selectedIds: Set<string>;
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

/** Complete schematic state */
export interface Schematic {
	components: Component[];
	wires: Wire[];
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

