/**
 * Waveform viewer types
 */

export interface TraceData {
	id: string;
	name: string;
	type: 'voltage' | 'current' | 'time' | 'frequency' | 'notype';
	values: number[];
	color: TraceColor;
	visible: boolean;
	yScale: number;
	yOffset: number;
}

export interface TraceColor {
	r: number;
	g: number;
	b: number;
	a: number;
}

export interface ViewBounds {
	xMin: number;
	xMax: number;
	yMin: number;
	yMax: number;
}

export interface Cursor {
	id: 'A' | 'B';
	x: number;
	visible: boolean;
	color: string;
}

export interface CursorReadout {
	cursorA: { x: number; values: Map<string, number> } | null;
	cursorB: { x: number; values: Map<string, number> } | null;
	delta: { x: number; values: Map<string, number> } | null;
}

export interface WaveformState {
	traces: TraceData[];
	timeData: number[];
	bounds: ViewBounds;
	cursors: [Cursor, Cursor];
	selectedTraceId: string | null;
}

// Default trace colors (LTSpice-inspired)
export const TRACE_COLORS: TraceColor[] = [
	{ r: 0, g: 1, b: 0, a: 1 },       // Green
	{ r: 1, g: 0, b: 0, a: 1 },       // Red
	{ r: 0, g: 0.5, b: 1, a: 1 },     // Blue
	{ r: 1, g: 0, b: 1, a: 1 },       // Magenta
	{ r: 0, g: 1, b: 1, a: 1 },       // Cyan
	{ r: 1, g: 1, b: 0, a: 1 },       // Yellow
	{ r: 1, g: 0.5, b: 0, a: 1 },     // Orange
	{ r: 0.5, g: 1, b: 0, a: 1 },     // Lime
	{ r: 0.5, g: 0, b: 1, a: 1 },     // Purple
	{ r: 1, g: 0, b: 0.5, a: 1 },     // Pink
];

export function getTraceColor(index: number): TraceColor {
	return TRACE_COLORS[index % TRACE_COLORS.length];
}

