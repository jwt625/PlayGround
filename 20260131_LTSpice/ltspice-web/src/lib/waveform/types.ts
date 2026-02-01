/** Waveform viewer types */

export interface TraceData {
	id: string;
	name: string;
	type: 'voltage' | 'current' | 'time' | 'frequency' | 'notype';
	values: number[];
	color: TraceColor;
	visible: boolean;
}

export interface TraceColor {
	r: number; g: number; b: number; a: number;
}

export interface ViewBounds {
	xMin: number; xMax: number; yMin: number; yMax: number;
}

export interface Cursor {
	id: 'A' | 'B';
	x: number;
	visible: boolean;
	color: string;
}

/** Waveform tab - contains traces for one panel */
export interface WaveformTab {
	id: string;
	name: string;
	traces: TraceData[];
}

const TRACE_COLORS: TraceColor[] = [
	{ r: 0, g: 1, b: 0, a: 1 },
	{ r: 1, g: 0, b: 0, a: 1 },
	{ r: 0, g: 0.5, b: 1, a: 1 },
	{ r: 1, g: 0, b: 1, a: 1 },
	{ r: 0, g: 1, b: 1, a: 1 },
	{ r: 1, g: 1, b: 0, a: 1 },
	{ r: 1, g: 0.5, b: 0, a: 1 },
	{ r: 0.5, g: 1, b: 0, a: 1 },
];

export function getTraceColor(index: number): TraceColor {
	return TRACE_COLORS[index % TRACE_COLORS.length];
}

