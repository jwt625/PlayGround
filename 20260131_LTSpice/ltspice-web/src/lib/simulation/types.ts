/**
 * Simulation result types
 */

export interface ComplexNumber {
	real: number;
	img: number;
}

export interface RealDataType {
	name: string;
	type: 'voltage' | 'current' | 'time' | 'frequency' | 'notype';
	values: number[];
}

export interface ComplexDataType {
	name: string;
	type: 'voltage' | 'current' | 'time' | 'frequency' | 'notype';
	values: ComplexNumber[];
}

export interface RealResult {
	header: string;
	numVariables: number;
	variableNames: string[];
	numPoints: number;
	dataType: 'real';
	data: RealDataType[];
}

export interface ComplexResult {
	header: string;
	numVariables: number;
	variableNames: string[];
	numPoints: number;
	dataType: 'complex';
	data: ComplexDataType[];
}

export type SimulationResult = RealResult | ComplexResult;

export interface SimulationError {
	message: string;
	errors: string[];
}

export interface SimulationStatus {
	initialized: boolean;
	running: boolean;
	error: string | null;
}

/**
 * API for the simulation worker
 */
export interface SimulationWorkerAPI {
	init(): Promise<void>;
	run(netlist: string): Promise<SimulationResult>;
	getStatus(): SimulationStatus;
	getErrors(): string[];
}

