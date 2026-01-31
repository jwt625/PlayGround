/** Simulation result types */

export interface ComplexNumber { real: number; img: number; }

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

interface BaseResult {
	header: string;
	numVariables: number;
	variableNames: string[];
	numPoints: number;
}

export interface RealResult extends BaseResult { dataType: 'real'; data: RealDataType[]; }
export interface ComplexResult extends BaseResult { dataType: 'complex'; data: ComplexDataType[]; }
export type SimulationResult = RealResult | ComplexResult;

export interface SimulationStatus { initialized: boolean; running: boolean; error: string | null; }

export interface SimulationWorkerAPI {
	init(): Promise<void>;
	run(netlist: string): Promise<SimulationResult>;
	getStatus(): SimulationStatus;
	getErrors(): string[];
}

