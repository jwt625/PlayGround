/**
 * Simulation Engine API
 * Provides async interface to the NGSpice Web Worker
 */

import * as Comlink from 'comlink';
import type { SimulationResult, SimulationStatus, SimulationWorkerAPI } from './types';

export type { SimulationResult, SimulationStatus, RealResult, ComplexResult, RealDataType, ComplexDataType } from './types';

let workerInstance: Worker | null = null;
let apiInstance: Comlink.Remote<SimulationWorkerAPI> | null = null;

/**
 * Get or create the simulation worker instance
 */
function getWorker(): Comlink.Remote<SimulationWorkerAPI> {
	if (!apiInstance) {
		// Create the worker - using Vite's worker import syntax
		workerInstance = new Worker(
			new URL('./simulation.worker.ts', import.meta.url),
			{ type: 'module' }
		);
		apiInstance = Comlink.wrap<SimulationWorkerAPI>(workerInstance);
	}
	return apiInstance;
}

/**
 * Initialize the simulation engine
 * Must be called before running simulations
 */
export async function initSimulation(): Promise<void> {
	const api = getWorker();
	await api.init();
}

/**
 * Run a simulation with the given netlist
 * @param netlist - SPICE netlist string
 * @returns Simulation results
 */
export async function runSimulation(netlist: string): Promise<SimulationResult> {
	const api = getWorker();
	return await api.run(netlist);
}

/**
 * Get the current simulation status
 */
export async function getSimulationStatus(): Promise<SimulationStatus> {
	const api = getWorker();
	return await api.getStatus();
}

/**
 * Get any errors from the last simulation
 */
export async function getSimulationErrors(): Promise<string[]> {
	const api = getWorker();
	return await api.getErrors();
}

/**
 * Terminate the simulation worker
 * Call this when cleaning up
 */
export function terminateSimulation(): void {
	if (workerInstance) {
		workerInstance.terminate();
		workerInstance = null;
		apiInstance = null;
	}
}

