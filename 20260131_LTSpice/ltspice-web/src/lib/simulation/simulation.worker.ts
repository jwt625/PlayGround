/**
 * Web Worker for running NGSpice simulations
 * Uses eecircuit-engine (NGSpice compiled to WASM)
 */

import { Simulation } from 'eecircuit-engine';
import * as Comlink from 'comlink';
import type { SimulationResult, SimulationStatus, SimulationWorkerAPI } from './types';

class SimulationWorker implements SimulationWorkerAPI {
	private sim: Simulation | null = null;
	private initialized = false;
	private running = false;
	private lastError: string | null = null;
	private errors: string[] = [];

	async init(): Promise<void> {
		if (this.initialized) {
			return;
		}

		try {
			this.sim = new Simulation();
			await this.sim.start();
			this.initialized = true;
			this.lastError = null;
			console.log('[SimWorker] NGSpice initialized');
			console.log('[SimWorker] Init info:', this.sim.getInitInfo());
		} catch (err) {
			this.lastError = err instanceof Error ? err.message : String(err);
			throw err;
		}
	}

	async run(netlist: string): Promise<SimulationResult> {
		if (!this.sim || !this.initialized) {
			throw new Error('Simulation engine not initialized. Call init() first.');
		}

		if (this.running) {
			throw new Error('Simulation already running.');
		}

		this.running = true;
		this.errors = [];
		this.lastError = null;

		try {
			console.log('[SimWorker] Setting netlist...');
			this.sim.setNetList(netlist);

			console.log('[SimWorker] Running simulation...');
			const result = await this.sim.runSim();

			// Check for errors
			this.errors = this.sim.getError();
			if (this.errors.length > 0) {
				console.warn('[SimWorker] Simulation warnings/errors:', this.errors);
			}

			console.log('[SimWorker] Simulation complete:', {
				numVariables: result.numVariables,
				numPoints: result.numPoints,
				variableNames: result.variableNames
			});

			return result as SimulationResult;
		} catch (err) {
			this.lastError = err instanceof Error ? err.message : String(err);
			this.errors = this.sim.getError();
			throw err;
		} finally {
			this.running = false;
		}
	}

	getStatus(): SimulationStatus {
		return {
			initialized: this.initialized,
			running: this.running,
			error: this.lastError
		};
	}

	getErrors(): string[] {
		return this.errors;
	}
}

// Expose the worker API via Comlink
Comlink.expose(new SimulationWorker());

