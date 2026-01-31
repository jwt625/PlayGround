<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { initSimulation, runSimulation, terminateSimulation, type SimulationResult } from '$lib/simulation';
	import { WaveformViewer, type TraceData, getTraceColor } from '$lib/waveform';

	let status = $state('Not initialized');
	let simResult = $state<SimulationResult | null>(null);
	let waveformTraces = $state<TraceData[]>([]);
	let timeData = $state<number[]>([]);
	let netlistInput = $state(`* Minimal RC Circuit Test
R1 in out 1k
C1 out 0 1u
Vin in 0 PULSE(0 5 0 1n 1n 0.5m 1m)
.tran 1u 5m
.end`);

	onMount(async () => {
		status = 'Initializing NGSpice...';
		try {
			await initSimulation();
			status = 'NGSpice ready';
		} catch (err) {
			status = `Init failed: ${err}`;
		}
	});

	onDestroy(() => {
		terminateSimulation();
	});

	async function runSim() {
		status = 'Running simulation...';
		simResult = null;
		waveformTraces = [];
		timeData = [];
		try {
			const result = await runSimulation(netlistInput);
			simResult = result;
			status = `Simulation complete: ${result.numPoints} points, ${result.numVariables} variables`;

			// Convert simulation result to waveform traces
			if (result.dataType === 'real') {
				const traces: TraceData[] = [];
				let colorIndex = 0;

				for (const data of result.data) {
					if (data.type === 'time') {
						// Time is the X axis
						timeData = data.values as number[];
					} else {
						// Other variables are traces
						traces.push({
							id: data.name,
							name: data.name,
							type: data.type,
							values: data.values as number[],
							color: getTraceColor(colorIndex++),
							visible: true,
							yScale: 1,
							yOffset: 0
						});
					}
				}
				waveformTraces = traces;
			}
		} catch (err) {
			status = `Simulation error: ${err}`;
		}
	}

	function handleKeyDown(e: KeyboardEvent) {
		// Ctrl+B to run simulation
		if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
			e.preventDefault();
			runSim();
		}
	}
</script>

<svelte:window onkeydown={handleKeyDown} />

<div class="app">
	<header class="toolbar">
		<span class="app-title">LTSpice Web</span>
		<button onclick={runSim} disabled={status.includes('Initializing') || status.includes('Running')}>
			Run Simulation (Ctrl+B)
		</button>
		{#if waveformTraces.length > 0}
			<span class="trace-count">{waveformTraces.length} traces</span>
		{/if}
	</header>
	<main class="workspace">
		<div class="left-panel">
			<div class="editor-panel">
				<h3>Netlist</h3>
				<textarea bind:value={netlistInput} spellcheck="false"></textarea>
			</div>
		</div>
		<div class="right-panel">
			<div class="waveform-panel">
				<h3>Waveform Viewer</h3>
				<div class="waveform-content">
					{#if waveformTraces.length > 0}
						<WaveformViewer traces={waveformTraces} {timeData} />
					{:else}
						<div class="placeholder-center">
							<p>Run a simulation to view waveforms</p>
							<p class="hint">Scroll to zoom, drag to pan, double-click to fit</p>
						</div>
					{/if}
				</div>
			</div>
			{#if simResult}
				<div class="info-panel">
					<h3>Simulation Info</h3>
					<div class="result-info">
						<p><strong>Variables:</strong> {simResult.variableNames.join(', ')}</p>
						<p><strong>Points:</strong> {simResult.numPoints}</p>
					</div>
				</div>
			{/if}
		</div>
	</main>
	<footer class="statusbar">
		<span>{status}</span>
	</footer>
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
		width: 100vw;
	}

	.toolbar {
		background: var(--toolbar-bg);
		border-bottom: 1px solid var(--border-primary);
		padding: var(--spacing-sm) var(--spacing-md);
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		height: 40px;
	}

	.toolbar button {
		background: var(--btn-primary-bg);
		color: var(--text-primary);
		border: none;
		padding: var(--spacing-xs) var(--spacing-md);
		border-radius: 3px;
		cursor: pointer;
		font-size: var(--font-size-sm);
	}

	.toolbar button:hover:not(:disabled) {
		background: var(--btn-primary-hover);
	}

	.toolbar button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.app-title {
		font-weight: 600;
		color: var(--text-primary);
	}

	.trace-count {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
	}

	.workspace {
		flex: 1;
		display: flex;
		flex-direction: row;
		overflow: hidden;
	}

	.left-panel {
		width: 350px;
		min-width: 250px;
		display: flex;
		flex-direction: column;
		border-right: 1px solid var(--border-primary);
	}

	.right-panel {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.editor-panel {
		flex: 1;
		display: flex;
		flex-direction: column;
		background: var(--bg-secondary);
	}

	.editor-panel h3,
	.waveform-panel h3,
	.info-panel h3 {
		padding: var(--spacing-sm) var(--spacing-md);
		margin: 0;
		font-size: var(--font-size-sm);
		background: var(--bg-tertiary);
		border-bottom: 1px solid var(--border-primary);
	}

	.editor-panel textarea {
		flex: 1;
		background: var(--bg-primary);
		color: var(--text-primary);
		border: none;
		padding: var(--spacing-md);
		font-family: var(--font-mono);
		font-size: var(--font-size-sm);
		resize: none;
	}

	.editor-panel textarea:focus {
		outline: none;
	}

	.waveform-panel {
		flex: 1;
		display: flex;
		flex-direction: column;
		background: var(--bg-secondary);
		min-height: 200px;
	}

	.waveform-content {
		flex: 1;
		position: relative;
		background: #000000;
	}

	.info-panel {
		height: 80px;
		background: var(--bg-secondary);
		border-top: 1px solid var(--border-primary);
	}

	.result-info {
		padding: var(--spacing-sm) var(--spacing-md);
		overflow: auto;
	}

	.result-info p {
		margin: var(--spacing-xs) 0;
		font-size: var(--font-size-xs);
		color: var(--text-secondary);
	}

	.placeholder-center {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		text-align: center;
		color: var(--text-muted);
	}

	.placeholder-center p {
		margin: var(--spacing-xs) 0;
	}

	.placeholder-center .hint {
		font-size: var(--font-size-xs);
		opacity: 0.7;
	}

	.statusbar {
		background: var(--statusbar-bg);
		color: var(--statusbar-text);
		padding: var(--spacing-xs) var(--spacing-md);
		font-size: var(--font-size-sm);
		height: 24px;
		display: flex;
		align-items: center;
	}
</style>
