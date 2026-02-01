<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { initSimulation, runSimulation, terminateSimulation, type SimulationResult } from '$lib/simulation';
	import { WaveformViewer, type TraceData, getTraceColor } from '$lib/waveform';
	import { NetlistEditor } from '$lib/editor';
	import { SchematicCanvas, type Schematic, type Probe } from '$lib/schematic';
	import { ResizablePanel } from '$lib/components';
	import { schematicToNetlist, generateNodeLabels } from '$lib/netlist';

	let status = $state('Not initialized');
	let simResult = $state<SimulationResult | null>(null);
	let waveformTraces = $state<TraceData[]>([]);
	let timeData = $state<number[]>([]);
	let schematic = $state<Schematic>({ components: [], wires: [], junctions: [] });
	let probes = $state<Probe[]>([]);

	let netlistCollapsed = $state(false);
	let schematicCollapsed = $state(false);
	let waveformCollapsed = $state(false);

	// Calculate initial panel sizes based on viewport
	// Schematic: 1/2 of total height, Waveform: 1/3 of total height, Info: 1/6 of total height
	const toolbarHeight = 40;
	const statusbarHeight = 24;

	function getInitialSizes() {
		if (typeof window === 'undefined') return { schematic: 400, waveform: 250, info: 100 };
		const availableHeight = window.innerHeight - toolbarHeight - statusbarHeight;
		// Schematic: 1/2, Waveform: 1/3, Info: 1/6
		const schematicHeight = Math.round(availableHeight / 2);
		const waveformHeight = Math.round(availableHeight / 3);
		const infoHeight = Math.round(availableHeight / 6);
		return { schematic: schematicHeight, waveform: waveformHeight, info: infoHeight };
	}

	let initialSizes = $state(getInitialSizes());
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
			updateWaveformFromResult(result);
		} catch (err) {
			status = `Simulation error: ${err}`;
		}
	}

	/** Update waveform traces from simulation result, filtered by probes */
	function updateWaveformFromResult(result: SimulationResult) {
		if (result.dataType !== 'real') return;

		const traces: TraceData[] = [];
		let colorIndex = 0;

		for (const data of result.data) {
			if (data.type === 'time') {
				timeData = data.values as number[];
			} else {
				// Check if this trace matches any probe
				const matchesProbe = probes.length === 0 || probes.some(probe => {
					const dataNameLower = data.name.toLowerCase();
					if (probe.type === 'voltage') {
						// Match v(node) format
						return dataNameLower === `v(${probe.node1.toLowerCase()})`;
					} else if (probe.type === 'voltage-diff') {
						// Match v(node1,node2) or v(node1)-v(node2)
						return dataNameLower === `v(${probe.node1.toLowerCase()},${probe.node2?.toLowerCase()})` ||
							   dataNameLower === `v(${probe.node1.toLowerCase()})` ||
							   dataNameLower === `v(${probe.node2?.toLowerCase()})`;
					} else if (probe.type === 'current') {
						// Match i(component) format
						return dataNameLower.includes(probe.node1.toLowerCase());
					}
					return false;
				});

				if (matchesProbe) {
					traces.push({
						id: data.name,
						name: data.name,
						type: data.type,
						values: data.values as number[],
						color: getTraceColor(colorIndex++),
						visible: true
					});
				}
			}
		}
		waveformTraces = traces;
	}

	/** Handle probe event from schematic canvas */
	function handleProbe(event: { type: string; node1: string; node2?: string; componentId?: string; label: string }) {
		const { type, node1, node2, componentId, label } = event;

		// Check if probe already exists
		const existingIndex = probes.findIndex(p => p.label === label);
		if (existingIndex >= 0) {
			// Remove existing probe (toggle off)
			probes = probes.filter((_, i) => i !== existingIndex);
			status = `Removed probe: ${label}`;
		} else {
			// Add new probe
			const newProbe: Probe = {
				id: crypto.randomUUID(),
				type: type as Probe['type'],
				node1,
				node2,
				componentId,
				label
			};
			probes = [...probes, newProbe];
			status = `Added probe: ${label}`;
		}

		// Update waveform if we have simulation results
		if (simResult) {
			updateWaveformFromResult(simResult);
		}
	}

	function generateNetlistFromSchematic() {
		if (schematic.components.length === 0) {
			status = 'No components in schematic';
			return;
		}
		const netlistText = schematicToNetlist(schematic, 'Generated from Schematic');
		netlistInput = netlistText;

		// Generate and attach node labels for display on schematic
		schematic.nodeLabels = generateNodeLabels(schematic);

		status = `Generated netlist: ${schematic.components.length} components, ${schematic.wires.length} wires, ${schematic.nodeLabels.length} nodes`;
	}

	function handleKeyDown(e: KeyboardEvent) {
		// Ctrl+B to run simulation
		if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
			e.preventDefault();
			runSim();
		}
		// Ctrl+N to generate netlist from schematic
		if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
			e.preventDefault();
			generateNetlistFromSchematic();
		}
		// Ctrl+S to save schematic
		if ((e.ctrlKey || e.metaKey) && e.key === 's') {
			e.preventDefault();
			saveSchematic();
		}
		// Ctrl+O to open schematic
		if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
			e.preventDefault();
			openSchematicDialog();
		}
	}

	/** Save schematic to JSON file */
	function saveSchematic() {
		if (schematic.components.length === 0 && schematic.wires.length === 0) {
			status = 'Nothing to save';
			return;
		}

		// Create a clean copy without nodeLabels (they're regenerated)
		const saveData = {
			version: 1,
			schematic: {
				components: schematic.components,
				wires: schematic.wires,
				junctions: schematic.junctions
			},
			netlist: netlistInput,
			savedAt: new Date().toISOString()
		};

		const json = JSON.stringify(saveData, null, 2);
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);

		const a = document.createElement('a');
		a.href = url;
		a.download = `schematic-${Date.now()}.json`;
		a.click();

		URL.revokeObjectURL(url);
		status = `Saved schematic: ${schematic.components.length} components, ${schematic.wires.length} wires`;
	}

	/** Open file dialog to load schematic */
	function openSchematicDialog() {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json';
		input.onchange = (e) => {
			const file = (e.target as HTMLInputElement).files?.[0];
			if (file) {
				loadSchematicFile(file);
			}
		};
		input.click();
	}

	/** Load schematic from file */
	async function loadSchematicFile(file: File) {
		try {
			const text = await file.text();
			const data = JSON.parse(text);

			if (!data.schematic) {
				status = 'Invalid schematic file: missing schematic data';
				return;
			}

			// Load schematic
			schematic = {
				components: data.schematic.components || [],
				wires: data.schematic.wires || [],
				junctions: data.schematic.junctions || []
			};

			// Load netlist if present
			if (data.netlist) {
				netlistInput = data.netlist;
			}

			// Clear probes
			probes = [];

			status = `Loaded: ${schematic.components.length} components, ${schematic.wires.length} wires`;
		} catch (err) {
			status = `Failed to load schematic: ${err}`;
		}
	}
</script>

<svelte:window onkeydown={handleKeyDown} />

<div class="app">
	<header class="toolbar">
		<span class="app-title">LTSpice Web</span>
		<button onclick={openSchematicDialog} title="Open schematic file">
			Open (Ctrl+O)
		</button>
		<button onclick={saveSchematic} disabled={schematic.components.length === 0 && schematic.wires.length === 0} title="Save schematic to file">
			Save (Ctrl+S)
		</button>
		<span class="toolbar-separator"></span>
		<button onclick={generateNetlistFromSchematic} disabled={schematic.components.length === 0}>
			Generate Netlist (Ctrl+N)
		</button>
		<button onclick={runSim} disabled={status.includes('Initializing') || status.includes('Running')}>
			Run Simulation (Ctrl+B)
		</button>
		{#if waveformTraces.length > 0}
			<span class="trace-count">{waveformTraces.length} traces</span>
		{/if}
	</header>
	<main class="workspace">
		<ResizablePanel title="Netlist" direction="horizontal" initialSize={300} minSize={200} bind:collapsed={netlistCollapsed}>
			<div class="panel-fill">
				<NetlistEditor bind:value={netlistInput} />
			</div>
		</ResizablePanel>
		<div class="right-panel">
			<ResizablePanel title="Schematic" direction="vertical" initialSize={initialSizes.schematic} minSize={100} bind:collapsed={schematicCollapsed}>
				<div class="panel-fill dark">
					<SchematicCanvas bind:schematic onprobe={handleProbe} />
				</div>
			</ResizablePanel>
			<div class="waveform-and-info">
				<ResizablePanel title="Waveform" direction="vertical" initialSize={initialSizes.waveform} minSize={100} bind:collapsed={waveformCollapsed}>
					<div class="panel-fill dark">
						{#if waveformTraces.length > 0}
							<WaveformViewer traces={waveformTraces} {timeData} />
						{:else}
							<div class="placeholder-center">
								<p>Run a simulation to view waveforms</p>
								<p class="hint">Scroll to zoom, drag to pan, double-click to fit</p>
							</div>
						{/if}
					</div>
				</ResizablePanel>
				{#if simResult}
					<div class="info-panel" style="height: {initialSizes.info}px">
						<h3>Simulation Info</h3>
						<div class="result-info">
							<p><strong>Variables:</strong> {simResult.variableNames.join(', ')}</p>
							<p><strong>Points:</strong> {simResult.numPoints}</p>
						</div>
					</div>
				{/if}
			</div>
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

	.toolbar-separator {
		width: 1px;
		height: 20px;
		background: var(--border-primary);
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

	.right-panel {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.waveform-and-info {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.panel-fill {
		width: 100%;
		height: 100%;
		overflow: hidden;
	}

	.panel-fill.dark {
		background: #1a1a1a;
		position: relative;
	}

	.info-panel {
		background: var(--bg-secondary);
		border-top: 1px solid var(--border-primary);
		flex-shrink: 0;
		overflow: auto;
	}

	.info-panel h3 {
		padding: var(--spacing-sm) var(--spacing-md);
		margin: 0;
		font-size: var(--font-size-sm);
		background: var(--bg-tertiary);
		border-bottom: 1px solid var(--border-primary);
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
