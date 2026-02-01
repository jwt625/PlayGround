<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { initSimulation, runSimulation, terminateSimulation, type SimulationResult, type RealDataType } from '$lib/simulation';
	import { TabbedWaveformViewer, type TraceData, type WaveformTab, getTraceColor } from '$lib/waveform';
	import { NetlistEditor } from '$lib/editor';
	import { SchematicCanvas, type Schematic, type Probe } from '$lib/schematic';
	import { ResizablePanel } from '$lib/components';
	import { schematicToNetlist, generateNodeLabels, calculateComponentCurrent } from '$lib/netlist';

	let status = $state('Not initialized');
	let simResult = $state<SimulationResult | null>(null);
	let timeData = $state<number[]>([]);
	let schematic = $state<Schematic>({ components: [], wires: [], junctions: [] });
	let probes = $state<Probe[]>([]);

	// Waveform tabs
	let waveformTabs = $state<WaveformTab[]>([{ id: 'default', name: 'Plot 1', traces: [] }]);
	let activeTabId = $state('default');

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
		// Clear all tabs' traces
		waveformTabs = waveformTabs.map(tab => ({ ...tab, traces: [] }));
		timeData = [];
		try {
			const result = await runSimulation(netlistInput);
			simResult = result;
			status = `Simulation complete: ${result.numPoints} points, ${result.numVariables} variables`;

			// Extract time data from result
			for (const data of result.data) {
				if (data.type === 'time') {
					timeData = data.values as number[];
					break;
				}
			}
			// Note: Traces are added via probing, not automatically
		} catch (err) {
			status = `Simulation error: ${err}`;
		}
	}

	/** Check if a simulation data entry matches a specific probe */
	function dataMatchesProbe(dataName: string, probe: Probe): boolean {
		const dataNameLower = dataName.toLowerCase();
		const node1Lower = probe.node1.toLowerCase();

		if (probe.type === 'voltage') {
			return dataNameLower === `v(${node1Lower})`;
		} else if (probe.type === 'voltage-diff') {
			const node2Lower = probe.node2?.toLowerCase() || '';
			return dataNameLower === `v(${node1Lower})` || dataNameLower === `v(${node2Lower})`;
		} else if (probe.type === 'current') {
			return dataNameLower === `i(${node1Lower})` ||
				   dataNameLower.includes(`(${node1Lower})`) ||
				   dataNameLower.includes(`@${node1Lower}`);
		}
		return false;
	}

	/** Add traces for a specific probe to the active tab */
	function addProbeTracesToActiveTab(probe: Probe) {
		if (!simResult || simResult.dataType !== 'real') return;

		const activeTab = waveformTabs.find(t => t.id === activeTabId);
		if (!activeTab) return;

		const traces: TraceData[] = [];
		let colorIndex = activeTab.traces.length;

		// Ensure we have time data
		for (const data of simResult.data) {
			if (data.type === 'time' && timeData.length === 0) {
				timeData = data.values as number[];
				break;
			}
		}

		// First, try to find matching data in simulation results
		let foundInSimData = false;
		for (const data of simResult.data) {
			if (dataMatchesProbe(data.name, probe)) {
				const alreadyExists = activeTab.traces.some(t => t.id === data.name);
				if (!alreadyExists) {
					traces.push({
						id: data.name,
						name: data.name,
						type: data.type,
						values: data.values as number[],
						color: getTraceColor(colorIndex++),
						visible: true
					});
					foundInSimData = true;
				}
			}
		}

		// If current probe not found in sim data, try to calculate it
		if (!foundInSimData && probe.type === 'current' && probe.componentId && timeData.length > 0) {
			const comp = schematic.components.find(c => c.id === probe.componentId);
			if (comp && (comp.type === 'resistor' || comp.type === 'capacitor')) {
				const calculatedCurrent = calculateComponentCurrent(
					comp,
					schematic,
					simResult.data as RealDataType[],
					timeData
				);
				if (calculatedCurrent) {
					const alreadyExists = activeTab.traces.some(t => t.id === calculatedCurrent.name);
					if (!alreadyExists) {
						traces.push({
							id: calculatedCurrent.name,
							name: `${calculatedCurrent.name} (calc)`,
							type: calculatedCurrent.type,
							values: calculatedCurrent.values,
							color: getTraceColor(colorIndex++),
							visible: true
						});
					}
				}
			}
		}

		if (traces.length > 0) {
			waveformTabs = waveformTabs.map(tab =>
				tab.id === activeTabId
					? { ...tab, traces: [...tab.traces, ...traces] }
					: tab
			);
		}
	}

	/** Handle trace deletion from waveform panel */
	function handleDeleteTrace(tabId: string, traceId: string) {
		waveformTabs = waveformTabs.map(tab =>
			tab.id === tabId
				? { ...tab, traces: tab.traces.filter(t => t.id !== traceId) }
				: tab
		);
	}

	/** Get total trace count across all tabs */
	function getTotalTraceCount(): number {
		return waveformTabs.reduce((sum, tab) => sum + tab.traces.length, 0);
	}

	/** Handle probe event from schematic canvas */
	function handleProbe(event: { type: string; node1: string; node2?: string; componentId?: string; label: string }) {
		const { type, node1, node2, componentId, label } = event;

		// Create the probe object
		const newProbe: Probe = {
			id: crypto.randomUUID(),
			type: type as Probe['type'],
			node1,
			node2,
			componentId,
			label
		};

		// Check if probe already exists in the probes list (for toggle behavior)
		const existingIndex = probes.findIndex(p => p.label === label);
		if (existingIndex >= 0) {
			// Remove existing probe (toggle off) - but don't remove traces
			probes = probes.filter((_, i) => i !== existingIndex);
			status = `Removed probe from list: ${label}`;
		} else {
			// Add to probes list for tracking
			probes = [...probes, newProbe];
		}

		// Always try to add the trace to the active tab (if not already there)
		if (simResult) {
			addProbeTracesToActiveTab(newProbe);
			status = `Added probe: ${label}`;
		} else {
			status = `Probe added: ${label} (run simulation to see trace)`;
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
		{#if getTotalTraceCount() > 0}
			<span class="trace-count">{getTotalTraceCount()} traces</span>
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
						<TabbedWaveformViewer
							bind:tabs={waveformTabs}
							bind:activeTabId={activeTabId}
							{timeData}
							ondeletetrace={handleDeleteTrace}
						/>
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
