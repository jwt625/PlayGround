<script lang="ts">
	import WaveformViewer from './WaveformViewer.svelte';
	import type { WaveformTab, TraceData } from './types';

	let {
		tabs = $bindable<WaveformTab[]>([]),
		activeTabId = $bindable<string>(''),
		timeData = [],
		ondeletetrace
	}: {
		tabs: WaveformTab[];
		activeTabId: string;
		timeData: number[];
		ondeletetrace?: (tabId: string, traceId: string) => void;
	} = $props();

	let deleteMode = $state(false);

	function addTab() {
		const newTab: WaveformTab = {
			id: crypto.randomUUID(),
			name: `Plot ${tabs.length + 1}`,
			traces: []
		};
		tabs = [...tabs, newTab];
		activeTabId = newTab.id;
	}

	function closeTab(tabId: string) {
		if (tabs.length <= 1) return; // Keep at least one tab
		const idx = tabs.findIndex(t => t.id === tabId);
		tabs = tabs.filter(t => t.id !== tabId);
		if (activeTabId === tabId) {
			activeTabId = tabs[Math.max(0, idx - 1)]?.id || '';
		}
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === '5') {
			deleteMode = !deleteMode;
		} else if (e.key === 'Escape') {
			deleteMode = false;
		}
	}

	function handleLegendClick(trace: TraceData) {
		if (deleteMode && ondeletetrace) {
			ondeletetrace(activeTabId, trace.id);
		}
	}

	$effect(() => {
		// Ensure there's always at least one tab
		if (tabs.length === 0) {
			addTab();
		}
		// Ensure activeTabId is valid
		if (!tabs.find(t => t.id === activeTabId) && tabs.length > 0) {
			activeTabId = tabs[0].id;
		}
	});

	const activeTab = $derived(tabs.find(t => t.id === activeTabId));
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="tabbed-waveform" onkeydown={handleKeyDown} tabindex="0">
	<div class="tab-bar" role="tablist">
		{#each tabs as tab}
			<div class="tab-container">
				<button
					role="tab"
					class="tab"
					class:active={tab.id === activeTabId}
					aria-selected={tab.id === activeTabId}
					onclick={() => activeTabId = tab.id}
				>
					{tab.name}
				</button>
				{#if tabs.length > 1}
					<button type="button" class="close-btn" aria-label="Close tab" onclick={() => closeTab(tab.id)}>×</button>
				{/if}
			</div>
		{/each}
		<button class="tab add-tab" aria-label="Add new tab" onclick={addTab}>+</button>
		{#if deleteMode}
			<span class="delete-indicator">DELETE MODE (5 to toggle, click legend to delete)</span>
		{/if}
	</div>
	<div class="waveform-content" class:delete-mode={deleteMode}>
		{#if activeTab && activeTab.traces.length > 0}
			<WaveformViewer traces={activeTab.traces} {timeData} />
			{#if deleteMode}
				<div class="delete-overlay">
					<div class="legend-delete">
						{#each activeTab.traces as trace}
							<button 
								class="trace-delete-btn"
								style="color: rgb({trace.color.r * 255}, {trace.color.g * 255}, {trace.color.b * 255})"
								onclick={() => handleLegendClick(trace)}
							>
								🗑 {trace.name}
							</button>
						{/each}
					</div>
				</div>
			{/if}
		{:else}
			<div class="placeholder-center">
				<p>No traces in this tab</p>
				<p class="hint">Add probes in schematic (P key) to add traces</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.tabbed-waveform {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: #000;
	}

	.tab-bar {
		display: flex;
		align-items: center;
		background: #1a1a1a;
		border-bottom: 1px solid #333;
		padding: 0 4px;
		height: 28px;
		flex-shrink: 0;
	}

	.tab-container {
		display: flex;
		align-items: center;
		border-right: 1px solid #333;
	}

	.tab {
		background: #222;
		border: none;
		color: #888;
		padding: 4px 12px;
		font-size: 11px;
		font-family: monospace;
		cursor: pointer;
	}

	.tab:hover {
		background: #333;
		color: #fff;
	}

	.tab.active {
		background: #000;
		color: #fff;
		border-bottom: 2px solid #4a9eff;
	}

	.close-btn {
		background: transparent;
		border: none;
		font-size: 14px;
		line-height: 1;
		color: #888;
		opacity: 0.5;
		cursor: pointer;
		padding: 2px 6px;
	}

	.close-btn:hover {
		opacity: 1;
		color: #ff4444;
	}

	.add-tab {
		background: transparent;
		color: #666;
		font-size: 16px;
		padding: 4px 10px;
	}

	.add-tab:hover {
		color: #4a9eff;
		background: transparent;
	}

	.delete-indicator {
		margin-left: auto;
		color: #ff4444;
		font-size: 10px;
		font-family: monospace;
		padding: 2px 8px;
		background: rgba(255, 68, 68, 0.2);
	}

	.waveform-content {
		flex: 1;
		position: relative;
		min-height: 0;
	}

	.waveform-content.delete-mode {
		cursor: not-allowed;
	}

	.delete-overlay {
		position: absolute;
		top: 8px;
		right: 8px;
		z-index: 100;
	}

	.legend-delete {
		display: flex;
		flex-direction: column;
		gap: 4px;
		background: rgba(0, 0, 0, 0.9);
		padding: 8px;
		border: 1px solid #ff4444;
	}

	.trace-delete-btn {
		background: transparent;
		border: 1px solid currentColor;
		padding: 4px 8px;
		font-size: 11px;
		font-family: monospace;
		cursor: pointer;
		text-align: left;
	}

	.trace-delete-btn:hover {
		background: rgba(255, 68, 68, 0.3);
		border-color: #ff4444;
	}

	.placeholder-center {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: #666;
		font-family: monospace;
	}

	.placeholder-center .hint {
		font-size: 11px;
		color: #444;
		margin-top: 8px;
	}
</style>

