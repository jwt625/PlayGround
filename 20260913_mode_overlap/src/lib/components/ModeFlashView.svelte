<script lang="ts">
	import type { ValueRange } from '$lib/fields';
	import type { Colormap, FieldComponent, FieldGrid, GaussianMode, ScaleKind } from '$lib/types';
	import { flashSettings } from '$lib/flash.svelte';
	import { onMount } from 'svelte';
	import FieldCanvas from './FieldCanvas.svelte';

	interface Props {
		gridA: FieldGrid;
		valuesA: Float64Array;
		gridB: FieldGrid;
		valuesB: Float64Array;
		component: FieldComponent;
		colormap: Colormap;
		scale: ScaleKind;
		range: ValueRange;
		overlayA: GaussianMode;
		overlayB: GaussianMode;
		offsetXUm: number;
		offsetYUm: number;
	}

	let {
		gridA,
		valuesA,
		gridB,
		valuesB,
		component,
		colormap,
		scale,
		range,
		overlayA,
		overlayB,
		offsetXUm,
		offsetYUm
	}: Props = $props();

	let showB = $state(false);
	let reducedMotion = $state(false);

	const MIN_HALF_PERIOD_MS = 25;

	let halfPeriodMs = $derived(flashSettings.rateHz > 0 ? Math.max(MIN_HALF_PERIOD_MS, 500 / flashSettings.rateHz) : 0);

	onMount(() => {
		reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
		if (reducedMotion) flashSettings.flashing = false;
	});

	$effect(() => {
		if (!flashSettings.flashing || halfPeriodMs <= 0) return;
		const id = setInterval(() => {
			showB = !showB;
		}, halfPeriodMs);
		return () => clearInterval(id);
	});

	function toggle() {
		showB = !showB;
	}

	function handleKey(event: KeyboardEvent) {
		if (event.key !== ' ') return;
		const target = event.target as HTMLElement | null;
		const tag = target?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') return;
		event.preventDefault();
		toggle();
	}
</script>

<svelte:window onkeydown={handleKey} />

<div class="flash">
	<div class="flash-head">
		<div class="identity">
			<span class="tag" class:b={showB}>{showB ? 'B' : 'A'}</span>
			<span class="title">Mode {showB ? 'B' : 'A'}</span>
		</div>
		<div class="flash-controls">
			<button class="chip" class:active={flashSettings.flashing} onclick={() => (flashSettings.flashing = !flashSettings.flashing)}>
				{flashSettings.flashing ? 'Pause' : 'Flash A/B'}
			</button>
			<button class="chip" onclick={toggle} disabled={flashSettings.flashing && !reducedMotion}>Toggle</button>
			<label class="rate">
				<span>rate {flashSettings.rateHz.toFixed(1)} Hz</span>
				<input type="range" min="0" max="6" step="0.2" bind:value={flashSettings.rateHz} disabled={reducedMotion} />
			</label>
		</div>
	</div>

	<div class="stage">
		<FieldCanvas
			grid={showB ? gridB : gridA}
			values={showB ? valuesB : valuesA}
			{component}
			{colormap}
			{scale}
			{range}
			overlay={showB ? overlayB : overlayA}
			offsetXUm={showB ? offsetXUm : 0}
			offsetYUm={showB ? offsetYUm : 0}
			axisLabel="x"
		/>
	</div>

	<p class="hint">
		Both modes share one µm extent and one color scale so the flip is comparable. Space toggles
		when no control is focused. {reducedMotion ? 'Auto-flash is off (reduced-motion preference).' : ''}
	</p>
</div>

<style>
	.flash {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		align-items: center;
	}

	.flash-head {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: var(--spacing-md);
		width: 100%;
	}

	.identity {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.tag {
		font-family: var(--font-mono);
		font-size: 1rem;
		font-weight: 600;
		color: var(--accent);
		padding: 2px 12px;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--accent);
	}

	.tag.b {
		color: var(--warning);
		border-color: var(--warning);
	}

	.title {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	.flash-controls {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.chip {
		font-family: var(--font-mono);
		font-size: 0.8rem;
		padding: 2px 10px;
	}

	.chip.active {
		border-color: var(--accent);
		color: var(--accent);
	}

	.rate {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
		font-family: var(--font-mono);
		font-size: 0.75rem;
		color: var(--text-secondary);
	}

	.rate input {
		width: 140px;
		accent-color: var(--accent);
	}

	.stage {
		width: 100%;
		max-width: 560px;
	}

	.hint {
		font-size: 0.72rem;
		color: var(--text-muted);
		text-align: center;
		max-width: 560px;
	}
</style>
