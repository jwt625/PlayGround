<script lang="ts">
	import type { Colormap, ColorScale, FieldComponent, ScaleKind, ViewMode } from '$lib/types';

	interface Props {
		component: FieldComponent;
		colormap: Colormap;
		scale: ScaleKind;
		colorScale: ColorScale;
		fitIndependently: boolean;
		viewMode: ViewMode;
		revealed: boolean;
		showIntegrand: boolean;
	}

	const components: { id: FieldComponent; label: string; title: string }[] = [
		{ id: 'intensity', label: 'Intensity', title: '|u|²' },
		{ id: 'amplitude', label: 'Amplitude', title: '|u|' },
		{ id: 'real', label: 'Sign', title: 'Re(u): field sign' },
		{ id: 'phase', label: 'Phase', title: 'arg(u)' }
	];

	let {
		component = $bindable(),
		colormap = $bindable(),
		scale = $bindable(),
		colorScale = $bindable(),
		fitIndependently = $bindable(),
		viewMode = $bindable(),
		revealed,
		showIntegrand = $bindable()
	}: Props = $props();

	function pickComponent(id: FieldComponent) {
		component = id;
		colormap = id === 'phase' ? 'cyclic' : id === 'real' ? 'diverging' : 'magma';
	}
</script>

<div class="controls">
	<div class="group">
		<span class="group-label">Field</span>
		<div class="buttons">
			{#each components as item}
				<button
					class="chip"
					class:active={component === item.id}
					title={item.title}
					onclick={() => pickComponent(item.id)}
				>
					{item.label}
				</button>
			{/each}
			<button
				class="chip disabled"
				disabled
				title="Requires the validated vector solver (C3)"
			>
				Sz (vector)
			</button>
		</div>
	</div>

	<div class="group">
		<label class="group-label" for="colormap">Colormap</label>
		<select id="colormap" bind:value={colormap}>
			<option value="viridis">viridis (sequential)</option>
			<option value="magma">magma (sequential)</option>
			<option value="diverging">diverging (signed)</option>
			<option value="cyclic">cyclic (phase)</option>
			<option value="gray">gray</option>
		</select>
	</div>

	<div class="group">
		<span class="group-label">Scale</span>
		<div class="buttons">
			<button class="chip" class:active={scale === 'linear'} onclick={() => (scale = 'linear')}>linear</button>
			<button
				class="chip"
				class:active={scale === 'log'}
				disabled={component === 'phase' || component === 'real'}
				title={component === 'phase' || component === 'real' ? 'Log scale is undefined for signed fields' : 'log10 scale'}
				onclick={() => (scale = 'log')}
			>
				log
			</button>
		</div>
	</div>

	<div class="group">
		<span class="group-label">Color scale</span>
		<div class="buttons">
			<button class="chip" class:active={colorScale === 'shared'} onclick={() => (colorScale = 'shared')}>shared</button>
			<button
				class="chip"
				class:active={colorScale === 'individual'}
				disabled={viewMode === 'flash'}
				title={viewMode === 'flash' ? 'Flashing uses one shared scale to avoid brightness jumps' : 'Normalize each panel separately'}
				onclick={() => (colorScale = 'individual')}
			>
				per panel
			</button>
		</div>
	</div>

	<div class="group">
		<span class="group-label">Compare</span>
		<div class="buttons">
			<button
				class="chip"
				class:active={viewMode === 'flash'}
				onclick={() => {
					viewMode = 'flash';
					colorScale = 'shared';
				}}
			>
				flash A/B
			</button>
			<button class="chip" class:active={viewMode === 'paired'} onclick={() => (viewMode = 'paired')}>side by side</button>
		</div>
	</div>

	<div class="group">
		<span class="group-label">Extent</span>
		<div class="buttons">
			<button class="chip" class:active={!fitIndependently} onclick={() => (fitIndependently = false)}>shared µm</button>
			<button
				class="chip"
				class:active={fitIndependently && viewMode === 'paired'}
				disabled={viewMode === 'flash'}
				title={viewMode === 'flash' ? 'Flashing needs a shared extent to stay comparable' : 'Fit each panel to its own mode'}
				onclick={() => (fitIndependently = true)}
			>
				fit each
			</button>
		</div>
	</div>

	{#if revealed}
		<div class="group">
			<span class="group-label">Reveal</span>
			<button class="chip" class:active={showIntegrand} onclick={() => (showIntegrand = !showIntegrand)}>
				overlap integrand
			</button>
		</div>
	{/if}
</div>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.group {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.group-label {
		font-family: var(--font-mono);
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
	}

	.buttons {
		display: flex;
		gap: var(--spacing-xs);
	}

	.chip {
		padding: 2px 8px;
		font-size: 0.8rem;
		font-family: var(--font-mono);
		background-color: var(--bg-tertiary);
	}

	.chip.active {
		border-color: var(--accent);
		color: var(--accent);
	}

	.chip.disabled {
		cursor: not-allowed;
		opacity: 0.45;
	}

	select {
		background-color: var(--bg-tertiary);
		color: var(--text-primary);
		border: 1px solid var(--border);
		padding: 2px 6px;
		font-family: var(--font-mono);
		font-size: 0.8rem;
	}
</style>
