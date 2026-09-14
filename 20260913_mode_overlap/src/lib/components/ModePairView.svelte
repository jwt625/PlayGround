<script lang="ts">
	import { transformDefaults } from '$lib/engine';
	import {
		componentValues,
		mergeRanges,
		rangeOf,
		sampleField,
		sampleIntegrand,
		sharedExtent
	} from '$lib/fields';
	import type { CachedQuestion, Colormap, ColorScale, FieldComponent, ScaleKind, ViewMode } from '$lib/types';
	import FieldCanvas from './FieldCanvas.svelte';
	import FieldControls from './FieldControls.svelte';
	import ModeFlashView from './ModeFlashView.svelte';

	interface Props {
		question: CachedQuestion;
		revealed: boolean;
		samples?: number;
		compact?: boolean;
	}

	let { question, revealed, samples = 129, compact = false }: Props = $props();

	let component = $state<FieldComponent>('intensity');
	let colormap = $state<Colormap>('magma');
	let scale = $state<ScaleKind>('linear');
	let colorScale = $state<ColorScale>('shared');
	let fitIndependently = $state(false);
	let viewMode = $state<ViewMode>('flash');
	let showIntegrand = $state(false);

	const identity = transformDefaults({});

	let extents = $derived.by(() => {
		const pair = sharedExtent(question.a, question.b);
		const fit = fitIndependently && viewMode === 'paired';
		return {
			a: fit ? sharedExtent(question.a, question.a) : pair,
			b: fit ? sharedExtent(question.b, question.b) : pair
		};
	});

	let gridA = $derived(
		sampleField({
			mode: question.a,
			transform: identity,
			wavelengthUm: question.wavelengthUm,
			extentXUm: extents.a.x,
			extentYUm: extents.a.y,
			samples
		})
	);
	let gridB = $derived(
		sampleField({
			mode: question.b,
			transform: question.transform,
			wavelengthUm: question.wavelengthUm,
			extentXUm: extents.b.x,
			extentYUm: extents.b.y,
			samples
		})
	);
	let valuesA = $derived(componentValues(gridA, component));
	let valuesB = $derived(componentValues(gridB, component));
	let rangeA = $derived(rangeOf(valuesA, component));
	let rangeB = $derived(rangeOf(valuesB, component));
	let sharedRange = $derived(mergeRanges(rangeA, rangeB, component));

	let integrand = $derived.by(() => {
		if (!revealed || !showIntegrand) return null;
		const extent = sharedExtent(question.a, question.b);
		return sampleIntegrand(
			{
				mode: question.a,
				transform: identity,
				wavelengthUm: question.wavelengthUm,
				extentXUm: extent.x,
				extentYUm: extent.y,
				samples
			},
			{
				mode: question.b,
				transform: question.transform,
				wavelengthUm: question.wavelengthUm,
				extentXUm: extent.x,
				extentYUm: extent.y,
				samples
			},
			samples
		);
	});

	let integrandRange = $derived.by(() => {
		if (!integrand) return { min: -1, max: 1, symmetric: true };
		let bound = 1e-12;
		for (const value of integrand.re) bound = Math.max(bound, Math.abs(value));
		return { min: -bound, max: bound, symmetric: true };
	});

	function transformSummary(): string {
		const t = question.transform;
		const parts: string[] = [];
		if (t.dxUm || t.dyUm) parts.push(`offset (${t.dxUm.toFixed(3)}, ${t.dyUm.toFixed(3)}) µm`);
		if (t.thetaXRad || t.thetaYRad)
			parts.push(`tilt (${((t.thetaXRad * 180) / Math.PI).toFixed(2)}, ${((t.thetaYRad * 180) / Math.PI).toFixed(2)})°`);
		if (t.polarizationRad) parts.push(`polarization ${((t.polarizationRad * 180) / Math.PI).toFixed(1)}°`);
		return parts.length ? parts.join(', ') : 'aligned, co-polarized';
	}
</script>

{#if !compact}
	<FieldControls
		bind:component
		bind:colormap
		bind:scale
		bind:colorScale
		bind:fitIndependently
		bind:viewMode
		{revealed}
		bind:showIntegrand
	/>
{/if}

{#if viewMode === 'flash' && !compact}
	<ModeFlashView
		{gridA}
		{valuesA}
		{gridB}
		{valuesB}
		{component}
		{colormap}
		{scale}
		range={sharedRange}
		overlayA={question.a}
		overlayB={question.b}
		offsetXUm={question.transform.dxUm}
		offsetYUm={question.transform.dyUm}
	/>
	<div class="mfd-row">
		<span>MFD A {question.a.mfdXUm.toFixed(2)} × {question.a.mfdYUm.toFixed(2)} µm</span>
		<span>MFD B {question.b.mfdXUm.toFixed(2)} × {question.b.mfdYUm.toFixed(2)} µm</span>
	</div>
{:else}
	<div class="pair" class:compact>
		<div class="panel">
			<div class="panel-head">
				<span class="tag">A</span>
				<span class="name">Mode A</span>
			</div>
			<FieldCanvas
				grid={gridA}
				values={valuesA}
				{component}
				{colormap}
				{scale}
				range={colorScale === 'shared' ? sharedRange : rangeA}
				overlay={question.a}
				offsetXUm={0}
				offsetYUm={0}
				pixelRatio={compact ? 1 : undefined}
				axisLabel="x"
			/>
			{#if !compact}
				<p class="meta">MFD {question.a.mfdXUm.toFixed(2)} × {question.a.mfdYUm.toFixed(2)} µm</p>
			{/if}
		</div>

		<div class="panel">
			<div class="panel-head">
				<span class="tag">B</span>
				<span class="name">Mode B</span>
			</div>
			<FieldCanvas
				grid={gridB}
				values={valuesB}
				{component}
				{colormap}
				{scale}
				range={colorScale === 'shared' ? sharedRange : rangeB}
				overlay={question.b}
				offsetXUm={question.transform.dxUm}
				offsetYUm={question.transform.dyUm}
				pixelRatio={compact ? 1 : undefined}
				axisLabel="x"
			/>
			{#if !compact}
				<p class="meta">MFD {question.b.mfdXUm.toFixed(2)} × {question.b.mfdYUm.toFixed(2)} µm</p>
			{/if}
		</div>
	</div>
{/if}

{#if !compact}
	<div class="question-meta">
		<span>λ = {question.wavelengthUm.toFixed(3)} µm</span>
		<span>{transformSummary()}</span>
		<span>scalar paraxial Gaussian, common waist plane</span>
		<span>MFD is mode field diameter, not core geometry</span>
	</div>
{/if}

{#if integrand}
	<div class="integrand">
		<div class="panel-head">
			<span class="tag">∫</span>
			<span class="name">u<sub>A</sub>*·u<sub>B</sub> (signed real part)</span>
		</div>
		<FieldCanvas
			grid={integrand.grid}
			values={integrand.re}
			component="real"
			colormap="diverging"
			scale="linear"
			range={integrandRange}
			overlay={null}
			axisLabel="x"
		/>
		<p class="meta">Display aid only: the authoritative η is the analytic normalized overlap.</p>
	</div>
{/if}

<style>
	.pair {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-md);
	}

	.pair.compact {
		gap: var(--spacing-sm);
	}

	.panel {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		min-width: 0;
	}

	.panel-head {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.tag {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		color: var(--accent);
		padding: 1px 6px;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
	}

	.name {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	.meta {
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--text-secondary);
	}

	.question-meta {
		display: flex;
		flex-wrap: wrap;
		gap: var(--spacing-md);
		margin-top: var(--spacing-sm);
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--text-secondary);
	}

	.mfd-row {
		display: flex;
		justify-content: center;
		flex-wrap: wrap;
		gap: var(--spacing-lg);
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--text-secondary);
	}

	.integrand {
		margin-top: var(--spacing-lg);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		max-width: 520px;
	}

	@media (max-width: 700px) {
		.pair {
			grid-template-columns: 1fr;
		}
	}
</style>
