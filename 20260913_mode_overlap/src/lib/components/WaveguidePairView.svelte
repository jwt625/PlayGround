<script lang="ts">
	import { colormapRGB } from '$lib/colors';
	import { normalizeValue } from '$lib/fields';
	import type { Colormap, ColorScale, FieldComponent, ScaleKind, StoredFieldView, ViewMode } from '$lib/types';
	import type { CachedQuestion } from '$lib/types';
	import { flashSettings } from '$lib/flash.svelte';
	import { onMount } from 'svelte';
	import FieldControls from './FieldControls.svelte';

	interface Props {
		question: CachedQuestion;
		revealed: boolean;
	}

	let { question, revealed }: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let panelA: HTMLCanvasElement | undefined = $state();
	let panelB: HTMLCanvasElement | undefined = $state();
	let integrandCanvas: HTMLCanvasElement | undefined = $state();

	let showB = $state(false);
	let reducedMotion = $state(false);

	let component = $state<FieldComponent>('intensity');
	let colormap = $state<Colormap>('magma');
	let scale = $state<ScaleKind>('linear');
	let colorScale = $state<ColorScale>('shared');
	let viewMode = $state<ViewMode>('flash');
	let fitIndependently = $state(false);
	let showIntegrand = $state(false);

	let view = $derived(question.view);
	let halfPeriodMs = $derived(flashSettings.rateHz > 0 ? Math.max(25, 500 / flashSettings.rateHz) : 0);

	onMount(() => {
		reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
		if (reducedMotion) flashSettings.flashing = false;
	});

	$effect(() => {
		if (!flashSettings.flashing || halfPeriodMs <= 0 || viewMode !== 'flash') return;
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
		const tag = (event.target as HTMLElement | null)?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') return;
		event.preventDefault();
		toggle();
	}

	function componentValue(psi: number, comp: FieldComponent): number {
		switch (comp) {
			case 'intensity':
				return psi * psi;
			case 'amplitude':
				return Math.abs(psi);
			case 'real':
				return psi;
			case 'phase':
				return psi < 0 ? Math.PI : 0;
		}
	}

	function sampleRaw(v: StoredFieldView, x: number, y: number): number {
		if (x < v.x0 || x > v.x1 || y < v.y0 || y > v.y1) return 0;
		const fx = ((x - v.x0) / (v.x1 - v.x0)) * (v.nx - 1);
		const fy = ((y - v.y0) / (v.y1 - v.y0)) * (v.ny - 1);
		const i0 = Math.max(0, Math.min(v.nx - 1, Math.floor(fx)));
		const i1 = Math.min(v.nx - 1, i0 + 1);
		const j0 = Math.max(0, Math.min(v.ny - 1, Math.floor(fy)));
		const j1 = Math.min(v.ny - 1, j0 + 1);
		const tx = fx - i0;
		const ty = fy - j0;
		const v00 = v.field[j0 * v.nx + i0];
		const v10 = v.field[j0 * v.nx + i1];
		const v01 = v.field[j1 * v.nx + i0];
		const v11 = v.field[j1 * v.nx + i1];
		return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
	}

	function rangeFor(v: StoredFieldView, comp: FieldComponent) {
		if (comp === 'phase') return { min: -Math.PI, max: Math.PI, symmetric: true };
		let min = Infinity;
		let max = -Infinity;
		for (const psi of v.field) {
			const value = componentValue(psi, comp);
			if (value < min) min = value;
			if (value > max) max = value;
		}
		if (comp === 'real') {
			const bound = Math.max(Math.abs(min), Math.abs(max)) || 1;
			return { min: -bound, max: bound, symmetric: true };
		}
		return { min: 0, max: max || 1, symmetric: false };
	}

	function mergeRanges(a: ReturnType<typeof rangeFor>, b: ReturnType<typeof rangeFor>) {
		if (a.symmetric || b.symmetric) {
			const bound = Math.max(Math.abs(a.min), Math.abs(a.max), Math.abs(b.min), Math.abs(b.max)) || 1;
			return { min: -bound, max: bound, symmetric: true };
		}
		return { min: Math.min(a.min, b.min), max: Math.max(a.max, b.max), symmetric: false };
	}

	function extentFor(v: StoredFieldView) {
		return { x0: v.x0, x1: v.x1, y0: v.y0, y1: v.y1 };
	}

	function renderLayer(
		v: StoredFieldView,
		extent: { x0: number; x1: number; y0: number; y1: number },
		range: ReturnType<typeof rangeFor>,
		width: number,
		height: number
	): HTMLCanvasElement {
		const layer = document.createElement('canvas');
		layer.width = width;
		layer.height = height;
		const ctx = layer.getContext('2d');
		if (!ctx) return layer;
		const image = ctx.createImageData(width, height);
		for (let py = 0; py < height; py++) {
			const y = extent.y1 - ((py + 0.5) / height) * (extent.y1 - extent.y0);
			for (let px = 0; px < width; px++) {
				const x = extent.x0 + ((px + 0.5) / width) * (extent.x1 - extent.x0);
				const value = componentValue(sampleRaw(v, x, y), component);
				const t = normalizeValue(value, range, scale);
				const [r, g, b] = colormapRGB(colormap, t);
				const index = (py * width + px) * 4;
				image.data[index] = r;
				image.data[index + 1] = g;
				image.data[index + 2] = b;
				image.data[index + 3] = 255;
			}
		}
		ctx.putImageData(image, 0, 0);
		return layer;
	}

	function overlay(ctx: CanvasRenderingContext2D, v: StoredFieldView, extent: ReturnType<typeof extentFor>, width: number, height: number, ratio: number) {
		const toX = (x: number) => ((x - extent.x0) / (extent.x1 - extent.x0)) * width;
		const toY = (y: number) => ((extent.y1 - y) / (extent.y1 - extent.y0)) * height;
		ctx.strokeStyle = 'rgba(255,255,255,0.85)';
		ctx.setLineDash([4 * ratio, 3 * ratio]);
		ctx.lineWidth = ratio;
		ctx.strokeRect(toX(v.core.x0), toY(v.core.y1), toX(v.core.x1) - toX(v.core.x0), toY(v.core.y0) - toY(v.core.y1));
		ctx.setLineDash([]);
	}

	let layers: { sig: string; a: HTMLCanvasElement; b: HTMLCanvasElement } | null = null;

	function sharedExtent() {
		if (!view) return { x0: -1, x1: 1, y0: -1, y1: 1 };
		return {
			x0: Math.min(view.a.x0, view.b.x0),
			x1: Math.max(view.a.x1, view.b.x1),
			y0: Math.min(view.a.y0, view.b.y0),
			y1: Math.max(view.a.y1, view.b.y1)
		};
	}

	function drawFlash() {
		if (!canvas || !view) return;
		const extent = sharedExtent();
		const aspect = (extent.y1 - extent.y0) / (extent.x1 - extent.x0);
		const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
		const width = Math.max(1, Math.round(420 * ratio));
		const height = Math.max(1, Math.round(420 * aspect * ratio));
		if (canvas.width !== width) canvas.width = width;
		if (canvas.height !== height) canvas.height = height;
		canvas.style.aspectRatio = `${extent.x1 - extent.x0} / ${extent.y1 - extent.y0}`;
		const rangeA = rangeFor(view.a, component);
		const rangeB = rangeFor(view.b, component);
		const rA = colorScale === 'shared' ? mergeRanges(rangeA, rangeB) : rangeA;
		const rB = colorScale === 'shared' ? mergeRanges(rangeA, rangeB) : rangeB;
		const sig = `${view.a.id}|${view.b.id}|${component}|${colormap}|${scale}|${colorScale}|${width}x${height}|${rA.min},${rA.max}`;
		if (!layers || layers.sig !== sig) {
			layers = { sig, a: renderLayer(view.a, extent, rA, width, height), b: renderLayer(view.b, extent, rB, width, height) };
		}
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, width, height);
		ctx.drawImage(showB ? layers.b : layers.a, 0, 0);
		overlay(ctx, showB ? view.b : view.a, extent, width, height, ratio);
	}

	function drawPaired(canvasEl: HTMLCanvasElement | undefined, v: StoredFieldView, other: StoredFieldView) {
		if (!canvasEl) return;
		const extent = fitIndependently ? extentFor(v) : sharedExtent();
		const aspect = (extent.y1 - extent.y0) / (extent.x1 - extent.x0);
		const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
		const width = Math.max(1, Math.round(300 * ratio));
		const height = Math.max(1, Math.round(300 * aspect * ratio));
		canvasEl.width = width;
		canvasEl.height = height;
		canvasEl.style.aspectRatio = `${extent.x1 - extent.x0} / ${extent.y1 - extent.y0}`;
		const range = colorScale === 'shared' ? mergeRanges(rangeFor(v, component), rangeFor(other, component)) : rangeFor(v, component);
		const layer = renderLayer(v, extent, range, width, height);
		const ctx = canvasEl.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, width, height);
		ctx.drawImage(layer, 0, 0);
		overlay(ctx, v, extent, width, height, ratio);
	}

	function drawIntegrand() {
		if (!integrandCanvas || !view) return;
		const extent = sharedExtent();
		const aspect = (extent.y1 - extent.y0) / (extent.x1 - extent.x0);
		const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
		const width = Math.max(1, Math.round(420 * ratio));
		const height = Math.max(1, Math.round(420 * aspect * ratio));
		integrandCanvas.width = width;
		integrandCanvas.height = height;
		integrandCanvas.style.aspectRatio = `${extent.x1 - extent.x0} / ${extent.y1 - extent.y0}`;
		let bound = 1e-12;
		for (const value of view.a.field) bound = Math.max(bound, value * value);
		const image = new ImageData(width, height);
		for (let py = 0; py < height; py++) {
			const y = extent.y1 - ((py + 0.5) / height) * (extent.y1 - extent.y0);
			for (let px = 0; px < width; px++) {
				const x = extent.x0 + ((px + 0.5) / width) * (extent.x1 - extent.x0);
				const product = sampleRaw(view.a, x, y) * sampleRaw(view.b, x, y);
				const t = (product / bound) * 0.5 + 0.5;
				const [r, g, b] = colormapRGB('diverging', t);
				const index = (py * width + px) * 4;
				image.data[index] = r;
				image.data[index + 1] = g;
				image.data[index + 2] = b;
				image.data[index + 3] = 255;
			}
		}
		integrandCanvas.getContext('2d')?.putImageData(image, 0, 0);
	}

	$effect(() => {
		void question;
		void revealed;
		void showB;
		void component;
		void colormap;
		void scale;
		void colorScale;
		void viewMode;
		void fitIndependently;
		void showIntegrand;
		if (!view) return;
		if (viewMode === 'flash') drawFlash();
		else {
			drawPaired(panelA, view.a, view.b);
			drawPaired(panelB, view.b, view.a);
		}
		if (revealed && showIntegrand) drawIntegrand();
	});
</script>

<svelte:window onkeydown={handleKey} />

{#if view}
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

	{#if viewMode === 'flash'}
		<div class="flash">
			<div class="flash-head">
				<div class="identity">
					<span class="tag" class:b={showB}>{showB ? 'B' : 'A'}</span>
					<span class="title">{showB ? view.b.id : view.a.id}</span>
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
			<div class="stage"><canvas bind:this={canvas}></canvas></div>
			<p class="hint">
				Shared µm extent and color scale when selected; dashed line = core outline. Space toggles.
				{reducedMotion ? ' Auto-flash is off (reduced-motion preference).' : ''}
			</p>
		</div>
	{:else}
		<div class="pair">
			<figure>
				<canvas bind:this={panelA}></canvas>
				<figcaption>Mode A · {view.a.id}</figcaption>
			</figure>
			<figure>
				<canvas bind:this={panelB}></canvas>
				<figcaption>Mode B · {view.b.id}</figcaption>
			</figure>
		</div>
	{/if}

	{#if revealed && showIntegrand}
		<div class="integrand">
			<div class="flash-head">
				<div class="identity"><span class="tag">∫</span><span class="title">overlap integrand a·b</span></div>
			</div>
			<div class="stage"><canvas bind:this={integrandCanvas}></canvas></div>
			<p class="hint">Signed product of the two solved fields; the authoritative η is computed separately.</p>
		</div>
	{/if}
{/if}

<style>
	.flash, .integrand {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		align-items: center;
		margin-top: var(--spacing-sm);
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

	canvas {
		width: 100%;
		display: block;
		background: #000;
		border: 1px solid var(--border);
	}

	.pair {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-md);
		margin-top: var(--spacing-sm);
	}

	figure {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		margin: 0;
		min-width: 0;
	}

	figcaption {
		font-family: var(--font-mono);
		font-size: 0.7rem;
		color: var(--text-secondary);
		text-align: center;
	}

	.hint {
		font-size: 0.72rem;
		color: var(--text-muted);
		text-align: center;
		max-width: 560px;
	}

	@media (max-width: 700px) {
		.pair {
			grid-template-columns: 1fr;
		}
	}
</style>
