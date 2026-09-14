<script lang="ts">
	import { colormapRGB } from '$lib/colors';
	import type { ModeGalleryEntry } from '$lib/types';

	interface Props {
		entry: ModeGalleryEntry;
		large?: boolean;
	}

	let { entry, large = false }: Props = $props();

	let indexCanvas: HTMLCanvasElement | undefined = $state();
	let fieldCanvas: HTMLCanvasElement | undefined = $state();

	const INDEX_MIN = 1.0;
	const INDEX_MAX = 3.6;

	function size(canvas: HTMLCanvasElement, widthPx: number): { width: number; height: number; ctx: CanvasRenderingContext2D } | null {
		const aspect = (entry.axes.y1 - entry.axes.y0) / (entry.axes.x1 - entry.axes.x0);
		const width = widthPx;
		const height = Math.max(1, Math.round(width * aspect));
		const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
		canvas.width = Math.round(width * ratio);
		canvas.height = Math.round(height * ratio);
		canvas.style.aspectRatio = `${entry.axes.x1 - entry.axes.x0} / ${entry.axes.y1 - entry.axes.y0}`;
		const ctx = canvas.getContext('2d');
		if (!ctx) return null;
		ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		return { width, height, ctx };
	}

	function drawIndex() {
		if (!indexCanvas) return;
		const sized = size(indexCanvas, large ? 460 : 240);
		if (!sized) return;
		const { width, height, ctx } = sized;
		const { x0, x1, y0, y1 } = entry.axes;
		const toX = (x: number) => ((x - x0) / (x1 - x0)) * width;
		const toY = (y: number) => ((y1 - y) / (y1 - y0)) * height;
		const colorOf = (n: number) => {
			const [r, g, b] = colormapRGB('viridis', (n - INDEX_MIN) / (INDEX_MAX - INDEX_MIN));
			return `rgb(${r},${g},${b})`;
		};
		ctx.fillStyle = colorOf(entry.backgroundIndex);
		ctx.fillRect(0, 0, width, height);
		for (const region of entry.indexRegions) {
			const rx0 = Math.max(x0, region.x0);
			const rx1 = Math.min(x1, region.x1);
			const ry0 = Math.max(y0, region.y0);
			const ry1 = Math.min(y1, region.y1);
			if (rx1 <= rx0 || ry1 <= ry0) continue;
			ctx.fillStyle = colorOf(region.index);
			ctx.fillRect(toX(rx0), toY(ry1), toX(rx1) - toX(rx0), toY(ry0) - toY(ry1));
		}
		ctx.strokeStyle = 'rgba(255,255,255,0.85)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 3]);
		ctx.strokeRect(
			toX(entry.core.x0),
			toY(entry.core.y1),
			toX(entry.core.x1) - toX(entry.core.x0),
			toY(entry.core.y0) - toY(entry.core.y1)
		);
		ctx.setLineDash([]);
	}

	function drawField() {
		if (!fieldCanvas) return;
		const sized = size(fieldCanvas, large ? 460 : 240);
		if (!sized) return;
		const { width: cssWidth, height: cssHeight, ctx } = sized;
		const { x0, x1, y0, y1, nx, ny } = entry.axes;
		// putImageData ignores the DPR transform, so build the image at device
		// resolution and sample physical coordinates per device pixel.
		const width = fieldCanvas.width;
		const height = fieldCanvas.height;
		const intensity = entry.field.map((value) => value * value);
		let max = 0;
		for (const value of intensity) max = Math.max(max, value);
		max = max || 1;
		const image = ctx.createImageData(width, height);
		for (let py = 0; py < height; py++) {
			const y = y1 - ((py + 0.5) / height) * (y1 - y0);
			const fy = ((y - y0) / (y1 - y0)) * (ny - 1);
			const j0 = Math.max(0, Math.min(ny - 1, Math.floor(fy)));
			const j1 = Math.min(ny - 1, j0 + 1);
			const ty = fy - j0;
			for (let px = 0; px < width; px++) {
				const x = x0 + ((px + 0.5) / width) * (x1 - x0);
				const fx = ((x - x0) / (x1 - x0)) * (nx - 1);
				const i0 = Math.max(0, Math.min(nx - 1, Math.floor(fx)));
				const i1 = Math.min(nx - 1, i0 + 1);
				const tx = fx - i0;
				const v00 = intensity[j0 * nx + i0];
				const v10 = intensity[j0 * nx + i1];
				const v01 = intensity[j1 * nx + i0];
				const v11 = intensity[j1 * nx + i1];
				const value = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
				const t = Math.sqrt(Math.max(0, Math.min(1, value / max)));
				const [r, g, b] = colormapRGB('magma', t);
				const index = (py * width + px) * 4;
				image.data[index] = r;
				image.data[index + 1] = g;
				image.data[index + 2] = b;
				image.data[index + 3] = 255;
			}
		}
		ctx.putImageData(image, 0, 0);
		const toX = (x: number) => ((x - x0) / (x1 - x0)) * cssWidth;
		const toY = (y: number) => ((y1 - y) / (y1 - y0)) * cssHeight;
		ctx.strokeStyle = 'rgba(255,255,255,0.85)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 3]);
		ctx.strokeRect(
			toX(entry.core.x0),
			toY(entry.core.y1),
			toX(entry.core.x1) - toX(entry.core.x0),
			toY(entry.core.y0) - toY(entry.core.y1)
		);
		ctx.setLineDash([]);
	}

	$effect(() => {
		void entry;
		void large;
		drawIndex();
		drawField();
	});
</script>

<div class="wg" class:large>
	<div class="plots">
		<figure>
			<canvas bind:this={indexCanvas}></canvas>
			<figcaption>n cross-section (dashed = core)</figcaption>
		</figure>
		<figure>
			<canvas bind:this={fieldCanvas}></canvas>
			<figcaption>|E|² (quasi-TE)</figcaption>
		</figure>
	</div>
	<dl class="meta">
		<div><dt>neff</dt><dd class="mono">{entry.neff.toFixed(4)}</dd></div>
		<div><dt>confinement</dt><dd class="mono">{entry.confinement.toFixed(3)}</dd></div>
		<div><dt>size</dt><dd class="mono">{entry.dimensions.widthUm} × {entry.dimensions.heightUm} µm{entry.dimensions.slabUm ? ` (slab ${entry.dimensions.slabUm})` : ''}</dd></div>
		<div><dt>λ</dt><dd class="mono">{entry.wavelengthUm} µm</dd></div>
		<div><dt>materials</dt><dd>{entry.materialLabel} / {entry.cladding}</dd></div>
		<div><dt>residual</dt><dd class="mono">{entry.residual.toExponential(1)}</dd></div>
	</dl>
	{#if entry.confinement < 0.6}
		<p class="warn">Weak confinement / near cutoff — treat neff as indicative only.</p>
	{/if}
	{#if entry.residual > 1e-6}
		<p class="warn">Iteration not fully converged (residual {entry.residual.toExponential(1)}).</p>
	{/if}
</div>

<style>
	.wg {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.plots {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-sm);
	}

	figure {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin: 0;
		min-width: 0;
	}

	canvas {
		width: 100%;
		display: block;
		background: #000;
		border: 1px solid var(--border);
	}

	figcaption {
		font-family: var(--font-mono);
		font-size: 0.65rem;
		color: var(--text-muted);
		text-align: center;
	}

	.meta {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 2px var(--spacing-sm);
		margin: 0;
		font-size: 0.72rem;
	}

	.meta > div {
		display: flex;
		justify-content: space-between;
		gap: var(--spacing-sm);
		border-bottom: 1px solid var(--border);
		padding-bottom: 1px;
	}

	dt {
		color: var(--text-muted);
	}

	dd {
		margin: 0;
	}

	.mono {
		font-family: var(--font-mono);
	}

	.warn {
		font-size: 0.7rem;
		color: var(--warning);
	}

	.large .meta {
		font-size: 0.8rem;
	}
</style>
