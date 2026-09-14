<script lang="ts">
	import { colormapRGB } from '$lib/colors';
	import { normalizeValue, type ValueRange } from '$lib/fields';
	import type { Colormap, FieldComponent, FieldGrid, GaussianMode, ScaleKind } from '$lib/types';

	interface Props {
		grid: FieldGrid;
		values: Float64Array;
		component: FieldComponent;
		colormap: Colormap;
		scale: ScaleKind;
		range: ValueRange;
		overlay?: GaussianMode | null;
		offsetXUm?: number;
		offsetYUm?: number;
		axisLabel?: string;
		pixelRatio?: number;
	}

	let {
		grid,
		values,
		component,
		colormap,
		scale,
		range,
		overlay = null,
		offsetXUm = 0,
		offsetYUm = 0,
		axisLabel = '',
		pixelRatio
	}: Props = $props();

	let wrapper: HTMLDivElement;
	let canvas: HTMLCanvasElement;
	let width = $state(0);

	function bilinear(gx: number, gy: number): number {
		const samples = grid.samples;
		const x0 = Math.max(0, Math.min(samples - 1, Math.floor(gx)));
		const y0 = Math.max(0, Math.min(samples - 1, Math.floor(gy)));
		const x1 = Math.min(samples - 1, x0 + 1);
		const y1 = Math.min(samples - 1, y0 + 1);
		const fx = gx - x0;
		const fy = gy - y0;
		const v00 = values[y0 * samples + x0];
		const v10 = values[y0 * samples + x1];
		const v01 = values[y1 * samples + x0];
		const v11 = values[y1 * samples + x1];
		return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
	}

	function draw() {
		if (!canvas || width <= 0) return;
		const ratio = pixelRatio ?? Math.min(globalThis.devicePixelRatio || 1, 2);
		const size = Math.max(1, Math.round(width * ratio));
		if (canvas.width !== size || canvas.height !== size) {
			canvas.width = size;
			canvas.height = size;
		}
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const image = ctx.createImageData(size, size);
		const samples = grid.samples;
		const extentX = grid.extentXUm;
		const extentY = grid.extentYUm;
		for (let py = 0; py < size; py++) {
			const y = extentY - (2 * extentY * (py + 0.5)) / size;
			const gy = ((y + extentY) / (2 * extentY)) * (samples - 1);
			for (let px = 0; px < size; px++) {
				const x = -extentX + (2 * extentX * (px + 0.5)) / size;
				const gx = ((x + extentX) / (2 * extentX)) * (samples - 1);
				const value = bilinear(gx, gy);
				const t = normalizeValue(value, range, scale);
				const [r, g, b] = colormapRGB(colormap, t);
				const index = (py * size + px) * 4;
				image.data[index] = r;
				image.data[index + 1] = g;
				image.data[index + 2] = b;
				image.data[index + 3] = 255;
			}
		}
		ctx.putImageData(image, 0, 0);

		// Geometry overlay: MFD ellipse in physical coordinates.
		if (overlay) {
			const toPxX = (xUm: number) => ((xUm + extentX) / (2 * extentX)) * size;
			const toPxY = (yUm: number) => ((extentY - yUm) / (2 * extentY)) * size;
			ctx.beginPath();
			ctx.ellipse(
				toPxX(offsetXUm),
				toPxY(offsetYUm),
				Math.max(1, (overlay.mfdXUm / 2 / (2 * extentX)) * size),
				Math.max(1, (overlay.mfdYUm / 2 / (2 * extentY)) * size),
				0,
				0,
				2 * Math.PI
			);
			ctx.strokeStyle = 'rgba(224,224,224,0.85)';
			ctx.lineWidth = Math.max(1, ratio);
			ctx.setLineDash([4 * ratio, 3 * ratio]);
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.beginPath();
			ctx.moveTo(toPxX(offsetXUm) - 4 * ratio, toPxY(offsetYUm));
			ctx.lineTo(toPxX(offsetXUm) + 4 * ratio, toPxY(offsetYUm));
			ctx.moveTo(toPxX(offsetXUm), toPxY(offsetYUm) - 4 * ratio);
			ctx.lineTo(toPxX(offsetXUm), toPxY(offsetYUm) + 4 * ratio);
			ctx.strokeStyle = 'rgba(74,158,255,0.9)';
			ctx.stroke();
		}

		// Physical axes ticks in µm.
		ctx.font = `${Math.round(10 * ratio)}px monospace`;
		ctx.fillStyle = 'rgba(224,224,224,0.75)';
		ctx.strokeStyle = 'rgba(42,42,42,0.9)';
		ctx.lineWidth = 1;
		const ticks = [-1, -0.5, 0, 0.5, 1];
		for (const t of ticks) {
			const px = ((t + 1) / 2) * size;
			const py = ((1 - t) / 2) * size;
			ctx.beginPath();
			ctx.moveTo(px, 0);
			ctx.lineTo(px, size);
			ctx.moveTo(0, py);
			ctx.lineTo(size, py);
			ctx.stroke();
		}
		ctx.textAlign = 'left';
		ctx.fillText(`+${extentX.toFixed(1)}`, 3 * ratio, 11 * ratio);
		ctx.textAlign = 'right';
		ctx.fillText(axisLabel ? `${axisLabel} µm` : 'µm', size - 3 * ratio, size - 4 * ratio);
	}

	$effect(() => {
		// Track every display input so the canvas redraws when any changes.
		void grid;
		void values;
		void component;
		void colormap;
		void scale;
		void range;
		void overlay;
		void offsetXUm;
		void offsetYUm;
		void width;
		draw();
	});
</script>

<div class="field" bind:this={wrapper} bind:clientWidth={width}>
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.field {
		position: relative;
		width: 100%;
		aspect-ratio: 1;
		background-color: #000;
		border: 1px solid var(--border);
		overflow: hidden;
	}

	canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
</style>
