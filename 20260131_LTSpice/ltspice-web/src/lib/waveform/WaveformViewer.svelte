<script lang="ts">
	import { onMount, onDestroy, untrack } from 'svelte';
	import { WebglPlot, WebglLine, ColorRGBA } from 'webgl-plot';
	import type { TraceData, ViewBounds, Cursor, CursorReadout } from './types';

	// Props
	let {
		traces = [],
		timeData = [],
		onCursorChange = (_readout: CursorReadout) => {}
	}: {
		traces: TraceData[];
		timeData: number[];
		onCursorChange?: (readout: CursorReadout) => void;
	} = $props();

	// Notify parent of cursor changes
	function notifyCursorChange() {
		const readout: CursorReadout = {
			cursorA: cursors[0].visible ? { x: cursors[0].x, values: new Map() } : null,
			cursorB: cursors[1].visible ? { x: cursors[1].x, values: new Map() } : null,
			delta: null
		};
		onCursorChange(readout);
	}

	// State
	let canvas: HTMLCanvasElement;
	let overlayCanvas: HTMLCanvasElement;
	let wglp: WebglPlot | null = null;
	let lines: Map<string, WebglLine> = new Map();
	let animationId: number | null = null;
	let initialized = false;

	// View state - use regular variables to avoid reactivity loops
	let bounds: ViewBounds = { xMin: 0, xMax: 1, yMin: -1, yMax: 1 };
	let cursors: [Cursor, Cursor] = $state([
		{ id: 'A', x: 0.25, visible: false, color: '#ffff00' },
		{ id: 'B', x: 0.75, visible: false, color: '#00ffff' }
	]);

	// Interaction state
	let isDragging = false;
	let dragStart = { x: 0, y: 0 };
	let dragMode: 'pan' | 'zoom-x' | 'zoom-y' | 'cursor-a' | 'cursor-b' | null = null;

	onMount(() => {
		initWebGL();
		window.addEventListener('resize', handleResize);
	});

	onDestroy(() => {
		if (animationId) cancelAnimationFrame(animationId);
		window.removeEventListener('resize', handleResize);
	});

	function initWebGL() {
		if (!canvas) return;

		const dpr = window.devicePixelRatio || 1;
		canvas.width = canvas.clientWidth * dpr;
		canvas.height = canvas.clientHeight * dpr;

		if (overlayCanvas) {
			overlayCanvas.width = canvas.width;
			overlayCanvas.height = canvas.height;
		}

		try {
			wglp = new WebglPlot(canvas, { antialias: true, transparent: false });
			initialized = true;
			updateTraces();
			render();
		} catch (err) {
			console.error('Failed to initialize WebGL:', err);
		}
	}

	function handleResize() {
		if (!canvas || !wglp) return;
		const dpr = window.devicePixelRatio || 1;
		canvas.width = canvas.clientWidth * dpr;
		canvas.height = canvas.clientHeight * dpr;
		if (overlayCanvas) {
			overlayCanvas.width = canvas.width;
			overlayCanvas.height = canvas.height;
		}
		wglp.viewport(0, 0, canvas.width, canvas.height);
		render();
	}

	// React to trace changes - use untrack to prevent bounds from causing loops
	$effect(() => {
		// Read reactive props
		const tracesLen = traces.length;
		const timeLen = timeData.length;

		// Only proceed if we have data and WebGL is initialized
		if (tracesLen > 0 && timeLen > 0 && initialized && wglp) {
			// Use untrack to prevent autoscale's bounds mutation from triggering this effect again
			untrack(() => {
				try {
					updateTraces();
					autoscale();
					render();
				} catch (err) {
					console.error('Error updating waveform:', err);
				}
			});
		}
	});

	function updateTraces() {
		if (!wglp || !timeData.length) return;

		wglp.removeDataLines();
		lines.clear();

		for (let i = 0; i < traces.length; i++) {
			const trace = traces[i];
			if (!trace.visible) continue;

			const numPoints = Math.min(trace.values.length, timeData.length);
			const color = new ColorRGBA(trace.color.r, trace.color.g, trace.color.b, trace.color.a);
			const line = new WebglLine(color, numPoints);

			// Set data points
			for (let j = 0; j < numPoints; j++) {
				line.setX(j, timeData[j]);
				line.setY(j, trace.values[j]);
			}

			wglp.addDataLine(line);
			lines.set(trace.id, line);
		}
	}

	function autoscale() {
		if (!timeData.length || !traces.length) return;

		let xMin = timeData[0];
		let xMax = timeData[timeData.length - 1];
		let yMin = Infinity;
		let yMax = -Infinity;

		for (const trace of traces) {
			if (!trace.visible) continue;
			for (const v of trace.values) {
				if (v < yMin) yMin = v;
				if (v > yMax) yMax = v;
			}
		}

		// Add 10% padding
		const yPad = (yMax - yMin) * 0.1 || 0.1;
		yMin -= yPad;
		yMax += yPad;

		bounds = { xMin, xMax, yMin, yMax };
	}

	function render() {
		if (!wglp) return;

		// Calculate scale and offset for WebGL (-1 to 1 range)
		const xRange = bounds.xMax - bounds.xMin || 1;
		const yRange = bounds.yMax - bounds.yMin || 1;

		wglp.gScaleX = 2 / xRange;
		wglp.gScaleY = 2 / yRange;
		wglp.gOffsetX = -(bounds.xMin + bounds.xMax) / xRange;
		wglp.gOffsetY = -(bounds.yMin + bounds.yMax) / yRange;

		wglp.update();
		drawOverlay();
	}

	function drawOverlay() {
		if (!overlayCanvas) return;
		const ctx = overlayCanvas.getContext('2d');
		if (!ctx) return;

		const w = overlayCanvas.width;
		const h = overlayCanvas.height;
		ctx.clearRect(0, 0, w, h);

		drawGrid(ctx, w, h);

		for (const cursor of cursors) {
			if (cursor.visible) {
				drawCursor(ctx, cursor, w, h);
			}
		}

		drawAxisLabels(ctx, w, h);
	}

	function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
		ctx.strokeStyle = '#333333';
		ctx.lineWidth = 1;

		// Vertical grid lines
		const xRange = bounds.xMax - bounds.xMin;
		const xStep = calculateGridStep(xRange, w / 100);
		for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= bounds.xMax; x += xStep) {
			const px = ((x - bounds.xMin) / xRange) * w;
			ctx.beginPath();
			ctx.moveTo(px, 0);
			ctx.lineTo(px, h);
			ctx.stroke();
		}

		// Horizontal grid lines
		const yRange = bounds.yMax - bounds.yMin;
		const yStep = calculateGridStep(yRange, h / 80);
		for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= bounds.yMax; y += yStep) {
			const py = h - ((y - bounds.yMin) / yRange) * h;
			ctx.beginPath();
			ctx.moveTo(0, py);
			ctx.lineTo(w, py);
			ctx.stroke();
		}
	}

	function calculateGridStep(range: number, targetDivisions: number): number {
		const rawStep = range / targetDivisions;
		const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
		const normalized = rawStep / magnitude;
		let step: number;
		if (normalized < 2) step = 1;
		else if (normalized < 5) step = 2;
		else step = 5;
		return step * magnitude;
	}

	function drawCursor(ctx: CanvasRenderingContext2D, cursor: Cursor, w: number, h: number) {
		const xRange = bounds.xMax - bounds.xMin;
		const px = ((cursor.x - bounds.xMin) / xRange) * w;

		ctx.strokeStyle = cursor.color;
		ctx.lineWidth = 2;
		ctx.setLineDash([5, 5]);
		ctx.beginPath();
		ctx.moveTo(px, 0);
		ctx.lineTo(px, h);
		ctx.stroke();
		ctx.setLineDash([]);

		// Cursor label
		ctx.fillStyle = cursor.color;
		ctx.font = '12px monospace';
		ctx.fillText(cursor.id, px + 4, 16);
	}

	function drawAxisLabels(ctx: CanvasRenderingContext2D, w: number, h: number) {
		ctx.fillStyle = '#888888';
		ctx.font = '10px monospace';

		// X-axis labels
		const xRange = bounds.xMax - bounds.xMin;
		const xStep = calculateGridStep(xRange, w / 100);
		for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= bounds.xMax; x += xStep) {
			const px = ((x - bounds.xMin) / xRange) * w;
			ctx.fillText(formatValue(x, 's'), px + 2, h - 4);
		}

		// Y-axis labels
		const yRange = bounds.yMax - bounds.yMin;
		const yStep = calculateGridStep(yRange, h / 80);
		for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= bounds.yMax; y += yStep) {
			const py = h - ((y - bounds.yMin) / yRange) * h;
			ctx.fillText(formatValue(y, 'V'), 4, py - 2);
		}
	}

	function formatValue(value: number, unit: string): string {
		const abs = Math.abs(value);
		if (abs === 0) return `0${unit}`;
		if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M${unit}`;
		if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}k${unit}`;
		if (abs >= 1) return `${value.toFixed(2)}${unit}`;
		if (abs >= 1e-3) return `${(value * 1e3).toFixed(1)}m${unit}`;
		if (abs >= 1e-6) return `${(value * 1e6).toFixed(1)}u${unit}`;
		if (abs >= 1e-9) return `${(value * 1e9).toFixed(1)}n${unit}`;
		return `${(value * 1e12).toFixed(1)}p${unit}`;
	}

	// Mouse event handlers
	function handleMouseDown(e: MouseEvent) {
		isDragging = true;
		dragStart = { x: e.offsetX, y: e.offsetY };

		if (e.shiftKey) {
			dragMode = 'zoom-x';
		} else if (e.ctrlKey || e.metaKey) {
			dragMode = 'zoom-y';
		} else {
			dragMode = 'pan';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isDragging || !dragMode) return;

		const dx = e.offsetX - dragStart.x;
		const dy = e.offsetY - dragStart.y;
		const w = canvas.clientWidth;
		const h = canvas.clientHeight;

		if (dragMode === 'pan') {
			const xRange = bounds.xMax - bounds.xMin;
			const yRange = bounds.yMax - bounds.yMin;
			const xShift = -(dx / w) * xRange;
			const yShift = (dy / h) * yRange;
			bounds = {
				xMin: bounds.xMin + xShift,
				xMax: bounds.xMax + xShift,
				yMin: bounds.yMin + yShift,
				yMax: bounds.yMax + yShift
			};
		}

		dragStart = { x: e.offsetX, y: e.offsetY };
		render();
	}

	function handleMouseUp() {
		isDragging = false;
		dragMode = null;
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
		const rect = canvas.getBoundingClientRect();
		const mouseX = (e.clientX - rect.left) / rect.width;
		const mouseY = 1 - (e.clientY - rect.top) / rect.height;

		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;
		const xPos = bounds.xMin + mouseX * xRange;
		const yPos = bounds.yMin + mouseY * yRange;

		if (e.shiftKey) {
			// Zoom Y only
			const newYRange = yRange * zoomFactor;
			bounds = {
				...bounds,
				yMin: yPos - mouseY * newYRange,
				yMax: yPos + (1 - mouseY) * newYRange
			};
		} else if (e.ctrlKey || e.metaKey) {
			// Zoom X only
			const newXRange = xRange * zoomFactor;
			bounds = {
				...bounds,
				xMin: xPos - mouseX * newXRange,
				xMax: xPos + (1 - mouseX) * newXRange
			};
		} else {
			// Zoom both
			const newXRange = xRange * zoomFactor;
			const newYRange = yRange * zoomFactor;
			bounds = {
				xMin: xPos - mouseX * newXRange,
				xMax: xPos + (1 - mouseX) * newXRange,
				yMin: yPos - mouseY * newYRange,
				yMax: yPos + (1 - mouseY) * newYRange
			};
		}

		render();
	}

	function handleDoubleClick() {
		autoscale();
		render();
	}

	// Keyboard handlers
	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'a' || e.key === 'A') {
			cursors[0].visible = !cursors[0].visible;
			notifyCursorChange();
			render();
		} else if (e.key === 'b' || e.key === 'B') {
			cursors[1].visible = !cursors[1].visible;
			notifyCursorChange();
			render();
		} else if (e.key === 'f' || e.key === 'F') {
			autoscale();
			render();
		}
	}

	export function setData(_newTraces: TraceData[], _newTimeData: number[]) {
		// External API to set data - handled via props reactivity
	}

	export function zoomToFit() {
		autoscale();
		render();
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	class="waveform-container"
	onmousedown={handleMouseDown}
	onmousemove={handleMouseMove}
	onmouseup={handleMouseUp}
	onmouseleave={handleMouseUp}
	onwheel={handleWheel}
	ondblclick={handleDoubleClick}
	onkeydown={handleKeyDown}
	tabindex="0"
	role="application"
	aria-label="Waveform viewer - use mouse to pan/zoom, A/B keys for cursors"
>
	<canvas bind:this={canvas} class="waveform-canvas"></canvas>
	<canvas bind:this={overlayCanvas} class="overlay-canvas"></canvas>
	<div class="legend">
		{#each traces as trace}
			{#if trace.visible}
				<div class="legend-item" style="color: rgb({trace.color.r * 255}, {trace.color.g * 255}, {trace.color.b * 255})">
					<span class="legend-color"></span>
					{trace.name}
				</div>
			{/if}
		{/each}
	</div>
</div>

<style>
	.waveform-container {
		position: relative;
		width: 100%;
		height: 100%;
		background: #000000;
		overflow: hidden;
	}

	.waveform-canvas,
	.overlay-canvas {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}

	.overlay-canvas {
		pointer-events: none;
	}

	.legend {
		position: absolute;
		top: 8px;
		right: 8px;
		background: rgba(0, 0, 0, 0.7);
		padding: 4px 8px;
		border-radius: 4px;
		font-size: 11px;
		font-family: monospace;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 2px 0;
	}

	.legend-color {
		width: 12px;
		height: 2px;
		background: currentColor;
	}
</style>

