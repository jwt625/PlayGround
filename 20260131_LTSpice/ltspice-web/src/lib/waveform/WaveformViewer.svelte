<script lang="ts">
	import { onMount, onDestroy, untrack } from 'svelte';
	import { WebglPlot, WebglLine, ColorRGBA } from 'webgl-plot';
	import type { TraceData, ViewBounds, Cursor } from './types';

	let { traces = [], timeData = [] }: { traces: TraceData[]; timeData: number[] } = $props();

	let gridCanvas: HTMLCanvasElement;
	let canvas: HTMLCanvasElement;
	let overlayCanvas: HTMLCanvasElement;
	let wglp: WebglPlot | null = null;
	let initialized = false;

	let bounds: ViewBounds = { xMin: 0, xMax: 1, yMin: -1, yMax: 1 };
	let cursors: [Cursor, Cursor] = $state([
		{ id: 'A', x: 0.25, visible: false, color: '#ffff00' },
		{ id: 'B', x: 0.75, visible: false, color: '#00ffff' }
	]);

	let showGrid = $state(true);
	let zoomStart: { x: number; y: number } | null = null;
	let zoomEnd: { x: number; y: number } | null = null;
	let zoomType: 'x' | 'y' | 'rect' | null = $state(null);
	let mousePos = $state({ x: 0, y: 0, dataX: 0, dataY: 0 });
	let showTooltip = $state(false);
	let isDragging = false;
	let dragStart = { x: 0, y: 0 };
	let dragMode: 'pan' | 'zoom' | null = null;

	const getDpr = () => window.devicePixelRatio || 1;

	onMount(() => {
		initWebGL();
		window.addEventListener('resize', handleResize);
	});

	onDestroy(() => window.removeEventListener('resize', handleResize));

	function initWebGL() {
		if (!canvas) return;
		const d = getDpr();
		const w = canvas.clientWidth * d, h = canvas.clientHeight * d;
		canvas.width = w; canvas.height = h;
		if (gridCanvas) { gridCanvas.width = w; gridCanvas.height = h; }
		if (overlayCanvas) { overlayCanvas.width = w; overlayCanvas.height = h; }
		try {
			wglp = new WebglPlot(canvas, { antialias: true, transparent: true });
			initialized = true;
			updateTraces();
			render();
		} catch (err) {
			console.error('WebGL init failed:', err);
		}
	}

	function handleResize() {
		if (!canvas || !wglp) return;
		const d = getDpr();
		const w = canvas.clientWidth * d, h = canvas.clientHeight * d;
		canvas.width = w; canvas.height = h;
		if (gridCanvas) { gridCanvas.width = w; gridCanvas.height = h; }
		if (overlayCanvas) { overlayCanvas.width = w; overlayCanvas.height = h; }
		wglp.viewport(0, 0, w, h);
		render();
	}

	$effect(() => {
		if (traces.length > 0 && timeData.length > 0 && initialized && wglp) {
			untrack(() => { updateTraces(); autoscale(); render(); });
		}
	});

	function updateTraces() {
		if (!wglp || !timeData.length) return;
		wglp.removeDataLines();
		for (const trace of traces) {
			if (!trace.visible) continue;
			const n = Math.min(trace.values.length, timeData.length);
			const line = new WebglLine(new ColorRGBA(trace.color.r, trace.color.g, trace.color.b, trace.color.a), n);
			for (let j = 0; j < n; j++) { line.setX(j, timeData[j]); line.setY(j, trace.values[j]); }
			wglp.addDataLine(line);
		}
	}

	function autoscale() {
		if (!timeData.length || !traces.length) return;
		let yMin = Infinity, yMax = -Infinity;
		for (const t of traces) {
			if (!t.visible) continue;
			for (const v of t.values) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
		}
		const pad = (yMax - yMin) * 0.1 || 0.1;
		bounds = { xMin: timeData[0], xMax: timeData[timeData.length - 1], yMin: yMin - pad, yMax: yMax + pad };
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
		drawGridLayer();
		drawOverlay();
	}

	function drawGridLayer() {
		if (!gridCanvas) return;
		const ctx = gridCanvas.getContext('2d');
		if (!ctx) return;

		const w = gridCanvas.width;
		const h = gridCanvas.height;
		ctx.clearRect(0, 0, w, h);

		if (showGrid) {
			drawGrid(ctx, w, h);
		}
	}

	function drawOverlay() {
		if (!overlayCanvas) return;
		const ctx = overlayCanvas.getContext('2d');
		if (!ctx) return;

		const w = overlayCanvas.width;
		const h = overlayCanvas.height;
		ctx.clearRect(0, 0, w, h);

		if (showTooltip && !zoomType && traces.length > 0) drawHoverLine(ctx, h);
		for (const c of cursors) if (c.visible) drawCursor(ctx, c, w, h);
		drawAxisLabels(ctx, w, h);
		if (zoomType && zoomStart && zoomEnd) drawZoomSelection(ctx, w, h);
	}

	const rgb = (c: {r:number,g:number,b:number}, a=1) => a < 1 ? `rgba(${c.r*255},${c.g*255},${c.b*255},${a})` : `rgb(${c.r*255},${c.g*255},${c.b*255})`;
	const toY = (v: number, h: number) => h - ((v - bounds.yMin) / (bounds.yMax - bounds.yMin)) * h;
	const toX = (v: number, w: number) => ((v - bounds.xMin) / (bounds.xMax - bounds.xMin)) * w;

	function niceStep(range: number, divs: number): number {
		if (divs <= 0 || range <= 0) return range || 1;
		const raw = range / divs, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
		return mag * (n < 1.25 ? 1 : n < 1.75 ? 1.5 : n < 2.25 ? 2 : n < 3.5 ? 2.5 : n < 7.5 ? 5 : 10);
	}

	function getTraceValuesAtX(x: number): Map<string, number> {
		const values = new Map<string, number>();
		if (!timeData.length) return values;
		let idx = timeData.findIndex(t => t >= x);
		if (idx < 0) idx = timeData.length - 1;
		if (idx > 0 && idx < timeData.length) {
			const t0 = timeData[idx - 1], t1 = timeData[idx], r = t1 !== t0 ? (x - t0) / (t1 - t0) : 0;
			for (const t of traces) if (t.visible) values.set(t.name, t.values[idx - 1] + (t.values[idx] - t.values[idx - 1]) * r);
		} else {
			for (const t of traces) if (t.visible) values.set(t.name, t.values[idx] || 0);
		}
		return values;
	}

	function formatValue(v: number, u: string): string {
		const a = Math.abs(v);
		if (a === 0) return `0${u}`;
		if (a >= 1e6) return `${(v/1e6).toFixed(1)}M${u}`;
		if (a >= 1e3) return `${(v/1e3).toFixed(1)}k${u}`;
		if (a >= 1) return `${v.toFixed(2)}${u}`;
		if (a >= 1e-3) return `${(v*1e3).toFixed(1)}m${u}`;
		if (a >= 1e-6) return `${(v*1e6).toFixed(1)}u${u}`;
		if (a >= 1e-9) return `${(v*1e9).toFixed(1)}n${u}`;
		return `${(v*1e12).toFixed(1)}p${u}`;
	}

	function drawHoverLine(ctx: CanvasRenderingContext2D, h: number) {
		const d = getDpr(), px = mousePos.x * d;
		ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
		ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke(); ctx.setLineDash([]);
		for (const [name, val] of getTraceValuesAtX(mousePos.dataX)) {
			const t = traces.find(tr => tr.name === name); if (!t?.visible) continue;
			const py = toY(val, h);
			ctx.fillStyle = rgb(t.color); ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
		}
	}

	function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
		const d = getDpr(), sp = 40 * d;
		ctx.strokeStyle = '#333'; ctx.lineWidth = d;
		const xR = bounds.xMax - bounds.xMin, yR = bounds.yMax - bounds.yMin;
		const xS = niceStep(xR, Math.max(4, Math.round(w/sp)));
		const yS = niceStep(yR, Math.max(4, Math.round(h/sp)));
		for (let x = Math.ceil(bounds.xMin/xS)*xS; x <= bounds.xMax; x += xS) {
			const px = toX(x, w); ctx.beginPath(); ctx.moveTo(Math.round(px)+0.5, 0); ctx.lineTo(Math.round(px)+0.5, h); ctx.stroke();
		}
		for (let y = Math.ceil(bounds.yMin/yS)*yS; y <= bounds.yMax; y += yS) {
			const py = toY(y, h); ctx.beginPath(); ctx.moveTo(0, Math.round(py)+0.5); ctx.lineTo(w, Math.round(py)+0.5); ctx.stroke();
		}
	}

	function drawCursor(ctx: CanvasRenderingContext2D, cur: Cursor, w: number, h: number) {
		const px = toX(cur.x, w), vals = getTraceValuesAtX(cur.x);
		ctx.strokeStyle = cur.color; ctx.lineWidth = 2; ctx.setLineDash([5,5]);
		ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke(); ctx.setLineDash([]);
		for (const [name, val] of vals) {
			const t = traces.find(tr => tr.name === name); if (!t?.visible) continue;
			const py = toY(val, h);
			ctx.fillStyle = rgb(t.color); ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI*2); ctx.fill();
			ctx.strokeStyle = rgb(t.color, 0.5); ctx.lineWidth = 1; ctx.setLineDash([3,3]);
			ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(0, py); ctx.stroke(); ctx.setLineDash([]);
		}
		const bx = (px + 128 > w) ? px - 128 : px + 8, by = 20, lh = 14;
		ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.fillRect(bx, by, 120, 20 + vals.size * lh);
		ctx.strokeStyle = cur.color; ctx.lineWidth = 1; ctx.strokeRect(bx, by, 120, 20 + vals.size * lh);
		ctx.fillStyle = cur.color; ctx.font = 'bold 11px monospace';
		ctx.fillText(`${cur.id}: ${formatValue(cur.x, 's')}`, bx + 4, by + 14);
		ctx.font = '10px monospace'; let yo = by + 28;
		for (const [name, val] of vals) {
			const t = traces.find(tr => tr.name === name); if (!t?.visible) continue;
			ctx.fillStyle = rgb(t.color); ctx.fillText(formatValue(val, 'V'), bx + 4, yo); yo += lh;
		}
	}

	function drawZoomSelection(ctx: CanvasRenderingContext2D, canvasW: number, canvasH: number) {
		if (!zoomStart || !zoomEnd || !zoomType) return;
		ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
		ctx.fillStyle = 'rgba(255,255,255,0.1)';

		if (zoomType === 'x') {
			// Horizontal zoom - full height band
			const x = Math.min(zoomStart.x, zoomEnd.x);
			const w = Math.abs(zoomEnd.x - zoomStart.x);
			ctx.fillRect(x, 0, w, canvasH);
			ctx.strokeRect(x, 0, w, canvasH);
		} else if (zoomType === 'y') {
			// Vertical zoom - full width band
			const y = Math.min(zoomStart.y, zoomEnd.y);
			const h = Math.abs(zoomEnd.y - zoomStart.y);
			ctx.fillRect(0, y, canvasW, h);
			ctx.strokeRect(0, y, canvasW, h);
		} else {
			// Rectangle zoom
			const x = Math.min(zoomStart.x, zoomEnd.x), y = Math.min(zoomStart.y, zoomEnd.y);
			const w = Math.abs(zoomEnd.x - zoomStart.x), h = Math.abs(zoomEnd.y - zoomStart.y);
			ctx.strokeRect(x, y, w, h);
			ctx.fillRect(x, y, w, h);
		}
		ctx.setLineDash([]);
	}

	function drawAxisLabels(ctx: CanvasRenderingContext2D, w: number, h: number) {
		const d = getDpr(), sp = 40 * d;
		ctx.fillStyle = '#888'; ctx.font = `${10*d}px monospace`;
		const xR = bounds.xMax - bounds.xMin, yR = bounds.yMax - bounds.yMin;
		const xS = niceStep(xR, Math.max(4, Math.round(w/sp))), yS = niceStep(yR, Math.max(4, Math.round(h/sp)));
		for (let x = Math.ceil(bounds.xMin/xS)*xS; x <= bounds.xMax; x += xS) ctx.fillText(formatValue(x, 's'), toX(x,w)+2, h-4*d);
		for (let y = Math.ceil(bounds.yMin/yS)*yS; y <= bounds.yMax; y += yS) ctx.fillText(formatValue(y, 'V'), 4*d, toY(y,h)-2*d);
	}

	function handleMouseDown(e: MouseEvent) {
		const d = getDpr();
		dragStart = { x: e.offsetX * d, y: e.offsetY * d };

		if (e.button === 0) {
			// Left click: zoom mode (Plotly-style)
			isDragging = true;
			dragMode = 'zoom';
			zoomStart = zoomEnd = { x: e.offsetX * d, y: e.offsetY * d };
			zoomType = null;
		} else if (e.button === 1) {
			// Middle click: pan
			isDragging = true;
			dragMode = 'pan';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const d = getDpr(), w = canvas.clientWidth, h = canvas.clientHeight;
		const xRange = bounds.xMax - bounds.xMin, yRange = bounds.yMax - bounds.yMin;
		mousePos = { x: e.offsetX, y: e.offsetY, dataX: bounds.xMin + (e.offsetX / w) * xRange, dataY: bounds.yMax - (e.offsetY / h) * yRange };
		showTooltip = true;

		if (!isDragging || !dragMode) { render(); return; }

		if (dragMode === 'zoom') {
			zoomEnd = { x: e.offsetX * d, y: e.offsetY * d };

			// Determine zoom type based on drag direction (Plotly-style)
			if (zoomStart) {
				const dx = Math.abs(zoomEnd.x - zoomStart.x);
				const dy = Math.abs(zoomEnd.y - zoomStart.y);
				const minDrag = 10; // Minimum pixels to start zoom

				if (dx < minDrag && dy < minDrag) {
					zoomType = null; // Not enough movement yet
				} else {
					const ratio = dx / (dy + 0.001); // Avoid division by zero
					if (ratio > 3) {
						zoomType = 'x'; // Mostly horizontal
					} else if (ratio < 0.33) {
						zoomType = 'y'; // Mostly vertical
					} else {
						zoomType = 'rect'; // Diagonal
					}
				}
			}
		} else if (dragMode === 'pan') {
			const dx = e.offsetX * d - dragStart.x, dy = e.offsetY * d - dragStart.y;
			bounds = { xMin: bounds.xMin - (dx / (w * d)) * xRange, xMax: bounds.xMax - (dx / (w * d)) * xRange,
				yMin: bounds.yMin + (dy / (h * d)) * yRange, yMax: bounds.yMax + (dy / (h * d)) * yRange };
			dragStart = { x: e.offsetX * d, y: e.offsetY * d };
		}
		render();
	}

	function handleMouseUp() {
		if (dragMode === 'zoom' && zoomStart && zoomEnd && zoomType) {
			const w = canvas.width, h = canvas.height;
			const xRange = bounds.xMax - bounds.xMin, yRange = bounds.yMax - bounds.yMin;
			const x1 = Math.min(zoomStart.x, zoomEnd.x), x2 = Math.max(zoomStart.x, zoomEnd.x);
			const y1 = Math.min(zoomStart.y, zoomEnd.y), y2 = Math.max(zoomStart.y, zoomEnd.y);

			if (zoomType === 'x' && x2 - x1 > 10) {
				// X-only zoom
				bounds = { ...bounds,
					xMin: bounds.xMin + (x1/w)*xRange,
					xMax: bounds.xMin + (x2/w)*xRange };
			} else if (zoomType === 'y' && y2 - y1 > 10) {
				// Y-only zoom
				bounds = { ...bounds,
					yMin: bounds.yMax - (y2/h)*yRange,
					yMax: bounds.yMax - (y1/h)*yRange };
			} else if (zoomType === 'rect' && x2 - x1 > 10 && y2 - y1 > 10) {
				// Rectangle zoom
				bounds = { xMin: bounds.xMin + (x1/w)*xRange, xMax: bounds.xMin + (x2/w)*xRange,
					yMin: bounds.yMax - (y2/h)*yRange, yMax: bounds.yMax - (y1/h)*yRange };
			}
			render();
		}
		zoomStart = zoomEnd = null;
		zoomType = null;
		isDragging = false;
		dragMode = null;
	}

	function handleMouseLeave() { showTooltip = false; handleMouseUp(); }

	function handleWheel(e: WheelEvent) {
		e.preventDefault();

		// Normalize deltaY across browsers - positive = zoom out, negative = zoom in
		const delta = e.deltaY !== 0 ? e.deltaY : e.deltaX;
		const zoomIn = delta < 0;
		const zoomFactor = zoomIn ? 0.9 : 1.1; // < 1 = zoom in (smaller range), > 1 = zoom out

		const rect = canvas.getBoundingClientRect();
		const mouseX = (e.clientX - rect.left) / rect.width;
		const mouseY = 1 - (e.clientY - rect.top) / rect.height;

		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;
		const xPos = bounds.xMin + mouseX * xRange;
		const yPos = bounds.yMin + mouseY * yRange;

		if (e.shiftKey) {
			// Zoom Y only - apply zoom factor to Y range only
			const newYRange = yRange * zoomFactor;
			const newYMin = yPos - mouseY * newYRange;
			const newYMax = yPos + (1 - mouseY) * newYRange;
			bounds = {
				...bounds,
				yMin: newYMin,
				yMax: newYMax
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

	function handleKeyDown(e: KeyboardEvent) {
		const xRange = bounds.xMax - bounds.xMin, yRange = bounds.yMax - bounds.yMin;
		const xc = (bounds.xMin + bounds.xMax) / 2, yc = (bounds.yMin + bounds.yMax) / 2;
		const zf = 0.8, pan = 0.1;

		if (e.key === 'a' || e.key === 'A') { cursors[0].visible = !cursors[0].visible; render(); }
		else if (e.key === 'b' || e.key === 'B') { cursors[1].visible = !cursors[1].visible; render(); }
		else if (e.key === 'f' || e.key === 'F') { autoscale(); render(); }
		else if (e.key === 'g' || e.key === 'G') { showGrid = !showGrid; render(); }
		else if (e.key === 'z' || e.key === 'Z') { zoomStart = zoomEnd = null; zoomType = null; }
		else if (e.key === '+' || e.key === '=') {
			e.preventDefault();
			bounds = { xMin: xc - xRange*zf/2, xMax: xc + xRange*zf/2, yMin: yc - yRange*zf/2, yMax: yc + yRange*zf/2 };
			render();
		} else if (e.key === '-' || e.key === '_') {
			e.preventDefault();
			bounds = { xMin: xc - xRange/zf/2, xMax: xc + xRange/zf/2, yMin: yc - yRange/zf/2, yMax: yc + yRange/zf/2 };
			render();
		} else if (e.key === 'ArrowLeft') { e.preventDefault(); bounds = { ...bounds, xMin: bounds.xMin - xRange*pan, xMax: bounds.xMax - xRange*pan }; render(); }
		else if (e.key === 'ArrowRight') { e.preventDefault(); bounds = { ...bounds, xMin: bounds.xMin + xRange*pan, xMax: bounds.xMax + xRange*pan }; render(); }
		else if (e.key === 'ArrowUp') { e.preventDefault(); bounds = { ...bounds, yMin: bounds.yMin + yRange*pan, yMax: bounds.yMax + yRange*pan }; render(); }
		else if (e.key === 'ArrowDown') { e.preventDefault(); bounds = { ...bounds, yMin: bounds.yMin - yRange*pan, yMax: bounds.yMax - yRange*pan }; render(); }
		else if (e.key === 'Escape') { zoomStart = zoomEnd = null; zoomType = null; isDragging = false; dragMode = null; render(); }
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	class="waveform-container"
	class:zooming={zoomType !== null}
	onmousedown={handleMouseDown}
	onmousemove={handleMouseMove}
	onmouseup={handleMouseUp}
	onmouseleave={handleMouseLeave}
	onwheel={handleWheel}
	ondblclick={handleDoubleClick}
	onkeydown={handleKeyDown}
	tabindex="0"
	role="application"
	aria-label="Waveform viewer - use mouse to pan/zoom, A/B keys for cursors"
>
	<canvas bind:this={gridCanvas} class="grid-canvas"></canvas>
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

	{#if showTooltip && !zoomType && traces.length > 0}
		<div class="tooltip" style="left: {mousePos.x + 15}px; top: {mousePos.y + 15}px;">
			<div class="tooltip-header">X: {formatValue(mousePos.dataX, 's')}</div>
			{#each traces as trace}
				{#if trace.visible}
					{@const values = getTraceValuesAtX(mousePos.dataX)}
					{@const val = values.get(trace.name)}
					{#if val !== undefined}
						<div class="tooltip-row" style="color: rgb({trace.color.r * 255}, {trace.color.g * 255}, {trace.color.b * 255})">
							{trace.name}: {formatValue(val, 'V')}
						</div>
					{/if}
				{/if}
			{/each}
		</div>
	{/if}

	{#if zoomType}
		<div class="zoom-mode-indicator">
			{#if zoomType === 'x'}X ZOOM{:else if zoomType === 'y'}Y ZOOM{:else}BOX ZOOM{/if}
		</div>
	{/if}
</div>

<style>
	.waveform-container {
		position: relative;
		width: 100%;
		height: 100%;
		background: #000000;
		overflow: hidden;
		cursor: crosshair;
	}

	.waveform-container.zooming {
		cursor: crosshair;
	}

	.grid-canvas,
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
		font-size: 11px;
		font-family: monospace;
		pointer-events: none;
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

	.tooltip {
		position: absolute;
		background: rgba(0, 0, 0, 0.9);
		border: 1px solid #444;
		padding: 6px 10px;
		font-family: monospace;
		font-size: 10px;
		pointer-events: none;
		z-index: 100;
		max-width: 200px;
		white-space: nowrap;
	}

	.tooltip-header {
		color: #aaa;
		margin-bottom: 4px;
		padding-bottom: 4px;
		border-bottom: 1px solid #333;
	}

	.tooltip-row {
		padding: 1px 0;
	}

	.zoom-mode-indicator {
		position: absolute;
		top: 8px;
		left: 50%;
		transform: translateX(-50%);
		background: rgba(255, 200, 0, 0.9);
		color: #000;
		padding: 4px 12px;
		font-family: monospace;
		font-size: 11px;
		font-weight: bold;
		pointer-events: none;
	}
</style>

