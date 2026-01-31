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
	let gridCanvas: HTMLCanvasElement;
	let canvas: HTMLCanvasElement;
	let overlayCanvas: HTMLCanvasElement;
	let containerEl: HTMLDivElement;
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

	// UI state
	let showGrid = $state(true);
	let zoomRectMode = $state(false);
	let zoomRectStart: { x: number; y: number } | null = null;
	let zoomRectEnd: { x: number; y: number } | null = null;
	let mousePos = $state({ x: 0, y: 0, dataX: 0, dataY: 0 });
	let showTooltip = $state(false);

	// Interaction state
	let isDragging = false;
	let dragStart = { x: 0, y: 0 };
	let dragMode: 'pan' | 'zoom-x' | 'zoom-y' | 'zoom-rect' | 'cursor-a' | 'cursor-b' | null = null;

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
		if (gridCanvas) {
			gridCanvas.width = canvas.width;
			gridCanvas.height = canvas.height;
		}
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

		// Draw hover crosshair line when tooltip is visible
		if (showTooltip && !zoomRectMode && traces.length > 0) {
			drawHoverLine(ctx, w, h);
		}

		for (const cursor of cursors) {
			if (cursor.visible) {
				drawCursor(ctx, cursor, w, h);
			}
		}

		drawAxisLabels(ctx, w, h);

		// Draw zoom rectangle if in zoom-rect mode
		if (zoomRectMode && zoomRectStart && zoomRectEnd) {
			drawZoomRect(ctx, w, h);
		}
	}

	function drawHoverLine(ctx: CanvasRenderingContext2D, _w: number, h: number) {
		const yRange = bounds.yMax - bounds.yMin;
		const dpr = window.devicePixelRatio || 1;
		const px = mousePos.x * dpr;

		// Draw vertical dashed line at mouse X position
		ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 4]);
		ctx.beginPath();
		ctx.moveTo(px, 0);
		ctx.lineTo(px, h);
		ctx.stroke();
		ctx.setLineDash([]);

		// Draw markers on each trace at the hover X position
		const traceValues = getTraceValuesAtX(mousePos.dataX);
		for (const [traceName, value] of traceValues) {
			const trace = traces.find(t => t.name === traceName);
			if (!trace || !trace.visible) continue;

			const py = h - ((value - bounds.yMin) / yRange) * h;

			// Draw small circle marker
			ctx.fillStyle = `rgb(${trace.color.r * 255}, ${trace.color.g * 255}, ${trace.color.b * 255})`;
			ctx.beginPath();
			ctx.arc(px, py, 3, 0, Math.PI * 2);
			ctx.fill();

			// Draw horizontal dashed line to Y axis
			ctx.strokeStyle = `rgba(${trace.color.r * 255}, ${trace.color.g * 255}, ${trace.color.b * 255}, 0.3)`;
			ctx.lineWidth = 1;
			ctx.setLineDash([2, 2]);
			ctx.beginPath();
			ctx.moveTo(px, py);
			ctx.lineTo(0, py);
			ctx.stroke();
			ctx.setLineDash([]);
		}
	}

	function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
		const dpr = window.devicePixelRatio || 1;

		// Grid styling - subtle dark gray, 1 CSS pixel wide
		ctx.strokeStyle = '#333333';
		ctx.lineWidth = dpr; // 1 CSS pixel

		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;

		// Target pixel spacing between grid lines
		const targetSpacingCss = 40; // CSS pixels
		const targetSpacingCanvas = targetSpacingCss * dpr;

		// X axis: how many divisions fit at target spacing?
		const xTargetDivisions = Math.max(4, Math.round(w / targetSpacingCanvas));
		const xStep = calculateNiceStep(xRange, xTargetDivisions);

		for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= bounds.xMax; x += xStep) {
			const px = ((x - bounds.xMin) / xRange) * w;
			ctx.beginPath();
			ctx.moveTo(Math.round(px) + 0.5, 0);
			ctx.lineTo(Math.round(px) + 0.5, h);
			ctx.stroke();
		}

		// Y axis: 2x density compared to X
		const yTargetDivisions = Math.max(4, Math.round(h / targetSpacingCanvas) * 2);
		const yStep = calculateNiceStep(yRange, yTargetDivisions);

		for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= bounds.yMax; y += yStep) {
			const py = h - ((y - bounds.yMin) / yRange) * h;
			ctx.beginPath();
			ctx.moveTo(0, Math.round(py) + 0.5);
			ctx.lineTo(w, Math.round(py) + 0.5);
			ctx.stroke();
		}
	}

	// Calculate a "nice" step value that gives approximately targetDivisions
	// Nice values are 1, 1.5, 2, 2.5, 5 × 10^n - more options for finer control
	function calculateNiceStep(range: number, targetDivisions: number): number {
		if (targetDivisions <= 0 || range <= 0) return range || 1;

		const rawStep = range / targetDivisions;
		const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
		const normalized = rawStep / magnitude;

		// Pick the nice number that gets us closest to target divisions
		// More granular nice numbers: 1, 1.5, 2, 2.5, 5, 10
		let niceStep: number;
		if (normalized < 1.25) niceStep = 1;
		else if (normalized < 1.75) niceStep = 1.5;
		else if (normalized < 2.25) niceStep = 2;
		else if (normalized < 3.5) niceStep = 2.5;
		else if (normalized < 7.5) niceStep = 5;
		else niceStep = 10;

		return niceStep * magnitude;
	}

	function drawCursor(ctx: CanvasRenderingContext2D, cursor: Cursor, w: number, h: number) {
		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;
		const px = ((cursor.x - bounds.xMin) / xRange) * w;

		ctx.strokeStyle = cursor.color;
		ctx.lineWidth = 2;
		ctx.setLineDash([5, 5]);
		ctx.beginPath();
		ctx.moveTo(px, 0);
		ctx.lineTo(px, h);
		ctx.stroke();
		ctx.setLineDash([]);

		// Get trace values at cursor X position
		const traceValues = getTraceValuesAtX(cursor.x);

		// Draw markers on traces and build tooltip content
		ctx.font = '11px monospace';

		for (const [traceName, value] of traceValues) {
			const trace = traces.find(t => t.name === traceName);
			if (!trace || !trace.visible) continue;

			// Draw marker dot on trace
			const py = h - ((value - bounds.yMin) / yRange) * h;
			ctx.fillStyle = `rgb(${trace.color.r * 255}, ${trace.color.g * 255}, ${trace.color.b * 255})`;
			ctx.beginPath();
			ctx.arc(px, py, 4, 0, Math.PI * 2);
			ctx.fill();

			// Draw horizontal dashed line from marker to Y axis
			ctx.strokeStyle = `rgba(${trace.color.r * 255}, ${trace.color.g * 255}, ${trace.color.b * 255}, 0.5)`;
			ctx.lineWidth = 1;
			ctx.setLineDash([3, 3]);
			ctx.beginPath();
			ctx.moveTo(px, py);
			ctx.lineTo(0, py);
			ctx.stroke();
			ctx.setLineDash([]);
		}

		// Draw cursor info box
		const boxX = px + 8;
		const boxY = 20;
		const lineHeight = 14;
		const boxHeight = 20 + traceValues.size * lineHeight;
		const boxWidth = 120;

		// Adjust box position if too close to right edge
		const adjustedBoxX = (boxX + boxWidth > w) ? px - boxWidth - 8 : boxX;

		// Background
		ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
		ctx.fillRect(adjustedBoxX, boxY, boxWidth, boxHeight);
		ctx.strokeStyle = cursor.color;
		ctx.lineWidth = 1;
		ctx.strokeRect(adjustedBoxX, boxY, boxWidth, boxHeight);

		// Cursor label and X value
		ctx.fillStyle = cursor.color;
		ctx.font = 'bold 11px monospace';
		ctx.fillText(`${cursor.id}: ${formatValue(cursor.x, 's')}`, adjustedBoxX + 4, boxY + 14);

		// Trace values
		ctx.font = '10px monospace';
		let yOffset = boxY + 28;
		for (const [traceName, value] of traceValues) {
			const trace = traces.find(t => t.name === traceName);
			if (!trace || !trace.visible) continue;
			ctx.fillStyle = `rgb(${trace.color.r * 255}, ${trace.color.g * 255}, ${trace.color.b * 255})`;
			ctx.fillText(`${formatValue(value, 'V')}`, adjustedBoxX + 4, yOffset);
			yOffset += lineHeight;
		}
	}

	function getTraceValuesAtX(x: number): Map<string, number> {
		const values = new Map<string, number>();
		if (!timeData.length) return values;

		// Find the index in timeData closest to x
		let idx = 0;
		for (let i = 0; i < timeData.length; i++) {
			if (timeData[i] >= x) {
				idx = i;
				break;
			}
			idx = i;
		}

		// Interpolate if possible
		if (idx > 0 && idx < timeData.length) {
			const t0 = timeData[idx - 1];
			const t1 = timeData[idx];
			const ratio = (t1 !== t0) ? (x - t0) / (t1 - t0) : 0;

			for (const trace of traces) {
				if (!trace.visible) continue;
				const v0 = trace.values[idx - 1];
				const v1 = trace.values[idx];
				const interpolated = v0 + (v1 - v0) * ratio;
				values.set(trace.name, interpolated);
			}
		} else {
			for (const trace of traces) {
				if (!trace.visible) continue;
				values.set(trace.name, trace.values[idx] || 0);
			}
		}

		return values;
	}

	function drawZoomRect(ctx: CanvasRenderingContext2D, _w: number, _h: number) {
		if (!zoomRectStart || !zoomRectEnd) return;

		const x = Math.min(zoomRectStart.x, zoomRectEnd.x);
		const y = Math.min(zoomRectStart.y, zoomRectEnd.y);
		const w = Math.abs(zoomRectEnd.x - zoomRectStart.x);
		const h = Math.abs(zoomRectEnd.y - zoomRectStart.y);

		ctx.strokeStyle = '#ffffff';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 4]);
		ctx.strokeRect(x, y, w, h);
		ctx.setLineDash([]);

		ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
		ctx.fillRect(x, y, w, h);
	}

	function drawAxisLabels(ctx: CanvasRenderingContext2D, w: number, h: number) {
		const dpr = window.devicePixelRatio || 1;
		ctx.fillStyle = '#888888';
		ctx.font = `${10 * dpr}px monospace`;

		// Use same grid step calculation as drawGrid for consistency
		const targetSpacingCss = 40;
		const targetSpacingCanvas = targetSpacingCss * dpr;

		// X-axis labels
		const xRange = bounds.xMax - bounds.xMin;
		const xTargetDivisions = Math.max(4, Math.round(w / targetSpacingCanvas));
		const xStep = calculateNiceStep(xRange, xTargetDivisions);
		for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= bounds.xMax; x += xStep) {
			const px = ((x - bounds.xMin) / xRange) * w;
			ctx.fillText(formatValue(x, 's'), px + 2, h - 4 * dpr);
		}

		// Y-axis labels
		const yRange = bounds.yMax - bounds.yMin;
		const yTargetDivisions = Math.max(4, Math.round(h / targetSpacingCanvas));
		const yStep = calculateNiceStep(yRange, yTargetDivisions);
		for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= bounds.yMax; y += yStep) {
			const py = h - ((y - bounds.yMin) / yRange) * h;
			ctx.fillText(formatValue(y, 'V'), 4 * dpr, py - 2 * dpr);
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
		const dpr = window.devicePixelRatio || 1;
		dragStart = { x: e.offsetX * dpr, y: e.offsetY * dpr };

		if (zoomRectMode) {
			dragMode = 'zoom-rect';
			zoomRectStart = { x: e.offsetX * dpr, y: e.offsetY * dpr };
			zoomRectEnd = { x: e.offsetX * dpr, y: e.offsetY * dpr };
		} else if (e.shiftKey) {
			dragMode = 'zoom-x';
		} else if (e.ctrlKey || e.metaKey) {
			dragMode = 'zoom-y';
		} else {
			dragMode = 'pan';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const dpr = window.devicePixelRatio || 1;
		const w = canvas.clientWidth;
		const h = canvas.clientHeight;

		// Update mouse position for tooltip
		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;
		mousePos = {
			x: e.offsetX,
			y: e.offsetY,
			dataX: bounds.xMin + (e.offsetX / w) * xRange,
			dataY: bounds.yMax - (e.offsetY / h) * yRange
		};
		showTooltip = true;

		if (!isDragging || !dragMode) {
			// Still render to update hover line even when not dragging
			render();
			return;
		}

		const dx = e.offsetX * dpr - dragStart.x;
		const dy = e.offsetY * dpr - dragStart.y;

		if (dragMode === 'zoom-rect') {
			zoomRectEnd = { x: e.offsetX * dpr, y: e.offsetY * dpr };
			render();
		} else if (dragMode === 'pan') {
			const xShift = -(dx / (w * dpr)) * xRange;
			const yShift = (dy / (h * dpr)) * yRange;
			bounds = {
				xMin: bounds.xMin + xShift,
				xMax: bounds.xMax + xShift,
				yMin: bounds.yMin + yShift,
				yMax: bounds.yMax + yShift
			};
			dragStart = { x: e.offsetX * dpr, y: e.offsetY * dpr };
			render();
		} else {
			dragStart = { x: e.offsetX * dpr, y: e.offsetY * dpr };
			render();
		}
	}

	function handleMouseUp() {
		if (dragMode === 'zoom-rect' && zoomRectStart && zoomRectEnd) {
			// Apply zoom to rectangle
			const w = canvas.width;
			const h = canvas.height;
			const xRange = bounds.xMax - bounds.xMin;
			const yRange = bounds.yMax - bounds.yMin;

			const x1 = Math.min(zoomRectStart.x, zoomRectEnd.x);
			const x2 = Math.max(zoomRectStart.x, zoomRectEnd.x);
			const y1 = Math.min(zoomRectStart.y, zoomRectEnd.y);
			const y2 = Math.max(zoomRectStart.y, zoomRectEnd.y);

			// Only zoom if rectangle is big enough
			if (x2 - x1 > 10 && y2 - y1 > 10) {
				const newXMin = bounds.xMin + (x1 / w) * xRange;
				const newXMax = bounds.xMin + (x2 / w) * xRange;
				const newYMax = bounds.yMax - (y1 / h) * yRange;
				const newYMin = bounds.yMax - (y2 / h) * yRange;

				bounds = { xMin: newXMin, xMax: newXMax, yMin: newYMin, yMax: newYMax };
			}

			zoomRectStart = null;
			zoomRectEnd = null;
			zoomRectMode = false;
			render();
		}

		isDragging = false;
		dragMode = null;
	}

	function handleMouseLeave() {
		showTooltip = false;
		handleMouseUp();
	}

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

	// Keyboard handlers
	function handleKeyDown(e: KeyboardEvent) {
		const panAmount = 0.1; // 10% of view
		const zoomFactor = 0.8; // 20% zoom per keypress
		const xRange = bounds.xMax - bounds.xMin;
		const yRange = bounds.yMax - bounds.yMin;
		const xCenter = (bounds.xMin + bounds.xMax) / 2;
		const yCenter = (bounds.yMin + bounds.yMax) / 2;

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
		} else if (e.key === 'g' || e.key === 'G') {
			// Toggle grid
			showGrid = !showGrid;
			render();
		} else if (e.key === 'z' || e.key === 'Z') {
			// Toggle zoom rectangle mode
			zoomRectMode = !zoomRectMode;
			zoomRectStart = null;
			zoomRectEnd = null;
		} else if (e.key === '+' || e.key === '=') {
			// Zoom in (centered)
			e.preventDefault();
			const newXRange = xRange * zoomFactor;
			const newYRange = yRange * zoomFactor;
			bounds = {
				xMin: xCenter - newXRange / 2,
				xMax: xCenter + newXRange / 2,
				yMin: yCenter - newYRange / 2,
				yMax: yCenter + newYRange / 2
			};
			render();
		} else if (e.key === '-' || e.key === '_') {
			// Zoom out (centered)
			e.preventDefault();
			const newXRange = xRange / zoomFactor;
			const newYRange = yRange / zoomFactor;
			bounds = {
				xMin: xCenter - newXRange / 2,
				xMax: xCenter + newXRange / 2,
				yMin: yCenter - newYRange / 2,
				yMax: yCenter + newYRange / 2
			};
			render();
		} else if (e.key === 'ArrowLeft') {
			// Pan left
			e.preventDefault();
			const shift = xRange * panAmount;
			bounds = { ...bounds, xMin: bounds.xMin - shift, xMax: bounds.xMax - shift };
			render();
		} else if (e.key === 'ArrowRight') {
			// Pan right
			e.preventDefault();
			const shift = xRange * panAmount;
			bounds = { ...bounds, xMin: bounds.xMin + shift, xMax: bounds.xMax + shift };
			render();
		} else if (e.key === 'ArrowUp') {
			// Pan up
			e.preventDefault();
			const shift = yRange * panAmount;
			bounds = { ...bounds, yMin: bounds.yMin + shift, yMax: bounds.yMax + shift };
			render();
		} else if (e.key === 'ArrowDown') {
			// Pan down
			e.preventDefault();
			const shift = yRange * panAmount;
			bounds = { ...bounds, yMin: bounds.yMin - shift, yMax: bounds.yMax - shift };
			render();
		} else if (e.key === 'Escape') {
			// Cancel zoom rect mode
			zoomRectMode = false;
			zoomRectStart = null;
			zoomRectEnd = null;
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
	bind:this={containerEl}
	class="waveform-container"
	class:zoom-rect-mode={zoomRectMode}
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

	{#if showTooltip && !zoomRectMode && traces.length > 0}
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

	{#if zoomRectMode}
		<div class="zoom-mode-indicator">ZOOM RECT (Z to cancel)</div>
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

	.waveform-container.zoom-rect-mode {
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

