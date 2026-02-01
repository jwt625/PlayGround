<script lang="ts">
	import { onMount } from 'svelte';
	import type { ViewTransform, GridSettings, Point, Schematic } from './types';
	import { DEFAULT_VIEW, DEFAULT_GRID } from './types';

	let { schematic = { components: [], wires: [] } }: { schematic: Schematic } = $props();

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D | null = null;

	let view: ViewTransform = $state({ ...DEFAULT_VIEW });
	let grid: GridSettings = $state({ ...DEFAULT_GRID });
	let isDragging = $state(false);
	let dragStart: Point | null = null;
	let mousePos: Point = $state({ x: 0, y: 0 });
	let schematicPos: Point = $state({ x: 0, y: 0 });

	const getDpr = () => (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1;

	onMount(() => {
		ctx = canvas.getContext('2d');
		resize();
		window.addEventListener('resize', resize);

		return () => {
			window.removeEventListener('resize', resize);
		};
	});

	function resize() {
		if (!canvas) return;
		const dpr = getDpr();
		const rect = canvas.getBoundingClientRect();
		canvas.width = rect.width * dpr;
		canvas.height = rect.height * dpr;
		render();
	}

	// Convert screen coords to schematic coords
	function screenToSchematic(sx: number, sy: number): Point {
		const dpr = getDpr();
		return {
			x: (sx * dpr - view.offsetX) / view.scale,
			y: (sy * dpr - view.offsetY) / view.scale
		};
	}

	// Convert schematic coords to screen coords
	function schematicToScreen(px: number, py: number): Point {
		const dpr = getDpr();
		return {
			x: (px * view.scale + view.offsetX) / dpr,
			y: (py * view.scale + view.offsetY) / dpr
		};
	}

	// Snap to grid
	function snapToGrid(p: Point): Point {
		if (!grid.snapEnabled) return p;
		return {
			x: Math.round(p.x / grid.size) * grid.size,
			y: Math.round(p.y / grid.size) * grid.size
		};
	}

	function render() {
		if (!ctx || !canvas) return;
		const w = canvas.width, h = canvas.height;

		// Clear
		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, w, h);

		// Apply transform
		ctx.save();
		ctx.translate(view.offsetX, view.offsetY);
		ctx.scale(view.scale, view.scale);

		// Draw grid
		if (grid.visible) drawGrid(w, h);

		// Draw origin crosshair
		drawOrigin();

		// Draw wires
		drawWires();

		// Draw components
		drawComponents();

		ctx.restore();
	}

	function drawGrid(w: number, h: number) {
		if (!ctx) return;
		const gs = grid.size;

		// Calculate visible range in schematic coords
		const topLeft = screenToSchematic(0, 0);
		const bottomRight = { x: (w - view.offsetX) / view.scale, y: (h - view.offsetY) / view.scale };

		const startX = Math.floor(topLeft.x / gs) * gs;
		const startY = Math.floor(topLeft.y / gs) * gs;
		const endX = Math.ceil(bottomRight.x / gs) * gs;
		const endY = Math.ceil(bottomRight.y / gs) * gs;

		// Draw dots at grid intersections
		ctx.fillStyle = '#444';
		const dotSize = Math.max(1, 2 / view.scale);

		for (let x = startX; x <= endX; x += gs) {
			for (let y = startY; y <= endY; y += gs) {
				ctx.fillRect(x - dotSize/2, y - dotSize/2, dotSize, dotSize);
			}
		}
	}

	function drawOrigin() {
		if (!ctx) return;
		const size = 20;
		ctx.strokeStyle = '#666';
		ctx.lineWidth = 1 / view.scale;
		ctx.beginPath();
		ctx.moveTo(-size, 0); ctx.lineTo(size, 0);
		ctx.moveTo(0, -size); ctx.lineTo(0, size);
		ctx.stroke();
	}

	function drawWires() {
		if (!ctx) return;
		ctx.strokeStyle = '#00ff00';
		ctx.lineWidth = 2 / view.scale;
		ctx.lineCap = 'round';

		for (const wire of schematic.wires) {
			ctx.beginPath();
			ctx.moveTo(wire.x1, wire.y1);
			ctx.lineTo(wire.x2, wire.y2);
			ctx.stroke();
		}
	}

	function drawComponents() {
		if (!ctx) return;
		// Placeholder - will be implemented in Phase 5
		for (const comp of schematic.components) {
			ctx.fillStyle = '#569cd6';
			ctx.fillRect(comp.x - 10, comp.y - 10, 20, 20);
			ctx.fillStyle = '#fff';
			ctx.font = `${12 / view.scale}px monospace`;
			ctx.fillText(comp.attributes['InstName'] || comp.type, comp.x - 8, comp.y + 4);
		}
	}

	function getHudText(): string {
		const snapped = snapToGrid(schematicPos);
		return `(${snapped.x}, ${snapped.y}) | Zoom: ${(view.scale * 100).toFixed(0)}%`;
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button === 0 || e.button === 1) {  // Left or middle click
			isDragging = true;
			dragStart = { x: e.offsetX, y: e.offsetY };
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const dpr = getDpr();
		mousePos = { x: e.offsetX, y: e.offsetY };
		schematicPos = screenToSchematic(e.offsetX, e.offsetY);

		if (isDragging && dragStart) {
			const dx = (e.offsetX - dragStart.x) * dpr;
			const dy = (e.offsetY - dragStart.y) * dpr;
			view.offsetX += dx;
			view.offsetY += dy;
			dragStart = { x: e.offsetX, y: e.offsetY };
		}
		render();
	}

	function handleMouseUp() {
		isDragging = false;
		dragStart = null;
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const dpr = getDpr();
		const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
		const newScale = Math.max(0.1, Math.min(10, view.scale * zoomFactor));

		// Zoom toward mouse position
		const mx = e.offsetX * dpr;
		const my = e.offsetY * dpr;
		view.offsetX = mx - (mx - view.offsetX) * (newScale / view.scale);
		view.offsetY = my - (my - view.offsetY) * (newScale / view.scale);
		view.scale = newScale;

		render();
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === 'g' || e.key === 'G') {
			grid.visible = !grid.visible;
			render();
		} else if (e.key === 'Home' || e.key === 'f' || e.key === 'F') {
			// Reset view
			view = { offsetX: canvas.width / 2, offsetY: canvas.height / 2, scale: 1 };
			render();
		} else if (e.key === '+' || e.key === '=') {
			view.scale = Math.min(10, view.scale * 1.2);
			render();
		} else if (e.key === '-' || e.key === '_') {
			view.scale = Math.max(0.1, view.scale / 1.2);
			render();
		}
	}

	// Re-render when schematic changes
	$effect(() => {
		if (schematic && ctx) {
			render();
		}
	});
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="schematic-container">
	<canvas
		bind:this={canvas}
		class="schematic-canvas"
		onmousedown={handleMouseDown}
		onmousemove={handleMouseMove}
		onmouseup={handleMouseUp}
		onmouseleave={handleMouseUp}
		onwheel={handleWheel}
		onkeydown={handleKeyDown}
		tabindex="0"
	></canvas>
	<div class="hud">{getHudText()}</div>
</div>

<style>
	.schematic-container {
		position: relative;
		width: 100%;
		height: 100%;
	}

	.schematic-canvas {
		width: 100%;
		height: 100%;
		display: block;
		cursor: crosshair;
		outline: none;
	}

	.schematic-canvas:active {
		cursor: grabbing;
	}

	.hud {
		position: absolute;
		bottom: 4px;
		left: 4px;
		background: rgba(0, 0, 0, 0.7);
		color: #888;
		font-family: monospace;
		font-size: 11px;
		padding: 2px 6px;
		pointer-events: none;
	}
</style>

