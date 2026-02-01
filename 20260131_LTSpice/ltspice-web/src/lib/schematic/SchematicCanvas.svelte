<script lang="ts">
	import { onMount } from 'svelte';
	import type { ViewTransform, GridSettings, Point, Schematic, Component, ComponentType, Rotation } from './types';
	import { DEFAULT_VIEW, DEFAULT_GRID } from './types';
	import { renderComponent, hitTestComponent, nextRotation } from './component-renderer';
	import { COMPONENT_DEFS, getComponentByShortcut } from './component-defs';

	let { schematic = $bindable({ components: [], wires: [] }) }: { schematic: Schematic } = $props();

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D | null = null;

	let view: ViewTransform = $state({ ...DEFAULT_VIEW });
	let grid: GridSettings = $state({ ...DEFAULT_GRID });
	let isDragging = $state(false);
	let dragStart: Point | null = null;
	let mousePos: Point = $state({ x: 0, y: 0 });
	let schematicPos: Point = $state({ x: 0, y: 0 });

	// Interaction mode
	type Mode = 'select' | 'pan' | 'place';
	let mode: Mode = $state('select');

	// Selection state
	let selectedIds: Set<string> = $state(new Set());

	// Component placement state
	let placingType: ComponentType | null = $state(null);
	let placingRotation: Rotation = $state(0);
	let placingMirror: boolean = $state(false);

	// Component counter for generating unique IDs
	let componentCounters: Record<string, number> = $state({});

	const getDpr = () => (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1;

	let resizeObserver: ResizeObserver | null = null;
	let container: HTMLDivElement;

	onMount(() => {
		ctx = canvas.getContext('2d');

		// Use ResizeObserver on the CONTAINER (not canvas) for accurate size tracking
		resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				if (entry.target === container) {
					resize(entry.contentRect.width, entry.contentRect.height);
				}
			}
		});
		resizeObserver.observe(container);

		// Initial size from container
		const rect = container.getBoundingClientRect();
		resize(rect.width, rect.height);

		return () => {
			resizeObserver?.disconnect();
		};
	});

	function resize(displayWidth: number, displayHeight: number) {
		if (!canvas || displayWidth <= 0 || displayHeight <= 0) return;
		const dpr = getDpr();

		// Set canvas internal resolution to match display size * dpr
		const newWidth = Math.floor(displayWidth * dpr);
		const newHeight = Math.floor(displayHeight * dpr);

		// Adjust view offset to keep the center point stable
		if (canvas.width > 0 && canvas.height > 0) {
			view.offsetX += (newWidth - canvas.width) / 2;
			view.offsetY += (newHeight - canvas.height) / 2;
		}

		canvas.width = newWidth;
		canvas.height = newHeight;

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

		// Draw all placed components
		for (const comp of schematic.components) {
			const isSelected = selectedIds.has(comp.id);
			renderComponent(ctx, comp, view.scale, isSelected, false);
		}

		// Draw ghost component if placing
		if (placingType) {
			const snapped = snapToGrid(schematicPos);
			const ghostComp: Component = {
				id: 'ghost',
				type: placingType,
				x: snapped.x,
				y: snapped.y,
				rotation: placingRotation,
				mirror: placingMirror,
				attributes: {},
				pins: []
			};
			renderComponent(ctx, ghostComp, view.scale, false, true);
		}
	}

	function getHudText(): string {
		const snapped = snapToGrid(schematicPos);
		let text = `(${snapped.x}, ${snapped.y}) | Zoom: ${(view.scale * 100).toFixed(0)}%`;

		if (placingType) {
			const def = COMPONENT_DEFS[placingType];
			text += ` | Placing: ${def.name}`;
			if (placingRotation !== 0) text += ` R${placingRotation}°`;
			if (placingMirror) text += ' M';
			text += ' | Ctrl+R=rotate, Ctrl+E=mirror, Esc=cancel';
		} else if (selectedIds.size > 0) {
			text += ` | Selected: ${selectedIds.size}`;
			text += ' | Ctrl+R=rotate, Ctrl+E=mirror, Del=delete';
		}

		return text;
	}

	function handleMouseDown(e: MouseEvent) {
		canvas.focus();

		if (e.button === 0) {  // Left click
			if (placingType) {
				// Place component
				placeComponent();
				return;
			}

			// Check for component selection
			const clickPos = screenToSchematic(e.offsetX, e.offsetY);
			let clickedComp: Component | null = null;

			// Find topmost component under cursor (iterate in reverse for z-order)
			for (let i = schematic.components.length - 1; i >= 0; i--) {
				const comp = schematic.components[i];
				if (hitTestComponent(comp, clickPos.x, clickPos.y)) {
					clickedComp = comp;
					break;
				}
			}

			if (clickedComp) {
				if (e.shiftKey) {
					// Toggle selection
					if (selectedIds.has(clickedComp.id)) {
						selectedIds.delete(clickedComp.id);
					} else {
						selectedIds.add(clickedComp.id);
					}
					selectedIds = new Set(selectedIds);  // Trigger reactivity
				} else {
					// Single select
					selectedIds = new Set([clickedComp.id]);
				}
				render();
				return;
			}

			// Click on empty space - prepare for potential pan, clear selection
			if (!e.shiftKey) {
				selectedIds = new Set();
			}
			// Don't set isDragging yet - wait for mouse move to confirm it's a drag
			dragStart = { x: e.offsetX, y: e.offsetY };
			render();
		} else if (e.button === 1) {  // Middle click - always pan
			isDragging = true;
			dragStart = { x: e.offsetX, y: e.offsetY };
		}
	}

	function placeComponent() {
		if (!placingType) return;

		const snapped = snapToGrid(schematicPos);
		const def = COMPONENT_DEFS[placingType];

		// Generate instance name
		const prefix = placingType === 'ground' ? '' : placingType[0].toUpperCase();
		const count = (componentCounters[placingType] || 0) + 1;
		componentCounters[placingType] = count;
		const instName = prefix ? `${prefix}${count}` : '';

		// Create new component
		const newComp: Component = {
			id: crypto.randomUUID(),
			type: placingType,
			x: snapped.x,
			y: snapped.y,
			rotation: placingRotation,
			mirror: placingMirror,
			attributes: {
				InstName: instName,
				Value: getDefaultValue(placingType)
			},
			pins: def.pins.map((p, i) => ({ ...p, id: `${i}` }))
		};

		schematic.components = [...schematic.components, newComp];
		render();
	}

	function getDefaultValue(type: ComponentType): string {
		switch (type) {
			case 'resistor': return '1k';
			case 'capacitor': return '1u';
			case 'inductor': return '1m';
			case 'voltage': return 'DC 5';
			case 'current': return 'DC 1m';
			case 'diode': return 'D';
			case 'npn': case 'pnp': return '2N2222';
			case 'nmos': case 'pmos': return 'NMOS';
			default: return '';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const dpr = getDpr();
		mousePos = { x: e.offsetX, y: e.offsetY };
		schematicPos = screenToSchematic(e.offsetX, e.offsetY);

		// Start dragging if mouse moved with button held (dragStart set but not isDragging yet)
		if (dragStart && !isDragging && (e.buttons & 1 || e.buttons & 4)) {
			const dx = Math.abs(e.offsetX - dragStart.x);
			const dy = Math.abs(e.offsetY - dragStart.y);
			// Only start drag if moved more than 3 pixels (prevents accidental drags)
			if (dx > 3 || dy > 3) {
				isDragging = true;
			}
		}

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
		// Escape cancels current operation
		if (e.key === 'Escape') {
			placingType = null;
			mode = 'select';
			selectedIds = new Set();
			render();
			return;
		}

		// Component shortcuts - can switch component while placing
		if (!e.ctrlKey && !e.metaKey) {
			const compDef = getComponentByShortcut(e.key);
			if (compDef) {
				placingType = compDef.type;
				placingRotation = 0;
				placingMirror = false;
				mode = 'place';
				render();
				return;
			}
		}

		// Ctrl+R to rotate while placing or rotate selected
		if ((e.key === 'r' || e.key === 'R') && (e.ctrlKey || e.metaKey)) {
			e.preventDefault();
			if (placingType) {
				placingRotation = nextRotation(placingRotation);
				render();
				return;
			}
			// Rotate selected components
			if (selectedIds.size > 0) {
				for (const comp of schematic.components) {
					if (selectedIds.has(comp.id)) {
						comp.rotation = nextRotation(comp.rotation);
					}
				}
				render();
				return;
			}
		}

		// Ctrl+E to mirror
		if ((e.key === 'e' || e.key === 'E') && (e.ctrlKey || e.metaKey)) {
			e.preventDefault();
			if (placingType) {
				placingMirror = !placingMirror;
				render();
				return;
			}
			if (selectedIds.size > 0) {
				for (const comp of schematic.components) {
					if (selectedIds.has(comp.id)) {
						comp.mirror = !comp.mirror;
					}
				}
				render();
				return;
			}
		}

		// Delete selected components
		if (e.key === 'Delete' || e.key === 'Backspace') {
			if (selectedIds.size > 0) {
				schematic.components = schematic.components.filter(c => !selectedIds.has(c.id));
				selectedIds = new Set();
				render();
				return;
			}
		}

		// Grid toggle (Ctrl+G or Shift+G to avoid conflict with Ground)
		if ((e.key === 'g' || e.key === 'G') && (e.ctrlKey || e.shiftKey)) {
			e.preventDefault();
			grid.visible = !grid.visible;
			render();
			return;
		}

		// View controls
		if (e.key === 'Home' || (e.key === 'f' && !placingType)) {
			view = { offsetX: canvas.width / 2, offsetY: canvas.height / 2, scale: 1 };
			render();
			return;
		}
		if (e.key === '+' || e.key === '=') {
			view.scale = Math.min(10, view.scale * 1.2);
			render();
			return;
		}
		if (e.key === '-' || e.key === '_') {
			view.scale = Math.max(0.1, view.scale / 1.2);
			render();
			return;
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
<div class="schematic-container" bind:this={container}>
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
		overflow: hidden;
	}

	.schematic-canvas {
		position: absolute;
		top: 0;
		left: 0;
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

