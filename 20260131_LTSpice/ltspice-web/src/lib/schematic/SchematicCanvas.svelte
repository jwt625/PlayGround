<script lang="ts">
	import { onMount } from 'svelte';
	import type { ViewTransform, GridSettings, Point, Schematic, Component, ComponentType, Rotation, EditorMode, WireDrawState, Wire, Junction } from './types';
	import { DEFAULT_VIEW, DEFAULT_GRID, DEFAULT_WIRE_DRAW, MODE_SHORTCUTS, getModeName } from './types';
	import { renderComponent, hitTestComponent, nextRotation } from './component-renderer';
	import { COMPONENT_DEFS, getComponentByShortcut } from './component-defs';
	import { COMPONENT_PREFIX } from '$lib/netlist/types';

	let { schematic = $bindable({ components: [], wires: [], junctions: [] }) }: { schematic: Schematic } = $props();

	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D | null = null;

	let view: ViewTransform = $state({ ...DEFAULT_VIEW });
	let grid: GridSettings = $state({ ...DEFAULT_GRID });
	let isDragging = $state(false);
	let dragStart: Point | null = null;
	let mousePos: Point = $state({ x: 0, y: 0 });
	let schematicPos: Point = $state({ x: 0, y: 0 });

	// Editor mode (select, wire, delete, move, place)
	let mode: EditorMode = $state('select');

	// Selection state
	let selectedIds: Set<string> = $state(new Set());
	let selectedWireIds: Set<string> = $state(new Set());

	// Component placement state
	let placingType: ComponentType | null = $state(null);
	let placingRotation: Rotation = $state(0);
	let placingMirror: boolean = $state(false);

	// Wire drawing state
	let wireDraw: WireDrawState = $state({ ...DEFAULT_WIRE_DRAW });

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
		ctx.lineCap = 'round';
		ctx.lineWidth = 2 / view.scale;

		// Draw existing wires
		for (const wire of schematic.wires) {
			const isSelected = selectedWireIds.has(wire.id);
			ctx.strokeStyle = isSelected ? '#ffff00' : '#00ff00';
			ctx.beginPath();
			ctx.moveTo(wire.x1, wire.y1);
			ctx.lineTo(wire.x2, wire.y2);
			ctx.stroke();
		}

		// Draw wire being drawn (preview)
		if (wireDraw.isDrawing && wireDraw.startPoint) {
			const snapped = snapToGrid(schematicPos);
			const segments = getWireSegments(wireDraw.startPoint, snapped, wireDraw.direction);

			ctx.strokeStyle = '#00ff0080';  // Semi-transparent green
			ctx.setLineDash([5 / view.scale, 5 / view.scale]);

			for (const seg of segments) {
				ctx.beginPath();
				ctx.moveTo(seg.x1, seg.y1);
				ctx.lineTo(seg.x2, seg.y2);
				ctx.stroke();
			}

			ctx.setLineDash([]);
		}

		// Draw junction dots where wires connect
		drawJunctions();
	}

	/** Get Manhattan-style wire segments from start to end */
	function getWireSegments(start: Point, end: Point, direction: 'horizontal-first' | 'vertical-first'): { x1: number; y1: number; x2: number; y2: number }[] {
		const segments: { x1: number; y1: number; x2: number; y2: number }[] = [];

		if (start.x === end.x && start.y === end.y) {
			return segments;  // No wire needed
		}

		if (start.x === end.x || start.y === end.y) {
			// Straight line
			segments.push({ x1: start.x, y1: start.y, x2: end.x, y2: end.y });
		} else if (direction === 'horizontal-first') {
			// Horizontal then vertical
			segments.push({ x1: start.x, y1: start.y, x2: end.x, y2: start.y });
			segments.push({ x1: end.x, y1: start.y, x2: end.x, y2: end.y });
		} else {
			// Vertical then horizontal
			segments.push({ x1: start.x, y1: start.y, x2: start.x, y2: end.y });
			segments.push({ x1: start.x, y1: end.y, x2: end.x, y2: end.y });
		}

		return segments;
	}

	/** Get world-space pin positions for a component */
	function getComponentPinPositions(comp: Component): Point[] {
		const def = COMPONENT_DEFS[comp.type];
		if (!def) return [];

		const rad = (comp.rotation * Math.PI) / 180;
		const cos = Math.cos(rad);
		const sin = Math.sin(rad);

		return def.pins.map(pin => {
			// Apply mirror
			let px = comp.mirror ? -pin.x : pin.x;
			let py = pin.y;
			// Apply rotation
			const rx = px * cos - py * sin;
			const ry = px * sin + py * cos;
			// Translate to world position
			return { x: comp.x + rx, y: comp.y + ry };
		});
	}

	/** Get all component pin positions in the schematic */
	function getAllPinPositions(): Set<string> {
		const pins = new Set<string>();
		for (const comp of schematic.components) {
			for (const pos of getComponentPinPositions(comp)) {
				pins.add(`${pos.x},${pos.y}`);
			}
		}
		return pins;
	}

	/** Draw junction dots - red for connections, following LTSpice style */
	function drawJunctions() {
		if (!ctx) return;

		const dotRadius = 4 / view.scale;
		const pinPositions = getAllPinPositions();

		// Count wire endpoints at each position
		const wireEndpoints: Map<string, number> = new Map();
		for (const wire of schematic.wires) {
			const key1 = `${wire.x1},${wire.y1}`;
			const key2 = `${wire.x2},${wire.y2}`;
			wireEndpoints.set(key1, (wireEndpoints.get(key1) || 0) + 1);
			wireEndpoints.set(key2, (wireEndpoints.get(key2) || 0) + 1);
		}

		// Draw red dots at:
		// 1. Explicit junctions (user-created for wire crossings)
		// 2. Wire endpoints that connect to component pins
		// 3. Wire endpoints where 3+ wire segments meet (T-junction)

		ctx.fillStyle = '#ff0000';  // Red like component terminals

		// Draw explicit junctions
		for (const junction of schematic.junctions) {
			ctx.beginPath();
			ctx.arc(junction.x, junction.y, dotRadius, 0, Math.PI * 2);
			ctx.fill();
		}

		// Draw connection dots at wire endpoints
		for (const [key, count] of wireEndpoints) {
			const [x, y] = key.split(',').map(Number);

			// Draw dot if:
			// - Wire endpoint is on a component pin (auto-connection)
			// - 3+ wire endpoints meet here (T-junction or more)
			if (pinPositions.has(key) || count >= 3) {
				ctx.beginPath();
				ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
				ctx.fill();
			}
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

		// Show current mode
		text += ` | ${getModeName(mode)}`;

		if (mode === 'place' && placingType) {
			const def = COMPONENT_DEFS[placingType];
			text += `: ${def.name}`;
			if (placingRotation !== 0) text += ` R${placingRotation}°`;
			if (placingMirror) text += ' M';
			text += ' | Ctrl+R=rotate, Ctrl+E=mirror, Esc=cancel';
		} else if (mode === 'wire') {
			if (wireDraw.isDrawing) {
				text += ' | Click to place segment, Space=toggle direction, Esc=cancel';
			} else {
				text += ' | Click to start wire, Esc=exit';
			}
		} else if (mode === 'delete') {
			text += ' | Click to delete (junctions/components/wires), Esc=exit';
		} else if (mode === 'duplicate') {
			text += ' | Click on item to duplicate, Esc=exit';
		} else if (mode === 'move') {
			text += ' | Drag to move, Esc=exit';
		} else if (selectedIds.size > 0 || selectedWireIds.size > 0) {
			const total = selectedIds.size + selectedWireIds.size;
			text += ` | Selected: ${total}`;
			text += ' | Ctrl+R=rotate, Ctrl+E=mirror, Del=delete';
		}

		// Always show mode shortcuts hint
		text += ' | 3=wire 5=del 6=dup 7=move';

		return text;
	}

	function handleMouseDown(e: MouseEvent) {
		canvas.focus();
		const clickPos = screenToSchematic(e.offsetX, e.offsetY);
		const snapped = snapToGrid(clickPos);

		if (e.button === 0) {  // Left click
			// Handle wire mode
			if (mode === 'wire') {
				handleWireClick(snapped);
				return;
			}

			// Handle delete mode
			if (mode === 'delete') {
				handleDeleteClick(clickPos);
				return;
			}

			// Handle duplicate mode
			if (mode === 'duplicate') {
				handleDuplicateClick(clickPos, snapped);
				return;
			}

			// Handle component placement
			if (mode === 'place' && placingType) {
				placeComponent();
				return;
			}

			// Select mode - check for component/wire selection
			const clickedComp = findComponentAt(clickPos);
			const clickedWire = findWireAt(clickPos);

			if (clickedComp) {
				if (e.shiftKey) {
					// Toggle selection
					if (selectedIds.has(clickedComp.id)) {
						selectedIds.delete(clickedComp.id);
					} else {
						selectedIds.add(clickedComp.id);
					}
					selectedIds = new Set(selectedIds);
				} else {
					selectedIds = new Set([clickedComp.id]);
					selectedWireIds = new Set();
				}
				render();
				return;
			}

			if (clickedWire) {
				if (e.shiftKey) {
					if (selectedWireIds.has(clickedWire.id)) {
						selectedWireIds.delete(clickedWire.id);
					} else {
						selectedWireIds.add(clickedWire.id);
					}
					selectedWireIds = new Set(selectedWireIds);
				} else {
					selectedWireIds = new Set([clickedWire.id]);
					selectedIds = new Set();
				}
				render();
				return;
			}

			// Click on empty space - prepare for potential pan, clear selection
			if (!e.shiftKey) {
				selectedIds = new Set();
				selectedWireIds = new Set();
			}
			dragStart = { x: e.offsetX, y: e.offsetY };
			render();
		} else if (e.button === 1) {  // Middle click - always pan
			isDragging = true;
			dragStart = { x: e.offsetX, y: e.offsetY };
		}
	}

	/** Handle click in wire drawing mode */
	function handleWireClick(snapped: Point) {
		if (!wireDraw.isDrawing) {
			// Start new wire - also check if starting on existing wire
			maybeCreateJunctionOnWire(snapped);
			wireDraw = {
				...wireDraw,
				isDrawing: true,
				startPoint: snapped,
				segments: []
			};
		} else if (wireDraw.startPoint) {
			// Place wire segment(s)
			const segments = getWireSegments(wireDraw.startPoint, snapped, wireDraw.direction);

			for (const seg of segments) {
				// Only add non-zero-length segments
				if (seg.x1 !== seg.x2 || seg.y1 !== seg.y2) {
					const newWire: Wire = {
						id: crypto.randomUUID(),
						x1: seg.x1,
						y1: seg.y1,
						x2: seg.x2,
						y2: seg.y2
					};
					schematic.wires = [...schematic.wires, newWire];
				}
			}

			// If ending on an existing wire, create junction
			maybeCreateJunctionOnWire(snapped);

			// Continue from end point for chained wires
			wireDraw = {
				...wireDraw,
				startPoint: snapped
			};
		}
		render();
	}

	/** Create junction if point is on an existing wire (not at endpoint) */
	function maybeCreateJunctionOnWire(point: Point) {
		for (const wire of schematic.wires) {
			if (isPointOnWireSegment(point, wire)) {
				createJunctionAt(point);
				return;
			}
		}
	}

	/** Check if point lies on wire segment (not at endpoints) */
	function isPointOnWireSegment(point: Point, wire: Wire): boolean {
		// Check if point is at an endpoint (no junction needed there)
		if ((point.x === wire.x1 && point.y === wire.y1) ||
			(point.x === wire.x2 && point.y === wire.y2)) {
			return false;
		}

		// Check if point is on the line segment
		// For Manhattan wires, this is simple: check if on horizontal or vertical segment
		if (wire.x1 === wire.x2) {
			// Vertical wire
			if (point.x === wire.x1) {
				const minY = Math.min(wire.y1, wire.y2);
				const maxY = Math.max(wire.y1, wire.y2);
				return point.y > minY && point.y < maxY;
			}
		} else if (wire.y1 === wire.y2) {
			// Horizontal wire
			if (point.y === wire.y1) {
				const minX = Math.min(wire.x1, wire.x2);
				const maxX = Math.max(wire.x1, wire.x2);
				return point.x > minX && point.x < maxX;
			}
		}
		return false;
	}

	/** Handle click in delete mode */
	function handleDeleteClick(clickPos: Point) {
		// Check for junction first
		const junction = findJunctionAt(clickPos);
		if (junction) {
			schematic.junctions = schematic.junctions.filter(j => j.id !== junction.id);
			render();
			return;
		}

		const comp = findComponentAt(clickPos);
		if (comp) {
			schematic.components = schematic.components.filter(c => c.id !== comp.id);
			render();
			return;
		}

		const wire = findWireAt(clickPos);
		if (wire) {
			schematic.wires = schematic.wires.filter(w => w.id !== wire.id);
			render();
		}
	}

	/** Find junction at position */
	function findJunctionAt(pos: Point, tolerance: number = 8): Junction | null {
		for (const junction of schematic.junctions) {
			const dist = Math.hypot(pos.x - junction.x, pos.y - junction.y);
			if (dist <= tolerance) {
				return junction;
			}
		}
		return null;
	}

	/** Create a junction at the given position */
	function createJunctionAt(pos: Point): boolean {
		// Check if junction already exists at this position
		const existing = schematic.junctions.find(j => j.x === pos.x && j.y === pos.y);
		if (existing) return false;

		const newJunction: Junction = {
			id: crypto.randomUUID(),
			x: pos.x,
			y: pos.y
		};
		schematic.junctions = [...schematic.junctions, newJunction];
		return true;
	}

	/** Handle click in duplicate mode - click on item to duplicate it */
	function handleDuplicateClick(clickPos: Point, snapped: Point) {
		const comp = findComponentAt(clickPos);
		if (comp) {
			// Duplicate component with offset
			const offset = grid.size * 2;  // Offset by 2 grid units
			const newComp: Component = {
				id: crypto.randomUUID(),
				type: comp.type,
				x: snapped.x + offset,
				y: snapped.y + offset,
				rotation: comp.rotation,
				mirror: comp.mirror,
				attributes: { ...comp.attributes },
				pins: comp.pins.map(p => ({ ...p }))
			};

			// Update instance name for the duplicate using correct SPICE prefix
			if (newComp.attributes['InstName']) {
				const count = (componentCounters[comp.type] || 0) + 1;
				componentCounters[comp.type] = count;
				const prefix = COMPONENT_PREFIX[comp.type] || '';
				newComp.attributes['InstName'] = prefix ? `${prefix}${count}` : '';
			}

			schematic.components = [...schematic.components, newComp];
			// Select the new component
			selectedIds = new Set([newComp.id]);
			selectedWireIds = new Set();
			render();
			return;
		}

		const wire = findWireAt(clickPos);
		if (wire) {
			// Duplicate wire with offset
			const offset = grid.size * 2;
			const newWire: Wire = {
				id: crypto.randomUUID(),
				x1: wire.x1 + offset,
				y1: wire.y1 + offset,
				x2: wire.x2 + offset,
				y2: wire.y2 + offset
			};
			schematic.wires = [...schematic.wires, newWire];
			selectedWireIds = new Set([newWire.id]);
			selectedIds = new Set();
			render();
		}
	}

	/** Find component at position */
	function findComponentAt(pos: Point): Component | null {
		for (let i = schematic.components.length - 1; i >= 0; i--) {
			const comp = schematic.components[i];
			if (hitTestComponent(comp, pos.x, pos.y)) {
				return comp;
			}
		}
		return null;
	}

	/** Find wire at position */
	function findWireAt(pos: Point, tolerance: number = 5): Wire | null {
		for (let i = schematic.wires.length - 1; i >= 0; i--) {
			const wire = schematic.wires[i];
			if (hitTestWire(wire, pos.x, pos.y, tolerance)) {
				return wire;
			}
		}
		return null;
	}

	/** Hit test for wire segment */
	function hitTestWire(wire: Wire, px: number, py: number, tolerance: number): boolean {
		// Distance from point to line segment
		const dx = wire.x2 - wire.x1;
		const dy = wire.y2 - wire.y1;
		const len2 = dx * dx + dy * dy;

		if (len2 === 0) {
			// Wire is a point
			return Math.hypot(px - wire.x1, py - wire.y1) <= tolerance;
		}

		// Project point onto line
		const t = Math.max(0, Math.min(1, ((px - wire.x1) * dx + (py - wire.y1) * dy) / len2));
		const projX = wire.x1 + t * dx;
		const projY = wire.y1 + t * dy;

		return Math.hypot(px - projX, py - projY) <= tolerance;
	}

	function placeComponent() {
		if (!placingType) return;

		const snapped = snapToGrid(schematicPos);
		const def = COMPONENT_DEFS[placingType];

		// Generate instance name using correct SPICE prefix
		// e.g., inductor -> L, current -> I, resistor -> R
		const prefix = COMPONENT_PREFIX[placingType] || '';
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
		// Escape cancels current operation and returns to select mode
		if (e.key === 'Escape') {
			cancelCurrentOperation();
			return;
		}

		// Mode shortcuts (number keys mapped from F-keys)
		// 3 = Wire (F3), 5 = Delete (F5), 7 = Move (F7)
		if (!e.ctrlKey && !e.metaKey && !e.altKey) {
			const modeKey = MODE_SHORTCUTS[e.key];
			if (modeKey) {
				setMode(modeKey);
				return;
			}

			// W key also enters wire mode (LTSpice shortcut)
			if (e.key === 'w' || e.key === 'W') {
				setMode('wire');
				return;
			}
		}

		// Space toggles wire direction while drawing
		if (e.key === ' ' && mode === 'wire' && wireDraw.isDrawing) {
			e.preventDefault();
			wireDraw = {
				...wireDraw,
				direction: wireDraw.direction === 'horizontal-first' ? 'vertical-first' : 'horizontal-first'
			};
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
				wireDraw = { ...DEFAULT_WIRE_DRAW };
				render();
				return;
			}
		}

		// Ctrl+R to rotate while placing or rotate selected
		if ((e.key === 'r' || e.key === 'R') && (e.ctrlKey || e.metaKey)) {
			e.preventDefault();
			if (mode === 'place' && placingType) {
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
			if (mode === 'place' && placingType) {
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

		// Delete selected components and wires
		if (e.key === 'Delete' || e.key === 'Backspace') {
			if (selectedIds.size > 0 || selectedWireIds.size > 0) {
				schematic.components = schematic.components.filter(c => !selectedIds.has(c.id));
				schematic.wires = schematic.wires.filter(w => !selectedWireIds.has(w.id));
				selectedIds = new Set();
				selectedWireIds = new Set();
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
		if (e.key === 'Home' || (e.key === 'f' && mode !== 'place')) {
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

	/** Set editor mode */
	function setMode(newMode: EditorMode) {
		mode = newMode;
		placingType = null;
		wireDraw = { ...DEFAULT_WIRE_DRAW };
		selectedIds = new Set();
		selectedWireIds = new Set();
		render();
	}

	/** Cancel current operation and return to select mode */
	function cancelCurrentOperation() {
		placingType = null;
		wireDraw = { ...DEFAULT_WIRE_DRAW };
		mode = 'select';
		selectedIds = new Set();
		selectedWireIds = new Set();
		render();
	}

	// Re-render when schematic changes
	$effect(() => {
		if (schematic && ctx) {
			render();
		}
	});

	/** Get cursor style based on current mode */
	function getCursorStyle(): string {
		if (isDragging) return 'grabbing';
		switch (mode) {
			case 'delete': return 'not-allowed';
			case 'wire': return 'crosshair';
			case 'move': return 'move';
			case 'place': return 'copy';
			case 'duplicate': return 'copy';
			default: return 'default';
		}
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="schematic-container" bind:this={container}>
	<canvas
		bind:this={canvas}
		class="schematic-canvas"
		style="cursor: {getCursorStyle()}"
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
		outline: none;
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

