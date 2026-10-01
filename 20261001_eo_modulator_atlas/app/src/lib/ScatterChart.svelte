
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Plot from './Plot.svelte';
	import DeviceTip from './DeviceTip.svelte';
	import { ui } from './state.svelte';
	import { groupColor, plotTheme, GROUP_ORDER } from './colors';
	import { paretoFront } from './logic';
	import type { Omitted, Pt } from './charts';

	interface PanelSpec {
		pts: Pt[];
		yTitle: string;
	}
	interface Props {
		letter: string;
		desc: string;
		panels: PanelSpec[];
		xTitle: string;
		xLog?: boolean;
		yLog?: boolean;
		allowX?: boolean;
		allowY?: boolean;
		omitted: Omitted;
		frontier?: boolean;
		xTickvals?: number[];
		xTicktext?: string[];
		xDtick?: number;
		extraTraces?: unknown[];
		xRange?: [number, number];
		headerExtra?: Snippet;
	}
	let {
		letter,
		desc,
		panels,
		xTitle,
		xLog = $bindable(false),
		yLog = $bindable(false),
		allowX = true,
		allowY = true,
		omitted,
		frontier = false,
		xTickvals,
		xTicktext,
		xDtick,
		extraTraces = [],
		xRange,
		headerExtra
	}: Props = $props();

	const th = $derived(plotTheme(ui.theme));
	const nPan = $derived(panels.length);
	const yName = (panel: number) => (nPan === 1 ? 'yaxis' : panel === 0 ? 'yaxis2' : 'yaxis');
	const yRef = (panel: number) => (nPan === 1 ? 'y' : panel === 0 ? 'y2' : 'y');

	interface Pin {
		id: string;
		panel: number;
	}
	let pins = $state<Pin[]>([]);
	let hover = $state<{ id: string; x: number; y: number } | null>(null);
	let wrap: HTMLDivElement;
	let gdEl: any = $state(null);

	const ptIndex = $derived.by(() => {
		const m = new Map<string, Pt>();
		for (const p of panels) for (const pt of p.pts) m.set(`${pt.panel}|${pt.id}`, pt);
		return m;
	});

	// drop pins whose point disappeared (filters changed)
	$effect(() => {
		const keep = pins.filter((p) => ptIndex.has(`${p.panel}|${p.id}`));
		if (keep.length !== pins.length) pins = keep;
	});

	function symbolFor(p: Pt): string {
		if (p.qy === 'lt') return 'arrow-bar-down';
		if (p.qy === 'gt') return 'arrow-bar-up';
		if (p.qx === 'lt') return 'arrow-bar-left';
		if (p.qx === 'gt') return 'arrow-bar-right';
		return p.sim ? 'diamond' : 'circle';
	}

	const traces = $derived.by(() => {
		const out: any[] = [];
		panels.forEach((panel, pi) => {
			const ax = { xaxis: 'x', yaxis: yRef(pi) };
			if (frontier && pi === 0) {
				const f = paretoFront(panel.pts, true, true);
				if (f.length > 1) {
					out.push({ type: 'scatter', mode: 'lines', x: f.map((p) => p.x), y: f.map((p) => p.y), line: { color: th.muted, width: 1, dash: 'dash' }, hoverinfo: 'skip', showlegend: false, ...ax });
				}
			}
			for (const g of GROUP_ORDER) {
				const pts = panel.pts.filter((p) => p.group === g);
				if (!pts.length) continue;
				const color = groupColor(g, ui.theme);
				out.push({
					type: 'scatter',
					mode: 'markers',
					x: pts.map((p) => p.x),
					y: pts.map((p) => p.y),
					customdata: pts.map((p) => p.id),
					meta: pi,
					marker: { color, size: pts.map((p) => (symbolFor(p).startsWith('arrow') ? 13 : 10)), symbol: pts.map(symbolFor), line: { color: th.surface, width: 1.5 } },
					hoverinfo: 'none',
					showlegend: false,
					...ax
				});
			}
		});
		for (const t of extraTraces) out.push(t);
		// pinned rings
		const ringsByPanel = new Map<number, Pt[]>();
		for (const p of pins) {
			const pt = ptIndex.get(`${p.panel}|${p.id}`);
			if (pt) ringsByPanel.set(p.panel, [...(ringsByPanel.get(p.panel) ?? []), pt]);
		}
		for (const [pi, pts] of ringsByPanel) {
			out.push({ type: 'scatter', mode: 'markers', x: pts.map((p) => p.x), y: pts.map((p) => p.y), marker: { symbol: 'circle-open', size: 18, color: th.ink, line: { color: th.ink, width: 1.5 } }, hoverinfo: 'skip', showlegend: false, xaxis: 'x', yaxis: yRef(pi) });
		}
		return out;
	});

	const layout = $derived.by(() => {
		const axBase = {
			gridcolor: th.grid,
			gridwidth: 1,
			linecolor: th.line,
			zeroline: false,
			ticks: 'outside',
			tickcolor: th.line,
			tickfont: { size: 10, color: th.ink2 },
			automargin: true,
			showline: true,
			mirror: false
		};
		const l: Record<string, any> = {
			paper_bgcolor: 'rgba(0,0,0,0)',
			plot_bgcolor: 'rgba(0,0,0,0)',
			margin: { l: 52, r: 12, t: 8, b: 8 },
			font: { family: 'system-ui, -apple-system, Segoe UI, sans-serif', size: 11, color: th.ink2 },
			showlegend: false,
			hovermode: 'closest',
			hoverdistance: 28,
			dragmode: 'zoom',
			xaxis: {
				...axBase,
				title: { text: xTitle, standoff: 4, font: { size: 11, color: th.ink2 } },
				type: xLog ? 'log' : 'linear',
				...(xTickvals ? { tickmode: 'array', tickvals: xTickvals, ticktext: xTicktext } : {}),
				...(xDtick ? { dtick: xDtick, tickformat: 'd' } : {}),
				...(xRange ? { range: xRange, autorange: false } : {}),
				anchor: 'y',
				domain: [0, 1]
			}
		};
		if (nPan === 1) {
			l.yaxis = { ...axBase, title: { text: panels[0].yTitle, standoff: 4, font: { size: 11, color: th.ink2 } }, type: yLog ? 'log' : 'linear' };
		} else {
			l.yaxis = { ...axBase, domain: [0, 0.46], title: { text: panels[1].yTitle, standoff: 4, font: { size: 11, color: th.ink2 } }, type: yLog ? 'log' : 'linear', anchor: 'x' };
			l.yaxis2 = { ...axBase, domain: [0.54, 1], title: { text: panels[0].yTitle, standoff: 4, font: { size: 11, color: th.ink2 } }, type: yLog ? 'log' : 'linear', anchor: 'x' };
		}
		return l;
	});

	// ----- hover and pin handling -----
	function evPt(e: any): { id: string; panel: number } | null {
		const p = e?.points?.[0];
		if (!p || typeof p.customdata !== 'string') return null;
		return { id: p.customdata, panel: Number(p.data?.meta ?? 0) };
	}

	function onhover(e: any) {
		const p = evPt(e);
		if (!p || !wrap) return;
		const r = wrap.getBoundingClientRect();
		hover = { id: p.id, x: e.event.clientX - r.left, y: e.event.clientY - r.top };
	}
	function onunhover() {
		hover = null;
	}

	let lastHit = 0;
	function onclick(e: any) {
		const p = evPt(e);
		if (!p) return;
		lastHit = performance.now();
		const exists = pins.some((x) => x.id === p.id && x.panel === p.panel);
		pins = exists ? pins.filter((x) => !(x.id === p.id && x.panel === p.panel)) : [...pins, p];
		hover = null;
	}
	let downXY: [number, number] | null = null;
	function pdown(e: PointerEvent) {
		downXY = [e.clientX, e.clientY];
	}
	function wclick(e: MouseEvent) {
		if ((e.target as HTMLElement).closest('.pin')) return;
		if (performance.now() - lastHit < 250) return;
		const moved = downXY ? Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 4 : false;
		if (!moved && pins.length) pins = [];
	}

	let pinXY = $state<{ key: string; id: string; px: number; py: number }[]>([]);
	function recompute(gd: any) {
		const fl = gd?._fullLayout;
		if (!fl?.xaxis || fl.xaxis._offset === undefined) return;
		const out: { key: string; id: string; px: number; py: number }[] = [];
		for (const p of pins) {
			const pt = ptIndex.get(`${p.panel}|${p.id}`);
			const ya = fl[yName(p.panel)];
			if (!pt || !ya) continue;
			const px = fl.xaxis._offset + fl.xaxis.d2p(pt.x);
			const py = ya._offset + ya.d2p(pt.y);
			if (Number.isFinite(px) && Number.isFinite(py)) out.push({ key: `${p.panel}|${p.id}`, id: p.id, px, py });
		}
		pinXY = out;
	}
	$effect(() => {
		void pins;
		void panels;
		if (gdEl) recompute(gdEl);
	});

	function place(x: number, y: number, w = 300, h = 210): { left: number; top: number } {
		const W = wrap?.clientWidth ?? 600;
		const H = wrap?.clientHeight ?? 400;
		let left = x + 14;
		if (left + w > W) left = Math.max(0, x - 14 - w);
		let top = y + 10;
		if (top + h > H) top = Math.max(0, H - h);
		return { left, top };
	}
</script>

<section class="panel">
	<header>
		<span class="letter" title={desc}>{letter}</span>
		<span class="badge num" title="{omitted.plotted} of {omitted.total} devices plotted. Not reported (omitted, never drawn at 0): x {omitted.missingX}, y {omitted.missingY}, both {omitted.missingBoth}.">
			{omitted.plotted}/{omitted.total}
		</span>
		<div class="grow"></div>
		{@render headerExtra?.()}
		{#if allowX}
			<span class="seg" title="x axis scale">
				<button class:on={!xLog} onclick={() => (xLog = false)}>x lin</button><button class:on={xLog} onclick={() => (xLog = true)}>x log</button>
			</span>
		{/if}
		{#if allowY}
			<span class="seg" title="y axis scale">
				<button class:on={!yLog} onclick={() => (yLog = false)}>y lin</button><button class:on={yLog} onclick={() => (yLog = true)}>y log</button>
			</span>
		{/if}
	</header>
	<div class="body" bind:this={wrap} onpointerdown={pdown} onclick={wclick} role="presentation">
		<Plot data={traces} {layout} {onhover} {onunhover} {onclick} onafterplot={recompute} bind:gd={gdEl} />
		{#if omitted.plotted === 0}<div class="empty muted">no plottable values</div>{/if}
		{#each pinXY as p (p.key)}
			{@const pos = place(p.px, p.py)}
			<div class="pin" style="left:{pos.left}px; top:{pos.top}px">
				<button class="x" aria-label="Unpin" title="Unpin" onclick={() => (pins = pins.filter((q) => `${q.panel}|${q.id}` !== p.key))}>
					<svg viewBox="0 0 10 10" width="8" height="8"><path d="M1 1 L9 9 M9 1 L1 9" stroke="currentColor" stroke-width="1.4" /></svg>
				</button>
				<DeviceTip id={p.id} />
			</div>
		{/each}
		{#if hover}
			{@const pos = place(hover.x, hover.y)}
			<div class="float" style="left:{pos.left}px; top:{pos.top}px"><DeviceTip id={hover.id} /></div>
		{/if}
	</div>
</section>

<style>
	.panel {
		background: var(--surface);
		border: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		height: 400px;
		min-width: 0;
	}
	header {
		display: flex;
		gap: 6px;
		align-items: center;
		height: 28px;
		padding: 0 6px;
		border-bottom: 1px solid var(--border);
		flex: none;
	}
	.letter {
		font-weight: 700;
		color: var(--ink-2);
		width: 16px;
		text-align: center;
		border: 1px solid var(--line);
	}
	.badge {
		color: var(--ink-3);
		font-size: 10px;
	}
	.grow {
		flex: 1;
	}
	.seg {
		display: inline-flex;
	}
	.seg button {
		height: 20px;
		font-size: 10px;
		padding: 0 6px;
	}
	.body {
		position: relative;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}
	.float {
		position: absolute;
		z-index: 10;
		pointer-events: none;
	}
	.pin {
		position: absolute;
		z-index: 9;
	}
	.pin :global(.tip) {
		pointer-events: auto;
	}
	.x {
		position: absolute;
		right: 2px;
		top: 2px;
		width: 18px;
		height: 18px;
		padding: 0;
		z-index: 2;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: 0;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
	}
</style>
