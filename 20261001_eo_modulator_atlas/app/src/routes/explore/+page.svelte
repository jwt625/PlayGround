<script lang="ts">
	import { filters, store, ui } from '../../lib/state.svelte';
	import { applyFilters, chartDevices, enumLabel } from '../../lib/logic';
	import {
		buildPoints,
		countryCounts,
		gBaud,
		gBw,
		gIl,
		gLen,
		gRate,
		gVpiIl,
		gVpil,
		groupStats,
		jitter,
		orgCounts,
		yearOf
	} from '../../lib/charts';
	import { GROUP_ORDER, groupColor, materialGroup } from '../../lib/colors';
	import FilterPanel from '../../lib/FilterPanel.svelte';
	import ScatterChart from '../../lib/ScatterChart.svelte';
	import YearStack from '../../lib/YearStack.svelte';
	import TopBars from '../../lib/TopBars.svelte';
	import HeatChart from '../../lib/HeatChart.svelte';
	import ScoreCard from '../../lib/ScoreCard.svelte';

	const a = $derived(store.atlas);
	const idx = $derived(store.index);
	const view = $derived(a && idx ? applyFilters(a, filters, idx) : null);
	const devs = $derived(view ? chartDevices(view, filters.allDevices) : []);
	const papers = $derived(idx?.paperById ?? new Map());

	// per-chart axis scale state
	let aX = $state(true), aY = $state(true);
	let bX = $state(true), bY = $state(false);
	let cY = $state(true);
	let dY = $state(true);
	let eY = $state(true);
	let fX = $state(true), fY = $state(true);
	let central = $state<'median' | 'mean'>('median');
	let orgKind = $state<'affil' | 'fab'>('affil');

	const A = $derived(buildPoints(devs, papers, gBw, gVpil, 0, true));
	const B = $derived(buildPoints(devs, papers, gVpil, gIl, 0, true));
	const D = $derived(buildPoints(devs, papers, yearOf, gBw, 0, true));
	const Ebaud = $derived(buildPoints(devs, papers, yearOf, gBaud, 0, true));
	const Erate = $derived(buildPoints(devs, papers, yearOf, gRate, 1, true));
	const F = $derived(buildPoints(devs, papers, gLen, gBw, 0, true));

	const years = $derived.by(() => {
		const ys = [...D.pts, ...Ebaud.pts, ...Erate.pts].map((p) => p.x);
		if (!ys.length) return { min: 0, max: 0, dtick: 1 };
		const min = Math.min(...ys);
		const max = Math.max(...ys);
		const span = max - min;
		return { min, max, dtick: span <= 12 ? 1 : span <= 30 ? 2 : 5 };
	});

	const C = $derived.by(() => {
		const base = buildPoints(devs, papers, () => ({ v: 0, qual: null, derived: false, basis: null, field: null }), gVpiIl, 0, true);
		const stats = groupStats(base.pts);
		const pos = new Map(stats.map((s) => [s.group, s.index]));
		const pts = base.pts.map((p) => ({ ...p, x: (pos.get(p.group) ?? 0) + 0.2 * jitter(p.id) }));
		return { pts, omitted: base.omitted, stats };
	});

	const groupLabel = (g: string) => (g === 'other_group' ? 'Other EO materials' : a ? enumLabel(a, 'eo_material', g) : g);

	function rgba(hex: string, al: number): string {
		const n = parseInt(hex.slice(1), 16);
		return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${al})`;
	}

	const cExtra = $derived.by(() => {
		const out: any[] = [];
		for (const s of C.stats) {
			const col = groupColor(s.group, ui.theme);
			out.push({
				type: 'scatter',
				mode: 'lines',
				x: [s.index - 0.36, s.index + 0.36, s.index + 0.36, s.index - 0.36, s.index - 0.36],
				y: [s.min, s.min, s.max, s.max, s.min],
				fill: 'toself',
				fillcolor: rgba(col, 0.14),
				line: { color: rgba(col, 0.6), width: 1 },
				hoverinfo: 'skip',
				showlegend: false
			});
			const v = central === 'median' ? s.median : s.mean;
			out.push({ type: 'scatter', mode: 'lines', x: [s.index - 0.36, s.index + 0.36], y: [v, v], line: { color: col, width: 4 }, hoverinfo: 'skip', showlegend: false });
		}
		return out;
	});

	const presentGroups = $derived([...new Set(devs.map((d) => materialGroup(d.eo_material)))].sort((x, y) => GROUP_ORDER.indexOf(x) - GROUP_ORDER.indexOf(y)));

	const orgMap = $derived(view ? orgCounts(view.papers, orgKind) : new Map<string, number>());
	const countryMap = $derived(view && a ? countryCounts(view.papers, (c) => enumLabel(a, 'country', c)) : new Map<string, number>());
</script>

<div class="page">
	<div class="side"><FilterPanel charts nShown={devs.length} /></div>
	<div class="main">
		<div class="legend">
			{#each presentGroups as g (g)}
				<span class="it" title={groupLabel(g)}><i class="sw" style="background:{groupColor(g, ui.theme)}"></i>{groupLabel(g)}</span>
			{/each}
			<span class="sep"></span>
			<span class="it" title="Measured"><svg viewBox="0 0 10 10" width="10" height="10"><circle cx="5" cy="5" r="4" fill="currentColor" /></svg>measured</span>
			<span class="it" title="Simulated, predicted or design target"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M5 0 L10 5 L5 10 L0 5 Z" fill="currentColor" /></svg>simulated / predicted</span>
			<span class="it" title="Upper / lower bound drawn as a bar-arrow in the bound direction"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M1 1 L1 9 M1 5 L9 5 M5 2 L1 5 L5 8" fill="none" stroke="currentColor" stroke-width="1.4" /></svg>bound</span>
			<span class="muted" title="Hover for details; click a point to pin its card, click empty space to unpin all. Values not reported are omitted, never drawn at 0.">
				{filters.allDevices ? 'all devices' : 'representative device per paper'}
			</span>
		</div>
		<div class="scroll">
			{#if store.error}
				<div class="msg">Failed to load data: {store.error}</div>
			{:else if !view}
				<div class="msg"><span class="spinner"></span></div>
			{:else}
				<div class="grid">
					<ScatterChart letter="a" desc="Vpi*L against 3 dB bandwidth; dashed line is the Pareto frontier (lower Vpi*L, higher bandwidth)" panels={[{ pts: A.pts, yTitle: 'Vpi*L (V*cm)' }]} xTitle="3 dB bandwidth (GHz)" bind:xLog={aX} bind:yLog={aY} omitted={A.omitted} frontier />
					<ScatterChart letter="b" desc="On-chip insertion loss against Vpi*L" panels={[{ pts: B.pts, yTitle: 'On-chip insertion loss (dB)' }]} xTitle="Vpi*L (V*cm)" bind:xLog={bX} bind:yLog={bY} omitted={B.omitted} />
					<ScatterChart
						letter="c"
						desc="Vpi*IL (Vpi times on-chip loss) by EO material: range band min to max, thick line = median or mean, points overlaid"
						panels={[{ pts: C.pts, yTitle: 'Vpi*IL (V*dB)' }]}
						xTitle="EO material"
						allowX={false}
						bind:yLog={cY}
						omitted={C.omitted}
						xTickvals={C.stats.map((s) => s.index)}
						xTicktext={C.stats.map((s) => `${groupLabel(s.group)} (${s.n})`)}
						xRange={C.stats.length ? [-0.6, C.stats.length - 0.4] : undefined}
						extraTraces={cExtra}
					>
						{#snippet headerExtra()}
							<span class="seg" title="Thick line statistic">
								<button class:on={central === 'median'} onclick={() => (central = 'median')}>median</button><button class:on={central === 'mean'} onclick={() => (central = 'mean')}>mean</button>
							</span>
						{/snippet}
					</ScatterChart>
					<ScatterChart letter="d" desc="3 dB bandwidth against publication year" panels={[{ pts: D.pts, yTitle: '3 dB bandwidth (GHz)' }]} xTitle="Publication year" allowX={false} bind:yLog={dY} omitted={D.omitted} xDtick={years.dtick} />
					<ScatterChart
						letter="e"
						desc="Maximum demonstrated symbol rate (top) and line rate (bottom) against publication year"
						panels={[
							{ pts: Ebaud.pts, yTitle: 'Max baud rate (GBd)' },
							{ pts: Erate.pts, yTitle: 'Max line rate (Gb/s)' }
						]}
						xTitle="Publication year"
						allowX={false}
						bind:yLog={eY}
						omitted={{ total: Ebaud.omitted.total + Erate.omitted.total, plotted: Ebaud.omitted.plotted + Erate.omitted.plotted, missingX: 0, missingY: Ebaud.omitted.missingY + Erate.omitted.missingY, missingBoth: 0 }}
						xDtick={years.dtick}
					/>
					<ScatterChart letter="f" desc="3 dB bandwidth against active length" panels={[{ pts: F.pts, yTitle: '3 dB bandwidth (GHz)' }]} xTitle="Active length (mm)" bind:xLog={fX} bind:yLog={fY} omitted={F.omitted} />
					<YearStack {view} />
					<div class="two">
						<TopBars letter="h" desc="Papers per organization (author affiliations, or fabricating organizations)" counts={orgMap}>
							{#snippet toggles()}
								<span class="seg" title="Organization role">
									<button class:on={orgKind === 'affil'} onclick={() => (orgKind = 'affil')}>affiliation</button><button class:on={orgKind === 'fab'} onclick={() => (orgKind = 'fab')}>fab</button>
								</span>
							{/snippet}
						</TopBars>
						<TopBars letter="h" desc="Papers per country (derived from affiliations)" counts={countryMap} />
					</div>
					<div class="full"><HeatChart {view} /></div>
					<div class="full"><ScoreCard /></div>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.page {
		display: grid;
		grid-template-columns: 220px minmax(0, 1fr);
		height: 100%;
	}
	.side {
		min-height: 0;
	}
	.main {
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 0 10px;
		height: 30px;
		flex: none;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		overflow: hidden;
		white-space: nowrap;
	}
	.it {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		color: var(--ink-2);
	}
	.sw {
		width: 10px;
		height: 10px;
		display: inline-block;
	}
	.sep {
		width: 1px;
		height: 14px;
		background: var(--line);
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 8px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
		align-items: start;
	}
	.two {
		grid-column: 1 / -1;
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
	}
	.full {
		grid-column: 1 / -1;
	}
	.msg {
		padding: 20px;
		color: var(--ink-3);
		display: flex;
		justify-content: center;
	}
	@media (max-width: 1100px) {
		.grid,
		.two {
			grid-template-columns: 1fr;
		}
	}
</style>
