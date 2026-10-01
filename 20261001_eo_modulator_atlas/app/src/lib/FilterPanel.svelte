
<script lang="ts">
	import { filters, resetFilters, store } from './state.svelte';
	import { countBy } from './logic';
	import type { RepMode } from './types';

	let { charts = false, nShown = 0 }: { charts?: boolean; nShown?: number } = $props();

	const a = $derived(store.atlas);

	// search with 0.3 s debounce
	let qLocal = $state('');
	let timer: ReturnType<typeof setTimeout> | undefined;
	let lastQ = '';
	function onInput(v: string) {
		qLocal = v;
		clearTimeout(timer);
		timer = setTimeout(() => {
			filters.q = v;
			lastQ = v;
		}, 300);
	}
	$effect(() => {
		// external changes (hash, reset) flow back into the box
		if (filters.q !== lastQ) {
			lastQ = filters.q;
			qLocal = filters.q;
		}
	});

	const open = $state<Record<string, boolean>>({ search: true, device: true, paper: true, flags: true, view: true });

	type Opt = { value: string; label: string; n: number };
	function opts(en: string, counts: Map<string, number>): Opt[] {
		if (!a) return [];
		return (a.enums[en] ?? [])
			.map((e) => ({ value: e.value, label: e.label, n: counts.get(e.value) ?? 0 }))
			.filter((o) => o.n > 0);
	}

	const materialOpts = $derived(a ? opts('eo_material', countBy(a.devices, (d) => d.eo_material)) : []);
	const classOpts = $derived(a ? opts('device_class', countBy(a.devices, (d) => d.device_class)) : []);
	const platformOpts = $derived.by(() => {
		if (!a) return [] as Opt[];
		const c = countBy(a.devices, (d) => d.waveguide_platform ?? 'unspecified');
		const o = opts('waveguide_platform', c);
		if (c.get('unspecified')) o.push({ value: 'unspecified', label: 'Unspecified', n: c.get('unspecified') ?? 0 });
		return o;
	});
	const sourceOpts = $derived(a ? opts('source_type', countBy(a.papers, (p) => p.source_type)) : []);
	const regionOpts = $derived(a ? opts('region', countBy(a.papers, (p) => p.regions_derived)) : []);
	const countryOpts = $derived(a ? opts('country', countBy(a.papers, (p) => p.countries_derived)) : []);
	const yearBounds = $derived.by(() => {
		if (!a || !a.papers.length) return { min: 0, max: 0 };
		const ys = a.papers.map((p) => p.year);
		return { min: Math.min(...ys), max: Math.max(...ys) };
	});

	function toggle(list: string[], v: string): string[] {
		return list.includes(v) ? list.filter((x) => x !== v) : [...list, v];
	}

	const repModes: { v: RepMode; l: string; t: string }[] = [
		{ v: 'default', l: 'Completeness, then FOM', t: 'Most complete record; ties by figure of merit, then lowest Vpi*L, then device id' },
		{ v: 'lowest_vpil', l: 'Lowest Vpi*L', t: 'Device with the lowest Vpi*L (reported or derived)' },
		{ v: 'highest_bw', l: 'Highest 3 dB BW', t: 'Device with the highest reported 3 dB bandwidth' },
		{ v: 'highest_fom', l: 'Highest FOM', t: 'Device with the highest figure of merit' },
		{ v: 'lowest_vpi_il', l: 'Lowest Vpi*IL', t: 'Device with the lowest Vpi times on-chip loss' }
	];

	function num(v: string): number | null {
		const n = Number(v);
		return v.trim() === '' || !Number.isFinite(n) ? null : Math.round(n);
	}
	const active = $derived(
		!!(
			filters.q ||
			filters.materials.length ||
			filters.classes.length ||
			filters.platforms.length ||
			filters.sourceTypes.length ||
			filters.regions.length ||
			filters.countries.length ||
			filters.yearMin !== null ||
			filters.yearMax !== null ||
			filters.measuredOnly ||
			filters.hasSim
		)
	);
</script>

{#snippet group(id: string, label: string, body: import('svelte').Snippet)}
	<section>
		<button class="gh" onclick={() => (open[id] = !open[id])} aria-expanded={open[id]}>
			<svg viewBox="0 0 10 10" width="8" height="8" class:rot={open[id]} aria-hidden="true"><path d="M3 1 L7 5 L3 9" fill="none" stroke="currentColor" stroke-width="1.5" /></svg>
			{label}
		</button>
		{#if open[id]}<div class="gb">{@render body()}</div>{/if}
	</section>
{/snippet}

{#snippet checks(list: Opt[], sel: string[], key: 'materials' | 'classes' | 'platforms' | 'sourceTypes' | 'regions' | 'countries')}
	<div class="checks">
		{#each list as o (o.value)}
			<label title="{o.label}: {o.n}">
				<input type="checkbox" checked={sel.includes(o.value)} onchange={() => (filters[key] = toggle(sel, o.value))} />
				<span class="lbl">{o.label}</span>
				<span class="n num">{o.n}</span>
			</label>
		{/each}
	</div>
{/snippet}

<aside>
	<div class="top">
		<span class="muted num" title="Rows passing the filters">{nShown} shown</span>
		<button onclick={resetFilters} disabled={!active} title="Clear all filters">Reset</button>
	</div>
	<div class="scroll">
		{#snippet searchBody()}
			<input type="search" placeholder="Search" value={qLocal} oninput={(e) => onInput(e.currentTarget.value)} title="Paper, authors, organizations, materials, DOI" />
		{/snippet}
		{@render group('search', 'Search', searchBody)}

		{#snippet deviceBody()}
			<div class="sub">EO material</div>
			{@render checks(materialOpts, filters.materials, 'materials')}
			<div class="sub">Device class</div>
			{@render checks(classOpts, filters.classes, 'classes')}
			<div class="sub">Platform</div>
			{@render checks(platformOpts, filters.platforms, 'platforms')}
		{/snippet}
		{@render group('device', 'Device', deviceBody)}

		{#snippet paperBody()}
			<div class="sub">Year</div>
			<div class="yr">
				<input type="number" placeholder={String(yearBounds.min)} value={filters.yearMin ?? ''} onchange={(e) => (filters.yearMin = num(e.currentTarget.value))} title="Earliest year" />
				<span class="muted">to</span>
				<input type="number" placeholder={String(yearBounds.max)} value={filters.yearMax ?? ''} onchange={(e) => (filters.yearMax = num(e.currentTarget.value))} title="Latest year" />
			</div>
			<div class="sub">Source type</div>
			{@render checks(sourceOpts, filters.sourceTypes, 'sourceTypes')}
			<div class="sub">Region</div>
			{@render checks(regionOpts, filters.regions, 'regions')}
			<div class="sub">Country</div>
			{@render checks(countryOpts, filters.countries, 'countries')}
		{/snippet}
		{@render group('paper', 'Paper', paperBody)}

		{#snippet flagBody()}
			<label class="row" title="Hide devices whose Vpi, bandwidth or loss is simulated, predicted or a design target">
				<input type="checkbox" checked={filters.measuredOnly} onchange={(e) => (filters.measuredOnly = e.currentTarget.checked)} />
				<span class="lbl">Measured only</span>
			</label>
			<label class="row" title="Only papers with a simulation config">
				<input type="checkbox" checked={filters.hasSim} onchange={(e) => (filters.hasSim = e.currentTarget.checked)} />
				<span class="lbl">Has simulation</span>
			</label>
		{/snippet}
		{@render group('flags', 'Flags', flagBody)}

		{#snippet viewBody()}
			{#if charts}
				<div class="seg" title="Charts of devices: every matching device, or one representative device per paper">
					<button class:on={!filters.allDevices} onclick={() => (filters.allDevices = false)}>Representative</button>
					<button class:on={filters.allDevices} onclick={() => (filters.allDevices = true)}>All devices</button>
				</div>
			{/if}
			<div class="sub">Representative device</div>
			<div class="checks">
				{#each repModes as m (m.v)}
					<label title={m.t}>
						<input type="radio" name="rep" checked={filters.rep === m.v} onchange={() => (filters.rep = m.v)} />
						<span class="lbl">{m.l}</span>
					</label>
				{/each}
			</div>
		{/snippet}
		{@render group('view', 'View', viewBody)}
	</div>
</aside>

<style>
	aside {
		display: flex;
		flex-direction: column;
		min-height: 0;
		height: 100%;
		background: var(--surface);
		border-right: 1px solid var(--border);
		width: 100%;
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 6px 8px;
		border-bottom: 1px solid var(--border);
		flex: none;
		height: 36px;
	}
	.scroll {
		overflow-y: auto;
		flex: 1;
		min-height: 0;
	}
	section {
		border-bottom: 1px solid var(--border);
	}
	.gh {
		width: 100%;
		text-align: left;
		display: flex;
		align-items: center;
		gap: 6px;
		background: transparent;
		border: 0;
		height: 26px;
		padding: 0 8px;
		color: var(--ink-2);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 10px;
	}
	.gh svg {
		transition: transform 0.1s;
	}
	.gh svg.rot {
		transform: rotate(90deg);
	}
	.gb {
		padding: 2px 8px 8px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.sub {
		color: var(--ink-3);
		font-size: 10px;
		margin-top: 4px;
	}
	.checks {
		display: flex;
		flex-direction: column;
		max-height: 150px;
		overflow-y: auto;
	}
	label {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 1px 2px;
		cursor: pointer;
	}
	label:hover {
		background: var(--hover);
	}
	.lbl {
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.n {
		color: var(--ink-3);
	}
	.yr {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.yr input {
		width: 64px;
	}
	input[type='search'] {
		width: 100%;
	}
	.seg {
		display: flex;
	}
	.seg button {
		flex: 1;
	}
	.row {
		padding: 2px;
	}
</style>
