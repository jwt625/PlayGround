
<script lang="ts">
	import { link } from '../../lib/paths';
	import { filters, store, tableState, ui } from '../../lib/state.svelte';
	import { applyFilters, cycleSort, sortRows, BASIS_MARK, BASIS_TIP } from '../../lib/logic';
	import { COLS, cellFor, sortValue, tableCsv, type Cell } from '../../lib/columns';
	import FilterPanel from '../../lib/FilterPanel.svelte';
	import Drawer from '../../lib/Drawer.svelte';
	import type { Device, Paper } from '../../lib/types';

	const a = $derived(store.atlas);
	const idx = $derived(store.index);
	const view = $derived(a && idx ? applyFilters(a, filters, idx) : null);

	interface Row {
		kind: 'paper' | 'device';
		paper: Paper;
		dev: Device;
		dim: boolean;
		isRep: boolean;
	}

	const sorted = $derived.by(() => {
		if (!view || !a) return [] as { paper: Paper; rep: Device }[];
		const base = view.papers.map((p) => ({ paper: p, rep: view.reps.get(p.paper_id) as Device }));
		return sortRows(base, tableState.sort, (r, key) => sortValue(a, key, r.paper, r.rep));
	});

	const rows = $derived.by(() => {
		if (!view || !idx) return [] as Row[];
		const out: Row[] = [];
		const open = new Set(tableState.expanded);
		for (const r of sorted) {
			out.push({ kind: 'paper', paper: r.paper, dev: r.rep, dim: false, isRep: true });
			if (open.has(r.paper.paper_id)) {
				for (const d of idx.devsByPaper.get(r.paper.paper_id) ?? []) {
					out.push({ kind: 'device', paper: r.paper, dev: d, dim: !view.matched.has(d.device_id), isRep: d.device_id === r.rep.device_id });
				}
			}
		}
		return out;
	});

	const visible = $derived(COLS.filter((c) => tableState.columns.includes(c.id)));
	const tableWidth = $derived(visible.reduce((s, c) => s + c.width, 0) + 24);
	const expandable = $derived(view?.papers.filter(p => p.n_devices > 1) ?? []);
	const allOpen = $derived(expandable.length > 0 && expandable.every((p) => tableState.expanded.includes(p.paper_id)));

	function toggleExpand(id: string) {
		tableState.expanded = tableState.expanded.includes(id) ? tableState.expanded.filter((x) => x !== id) : [...tableState.expanded, id];
	}
	function toggleAll() {
		if (!view) return;
		tableState.expanded = allOpen ? [] : expandable.map((p) => p.paper_id);
	}

	function sortState(id: string): 'asc' | 'desc' | null {
		return tableState.sort.find((k) => k.key === id)?.dir ?? null;
	}
	function sortRank(id: string): number {
		return tableState.sort.length > 1 ? tableState.sort.findIndex((k) => k.key === id) + 1 : 0;
	}
	function onSort(e: MouseEvent, id: string) {
		tableState.sort = cycleSort(tableState.sort, id, e.shiftKey);
	}

	function cell(id: string, r: Row): Cell {
		if (r.kind === 'device' && id === 'paper') {
			return { text: r.dev.device_label, tip: `${r.dev.device_id}${r.isRep ? '\nrepresentative device' : ''}` };
		}
		if (r.kind === 'device' && COLS.find((c) => c.id === id)?.level === 'paper') return { text: '', muted: true };
		return cellFor(a!, id, r.paper, r.dev);
	}

	function open(r: Row) {
		ui.drawerDevice = r.dev.device_id;
	}

	// column chooser
	let chooser = $state(false);
	function toggleCol(id: string) {
		tableState.columns = tableState.columns.includes(id) ? tableState.columns.filter((c) => c !== id) : [...tableState.columns, id];
		try {
			localStorage.setItem('eo-atlas-columns', JSON.stringify(tableState.columns));
		} catch {
			/* ignore */
		}
	}
	$effect(() => {
		try {
			const s = localStorage.getItem('eo-atlas-columns');
			if (s) {
				const c = JSON.parse(s) as string[];
				if (Array.isArray(c) && c.every((x) => COLS.some((d) => d.id === x))) tableState.columns = c;
			}
		} catch {
			/* ignore */
		}
	});

	// export
	function exportCsv() {
		if (!a) return;
		const blob = new Blob([tableCsv(a, visible, rows)], { type: 'text/csv' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = `eo-atlas-table-${new Date().toISOString().slice(0, 10)}.csv`;
		link.click();
		URL.revokeObjectURL(url);
	}
</script>

<div class="page">
	<div class="side"><FilterPanel nShown={view?.papers.length ?? 0} /></div>
	<div class="main">
		<div class="tools">
			<button onclick={toggleAll} title={allOpen ? 'Collapse all papers' : 'Expand every multi-device paper'}>{allOpen ? 'Collapse all' : 'Expand all'}</button>
			<div class="chooser">
				<button onclick={() => (chooser = !chooser)} class:on={chooser} title="Choose columns">Columns</button>
				{#if chooser}
					<div class="pop">
						{#each COLS as c (c.id)}
							<label title={c.tip}>
								<input type="checkbox" checked={tableState.columns.includes(c.id)} onchange={() => toggleCol(c.id)} />
								{c.label}{c.unit ? ` (${c.unit})` : ''}
							</label>
						{/each}
					</div>
				{/if}
			</div>
			<button onclick={exportCsv} title="Export the rows currently shown (including expanded devices) as CSV">Export CSV</button>
			<div class="grow"></div>
			<span class="muted legend" title={Object.entries(BASIS_MARK).map(([k, m]) => `${m} = ${BASIS_TIP[k]}`).join('\n') + '\n< > ~ = bound / approximate\nitalic = derived\nno marker = measured'}>
				{#each Object.entries(BASIS_MARK) as [k, m] (k)}<span><sup>{m}</sup> {BASIS_TIP[k]}</span>{/each}
				<span>&lt; &gt; ~ bound</span>
			</span>
		</div>

		<div class="scroller" role="presentation" onclick={() => (chooser = false)}>
			{#if store.error}
				<div class="msg">Failed to load data: {store.error}</div>
			{:else if !view}
				<div class="msg"><span class="spinner"></span></div>
			{:else}
				<table style="width:{tableWidth}px">
					<colgroup>
						<col style="width:24px" />
						{#each visible as c (c.id)}<col style="width:{c.width}px" />{/each}
					</colgroup>
					<thead>
						<tr>
							<th></th>
							{#each visible as c (c.id)}
								{@const s = sortState(c.id)}
								<th class:num={c.num} title="{c.tip}{c.unit ? '\nUnit: ' + c.unit : ''}\nClick to sort; shift-click adds a secondary key">
									<button class="sh" onclick={(e) => onSort(e, c.id)}>
										<span class="lt"><span class="lab">{c.label}</span>{#if c.unit}<span class="unit">{c.unit}</span>{/if}</span>
										<span class="arrow" class:act={s !== null}>
											{#if s === 'asc'}&#9650;{:else if s === 'desc'}&#9660;{:else}&#8597;{/if}{#if sortRank(c.id)}<sub>{sortRank(c.id)}</sub>{/if}
										</span>
									</button>
								</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each rows as r (r.kind + r.dev.device_id)}
							<tr class:sub={r.kind === 'device'} class:dim={r.dim} class:sel={ui.drawerDevice === r.dev.device_id} onclick={() => open(r)}>
								<td class="chev">
									{#if r.kind === 'paper' && r.paper.n_devices > 1}
										<button
											class="cb"
											aria-label="Expand devices"
											aria-expanded={tableState.expanded.includes(r.paper.paper_id)}
											title="{r.paper.n_devices} devices"
											onclick={(e) => {
												e.stopPropagation();
												toggleExpand(r.paper.paper_id);
											}}
										>
											<svg viewBox="0 0 10 10" width="8" height="8" class:rot={tableState.expanded.includes(r.paper.paper_id)}><path d="M3 1 L7 5 L3 9" fill="none" stroke="currentColor" stroke-width="1.5" /></svg>
										</button>
									{/if}
								</td>
								{#each visible as c (c.id)}
									{@const x = cell(c.id, r)}
									<td class:num={c.num} class:muted={x.muted} class:ml={x.multiline} title={x.tip ?? ''}>
										{#if x.href}
											<a href={link('/' + x.href)} onclick={(e) => e.stopPropagation()}>{x.text}</a>
										{:else}
											<span class:ital={x.derived} class:clamp={x.multiline} class="v">{x.text}</span>{#if x.basis && BASIS_MARK[x.basis]}<sup class="bm">{BASIS_MARK[x.basis]}</sup>{/if}{#if r.kind === 'paper' && c.id === 'paper' && r.paper.n_devices > 1}<span class="nd">{r.paper.n_devices}</span>{/if}
										{/if}
									</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
				{#if !rows.length}<div class="msg">No rows match the filters.</div>{/if}
			{/if}
		</div>
		<Drawer />
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
		position: relative;
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
	}
	.tools {
		display: flex;
		gap: 6px;
		align-items: center;
		padding: 5px 8px;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
		flex: none;
		height: 36px;
	}
	.grow {
		flex: 1;
	}
	.legend {
		display: flex;
		gap: 10px;
		font-size: 10px;
		white-space: nowrap;
		overflow: hidden;
	}
	.chooser {
		position: relative;
	}
	.pop {
		position: absolute;
		top: 28px;
		left: 0;
		z-index: 30;
		background: var(--surface);
		border: 1px solid var(--line);
		padding: 4px;
		width: 230px;
		max-height: 60vh;
		overflow-y: auto;
		box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
	}
	.pop label {
		display: flex;
		gap: 6px;
		align-items: center;
		padding: 2px 4px;
		cursor: pointer;
	}
	.pop label:hover {
		background: var(--hover);
	}
	.scroller {
		flex: 1;
		min-height: 0;
		overflow: auto;
		background: var(--surface);
	}
	table {
		border-collapse: separate;
		border-spacing: 0;
		table-layout: fixed;
	}
	thead th {
		position: sticky;
		top: 0;
		z-index: 5;
		background: var(--surface-2);
		border-bottom: 1px solid var(--line);
		height: 34px;
		padding: 0;
		text-align: left;
		font-weight: 600;
		color: var(--ink-2);
	}
	thead th.num .sh {
		justify-content: flex-end;
	}
	.sh {
		width: 100%;
		height: 34px;
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 0 6px;
		background: transparent;
		border: 0;
		text-align: left;
		overflow: hidden;
		white-space: nowrap;
	}
	.sh:hover {
		background: var(--hover);
	}
	.lt {
		display: flex;
		flex-direction: column;
		line-height: 1.15;
		min-width: 0;
		flex: 1;
	}
	.lab {
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.unit {
		color: var(--ink-3);
		font-weight: 400;
		font-size: 10px;
		min-height: 11px;
	}
	thead th.num .lt {
		align-items: flex-end;
	}
	.arrow {
		color: var(--ink-3);
		font-size: 9px;
		opacity: 0.5;
	}
	.arrow.act {
		color: var(--accent);
		opacity: 1;
	}
	tbody td {
		padding: 2px 6px;
		height: 30px;
		border-bottom: 1px solid var(--grid);
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
		vertical-align: middle;
	}
	tbody td.num {
		text-align: right;
	}
	.clamp {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
		white-space: normal;
		line-height: 1.2;
	}
	tbody tr {
		cursor: pointer;
	}
	tbody tr:hover td {
		background: var(--hover);
	}
	tr.sel td {
		background: var(--sel);
	}
	tr.sub td {
		background: var(--surface-2);
		color: var(--ink-2);
	}
	tr.sub td:first-child {
		box-shadow: inset 2px 0 0 var(--accent);
	}
	tr.dim td {
		opacity: 0.5;
	}
	.chev {
		padding: 0 !important;
		text-align: center;
	}
	.cb {
		width: 22px;
		height: 22px;
		padding: 0;
		background: transparent;
		border: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
	}
	.cb svg {
		transition: transform 0.1s;
	}
	.cb svg.rot {
		transform: rotate(90deg);
	}
	.ital {
		font-style: italic;
	}
	.bm {
		color: var(--warn);
		font-size: 9px;
		margin-left: 1px;
	}
	.nd {
		margin-left: 6px;
		color: var(--ink-3);
		border: 1px solid var(--line);
		padding: 0 3px;
		font-size: 10px;
	}
	.msg {
		padding: 20px;
		color: var(--ink-3);
		display: flex;
		justify-content: center;
	}
</style>
