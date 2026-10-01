
<script lang="ts">
	import { link } from './paths';
	import { store, ui } from './state.svelte';
	import { BASIS_MARK, BASIS_TIP, EM_DASH, enumLabel, fmt, qualPrefix, qualWord } from './logic';
	import type { Device } from './types';

	const a = $derived(store.atlas);
	const idx = $derived(store.index);
	const dev = $derived(ui.drawerDevice && idx ? (idx.devById.get(ui.drawerDevice) ?? null) : null);
	const paper = $derived(dev && idx ? (idx.paperById.get(dev.paper_id) ?? null) : null);
	const siblings = $derived(paper && idx ? (idx.devsByPaper.get(paper.paper_id) ?? []) : []);
	let hideEmpty = $state(false);

	function close() {
		ui.drawerDevice = null;
	}

	function onkey(e: KeyboardEvent) {
		if (e.key === 'Escape' && ui.drawerDevice) close();
	}

	function valueText(d: Device, name: string, type: string, enumName: string): string {
		if (name === 'qualifiers') {
			const e = Object.entries(d.qualifiers).map(([k, o]) => `${k}:${o}`);
			return e.length ? e.join('; ') : EM_DASH;
		}
		const v = d[name];
		if (v === null || v === undefined || (Array.isArray(v) && !v.length) || v === '') return EM_DASH;
		if (Array.isArray(v)) return v.join('; ');
		if (type === 'enum' && a) return enumLabel(a, enumName, String(v));
		if (type === 'float' && typeof v === 'number') return `${qualPrefix(d.qualifiers[name])}${String(v)}`;
		return String(v);
	}

	const derivedRows = $derived.by(() => {
		if (!dev) return [];
		const out: { key: string; label: string; v: string; unit: string; formula: string; warn: string; q: string }[] = [];
		const labels: Record<string, string> = {
			vpil_dc_vcm_derived: 'Vpi*L (DC, derived)',
			il_rf_total_db: 'RF loss total (derived)',
			vpi_il_vdb: 'Vpi*IL (derived)',
			fom: 'Figure of merit (derived)'
		};
		for (const k of ['vpil_dc_vcm_derived', 'il_rf_total_db', 'vpi_il_vdb', 'fom'] as const) {
			const x = dev.derived[k];
			if (!x) continue;
			const q = x.qualifier ? qualPrefix(x.qualifier) : x.qualifiers?.length ? '~' : '';
			out.push({
				key: k,
				label: labels[k],
				v: `${q}${fmt(x.value)}`,
				unit: x.unit,
				formula: x.formula,
				warn: [x.warning, x.qualifiers?.length ? `inputs with qualifiers: ${x.qualifiers.join(', ')}` : ''].filter(Boolean).join('; '),
				q: x.qualifier ? qualWord(x.qualifier) : ''
			});
		}
		return out;
	});

	const simId = $derived(paper?.sim_ids[0] ?? null);
</script>

<svelte:window onkeydown={onkey} />

{#if dev && paper && a}
	<div class="drawer" role="dialog" aria-label="Device detail">
		<div class="dh">
			<div class="ttl" title={paper.title}>{paper.title}</div>
			<button onclick={close} aria-label="Close" title="Close (Esc)">
				<svg viewBox="0 0 10 10" width="10" height="10"><path d="M1 1 L9 9 M9 1 L1 9" stroke="currentColor" stroke-width="1.4" /></svg>
			</button>
		</div>
		<div class="body">
			<div class="meta">
				<div class="k">Authors</div>
				<div title={paper.authors.join('; ')}>{paper.authors.slice(0, 4).join('; ')}{paper.authors.length > 4 ? `; +${paper.authors.length - 4}` : ''}</div>
				<div class="k">Venue</div>
				<div>{paper.venue ?? EM_DASH} <span class="muted num">{paper.year}</span></div>
				<div class="k">Link</div>
				<div class="links">
					{#if paper.doi}<a href="https://doi.org/{paper.doi}" target="_blank" rel="noreferrer" title="Open DOI">DOI {paper.doi}</a>{/if}
					{#if paper.arxiv_id}<a href="https://arxiv.org/abs/{paper.arxiv_id}" target="_blank" rel="noreferrer">arXiv {paper.arxiv_id}</a>{/if}
					{#if !paper.doi && !paper.arxiv_id}<a href={paper.url} target="_blank" rel="noreferrer">{paper.url}</a>{/if}
				</div>
				<div class="k">License</div>
				<div title="License as recorded for the source">{paper.license ?? EM_DASH}</div>
				<div class="k">Redistribution</div>
				<div title="Whether text and figures of the source may be redistributed with the atlas">{enumLabel(a, 'redistribution', paper.redistribution)}; access: {enumLabel(a, 'access', paper.access)}</div>
				<div class="k">Organizations</div>
				<div>{paper.orgs_affil.map((o) => o.org_name).join('; ') || EM_DASH}</div>
				<div class="k">Foundry / fab</div>
				<div>{paper.orgs_fab.map((o) => o.org_name).join('; ') || EM_DASH}</div>
				<div class="k">Simulation</div>
				<div>
					{#if simId}<a href={link(`/sim?id=${encodeURIComponent(simId)}`)}>Open sim route ({simId})</a>{:else}<span class="muted">none</span>{/if}
				</div>
			</div>

			{#if siblings.length > 1}
				<div class="tabs" role="tablist">
					{#each siblings as s (s.device_id)}
						<button role="tab" class:on={s.device_id === dev.device_id} onclick={() => (ui.drawerDevice = s.device_id)} title={s.device_label}>
							{s.device_id.replace(paper.paper_id + '-', '')}
						</button>
					{/each}
				</div>
			{/if}
			<div class="dtitle" title={dev.device_id}>{dev.device_label}</div>

			{#if derivedRows.length}
				<div class="sec">Derived (not stored in the database)</div>
				<table>
					<tbody>
						{#each derivedRows as r (r.key)}
							<tr title="{r.formula}{r.warn ? '\n' + r.warn : ''}">
								<td class="f">{r.label}</td>
								<td class="v num"><i>{r.v}</i></td>
								<td class="u">{r.unit}</td>
								<td class="b">derived</td>
								<td class="e muted">{r.formula}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}

			<label class="he"><input type="checkbox" bind:checked={hideEmpty} /> Hide empty</label>

			{#each a.meta.column_groups as grp (grp.name)}
				{@const cols = grp.fields.map((f) => a.columns.find((c) => c.name === f)).filter((c) => !!c)}
				{@const rows = cols.filter((c) => !hideEmpty || valueText(dev, c.name, c.type, c.enum) !== EM_DASH)}
				{#if rows.length}
					<div class="sec">{grp.name}</div>
					<table>
						<tbody>
							{#each rows as c (c.name)}
								{@const ev = dev.evidence[c.name]}
								{@const vt = valueText(dev, c.name, c.type, c.enum)}
								{@const q = dev.qualifiers[c.name]}
								{@const long = vt.length > 26 && c.type === 'str'}
								<tr title={[c.desc, q ? qualWord(q) : '', ev?.note ?? ''].filter(Boolean).join('\n')}>
									<td class="f">{c.label}</td>
									{#if long}<td class="v" colspan="2"></td>{:else}<td class="v num" class:muted={vt === EM_DASH} class:ital={ev?.derived}>{vt}</td>
									<td class="u">{vt === EM_DASH ? '' : c.unit}</td>{/if}
									<td class="b" title={ev?.basis ? (BASIS_TIP[ev.basis] ?? ev.basis) : ''}>
										{#if ev?.basis}<span class="chip" class:sim={['simulated', 'predicted', 'design_target'].includes(ev.basis)}>{ev.basis === 'extracted_from_figure' ? 'figure' : ev.basis}</span>{/if}
									</td>
									<td class="e" title={ev?.note ?? ''}>{ev?.locator ?? ''}{#if ev?.derived && ev.formula}<span class="muted"> {ev.formula}</span>{/if}</td>
								</tr>
								{#if long}<tr class="lr"><td colspan="5" title={ev?.note ?? ''}><div class="lv">{vt}</div></td></tr>{/if}
							{/each}
						</tbody>
					</table>
				{/if}
			{/each}

			<div class="foot muted" title="Basis markers in tables">
				Basis markers: {Object.entries(BASIS_MARK).map(([k, m]) => `${m} ${BASIS_TIP[k]}`).join(', ')}
			</div>
		</div>
	</div>
{/if}

<style>
	.drawer {
		position: absolute;
		top: 0;
		right: 0;
		bottom: 0;
		width: min(560px, 60%);
		background: var(--surface);
		border-left: 1px solid var(--line);
		display: flex;
		flex-direction: column;
		z-index: 20;
		box-shadow: -8px 0 16px rgba(0, 0, 0, 0.25);
	}
	.dh {
		display: flex;
		gap: 8px;
		align-items: flex-start;
		padding: 8px 10px;
		border-bottom: 1px solid var(--border);
		flex: none;
	}
	.ttl {
		flex: 1;
		font-weight: 600;
		font-size: 13px;
		line-height: 1.3;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.dh button {
		width: 24px;
		padding: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex: none;
	}
	.body {
		overflow-y: auto;
		min-height: 0;
		padding: 8px 10px 16px;
	}
	.meta {
		display: grid;
		grid-template-columns: 96px 1fr;
		gap: 2px 8px;
		margin-bottom: 8px;
	}
	.k {
		color: var(--ink-3);
	}
	.links {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
	}
	.tabs {
		display: flex;
		gap: 2px;
		flex-wrap: wrap;
		margin: 6px 0;
	}
	.dtitle {
		font-weight: 600;
		margin: 6px 0 2px;
	}
	.sec {
		margin: 10px 0 2px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 10px;
		color: var(--ink-3);
		border-bottom: 1px solid var(--border);
	}
	table {
		width: 100%;
		border-collapse: collapse;
	}
	td {
		padding: 1px 4px;
		vertical-align: top;
		border-bottom: 1px solid var(--grid);
	}
	tr:hover td {
		background: var(--hover);
	}
	.f {
		width: 34%;
		color: var(--ink-2);
	}
	.v {
		width: 24%;
		word-break: break-word;
	}
	.u {
		width: 9%;
		color: var(--ink-3);
	}
	.b {
		width: 11%;
	}
	.e {
		width: 22%;
		color: var(--ink-2);
		font-size: 11px;
	}
	.ital {
		font-style: italic;
	}
	.chip {
		font-size: 10px;
		padding: 0 4px;
		border: 1px solid var(--line);
		color: var(--ink-2);
		white-space: nowrap;
	}
	.chip.sim {
		border-color: var(--warn);
		color: var(--warn);
	}
	.he {
		display: flex;
		gap: 6px;
		align-items: center;
		margin-top: 8px;
	}
	.lv {
		color: var(--ink-2);
		white-space: pre-wrap;
		max-height: 110px;
		overflow-y: auto;
		font-size: 11px;
	}
	.foot {
		margin-top: 10px;
		font-size: 10px;
	}
</style>
