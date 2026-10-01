<script lang="ts">
	import { store } from './state.svelte';
	import { metricCell } from './columns';
	import { BASIS_MARK, BASIS_TIP, EM_DASH, enumLabel } from './logic';

	let { id }: { id: string } = $props();
	const a = $derived(store.atlas);
	const dev = $derived(store.index?.devById.get(id) ?? null);
	const paper = $derived(dev ? (store.index?.paperById.get(dev.paper_id) ?? null) : null);

	const keys: [string, string, string][] = [
		['vpil', 'Vpi*L', 'V*cm'],
		['vpi', 'Vpi', 'V'],
		['bw3db', '3 dB BW', 'GHz'],
		['il_onchip', 'IL on-chip', 'dB'],
		['il_f2f', 'IL fiber-fiber', 'dB'],
		['length', 'Length', 'mm'],
		['rf_loss', 'RF loss', 'dB/cm'],
		['rate', 'Max rate', 'Gb/s'],
		['vpi_il', 'Vpi*IL', 'V*dB'],
		['fom', 'FOM', 'GHz/V']
	];
	const rows = $derived(
		a && dev
			? keys
					.map(([k, l, u]) => ({ l, u, c: metricCell(a, k, dev) }))
					.filter((r) => r.c.text !== EM_DASH)
			: []
	);
	const basisText = $derived.by(() => {
		if (!dev) return '';
		const hb = dev.headline_basis;
		const set = new Set(Object.values(hb).filter(Boolean) as string[]);
		return [...set].map((b) => BASIS_TIP[b] ?? b).join(', ');
	});
</script>

{#if dev && paper && a}
	<div class="tip">
		<div class="tt">{paper.title}</div>
		<div class="sub">{paper.label} / {dev.device_label}</div>
		<div class="sub">{enumLabel(a, 'eo_material', dev.eo_material)}; {enumLabel(a, 'device_class', dev.device_class)}; {dev.is_sim ? 'simulated / predicted' : 'measured'}</div>
		<div class="org">{paper.orgs_affil.map((o) => o.org_name).join('; ') || EM_DASH}</div>
		<table>
			<tbody>
				{#each rows as r (r.l)}
					<tr>
						<td class="l">{r.l}</td>
						<td class="v num"><span class:ital={r.c.derived}>{r.c.text}</span>{#if r.c.basis && BASIS_MARK[r.c.basis]}<sup>{BASIS_MARK[r.c.basis]}</sup>{/if}</td>
						<td class="u">{r.u}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<div class="sub">basis: {basisText || EM_DASH}</div>
	</div>
{/if}

<style>
	.tip {
		background: var(--surface);
		border: 1px solid var(--line);
		padding: 6px 8px;
		width: 300px;
		box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
		font-size: 11px;
		pointer-events: none;
	}
	.tt {
		font-weight: 600;
		line-height: 1.25;
		margin-bottom: 2px;
	}
	.sub {
		color: var(--ink-2);
	}
	.org {
		color: var(--ink-3);
		margin-bottom: 3px;
	}
	table {
		border-collapse: collapse;
		margin: 3px 0;
	}
	td {
		padding: 0 6px 0 0;
	}
	.v {
		color: var(--ink);
		font-weight: 600;
		text-align: right;
		min-width: 56px;
	}
	.l,
	.u {
		color: var(--ink-3);
	}
	.ital {
		font-style: italic;
	}
	sup {
		color: var(--warn);
		font-size: 9px;
	}
</style>
