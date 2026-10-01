<script lang="ts">
	import { link } from '../lib/paths';
	import { store } from '../lib/state.svelte';
	import Bars from '../lib/Bars.svelte';
	import { enumLabel } from '../lib/logic';

	const a = $derived(store.atlas);
	const i = $derived(a?.integrity);

	function labelMap(en: string, keys: string[]): Record<string, string> {
		const m: Record<string, string> = {};
		if (!a) return m;
		for (const k of keys) m[k] = k === 'unspecified' ? 'Unspecified' : enumLabel(a, en, k);
		return m;
	}

	const tiles = $derived(
		i
			? [
					['Papers', i.counts.papers, 'Publications in the database'],
					['Devices', i.counts.devices, 'Device / operating-point rows'],
					['Organizations', i.counts.organizations, 'Distinct organizations named on papers'],
					['Countries', i.counts.countries, 'Countries derived from affiliations'],
					['Sim configs', i.counts.sim_configs, 'Simulation configs under sims/'],
					['With FOM', i.counts.devices_with_fom, 'Devices with bandwidth, Vpi and on-chip loss reported']
				]
			: []
	);
</script>

<div class="page">
	{#if store.error}
		<div class="msg">Failed to load data: {store.error}</div>
	{:else}
		<div class="tiles">
			{#each tiles as [l, n, t] (l)}
				<div class="tile" title={t as string}>
					<div class="tv num">{n}</div>
					<div class="tl">{l}</div>
				</div>
			{:else}
				{#each [0, 1, 2, 3, 4, 5] as k (k)}<div class="tile"><div class="tv">&nbsp;</div><div class="tl">&nbsp;</div></div>{/each}
			{/each}
		</div>
		<div class="grid">
			<section>
				<h3 title="Devices per waveguide platform">Platform</h3>
				{#if i}<Bars items={i.per_platform} labels={labelMap('waveguide_platform', Object.keys(i.per_platform))} />{/if}
			</section>
			<section>
				<h3 title="Devices per EO material">EO material</h3>
				{#if i}<Bars items={i.per_material} labels={labelMap('eo_material', Object.keys(i.per_material))} />{/if}
			</section>
			<section>
				<h3 title="Devices per class">Device class</h3>
				{#if i}<Bars items={i.per_device_class} labels={labelMap('device_class', Object.keys(i.per_device_class))} />{/if}
			</section>
			<section>
				<h3 title="Evidence entries per basis (every non-empty value carries a basis and a locator)">Basis</h3>
				{#if i}<Bars items={i.per_basis} labels={labelMap('basis', Object.keys(i.per_basis))} />{/if}
			</section>
			<section>
				<h3 title="Papers per source type">Source type</h3>
				{#if i}<Bars items={i.per_source_type} labels={labelMap('source_type', Object.keys(i.per_source_type))} />{/if}
			</section>
			<section>
				<h3 title="Papers per reproducibility grade">Repro grade</h3>
				{#if i}<Bars items={i.per_repro_grade} />{/if}
			</section>
		</div>
		<div class="links">
			<a href={link('/table')}>Table</a>
			<a href={link('/explore')}>Explore</a>
			<a href={link('/sim')}>Sim</a>
			<a href={link('/about')}>About</a>
		</div>
	{/if}
</div>

<style>
	.page {
		height: 100%;
		overflow-y: auto;
		padding: 10px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(6, minmax(0, 1fr));
		gap: 8px;
	}
	.tile {
		background: var(--surface);
		border: 1px solid var(--border);
		padding: 8px 10px;
		height: 62px;
	}
	.tv {
		font-size: 22px;
		font-weight: 600;
		line-height: 1.1;
	}
	.tl {
		color: var(--ink-3);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-size: 10px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 8px;
	}
	section {
		background: var(--surface);
		border: 1px solid var(--border);
		padding: 8px 10px;
		min-height: 120px;
	}
	h3 {
		margin: 0 0 6px;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--ink-3);
		font-weight: 600;
	}
	.links {
		display: flex;
		gap: 14px;
	}
	.msg {
		padding: 20px;
		color: var(--ink-3);
	}
	@media (max-width: 900px) {
		.tiles {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
