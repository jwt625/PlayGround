<script lang="ts">
	import Frame from './Frame.svelte';
	import Plot from './Plot.svelte';
	import { ui, store } from './state.svelte';
	import { plotTheme } from './colors';
	import { CORE_LABELS, completenessMatrix } from './charts';
	import type { View } from './logic';

	let { view }: { view: View } = $props();
	const th = $derived(plotTheme(ui.theme));
	const core = $derived(store.atlas?.meta.core_fields.map((c) => c.name) ?? []);
	const m = $derived(completenessMatrix(view, core));
	const height = $derived(Math.max(300, m.rows.length * 17 + 70));
	const traces = $derived([
		{
			type: 'heatmap',
			z: m.z,
			x: core.map((c) => CORE_LABELS[c] ?? c),
			y: m.rows.map((r) => r.paper.paper_id),
			text: m.rows.map((r, i) => core.map((c, j) => `${r.paper.title}<br>${r.paper.label} / ${r.dev.device_label}<br>${CORE_LABELS[c] ?? c}: ${m.z[i][j] ? 'reported' : 'not reported'}`)),
			hovertemplate: '%{text}<extra></extra>',
			colorscale: [[0, th.grid], [1, th.accent]],
			zmin: 0,
			zmax: 1,
			showscale: false,
			xgap: 2,
			ygap: 2
		}
	]);
	const layout = $derived({
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		margin: { l: 8, r: 12, t: 4, b: 4 },
		font: { family: 'system-ui, sans-serif', size: 11, color: th.ink2 },
		hoverlabel: { bgcolor: th.surface, bordercolor: th.line, font: { color: th.ink, size: 11 }, align: 'left' },
		xaxis: { side: 'top', tickfont: { size: 10, color: th.ink }, automargin: true, fixedrange: true, linecolor: th.line },
		yaxis: { type: 'category', autorange: 'reversed', tickmode: 'array', tickvals: m.rows.map((r) => r.paper.paper_id), ticktext: m.rows.map((r) => r.paper.label), tickfont: { size: 10, color: th.ink }, automargin: true, fixedrange: true }
	});
</script>

<Frame letter="i" desc="Disclosure completeness of each paper's representative device: which of the 7 core fields are reported" height={400} badge={`${m.rows.length}`} badgeTip="papers shown">
	{#snippet controls()}
		<span class="key"><i class="sw on"></i> reported <i class="sw"></i> not reported</span>
	{/snippet}
	<div class="sc"><div style="height:{height}px"><Plot data={traces} {layout} /></div></div>
</Frame>

<style>
	.sc {
		height: 100%;
		overflow-y: auto;
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		color: var(--ink-3);
		font-size: 10px;
	}
	.sw {
		display: inline-block;
		width: 10px;
		height: 10px;
		background: var(--grid);
		border: 1px solid var(--line);
	}
	.sw.on {
		background: var(--accent);
		border-color: var(--accent);
	}
</style>
