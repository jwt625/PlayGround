<script lang="ts">
	import Frame from './Frame.svelte';
	import Plot from './Plot.svelte';
	import { ui } from './state.svelte';
	import { groupColor, plotTheme } from './colors';
	import { papersPerYearByGroup } from './charts';
	import type { View } from './logic';
	import { store } from './state.svelte';

	let { view }: { view: View } = $props();
	const th = $derived(plotTheme(ui.theme));
	const data = $derived(papersPerYearByGroup(view));
	const label = (g: string) => (g === 'other_group' ? 'Other EO materials' : (store.atlas?.enums.eo_material.find((e) => e.value === g)?.label ?? g));

	const traces = $derived(
		data.groups.map((g) => ({
			type: 'bar',
			name: label(g),
			x: data.years,
			y: data.years.map((y) => data.counts.get(g)?.get(y) ?? 0),
			marker: { color: groupColor(g, ui.theme), line: { color: th.surface, width: 2 } },
			hovertemplate: `${label(g)}<br>%{x}: %{y} papers<extra></extra>`,
			textposition: 'none'
		}))
	);
	const layout = $derived({
		barmode: 'stack',
		bargap: 0.35,
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		margin: { l: 44, r: 12, t: 8, b: 8 },
		font: { family: 'system-ui, sans-serif', size: 11, color: th.ink2 },
		showlegend: false,
		hoverlabel: { bgcolor: th.surface, bordercolor: th.line, font: { color: th.ink, size: 11 } },
		xaxis: { title: { text: 'Publication year', standoff: 4 }, dtick: 1, tickformat: 'd', gridcolor: th.grid, linecolor: th.line, tickfont: { size: 10 }, automargin: true, zeroline: false },
		yaxis: { title: { text: 'Papers (count)', standoff: 4 }, gridcolor: th.grid, linecolor: th.line, tickfont: { size: 10 }, rangemode: 'tozero', dtick: 1, tickformat: 'd', automargin: true, zeroline: false }
	});
	const total = $derived(view.papers.length);
</script>

<Frame letter="g" desc="Papers per publication year, stacked by EO material of each paper's representative device" badge={`${total}`} badgeTip="papers in the current filter">
	<Plot data={traces} {layout} />
</Frame>
