<script lang="ts">
	import Frame from './Frame.svelte';
	import Plot from './Plot.svelte';
	import { ui } from './state.svelte';
	import { plotTheme } from './colors';
	import { topCounts } from './charts';

	let {
		letter,
		desc,
		counts,
		axisTitle = 'Papers (count)',
		toggles
	}: { letter: string; desc: string; counts: Map<string, number>; axisTitle?: string; toggles?: import('svelte').Snippet } = $props();

	let topN = $state<number | null>(10);
	const th = $derived(plotTheme(ui.theme));
	const rows = $derived(topCounts(counts, topN));
	const height = $derived(Math.max(300, rows.length * 18 + 56));
	const traces = $derived([
		{
			type: 'bar',
			orientation: 'h',
			y: rows.map((r) => r[0]),
			x: rows.map((r) => r[1]),
			text: rows.map((r) => String(r[1])),
			textposition: 'outside',
			cliponaxis: false,
			textfont: { color: th.ink2, size: 10 },
			width: 0.6,
			marker: { color: th.accent, line: { color: th.surface, width: 0 } },
			hovertemplate: '%{y}<br>%{x} papers<extra></extra>'
		}
	]);
	const layout = $derived({
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		margin: { l: 8, r: 28, t: 6, b: 8 },
		font: { family: 'system-ui, sans-serif', size: 11, color: th.ink2 },
		showlegend: false,
		hoverlabel: { bgcolor: th.surface, bordercolor: th.line, font: { color: th.ink, size: 11 } },
		xaxis: { title: { text: axisTitle, standoff: 4 }, gridcolor: th.grid, linecolor: th.line, tickfont: { size: 10 }, rangemode: 'tozero', tickformat: 'd', automargin: true, zeroline: false },
		yaxis: { autorange: 'reversed', automargin: true, tickfont: { size: 10, color: th.ink }, linecolor: th.line, type: 'category' }
	});
</script>

<Frame {letter} {desc} badge={`${counts.size}`} badgeTip="distinct entries in the current filter">
	{#snippet controls()}
		{@render toggles?.()}
		<span class="seg" title="Number of bars">
			<button class:on={topN === 10} onclick={() => (topN = 10)}>top 10</button><button class:on={topN === 25} onclick={() => (topN = 25)}>top 25</button><button class:on={topN === null} onclick={() => (topN = null)}>all</button>
		</span>
	{/snippet}
	<div class="sc">
		<div style="height:{height}px"><Plot data={traces} {layout} /></div>
	</div>
</Frame>

<style>
	.sc {
		height: 100%;
		overflow-y: auto;
	}
</style>
