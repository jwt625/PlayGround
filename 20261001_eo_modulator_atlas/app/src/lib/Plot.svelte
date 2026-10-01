<script lang="ts">
	import { onMount } from 'svelte';

	interface Props {
		data: unknown[];
		layout: Record<string, unknown>;
		onhover?: (e: any) => void;
		onunhover?: (e: any) => void;
		onclick?: (e: any) => void;
		onafterplot?: (gd: any) => void;
		gd?: any;
	}
	let { data, layout, onhover, onunhover, onclick, onafterplot, gd = $bindable(null) }: Props = $props();

	let el: HTMLDivElement;
	let P: any = $state(null);
	let bound = false;

	onMount(() => {
		let alive = true;
		import('plotly.js-dist-min').then((m) => {
			if (alive) P = (m as any).default ?? m;
		});
		const ro = new ResizeObserver(() => {
			if (P && el && el.clientWidth > 0) P.Plots.resize(el);
		});
		ro.observe(el);
		return () => {
			alive = false;
			ro.disconnect();
			if (P) P.purge(el);
		};
	});

	$effect(() => {
		if (!P) return;
		const d = data;
		const l = layout;
		P.react(el, d, l, {
			displaylogo: false,
			responsive: false,
			scrollZoom: false,
			doubleClick: 'reset',
			modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
			toImageButtonOptions: { format: 'png', scale: 2 }
		}).then(() => {
			gd = el;
			if (!bound) {
				bound = true;
				(el as any).on('plotly_hover', (e: any) => onhover?.(e));
				(el as any).on('plotly_unhover', (e: any) => onunhover?.(e));
				(el as any).on('plotly_click', (e: any) => onclick?.(e));
				(el as any).on('plotly_afterplot', () => onafterplot?.(el));
			}
			onafterplot?.(el);
		});
	});
</script>

<div class="plot" bind:this={el}></div>

<style>
	.plot {
		width: 100%;
		height: 100%;
	}
</style>
