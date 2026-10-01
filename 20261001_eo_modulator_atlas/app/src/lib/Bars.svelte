<script lang="ts">
	import { EM_DASH } from './logic';
	let { items, labels = {}, title = '' }: { items: Record<string, number>; labels?: Record<string, string>; title?: string } = $props();
	const max = $derived(Math.max(1, ...Object.values(items)));
	const entries = $derived(Object.entries(items));
</script>

<div class="bars" {title}>
	{#each entries as [k, n] (k)}
		<div class="r" title="{labels[k] ?? k}: {n}">
			<span class="l">{labels[k] ?? k}</span>
			<span class="t"><span class="f" style="width:{(100 * n) / max}%"></span></span>
			<span class="n num">{n}</span>
		</div>
	{/each}
	{#if !entries.length}<div class="muted">{EM_DASH}</div>{/if}
</div>

<style>
	.bars {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.r {
		display: grid;
		grid-template-columns: 150px minmax(0, 1fr) 28px;
		align-items: center;
		gap: 6px;
		height: 16px;
	}
	.l {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--ink-2);
	}
	.t {
		height: 8px;
		background: var(--grid);
		display: block;
	}
	.f {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	.n {
		text-align: right;
	}
</style>
