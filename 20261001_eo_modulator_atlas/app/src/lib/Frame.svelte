<script lang="ts">
	import type { Snippet } from 'svelte';
	let {
		letter,
		desc,
		badge = '',
		badgeTip = '',
		height = 400,
		controls,
		children
	}: { letter: string; desc: string; badge?: string; badgeTip?: string; height?: number; controls?: Snippet; children: Snippet } = $props();
</script>

<section class="panel" style="height:{height}px">
	<header>
		<span class="letter" title={desc}>{letter}</span>
		{#if badge}<span class="badge num" title={badgeTip}>{badge}</span>{/if}
		<div class="grow"></div>
		{@render controls?.()}
	</header>
	<div class="body">{@render children()}</div>
</section>

<style>
	.panel {
		background: var(--surface);
		border: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	header {
		display: flex;
		gap: 6px;
		align-items: center;
		height: 28px;
		padding: 0 6px;
		border-bottom: 1px solid var(--border);
		flex: none;
	}
	.letter {
		font-weight: 700;
		color: var(--ink-2);
		width: 16px;
		text-align: center;
		border: 1px solid var(--line);
	}
	.badge {
		color: var(--ink-3);
		font-size: 10px;
	}
	.grow {
		flex: 1;
	}
	.body {
		position: relative;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}
	:global(.seg) {
		display: inline-flex;
	}
	:global(.seg button) {
		height: 20px;
		font-size: 10px;
		padding: 0 6px;
	}
</style>
