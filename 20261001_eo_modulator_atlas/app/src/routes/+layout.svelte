<script lang="ts">
	import '../app.css';
	import { onMount, type Snippet } from 'svelte';
	import { link } from '../lib/paths';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { filters, loadAtlas, store, tableState, ui } from '../lib/state.svelte';
	import { fromHash, toHash } from '../lib/logic';
	import ThemeToggle from '../lib/ThemeToggle.svelte';

	let { children }: { children: Snippet } = $props();
	let ready = $state(false);

	const nav = [
		{ href: '/', label: 'Dashboard' },
		{ href: '/table', label: 'Table' },
		{ href: '/explore', label: 'Explore' },
		{ href: '/sim', label: 'Sim' },
		{ href: '/about', label: 'About' }
	];

	onMount(() => {
		const t = document.documentElement.getAttribute('data-theme');
		ui.theme = t === 'light' ? 'light' : 'dark';
		const { filters: f, extras } = fromHash(location.hash);
		Object.assign(filters, f);
		if (extras.sort.length) tableState.sort = extras.sort;
		ready = true;
		loadAtlas();
	});

	$effect(() => {
		if (!ready) return;
		const h = toHash(filters, { sort: tableState.sort });
		void page.url.pathname;
		const want = h ? `#${h}` : '';
		if (location.hash !== want) {
			try {
				void goto(location.pathname + location.search + want, { replace: true, shallow: true, reset: false });
			} catch {
				/* router not ready */
			}
		}
	});

	const isActive = (href: string) => {
		const root = link('/').replace(/\/$/, '');
		const p = page.url.pathname.replace(root, '') || '/';
		return href === '/' ? p === '/' : p.startsWith(href);
	};
</script>

<div class="app">
	<header>
		<a class="brand" href={link('/')} title="EO modulator atlas">EO Modulator Atlas</a>
		<nav>
			{#each nav as n}
				<a href={link(n.href)} class:active={isActive(n.href)}>{n.label}</a>
			{/each}
		</nav>
		<div class="grow"></div>
		{#if store.atlas}
			<span class="muted num" title="Papers / devices in the loaded database">
				{store.atlas.integrity.counts.papers} papers, {store.atlas.integrity.counts.devices} devices
			</span>
		{:else if store.loading}
			<span class="spinner"></span>
		{/if}
		<ThemeToggle />
	</header>
	<main>
		{@render children()}
	</main>
</div>

<style>
	.app {
		height: 100vh;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		grid-template-rows: var(--hdr) minmax(0, 1fr);
	}
	header {
		min-width: 0;
		display: flex;
		align-items: center;
		gap: 14px;
		padding: 0 10px;
		background: var(--surface);
		border-bottom: 1px solid var(--border);
	}
	.brand {
		color: var(--ink);
		font-weight: 600;
		letter-spacing: 0.02em;
	}
	.brand:hover {
		text-decoration: none;
	}
	nav {
		display: flex;
		gap: 2px;
		height: 100%;
	}
	nav a {
		display: flex;
		align-items: center;
		padding: 0 10px;
		color: var(--ink-2);
		border-bottom: 2px solid transparent;
	}
	nav a:hover {
		background: var(--hover);
		text-decoration: none;
	}
	nav a.active {
		color: var(--ink);
		border-bottom-color: var(--accent);
	}
	.grow {
		flex: 1;
	}
	main {
		min-height: 0;
		min-width: 0;
		overflow: hidden;
	}
	@media (max-width: 700px) {
		.app { grid-template-rows: auto minmax(0, 1fr); }
		header { flex-wrap: wrap; gap: 6px 10px; padding: 6px 10px; }
		.brand { white-space: nowrap; }
		header > .muted { display: none; }
		nav { order: 3; width: 100%; height: 30px; overflow-x: auto; }
		nav a { flex-shrink: 0; padding: 0 8px; }
	}
</style>
