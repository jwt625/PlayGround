<script lang="ts">
	import { resolve } from '$app/paths';
	import { searchAll, type SearchResponse, type SearchResult } from '$lib/api';

	let query = $state('');
	let loading = $state(false);
	let error = $state('');
	let results = $state<SearchResponse>({
		people: [],
		organizations: [],
		events: [],
		reminders: []
	});
	const sections = $derived(
		[
			{ label: 'People', items: results.people },
			{ label: 'Organizations', items: results.organizations },
			{ label: 'Events', items: results.events },
			{ label: 'Reminders', items: results.reminders }
		] satisfies Array<{ label: string; items: SearchResult[] }>
	);

	async function runSearch() {
		if (!query.trim()) return;
		loading = true;
		error = '';
		try {
			results = await searchAll(query, 20);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Search failed.';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Search | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Search</h1>
		</div>
		<p class="meta">People, orgs, events, reminders</p>
	</header>

	<section class="toolbar">
		<label>
			<span>Query</span>
			<input bind:value={query} placeholder="City, industry, person, note fragment" />
		</label>
		<button type="button" onclick={runSearch}>{loading ? 'Searching…' : 'Search'}</button>
	</section>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	<section class="grid">
		{#each sections as section (section.label)}
			<section class="panel">
				<div class="panel-header">
					<h2>{section.label}</h2>
					<span>{section.items.length}</span>
				</div>
				<ul>
					{#if section.items.length}
						{#each section.items as item (item.id)}
							<li>
								<strong>{item.title}</strong>
								<p>{item.subtitle || '—'}</p>
							</li>
						{/each}
					{:else}
						<li class="empty">No matches.</li>
					{/if}
				</ul>
			</section>
		{/each}
	</section>
</main>

<style>
	.shell {
		min-height: 100vh;
		padding: 1.25rem;
	}

	.topbar,
	.toolbar,
	.grid {
		display: grid;
		gap: 1rem;
	}

	.topbar {
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--line-strong);
	}

	.toolbar {
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		margin-top: 1rem;
	}

	.grid {
		grid-template-columns: repeat(2, minmax(0, 1fr));
		margin-top: 1rem;
	}

	.brand,
	.meta,
	.panel-header span,
	label span {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.12em;
		color: var(--muted);
		text-decoration: none;
	}

	h1 {
		margin: 0.35rem 0 0;
		font-size: clamp(2rem, 4vw, 3rem);
		letter-spacing: -0.05em;
	}

	h2 {
		margin: 0;
		font-size: 0.9rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	label {
		display: grid;
		gap: 0.35rem;
	}

	input,
	button {
		width: 100%;
		border: 1px solid var(--line);
		background: var(--panel);
		padding: 0.62rem 0.72rem;
		color: var(--text);
	}

	.panel,
	.notice {
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
		box-shadow: var(--shadow);
	}

	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 0.85rem;
		border-bottom: 1px solid var(--line);
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0.25rem 0.85rem 0.85rem;
	}

	li {
		padding: 0.65rem 0;
		border-bottom: 1px solid var(--line);
	}

	li:last-child {
		border-bottom: 0;
	}

	li strong,
	li p {
		display: block;
		margin: 0;
	}

	li p {
		margin-top: 0.2rem;
		color: var(--muted);
	}

	.notice {
		margin-top: 1rem;
		padding: 0.85rem;
	}

	.empty {
		color: var(--muted);
	}

	@media (max-width: 960px) {
		.toolbar,
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
