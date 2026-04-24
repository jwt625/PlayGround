<script lang="ts">
	import { resolve } from '$app/paths';
	import { importPeopleCsv } from '$lib/api';

	let file = $state<File | null>(null);
	let uploading = $state(false);
	let error = $state('');
	let result = $state<{ created: number; skipped: number; errors: string[] } | null>(null);

	async function submitImport() {
		if (!file) return;
		uploading = true;
		error = '';
		try {
			result = await importPeopleCsv(file);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Import failed.';
		} finally {
			uploading = false;
		}
	}
</script>

<svelte:head>
	<title>Imports | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Imports</h1>
		</div>
		<p class="meta">People CSV</p>
	</header>

	<section class="panel">
		<div class="panel-header">
			<h2>Upload file</h2>
			<span>Dedupes by display name and email</span>
		</div>
		<form onsubmit={(event) => {
			event.preventDefault();
			submitImport();
		}}>
			<label class="wide">
				<span>CSV file</span>
				<input accept=".csv,text/csv" onchange={(event) => (file = (event.currentTarget as HTMLInputElement).files?.[0] || null)} type="file" />
			</label>
			<button type="submit" disabled={!file || uploading}>{uploading ? 'Importing…' : 'Import people'}</button>
		</form>
	</section>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	{#if result}
		<section class="panel result">
			<div class="panel-header">
				<h2>Result</h2>
				<span>{result.created} created</span>
			</div>
			<div class="summary">
				<p>Created: {result.created}</p>
				<p>Skipped: {result.skipped}</p>
			</div>
			{#if result.errors.length}
				<ul>
					{#each result.errors as message (message)}
						<li>{message}</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}
</main>

<style>
	.shell { min-height: 100vh; padding: 1.25rem; }
	.topbar { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--line-strong); }
	.brand, .meta, .panel-header span, label span { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); text-decoration: none; }
	h1 { margin: 0.35rem 0 0; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -0.05em; }
	h2 { margin: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; }
	.panel, .notice { margin-top: 1rem; border: 1px solid var(--line-strong); background: var(--panel-strong); box-shadow: var(--shadow); }
	.panel-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0.85rem; border-bottom: 1px solid var(--line); }
	form { display: grid; gap: 1rem; padding: 0.85rem; }
	label { display: grid; gap: 0.35rem; }
	input, button { width: 100%; border: 1px solid var(--line); background: var(--panel); padding: 0.62rem 0.72rem; color: var(--text); }
	button { cursor: pointer; }
	.summary, ul { padding: 0.85rem; margin: 0; }
	ul { padding-top: 0; }
</style>
