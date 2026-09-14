<script lang="ts">
	import { base } from '$app/paths';
	import { loadBank, loadModeGallery } from '$lib/data';
	import { explainQuestion } from '$lib/explain';
	import { mismatchLossDB, formatLossDB } from '$lib/engine';
	import type { CachedQuestion, Manifest, ModeGalleryEntry, ModeGalleryManifest, PendingRecipe } from '$lib/types';
	import ModePairView from '$lib/components/ModePairView.svelte';
	import WaveguideModeCard from '$lib/components/WaveguideModeCard.svelte';
	import { onMount } from 'svelte';

	let all = $state<CachedQuestion[]>([]);
	let manifest = $state<Manifest | null>(null);
	let pending = $state<PendingRecipe[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let selectedCategory = $state<string | null>(null);
	let selected = $state<CachedQuestion | null>(null);
	let revealInModal = $state(false);
	let modeGallery = $state<ModeGalleryManifest | null>(null);
	let modeError = $state<string | null>(null);
	let selectedMode = $state<ModeGalleryEntry | null>(null);

	let categories = $derived.by(() => {
		const counts = new Map<string, number>();
		for (const question of all) counts.set(question.category, (counts.get(question.category) ?? 0) + 1);
		return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((a, b) => a.name.localeCompare(b.name));
	});

	let filtered = $derived(
		selectedCategory === null ? all : all.filter((question) => question.category === selectedCategory)
	);

	onMount(async () => {
		try {
			const bank = await loadBank((input) => fetch(input), base);
			manifest = bank.manifest;
			all = bank.questions;
			pending = bank.pending;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to load the shipped bank';
		} finally {
			loading = false;
		}
		try {
			modeGallery = await loadModeGallery((input) => fetch(input), base);
		} catch (cause) {
			modeError = cause instanceof Error ? cause.message : 'Failed to load the mode gallery';
		}
	});

	function open(question: CachedQuestion) {
		selected = question;
		revealInModal = false;
	}

	function close() {
		selected = null;
	}

	function handleKey(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			close();
			selectedMode = null;
		}
	}
</script>

<svelte:window onkeydown={handleKey} />

<div class="gallery">
	<header class="page-header">
		<h1>Gallery</h1>
		<p class="subtitle">Ready analytic mode pairs from the shipped bank; numerical guides stay locked until C3.</p>
	</header>

	{#if modeGallery && modeGallery.entries.length > 0}
		<section class="wg-section">
			<div class="wg-head">
				<h2>Waveguide modes — nominal dimensions</h2>
				<p class="note">{modeGallery.note}</p>
			</div>
			<div class="wg-grid">
				{#each modeGallery.entries as entry (entry.id)}
					<button class="wg-card" onclick={() => (selectedMode = entry)}>
						<div class="wg-title">
							<span class="mono">{entry.id}</span>
							<span class="badge">{entry.modeLabel}</span>
						</div>
						<WaveguideModeCard {entry} />
					</button>
				{/each}
			</div>
		</section>
	{:else if modeError}
		<p class="note">Mode gallery unavailable: {modeError}. Run <span class="mono">npm run cache:modes</span>.</p>
	{/if}

	{#if loading}
		<div class="state">Loading shipped bank…</div>
	{:else if error}
		<div class="state">
			<p class="text-error">{error}</p>
			<p class="hint">Run <span class="mono">npm run cache:build</span>, then reload.</p>
		</div>
	{:else}
		<div class="content">
			<aside class="sidebar">
				<h2>Categories</h2>
				<ul class="cat-list">
					<li>
						<button class="cat-btn" class:active={selectedCategory === null} onclick={() => (selectedCategory = null)}>
							<span class="cat-name">All analytic</span>
							<span class="cat-count">{all.length}</span>
						</button>
					</li>
					{#each categories as category}
						<li>
							<button
								class="cat-btn"
								class:active={selectedCategory === category.name}
								onclick={() => (selectedCategory = category.name)}
							>
								<span class="cat-name">{category.name}</span>
								<span class="cat-count">{category.count}</span>
							</button>
						</li>
					{/each}
				</ul>
				{#if pending.length}
					<div class="pending">
						<span class="pending-title">Locked recipes</span>
						<span class="pending-count">{pending.length} pending vector solver</span>
					</div>
				{/if}
			</aside>

			<section class="cards">
				<div class="sample-grid">
					{#each filtered as question (question.id)}
						<button class="card" onclick={() => open(question)}>
							<div class="card-head">
								<span class="card-title">{question.title}</span>
								<span class="card-cat">{question.category}</span>
							</div>
							<ModePairView {question} revealed={false} samples={65} compact />
						</button>
					{/each}
				</div>
			</section>
		</div>
	{/if}
</div>

{#if selected}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="modal-backdrop" onclick={(e) => e.target === e.currentTarget && close()}>
		<div class="modal">
			<div class="modal-head">
				<div>
					<span class="modal-title">{selected.title}</span>
					<span class="modal-cat">{selected.category}</span>
				</div>
				<button onclick={close} title="Close (Esc)">✕</button>
			</div>
			<div class="modal-body">
				<button onclick={() => (revealInModal = !revealInModal)}>
					{revealInModal ? 'Hide' : 'Reveal'} η and explanation
				</button>
				<ModePairView question={selected} revealed={revealInModal} />
				{#if revealInModal}
					<div class="answer">
						<span>η = {selected.answer.eta.toFixed(6)}</span>
						<span>
							mismatch loss = {selected.answer.eta === 0
								? '∞ (exact zero overlap)'
								: formatLossDB(mismatchLossDB(selected.answer.eta))}
						</span>
						<span>{explainQuestion(selected).sentence}</span>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

{#if selectedMode}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="modal-backdrop" onclick={(e) => e.target === e.currentTarget && (selectedMode = null)}>
		<div class="modal">
			<div class="modal-head">
				<div>
					<span class="modal-title">{selectedMode.id}</span>
					<span class="modal-cat">{selectedMode.materialLabel} / {selectedMode.cladding}</span>
				</div>
				<button onclick={() => (selectedMode = null)} title="Close (Esc)">✕</button>
			</div>
			<div class="modal-body">
				<p class="note">
					Scalar quasi-TE FEM at nominal dimensions. Experimental: not full-vector, SiN index is a
					placeholder, and neff is not a released value.
				</p>
				<WaveguideModeCard entry={selectedMode} large />
			</div>
		</div>
	</div>
{/if}

<style>
	.gallery {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.page-header h1 {
		margin-bottom: var(--spacing-xs);
	}

	.subtitle {
		color: var(--text-secondary);
	}

	.state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 240px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-secondary);
		gap: var(--spacing-sm);
	}

	.hint {
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.mono {
		font-family: var(--font-mono);
	}

	.content {
		display: grid;
		grid-template-columns: 240px 1fr;
		gap: var(--spacing-lg);
		align-items: start;
	}

	.sidebar {
		position: sticky;
		top: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		padding: var(--spacing-md);
	}

	.sidebar h2 {
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-sm);
	}

	.cat-list {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.cat-btn {
		width: 100%;
		display: flex;
		justify-content: space-between;
		padding: var(--spacing-xs) var(--spacing-sm);
		background: transparent;
		border: 1px solid transparent;
		color: var(--text-primary);
	}

	.cat-btn:hover {
		background-color: var(--bg-tertiary);
	}

	.cat-btn.active {
		border-color: var(--accent);
		background-color: var(--bg-tertiary);
	}

	.cat-name {
		font-size: 0.82rem;
	}

	.cat-count {
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--text-muted);
	}

	.pending {
		display: flex;
		flex-direction: column;
		margin-top: var(--spacing-md);
		padding-top: var(--spacing-sm);
		border-top: 1px solid var(--border);
	}

	.pending-title {
		font-size: 0.75rem;
		text-transform: uppercase;
		color: var(--warning);
	}

	.pending-count {
		font-size: 0.72rem;
		color: var(--text-muted);
	}

	.sample-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
		gap: var(--spacing-md);
	}

	.card {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		padding: var(--spacing-sm);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-primary);
		text-align: left;
		cursor: pointer;
	}

	.card:hover {
		border-color: var(--accent);
	}

	.card-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: var(--spacing-sm);
	}

	.card-title {
		font-size: 0.82rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.card-cat {
		font-family: var(--font-mono);
		font-size: 0.65rem;
		color: var(--text-muted);
	}

	.modal-backdrop {
		position: fixed;
		inset: 0;
		background-color: rgba(0, 0, 0, 0.85);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
		padding: var(--spacing-lg);
	}

	.modal {
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		max-width: 900px;
		max-height: 90vh;
		width: 100%;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.modal-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: var(--spacing-md);
		border-bottom: 1px solid var(--border);
	}

	.modal-title {
		font-weight: 600;
		margin-right: var(--spacing-sm);
	}

	.modal-cat {
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--text-muted);
	}

	.modal-body {
		padding: var(--spacing-md);
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: var(--spacing-md);
	}

	.answer {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		padding: var(--spacing-sm);
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
		font-family: var(--font-mono);
		font-size: 0.8rem;
	}

	.wg-section {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		padding-bottom: var(--spacing-lg);
		border-bottom: 1px solid var(--border);
	}

	.wg-head h2 {
		font-size: 1.1rem;
	}

	.note {
		font-size: 0.78rem;
		color: var(--text-muted);
		max-width: 760px;
	}

	.wg-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
		gap: var(--spacing-md);
	}

	.wg-card {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		padding: var(--spacing-sm);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-primary);
		text-align: left;
		cursor: pointer;
	}

	.wg-card:hover {
		border-color: var(--accent);
	}

	.wg-title {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.badge {
		font-family: var(--font-mono);
		font-size: 0.65rem;
		color: var(--accent);
		border: 1px solid var(--border);
		padding: 1px 6px;
	}

	@media (max-width: 800px) {
		.content {
			grid-template-columns: 1fr;
		}

		.wg-grid {
			grid-template-columns: 1fr;
		}

		.sidebar {
			position: static;
		}
	}
</style>
