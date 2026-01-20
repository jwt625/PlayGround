<script lang="ts">
	import { base } from '$app/paths';
	import type { Sample, SampleManifest } from '$lib/types';
	import SampleCard from '$lib/components/SampleCard.svelte';
	import SampleModal from '$lib/components/SampleModal.svelte';
	import { onMount } from 'svelte';

	// Level metadata
	const levelMeta = [
		{ id: 1, name: 'Foundations' },
		{ id: 2, name: 'Periodic Structures' },
		{ id: 3, name: 'Spot Arrays' },
		{ id: 4, name: 'Special Beams' },
		{ id: 5, name: 'Compound Patterns' },
		{ id: 6, name: 'Practical Applications' },
		{ id: 7, name: 'Shapes and Objects' }
	];

	let allSamples = $state<Sample[]>([]);
	let selectedLevel = $state<number | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let modalSample = $state<Sample | null>(null);

	// Compute level counts from loaded samples
	let levels = $derived(
		levelMeta.map(meta => ({
			...meta,
			count: allSamples.filter(s => s.level === meta.id).length
		}))
	);

	// Filter samples for selected level
	let filteredSamples = $derived(
		selectedLevel === null
			? []
			: allSamples.filter(s => s.level === selectedLevel)
	);

	onMount(async () => {
		try {
			const res = await fetch(`${base}/samples.json`);
			if (!res.ok) throw new Error(`Failed to load samples: ${res.status}`);
			const manifest: SampleManifest = await res.json();
			allSamples = manifest.samples;
			// Auto-select first level with samples
			const firstWithSamples = levels.find(l =>
				allSamples.some(s => s.level === l.id)
			);
			if (firstWithSamples) {
				selectedLevel = firstWithSamples.id;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	});

	function selectLevel(level: number) {
		selectedLevel = level;
	}

	function resolvePath(path: string): string {
		return `${base}/${path}`;
	}

	function openModal(sample: Sample) {
		modalSample = sample;
	}

	function closeModal() {
		modalSample = null;
	}
</script>

<div class="gallery">
	<header class="page-header">
		<h1>Gallery</h1>
		<p class="subtitle">Browse phase-intensity pairs by difficulty level</p>
	</header>

	<div class="content">
		<aside class="sidebar">
			<h2>Levels</h2>
			<ul class="level-list">
				{#each levels as level}
					<li>
						<button
							class="level-btn"
							class:active={selectedLevel === level.id}
							onclick={() => selectLevel(level.id)}
						>
							<span class="level-num">L{level.id}</span>
							<span class="level-name">{level.name}</span>
							<span class="level-count">{level.count}</span>
						</button>
					</li>
				{/each}
			</ul>
		</aside>

		<section class="samples">
			{#if loading}
				<div class="empty-state">
					<p>Loading samples...</p>
				</div>
			{:else if error}
				<div class="empty-state">
					<p class="text-error">{error}</p>
					<p class="hint">Make sure samples.json exists in static/</p>
				</div>
			{:else if selectedLevel === null}
				<div class="empty-state">
					<p>Select a level to view samples</p>
				</div>
			{:else if filteredSamples.length === 0}
				<div class="empty-state">
					<p>No samples generated yet for Level {selectedLevel}</p>
					<p class="hint">Run the Python generator to create training samples</p>
				</div>
			{:else}
				<div class="sample-grid">
					{#each filteredSamples as sample (sample.id)}
						<SampleCard
							sample={{
								...sample,
								phase_gif: resolvePath(sample.phase_gif),
								intensity_gif: resolvePath(sample.intensity_gif)
							}}
							onOpenModal={(s) => openModal({
								...s,
								phase_gif: resolvePath(sample.phase_gif),
								intensity_gif: resolvePath(sample.intensity_gif)
							})}
						/>
					{/each}
				</div>
			{/if}
		</section>
	</div>
</div>

<SampleModal sample={modalSample} onClose={closeModal} />

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

	.content {
		display: grid;
		grid-template-columns: 260px 1fr;
		gap: var(--spacing-lg);
	}

	.sidebar {
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		padding: var(--spacing-md);
	}

	.sidebar h2 {
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-md);
	}

	.level-list {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.level-btn {
		width: 100%;
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
		padding: var(--spacing-sm);
		background: transparent;
		border: 1px solid transparent;
		text-align: left;
		cursor: pointer;
		color: var(--text-primary);
	}

	.level-btn:hover {
		background-color: var(--bg-tertiary);
	}

	.level-btn.active {
		background-color: var(--bg-tertiary);
		border-color: var(--accent);
	}

	.level-num {
		font-family: var(--font-mono);
		font-size: 0.8rem;
		color: var(--accent);
		width: 24px;
	}

	.level-name {
		flex: 1;
		font-size: 0.9rem;
	}

	.level-count {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	.samples {
		min-height: 400px;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		min-height: 300px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-secondary);
	}

	.empty-state .hint {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-top: var(--spacing-sm);
	}

	.sample-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
		gap: var(--spacing-md);
	}
</style>

