<script lang="ts">
	import type { Sample } from '$lib/types';

	// Placeholder data until samples.json is generated
	const levels = [
		{ id: 1, name: 'Foundations', count: 15 },
		{ id: 2, name: 'Periodic Structures', count: 20 },
		{ id: 3, name: 'Spot Arrays', count: 20 },
		{ id: 4, name: 'Special Beams', count: 20 },
		{ id: 5, name: 'Compound Patterns', count: 15 },
		{ id: 6, name: 'Practical Applications', count: 15 },
		{ id: 7, name: 'Shapes and Objects', count: 40 }
	];

	let selectedLevel = $state<number | null>(null);
	let samples = $state<Sample[]>([]);

	function selectLevel(level: number) {
		selectedLevel = level;
		// TODO: Load samples for this level from samples.json
		samples = [];
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
			{#if selectedLevel === null}
				<div class="empty-state">
					<p>Select a level to view samples</p>
				</div>
			{:else if samples.length === 0}
				<div class="empty-state">
					<p>No samples generated yet for Level {selectedLevel}</p>
					<p class="hint">Run the Python generator to create training samples</p>
				</div>
			{:else}
				<div class="sample-grid">
					{#each samples as sample}
						<div class="sample-card">
							<!-- TODO: SampleCard component -->
							<p>{sample.name}</p>
						</div>
					{/each}
				</div>
			{/if}
		</section>
	</div>
</div>

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

