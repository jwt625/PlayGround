<script lang="ts">
	import type { Sample } from '$lib/types';
	import GifPlayer from './GifPlayer.svelte';

	interface Props {
		sample: Sample;
		showLabels?: boolean;
		onOpenModal?: (sample: Sample) => void;
	}

	let { sample, showLabels = true, onOpenModal }: Props = $props();

	// Shared play state for synchronized pause
	let isPlaying = $state(true);

	function handleGifClick() {
		if (onOpenModal) {
			onOpenModal(sample);
		}
	}
</script>

<div class="sample-card">
	{#if showLabels}
		<div class="sample-header">
			<span class="sample-level">L{sample.level}</span>
			<span class="sample-name">{sample.name}</span>
		</div>
	{/if}

	<div class="sample-images">
		<div class="image-container">
			<GifPlayer
				src={sample.phase_gif}
				alt="Phase mask"
				bind:playing={isPlaying}
				onclick={onOpenModal ? handleGifClick : undefined}
			/>
			{#if showLabels}
				<span class="image-label">Phase</span>
			{/if}
		</div>
		<div class="image-container">
			<GifPlayer
				src={sample.intensity_gif}
				alt="Intensity distribution"
				bind:playing={isPlaying}
				onclick={onOpenModal ? handleGifClick : undefined}
			/>
			{#if showLabels}
				<span class="image-label">Intensity</span>
			{/if}
		</div>
	</div>

	{#if showLabels && sample.description}
		<p class="sample-description">{sample.description}</p>
	{/if}
</div>

<style>
	.sample-card {
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		padding: var(--spacing-md);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.sample-header {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.sample-level {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		color: var(--accent);
		padding: 2px 6px;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
	}

	.sample-name {
		font-weight: 500;
		font-size: 0.9rem;
	}

	.sample-images {
		display: flex;
		gap: var(--spacing-sm);
		min-width: 0;
	}

	.image-container {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.image-label {
		font-size: 0.75rem;
		color: var(--text-secondary);
		text-align: center;
		font-family: var(--font-mono);
	}

	.sample-description {
		font-size: 0.85rem;
		color: var(--text-secondary);
		line-height: 1.5;
	}
</style>

