<script lang="ts">
	import type { Sample } from '$lib/types';
	import GifPlayer from './GifPlayer.svelte';

	interface Props {
		sample: Sample | null;
		onClose: () => void;
	}

	let { sample, onClose }: Props = $props();

	// Shared play state for synchronized pause in modal
	let isPlaying = $state(true);

	function handleBackdropClick(e: MouseEvent) {
		if (e.target === e.currentTarget) {
			onClose();
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			onClose();
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if sample}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="modal-backdrop" onclick={handleBackdropClick}>
		<div class="modal-content">
			<div class="modal-header">
				<div class="sample-info">
					<span class="sample-level">L{sample.level}</span>
					<span class="sample-name">{sample.name}</span>
				</div>
				<button class="close-btn" onclick={onClose} title="Close (Esc)">
					<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
						<line x1="18" y1="6" x2="6" y2="18" />
						<line x1="6" y1="6" x2="18" y2="18" />
					</svg>
				</button>
			</div>

			<div class="modal-body">
				<div class="gif-pair">
					<div class="gif-container">
						<GifPlayer src={sample.phase_gif} alt="Phase mask" bind:playing={isPlaying} />
						<span class="gif-label">Phase</span>
					</div>
					<div class="gif-container">
						<GifPlayer src={sample.intensity_gif} alt="Intensity distribution" bind:playing={isPlaying} />
						<span class="gif-label">Intensity</span>
					</div>
				</div>
				{#if sample.description}
					<p class="sample-description">{sample.description}</p>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
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

	.modal-content {
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		max-width: 900px;
		max-height: 90vh;
		width: 100%;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.modal-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--spacing-md);
		border-bottom: 1px solid var(--border);
	}

	.sample-info {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.sample-level {
		font-family: var(--font-mono);
		font-size: 0.8rem;
		color: var(--accent);
		padding: 2px 8px;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
	}

	.sample-name {
		font-weight: 600;
		font-size: 1.1rem;
	}

	.close-btn {
		width: 32px;
		height: 32px;
		padding: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		border: 1px solid transparent;
		color: var(--text-secondary);
		cursor: pointer;
	}

	.close-btn:hover {
		color: var(--text-primary);
		background-color: var(--bg-tertiary);
	}

	.modal-body {
		padding: var(--spacing-lg);
		overflow-y: auto;
	}

	.gif-pair {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-lg);
	}

	.gif-container {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.gif-label {
		text-align: center;
		font-family: var(--font-mono);
		font-size: 0.9rem;
		color: var(--text-secondary);
	}

	.sample-description {
		margin-top: var(--spacing-lg);
		color: var(--text-secondary);
		text-align: center;
		line-height: 1.6;
	}
</style>

