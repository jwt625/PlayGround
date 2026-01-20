<script lang="ts">
	interface Props {
		src: string;
		alt?: string;
	}

	let { src, alt = '' }: Props = $props();

	let isPlaying = $state(true);
	let imgElement = $state<HTMLImageElement | null>(null);
	let staticSrc = $state<string | null>(null);

	// Capture first frame when paused
	function captureFrame() {
		if (!imgElement) return;
		const canvas = document.createElement('canvas');
		canvas.width = imgElement.naturalWidth || 256;
		canvas.height = imgElement.naturalHeight || 256;
		const ctx = canvas.getContext('2d');
		if (ctx) {
			ctx.drawImage(imgElement, 0, 0);
			staticSrc = canvas.toDataURL('image/png');
		}
	}

	function togglePlay() {
		if (isPlaying) {
			captureFrame();
		} else {
			staticSrc = null;
		}
		isPlaying = !isPlaying;
	}
</script>

<div class="gif-player">
	<img
		bind:this={imgElement}
		src={isPlaying ? src : (staticSrc || src)}
		{alt}
		class="gif-image"
	/>
	<button class="play-toggle" onclick={togglePlay} title={isPlaying ? 'Pause' : 'Play'}>
		{#if isPlaying}
			<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
				<rect x="6" y="4" width="4" height="16" />
				<rect x="14" y="4" width="4" height="16" />
			</svg>
		{:else}
			<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
				<polygon points="5,3 19,12 5,21" />
			</svg>
		{/if}
	</button>
</div>

<style>
	.gif-player {
		position: relative;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
		overflow: hidden;
		width: 100%;
		aspect-ratio: 1 / 1;
	}

	.gif-image {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
	}

	.play-toggle {
		position: absolute;
		bottom: var(--spacing-xs);
		right: var(--spacing-xs);
		width: 28px;
		height: 28px;
		padding: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		background-color: rgba(0, 0, 0, 0.7);
		border: 1px solid var(--border);
		color: var(--text-primary);
		cursor: pointer;
		opacity: 0;
		transition: opacity 0.15s;
	}

	.gif-player:hover .play-toggle {
		opacity: 1;
	}

	.play-toggle:hover {
		background-color: rgba(0, 0, 0, 0.9);
	}
</style>

