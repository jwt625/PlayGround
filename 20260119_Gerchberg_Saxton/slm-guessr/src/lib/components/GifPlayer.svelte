<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	interface Props {
		src: string;
		alt?: string;
	}

	let { src, alt = '' }: Props = $props();

	let isPlaying = $state(true);
	let imgElement = $state<HTMLImageElement | null>(null);
	let canvasElement = $state<HTMLCanvasElement | null>(null);
	let animationId: number | null = null;

	// Continuously capture frames to canvas while playing
	function captureLoop() {
		if (!imgElement || !canvasElement || !isPlaying) return;
		const ctx = canvasElement.getContext('2d');
		if (ctx && imgElement.complete && imgElement.naturalWidth > 0) {
			canvasElement.width = imgElement.naturalWidth;
			canvasElement.height = imgElement.naturalHeight;
			ctx.drawImage(imgElement, 0, 0);
		}
		animationId = requestAnimationFrame(captureLoop);
	}

	function startCapturing() {
		if (animationId === null) {
			captureLoop();
		}
	}

	function stopCapturing() {
		if (animationId !== null) {
			cancelAnimationFrame(animationId);
			animationId = null;
		}
	}

	function togglePlay() {
		if (isPlaying) {
			// Pausing - stop capturing, canvas already has last frame
			stopCapturing();
		} else {
			// Resuming - start capturing again
			startCapturing();
		}
		isPlaying = !isPlaying;
	}

	onMount(() => {
		// Start capture loop when mounted
		startCapturing();
	});

	onDestroy(() => {
		stopCapturing();
	});
</script>

<div class="gif-player">
	<!-- Hidden img for GIF source -->
	<img
		bind:this={imgElement}
		{src}
		{alt}
		class="gif-source"
		class:hidden={!isPlaying}
	/>
	<!-- Canvas shows captured frame when paused -->
	<canvas
		bind:this={canvasElement}
		class="gif-canvas"
		class:hidden={isPlaying}
	></canvas>
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

	.gif-source,
	.gif-canvas {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
	}

	.hidden {
		display: none;
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

