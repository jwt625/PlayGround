<script lang="ts">
	interface Props {
		src: string;
		alt?: string;
		playing?: boolean;
		onclick?: () => void;
	}

	let { src, alt = '', playing = $bindable(true), onclick }: Props = $props();

	let imgElement = $state<HTMLImageElement | null>(null);
	let frozenSrc = $state<string | null>(null);

	// Capture current frame to data URL
	function captureFrame(): string | null {
		if (!imgElement || !imgElement.complete || imgElement.naturalWidth === 0) {
			return null;
		}
		const canvas = document.createElement('canvas');
		canvas.width = imgElement.naturalWidth;
		canvas.height = imgElement.naturalHeight;
		const ctx = canvas.getContext('2d');
		if (!ctx) return null;
		ctx.drawImage(imgElement, 0, 0);
		return canvas.toDataURL();
	}

	function togglePlay() {
		if (playing) {
			// Pausing - capture current frame first
			frozenSrc = captureFrame();
		}
		playing = !playing;
	}

	// When playing changes externally
	$effect(() => {
		if (!playing && !frozenSrc && imgElement) {
			// External pause without frozen frame - capture now
			frozenSrc = captureFrame();
		}
		if (playing) {
			// Clear frozen frame when resuming
			frozenSrc = null;
		}
	});

	function handleClick() {
		if (onclick) {
			onclick();
		}
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="gif-player" class:clickable={onclick} onclick={handleClick}>
	<img
		bind:this={imgElement}
		src={playing ? src : (frozenSrc || src)}
		{alt}
		class="gif-img"
	/>
	<button class="play-toggle" onclick={(e) => { e.stopPropagation(); togglePlay(); }} title={playing ? 'Pause' : 'Play'}>
		{#if playing}
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

	.gif-player.clickable {
		cursor: pointer;
	}

	.gif-player.clickable:hover {
		border-color: var(--accent);
	}

	.gif-img {
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

