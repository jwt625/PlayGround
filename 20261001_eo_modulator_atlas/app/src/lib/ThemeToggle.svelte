<script lang="ts">
	import { ui, setTheme } from './state.svelte';
</script>

<button
	class="theme"
	aria-label={ui.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
	title={ui.theme === 'dark' ? 'Light theme' : 'Dark theme'}
	onclick={() => setTheme(ui.theme === 'dark' ? 'light' : 'dark')}
>
	{#if ui.theme === 'dark'}
		<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
			<circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.4" />
			<g stroke="currentColor" stroke-width="1.4">
				<line x1="8" y1="1" x2="8" y2="3" /><line x1="8" y1="13" x2="8" y2="15" />
				<line x1="1" y1="8" x2="3" y2="8" /><line x1="13" y1="8" x2="15" y2="8" />
				<line x1="3" y1="3" x2="4.4" y2="4.4" /><line x1="11.6" y1="11.6" x2="13" y2="13" />
				<line x1="3" y1="13" x2="4.4" y2="11.6" /><line x1="11.6" y1="4.4" x2="13" y2="3" />
			</g>
		</svg>
	{:else}
		<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
			<path d="M13 9.6A5.6 5.6 0 0 1 6.4 3a5.6 5.6 0 1 0 6.6 6.6z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="miter" />
		</svg>
	{/if}
</button>

<style>
	.theme {
		width: 26px;
		height: 24px;
		padding: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		background: transparent;
	}
</style>
