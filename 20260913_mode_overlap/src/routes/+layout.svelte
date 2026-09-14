<script lang="ts">
	import '../app.css';
	import Header from '$lib/components/Header.svelte';

	let { children } = $props();
</script>

<div class="app">
	<Header />
	<main>
		{@render children()}
	</main>
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	}

	main {
		flex: 1;
		padding: var(--spacing-lg);
		max-width: 1200px;
		margin: 0 auto;
		width: 100%;
	}
</style>
