<script lang="ts">
	import '$lib/styles/theme.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>LTSpice Web</title>
</svelte:head>

{@render children()}
