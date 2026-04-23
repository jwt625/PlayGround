<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>Reminders | Kizuna</title>
</svelte:head>

<main>
	<a href={resolve('/')}>Kizuna</a>
	<h1>Reminders</h1>
	<p>The follow-up dashboard will land after the reminder API is implemented.</p>
</main>

<style>
	main {
		min-height: 100vh;
		padding: 4rem max(1.5rem, 7vw);
		background: #e8efe4;
		color: #1e2a24;
		font-family: Georgia, 'Times New Roman', serif;
	}

	a {
		color: inherit;
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		text-transform: uppercase;
		letter-spacing: 0.14em;
	}

	h1 {
		font-size: clamp(3rem, 10vw, 7rem);
		letter-spacing: -0.07em;
	}
</style>
