<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>People | Kizuna</title>
</svelte:head>

<main>
	<a href={resolve('/')}>Kizuna</a>
	<h1>People</h1>
	<p>
		The people workspace will connect to the FastAPI people endpoints in the next frontend phase.
	</p>
</main>

<style>
	main {
		min-height: 100vh;
		padding: 4rem max(1.5rem, 7vw);
		background: #f7efe0;
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
