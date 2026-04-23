<script lang="ts">
	import { resolve } from '$app/paths';

	const pillars = [
		'Capture people and context quickly',
		'Review relationship history before meetings',
		'Keep follow-ups visible without CRM heaviness'
	];
</script>

<svelte:head>
	<title>Kizuna</title>
	<meta
		name="description"
		content="A local-first relationship memory system for people, organizations, events, and follow-ups."
	/>
</svelte:head>

<main class="shell">
	<section class="hero">
		<p class="eyebrow">Local-first relationship memory</p>
		<h1>Kizuna keeps the thread of every important relationship close at hand.</h1>
		<p class="lede">
			Start with people, organizations, shared events, and reminders. Add imports, pipelines, and
			richer context once the daily loop feels solid.
		</p>
		<div class="actions">
			<a href={resolve('/people')}>Open people</a>
			<a class="secondary" href={resolve('/reminders')}>Review reminders</a>
		</div>
	</section>

	<section class="cards" aria-label="MVP priorities">
		{#each pillars as pillar (pillar)}
			<article>
				<span></span>
				<p>{pillar}</p>
			</article>
		{/each}
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background:
			radial-gradient(circle at 15% 20%, rgba(227, 180, 72, 0.24), transparent 28rem),
			linear-gradient(135deg, #f7efe0 0%, #e8efe4 46%, #dbe8ef 100%);
		color: #1e2a24;
		font-family: Georgia, 'Times New Roman', serif;
	}

	.shell {
		min-height: 100vh;
		padding: 6rem max(1.5rem, 7vw);
	}

	.hero {
		max-width: 58rem;
	}

	.eyebrow {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		color: #6c4c18;
	}

	h1 {
		margin: 0;
		font-size: clamp(3.4rem, 9vw, 7.8rem);
		line-height: 0.88;
		letter-spacing: -0.08em;
	}

	.lede {
		max-width: 42rem;
		font-size: clamp(1.15rem, 2vw, 1.45rem);
		line-height: 1.55;
		color: #44534a;
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		margin-top: 2rem;
	}

	a {
		border: 1px solid #1e2a24;
		border-radius: 999px;
		background: #1e2a24;
		color: #fff9ef;
		padding: 0.85rem 1.2rem;
		text-decoration: none;
	}

	a.secondary {
		background: transparent;
		color: #1e2a24;
	}

	.cards {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 1rem;
		margin-top: 5rem;
	}

	article {
		min-height: 11rem;
		border: 1px solid rgba(30, 42, 36, 0.18);
		border-radius: 2rem;
		background: rgba(255, 251, 242, 0.58);
		padding: 1.4rem;
		backdrop-filter: blur(18px);
	}

	article span {
		display: block;
		width: 2.4rem;
		height: 0.35rem;
		border-radius: 999px;
		background: #c47d32;
	}

	article p {
		margin-top: 3.2rem;
		font-size: 1.25rem;
		line-height: 1.35;
	}

	@media (max-width: 760px) {
		.shell {
			padding-top: 3.5rem;
		}

		.cards {
			grid-template-columns: 1fr;
			margin-top: 3rem;
		}
	}
</style>
