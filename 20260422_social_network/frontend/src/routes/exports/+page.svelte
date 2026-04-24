<script lang="ts">
	import { resolve } from '$app/paths';

	const apiBase = (import.meta.env.PUBLIC_API_BASE_URL || 'http://localhost:8000/api').replace(/\/$/, '');
	const exportsList = [
		{ label: 'People CSV', href: `${apiBase}/exports/people-csv`, note: 'Core relationship records' },
		{ label: 'Organizations CSV', href: `${apiBase}/exports/organizations-csv`, note: 'Organization directory' },
		{ label: 'Events CSV', href: `${apiBase}/exports/events-csv`, note: 'Interaction history export' },
		{ label: 'Reminders CSV', href: `${apiBase}/exports/reminders-csv`, note: 'Follow-up workload export' }
	];
</script>

<svelte:head>
	<title>Exports | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Exports</h1>
		</div>
		<p class="meta">Portable CSV snapshots</p>
	</header>

	<section class="panel">
		<div class="panel-header">
			<h2>Available exports</h2>
			<span>Backend-generated</span>
		</div>
		<ul>
			{#each exportsList as item (item.href)}
				<li>
					<div>
						<strong>{item.label}</strong>
						<p>{item.note}</p>
					</div>
					<button type="button" onclick={() => globalThis.open(item.href, '_blank', 'noopener,noreferrer')}>
						Open
					</button>
				</li>
			{/each}
		</ul>
	</section>
</main>

<style>
	.shell {
		min-height: 100vh;
		padding: 1.25rem;
	}

	.topbar {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		gap: 1rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--line-strong);
	}

	.brand,
	.meta,
	.panel-header span {
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.12em;
		color: var(--muted);
		text-decoration: none;
	}

	h1 {
		margin: 0.35rem 0 0;
		font-size: clamp(2rem, 4vw, 3rem);
		letter-spacing: -0.05em;
	}

	h2 {
		margin: 0;
		font-size: 0.9rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.panel {
		margin-top: 1rem;
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
		box-shadow: var(--shadow);
	}

	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 0.85rem;
		border-bottom: 1px solid var(--line);
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0.25rem 0.85rem 0.85rem;
	}

	li {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 1rem;
		padding: 0.75rem 0;
		border-bottom: 1px solid var(--line);
	}

	li:last-child {
		border-bottom: 0;
	}

	li strong,
	li p {
		display: block;
		margin: 0;
	}

	li p {
		margin-top: 0.2rem;
		color: var(--muted);
	}

	li button {
		border: 1px solid var(--line);
		background: var(--panel);
		padding: 0.55rem 0.7rem;
		cursor: pointer;
	}

	@media (max-width: 720px) {
		li {
			flex-direction: column;
			align-items: start;
		}
	}
</style>
