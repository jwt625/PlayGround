<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		listEvents,
		listOrganizations,
		listPeople,
		listReminders,
		type Event,
		type Person,
		type Reminder
	} from '$lib/api';

	const workflow = [
		'Capture a person with context',
		'Review recent relationships fast',
		'Track pending follow-up without dashboard noise'
	];

	let loading = $state(true);
	let error = $state('');
	let people = $state<Person[]>([]);
	let reminders = $state<Reminder[]>([]);
	let events = $state<Event[]>([]);
	let organizationCount = $state(0);

	onMount(async () => {
		try {
			const [peopleResult, remindersResult, organizationsResult, eventsResult] = await Promise.all([
				listPeople({ limit: 6 }),
				listReminders({ limit: 6 }),
				listOrganizations({ limit: 50 }),
				listEvents({ limit: 6 })
			]);
			people = peopleResult;
			reminders = remindersResult;
			organizationCount = organizationsResult.length;
			events = eventsResult;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load workspace.';
		} finally {
			loading = false;
		}
	});

	function formatDate(value: string | null) {
		if (!value) return 'No date';
		return new Date(value).toLocaleString([], {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	}
</script>

<svelte:head>
	<title>Kizuna</title>
	<meta
		name="description"
		content="A local-first relationship memory system for people, organizations, events, and follow-ups."
	/>
</svelte:head>

<main class="shell">
	<section class="topbar">
		<div>
			<p class="eyebrow">Kizuna</p>
			<h1>Relationship workspace</h1>
		</div>
		<nav>
			<a href={resolve('/people')}>People</a>
			<a href={resolve('/organizations')}>Organizations</a>
			<a href={resolve('/events')}>Events</a>
			<a href={resolve('/reminders')}>Reminders</a>
			<a href={resolve('/pipelines')}>Pipelines</a>
			<a href={resolve('/search')}>Search</a>
			<a href={resolve('/imports')}>Imports</a>
			<a href={resolve('/exports')}>Exports</a>
		</nav>
	</section>

	{#if error}
		<p class="notice error">{error}</p>
	{:else if loading}
		<p class="notice">Loading workspace…</p>
	{:else}
		<section class="stats" aria-label="Workspace summary">
			<article>
				<span class="label">People</span>
				<strong>{people.length}</strong>
				<p>Visible in the current pull.</p>
			</article>
			<article>
				<span class="label">Organizations</span>
				<strong>{organizationCount}</strong>
				<p>Stored organization records.</p>
			</article>
			<article>
				<span class="label">Open reminders</span>
				<strong>{reminders.filter((item) => item.status !== 'Done').length}</strong>
				<p>Upcoming follow-up workload.</p>
			</article>
		</section>

		<section class="grid">
			<section class="panel">
				<header>
					<h2>Daily loop</h2>
				</header>
				<ul class="list compact">
					{#each workflow as item (item)}
						<li>{item}</li>
					{/each}
				</ul>
			</section>

			<section class="panel">
				<header>
					<h2>Recent people</h2>
					<a href={resolve('/people')}>Open</a>
				</header>
				<div class="table">
					<div class="row head">
						<span>Name</span>
						<span>Category</span>
						<span>Location</span>
					</div>
					{#each people as person (person.id)}
						<div class="row">
							<span>{person.display_name}</span>
							<span>{person.relationship_category}</span>
							<span>{person.primary_location || '—'}</span>
						</div>
					{/each}
				</div>
			</section>

			<section class="panel">
				<header>
					<h2>Upcoming reminders</h2>
					<a href={resolve('/reminders')}>Open</a>
				</header>
				<ul class="list">
					{#each reminders as reminder (reminder.id)}
						<li>
							<div>
								<strong>{reminder.title}</strong>
								<span>{reminder.priority}</span>
							</div>
							<time>{formatDate(reminder.due_at)}</time>
						</li>
					{/each}
				</ul>
			</section>

			<section class="panel">
				<header>
					<h2>Recent events</h2>
					<a href={resolve('/events')}>Open</a>
				</header>
				<ul class="list">
					{#each events as event (event.id)}
						<li>
							<div>
								<strong>{event.title}</strong>
								<span>{event.type}</span>
							</div>
							<time>{formatDate(event.started_at)}</time>
						</li>
					{/each}
				</ul>
			</section>
		</section>
	{/if}
	
	<section class="footnote">
		<p>Local-first relationship memory with a deliberately dense interface.</p>
	</section>
</main>

<style>
	.shell {
		min-height: 100vh;
		padding: 1.25rem;
	}

	.topbar,
	.stats,
	.grid {
		display: grid;
		gap: 1rem;
	}

	.topbar {
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--line-strong);
	}

	nav {
		display: flex;
		gap: 0.5rem;
	}

	nav a,
	.panel header a {
		border: 1px solid var(--line);
		padding: 0.5rem 0.7rem;
		text-decoration: none;
		background: var(--panel);
	}

	.eyebrow {
		margin: 0 0 0.35rem;
		font-size: 0.72rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: var(--muted);
	}

	h1 {
		margin: 0;
		font-size: clamp(2rem, 5vw, 3.4rem);
		line-height: 0.95;
		letter-spacing: -0.05em;
	}

	h2 {
		margin: 0;
		font-size: 1rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.stats {
		grid-template-columns: repeat(3, minmax(0, 1fr));
		margin-top: 1rem;
	}

	.grid {
		grid-template-columns: 0.9fr 1.3fr 1fr;
		margin-top: 1rem;
	}

	article,
	.panel {
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
		box-shadow: var(--shadow);
	}

	.stats article {
		padding: 0.85rem;
	}

	.label {
		display: block;
		font-size: 0.72rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--muted);
	}

	.stats strong {
		display: block;
		margin-top: 0.45rem;
		font-size: 2.2rem;
		line-height: 1;
	}

	.stats p,
	.footnote p,
	.notice {
		margin: 0.4rem 0 0;
		color: var(--muted);
	}

	.panel header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 0.85rem;
		border-bottom: 1px solid var(--line);
	}

	.list,
	.table {
		padding: 0.25rem 0.85rem 0.85rem;
	}

	.list {
		list-style: none;
		margin: 0;
		padding-left: 0.85rem;
	}

	.list li {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
		padding: 0.65rem 0;
		border-bottom: 1px solid var(--line);
	}

	.list li:last-child {
		border-bottom: 0;
		padding-bottom: 0;
	}

	.list li div {
		display: grid;
		gap: 0.2rem;
	}

	.list li span,
	.list li time {
		color: var(--muted);
		font-size: 0.9rem;
	}

	.compact li {
		justify-content: flex-start;
	}

	.table {
		display: grid;
	}

	.row {
		display: grid;
		grid-template-columns: 1.5fr 0.8fr 0.9fr;
		gap: 0.75rem;
		padding: 0.65rem 0;
		border-bottom: 1px solid var(--line);
	}

	.row.head {
		padding-top: 0.5rem;
		font-size: 0.72rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--muted);
	}

	.notice {
		padding: 0.85rem;
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
	}

	.notice.error {
		color: var(--text);
	}

	.footnote {
		margin-top: 1rem;
		border-top: 1px solid var(--line);
		padding-top: 0.75rem;
	}

	@media (max-width: 760px) {
		.shell {
			padding: 1rem;
		}

		.topbar,
		.stats,
		.grid,
		.row {
			grid-template-columns: 1fr;
		}
	}
</style>
