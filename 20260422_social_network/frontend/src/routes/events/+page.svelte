<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { createEvent, listEvents, listPeople, type Event, type Person } from '$lib/api';

	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let events = $state<Event[]>([]);
	let people = $state<Person[]>([]);

	let form = $state({
		title: '',
		type: 'Meeting',
		started_at: '',
		duration_minutes: 30,
		context: '',
		summary: '',
		selected_person_ids: [] as string[]
	});

	onMount(async () => {
		await Promise.all([loadEvents(), loadPeople()]);
	});

	async function loadEvents() {
		loading = true;
		error = '';
		try {
			events = await listEvents({ limit: 100 });
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load events.';
		} finally {
			loading = false;
		}
	}

	async function loadPeople() {
		try {
			people = await listPeople({ limit: 100 });
			if (!form.selected_person_ids.length && people[0]) {
				form.selected_person_ids = [people[0].id];
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load people.';
		}
	}

	async function submitEvent() {
		saving = true;
		error = '';
		try {
			await createEvent({
				title: form.title,
				type: form.type,
				started_at: new Date(form.started_at).toISOString(),
				duration_minutes: Number(form.duration_minutes) || undefined,
				context: form.context || undefined,
				summary: form.summary || undefined,
				person_ids: form.selected_person_ids
			});
			form = {
				title: '',
				type: 'Meeting',
				started_at: '',
				duration_minutes: 30,
				context: '',
				summary: '',
				selected_person_ids: people[0] ? [people[0].id] : []
			};
			await loadEvents();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create event.';
		} finally {
			saving = false;
		}
	}

	function formatDateTime(value: string) {
		return new Date(value).toLocaleString([], {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
</script>

<svelte:head>
	<title>Events | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Events</h1>
		</div>
		<p class="meta">{events.length} logged</p>
	</header>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	<section class="workspace">
		<section class="panel">
			<div class="panel-header">
				<h2>Timeline</h2>
				<span>{loading ? 'Loading…' : 'Recent first'}</span>
			</div>
			<div class="table">
				<div class="row head">
					<span>Event</span>
					<span>Type</span>
					<span>When</span>
				</div>
				{#each events as event (event.id)}
					<div class="row">
						<span>
							<strong>{event.title}</strong>
							<small>{event.summary || event.context || 'No summary'}</small>
						</span>
						<span>{event.type}</span>
						<span>{formatDateTime(event.started_at)}</span>
					</div>
				{/each}
			</div>
		</section>

		<section class="panel">
			<div class="panel-header">
				<h2>Log event</h2>
				<span>Quick entry</span>
			</div>
			<form onsubmit={(event) => {
				event.preventDefault();
				submitEvent();
			}}>
				<label class="wide">
					<span>Title</span>
					<input bind:value={form.title} required />
				</label>
				<label>
					<span>Type</span>
					<select bind:value={form.type}>
						<option>Meeting</option>
						<option>One-on-one</option>
						<option>Call</option>
						<option>Email</option>
						<option>Message</option>
						<option>Meal</option>
						<option>Event attendance</option>
						<option>Work session</option>
						<option>Intro</option>
					</select>
				</label>
				<label>
					<span>Started at</span>
					<input bind:value={form.started_at} required type="datetime-local" />
				</label>
				<label>
					<span>Duration (min)</span>
					<input bind:value={form.duration_minutes} min="0" type="number" />
				</label>
				<label>
					<span>People</span>
					<select bind:value={form.selected_person_ids} multiple size="6">
						{#each people as person (person.id)}
							<option value={person.id}>{person.display_name}</option>
						{/each}
					</select>
				</label>
				<label class="wide">
					<span>Context</span>
					<input bind:value={form.context} />
				</label>
				<label class="wide">
					<span>Summary</span>
					<textarea bind:value={form.summary} rows="5"></textarea>
				</label>
				<button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Create event'}</button>
			</form>
		</section>
	</section>
</main>

<style>
	.shell {
		min-height: 100vh;
		padding: 1.25rem;
	}

	.topbar,
	.workspace,
	form {
		display: grid;
		gap: 1rem;
	}

	.topbar {
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--line-strong);
	}

	.workspace {
		grid-template-columns: 1.35fr 0.95fr;
		margin-top: 1rem;
	}

	.brand,
	.meta,
	.panel-header span,
	label span,
	.row.head {
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

	.table {
		display: grid;
		padding: 0 0.85rem 0.85rem;
	}

	.row {
		display: grid;
		grid-template-columns: 1.4fr 0.8fr 1fr;
		gap: 0.75rem;
		padding: 0.75rem 0;
		border-bottom: 1px solid var(--line);
	}

	.row strong,
	.row small {
		display: block;
	}

	.row small {
		margin-top: 0.2rem;
		color: var(--muted);
	}

	form {
		grid-template-columns: repeat(2, minmax(0, 1fr));
		padding: 0.85rem;
	}

	label {
		display: grid;
		gap: 0.35rem;
	}

	input,
	select,
	textarea,
	button {
		width: 100%;
		border: 1px solid var(--line);
		background: var(--panel);
		padding: 0.62rem 0.72rem;
		color: var(--text);
	}

	button {
		cursor: pointer;
	}

	.notice {
		margin-top: 1rem;
		padding: 0.85rem;
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
	}

	.wide {
		grid-column: 1 / -1;
	}

	@media (max-width: 960px) {
		.workspace,
		.row,
		form {
			grid-template-columns: 1fr;
		}
	}
</style>
