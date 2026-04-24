<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		completeReminder,
		createReminder,
		listReminders,
		snoozeReminder,
		type Reminder
	} from '$lib/api';

	const statusOptions = ['All', 'Open', 'Done'];

	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let status = $state('All');
	let reminders = $state<Reminder[]>([]);

	let form = $state({
		title: '',
		due_at: '',
		priority: 'Normal',
		entity_type: 'Person',
		notes: ''
	});

	onMount(loadReminders);

	async function loadReminders() {
		loading = true;
		error = '';
		try {
			reminders = await listReminders({
				status: status === 'All' ? undefined : status,
				limit: 100
			});
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load reminders.';
		} finally {
			loading = false;
		}
	}

	async function submitReminder() {
		saving = true;
		error = '';
		try {
			await createReminder({
				title: form.title,
				due_at: new Date(form.due_at).toISOString(),
				notes: form.notes || undefined,
				priority: form.priority,
				entity_type: form.entity_type
			});
			form = {
				title: '',
				due_at: '',
				priority: 'Normal',
				entity_type: 'Person',
				notes: ''
			};
			await loadReminders();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create reminder.';
		} finally {
			saving = false;
		}
	}

	async function markDone(reminder: Reminder) {
		try {
			await completeReminder(reminder.id);
			await loadReminders();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to update reminder.';
		}
	}

	async function snoozeOneDay(reminder: Reminder) {
		try {
			await snoozeReminder(reminder.id, new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString());
			await loadReminders();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to update reminder.';
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
	<title>Reminders | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Reminders</h1>
		</div>
		<p class="meta">{reminders.filter((item) => item.status !== 'Done').length} open</p>
	</header>

	<section class="toolbar">
		<label>
			<span>Status</span>
			<select bind:value={status}>
				{#each statusOptions as option (option)}
					<option value={option}>{option}</option>
				{/each}
			</select>
		</label>
		<button type="button" onclick={loadReminders}>Refresh</button>
	</section>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	<section class="workspace">
		<section class="panel">
			<div class="panel-header">
				<h2>Queue</h2>
				<span>{loading ? 'Loading…' : 'Sorted by due date'}</span>
			</div>
			<div class="table">
				<div class="row head">
					<span>Task</span>
					<span>Due</span>
					<span>Status</span>
					<span>Action</span>
				</div>
				{#each reminders as reminder (reminder.id)}
					<div class="row">
						<span>
							<strong>{reminder.title}</strong>
							<small>{reminder.priority} · {reminder.entity_type || 'General'}</small>
						</span>
						<span>{formatDateTime(reminder.due_at)}</span>
						<span>{reminder.status}</span>
						<span>
							{#if reminder.status === 'Done'}
								Closed
							{:else}
								<div class="inline-actions">
									<button type="button" onclick={() => markDone(reminder)}>Done</button>
									<button type="button" onclick={() => snoozeOneDay(reminder)}>+1d</button>
								</div>
							{/if}
						</span>
					</div>
				{/each}
			</div>
		</section>

		<section class="panel">
			<div class="panel-header">
				<h2>New reminder</h2>
				<span>Quick add</span>
			</div>
			<form onsubmit={(event) => {
				event.preventDefault();
				submitReminder();
			}}>
				<label class="wide">
					<span>Title</span>
					<input bind:value={form.title} required />
				</label>
				<label>
					<span>Due at</span>
					<input bind:value={form.due_at} required type="datetime-local" />
				</label>
				<label>
					<span>Priority</span>
					<select bind:value={form.priority}>
						<option>Low</option>
						<option>Normal</option>
						<option>High</option>
					</select>
				</label>
				<label>
					<span>Entity type</span>
					<select bind:value={form.entity_type}>
						<option>Person</option>
						<option>Organization</option>
						<option>General</option>
					</select>
				</label>
				<label class="wide">
					<span>Notes</span>
					<textarea bind:value={form.notes} rows="6"></textarea>
				</label>
				<button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Create reminder'}</button>
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
	.toolbar,
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

	.toolbar {
		grid-template-columns: 14rem auto;
		align-items: end;
		margin-top: 1rem;
	}

	.workspace {
		grid-template-columns: 1.5fr 0.9fr;
		margin-top: 1rem;
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
		grid-template-columns: 1.3fr 1fr 0.7fr auto;
		gap: 0.75rem;
		padding: 0.75rem 0;
		border-bottom: 1px solid var(--line);
		align-items: center;
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

	.inline-actions {
		display: flex;
		gap: 0.4rem;
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
		.toolbar,
		.workspace,
		.row,
		form {
			grid-template-columns: 1fr;
		}
	}
</style>
