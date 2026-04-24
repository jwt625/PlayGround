<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		addPersonLocation,
		addPersonOrganizationRole,
		addPersonTag,
		completeReminder,
		createEvent,
		createPerson,
		createReminder,
		getPerson,
		listOrganizations,
		listPeople,
		snoozeReminder,
		type Organization,
		type Person,
		type PersonDetail
	} from '$lib/api';

	const categoryOptions = ['All', 'New', 'Active', 'Warm', 'Watch'];

	let loading = $state(true);
	let saving = $state(false);
	let error = $state('');
	let query = $state('');
	let category = $state('All');
	let city = $state('');
	let people = $state<Person[]>([]);
	let organizations = $state<Organization[]>([]);
	let selectedId = $state('');
	let selectedPersonDetail = $state<PersonDetail | null>(null);

	let form = $state({
		display_name: '',
		given_name: '',
		family_name: '',
		primary_location: '',
		relationship_summary: '',
		how_we_met: '',
		contact_type: 'Email',
		contact_value: '',
		contact_label: '',
		profile_platform: 'LinkedIn',
		profile_url: '',
		notes: ''
	});

	let eventForm = $state({
		title: '',
		type: 'One-on-one',
		started_at: '',
		duration_minutes: 45,
		context: '',
		summary: ''
	});

	let reminderForm = $state({
		title: '',
		due_at: '',
		priority: 'Normal',
		notes: ''
	});

	let tagForm = $state({ name: '' });
	let locationForm = $state({ city: '', region: '', country: '' });
	let roleForm = $state({ organization_id: '', title: '', role_type: '' });

	const selectedPerson = $derived(people.find((person) => person.id === selectedId) ?? people[0]);

	$effect(() => {
		if (selectedId) {
			void loadSelectedPerson(selectedId);
		}
	});

	onMount(async () => {
		await Promise.all([loadPeople(), loadOrganizations()]);
	});

	async function loadPeople() {
		loading = true;
		error = '';
		try {
			people = await listPeople({
				q: query,
				relationship_category: category === 'All' ? undefined : category,
				city: city || undefined,
				limit: 100
			});
			if (!selectedPerson && people[0]) {
				selectedId = people[0].id;
			}
			if (selectedId && !people.some((person) => person.id === selectedId) && people[0]) {
				selectedId = people[0].id;
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load people.';
		} finally {
			loading = false;
		}
	}

	async function loadOrganizations() {
		try {
			organizations = await listOrganizations({ limit: 100 });
			if (!roleForm.organization_id && organizations[0]) {
				roleForm.organization_id = organizations[0].id;
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load organizations.';
		}
	}

	async function loadSelectedPerson(personId: string) {
		try {
			selectedPersonDetail = await getPerson(personId);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load person detail.';
		}
	}

	async function submitPerson() {
		saving = true;
		error = '';
		try {
			const created = await createPerson({
				display_name: form.display_name,
				given_name: form.given_name || undefined,
				family_name: form.family_name || undefined,
				primary_location: form.primary_location || undefined,
				relationship_summary: form.relationship_summary || undefined,
				how_we_met: form.how_we_met || undefined,
				notes: form.notes || undefined,
				contact_methods: form.contact_value
					? [
							{
								type: form.contact_type,
								value: form.contact_value,
								label: form.contact_label || undefined,
								is_primary: true
							}
						]
					: [],
				external_profiles: form.profile_url
					? [
							{
								platform: form.profile_platform,
								url_or_handle: form.profile_url
							}
						]
					: []
			});
			form = {
				display_name: '',
				given_name: '',
				family_name: '',
				primary_location: '',
				relationship_summary: '',
				how_we_met: '',
				contact_type: 'Email',
				contact_value: '',
				contact_label: '',
				profile_platform: 'LinkedIn',
				profile_url: '',
				notes: ''
			};
			await loadPeople();
			selectedId = created.id;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create person.';
		} finally {
			saving = false;
		}
	}

	async function submitEvent() {
		if (!selectedId || !eventForm.started_at || !eventForm.title) return;
		try {
			await createEvent({
				title: eventForm.title,
				type: eventForm.type,
				started_at: new Date(eventForm.started_at).toISOString(),
				duration_minutes: Number(eventForm.duration_minutes) || undefined,
				context: eventForm.context || undefined,
				summary: eventForm.summary || undefined,
				person_ids: [selectedId]
			});
			eventForm = {
				title: '',
				type: 'One-on-one',
				started_at: '',
				duration_minutes: 45,
				context: '',
				summary: ''
			};
			await Promise.all([loadPeople(), loadSelectedPerson(selectedId)]);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create event.';
		}
	}

	async function submitReminder() {
		if (!selectedId || !reminderForm.title || !reminderForm.due_at) return;
		try {
			await createReminder({
				title: reminderForm.title,
				due_at: new Date(reminderForm.due_at).toISOString(),
				priority: reminderForm.priority,
				notes: reminderForm.notes || undefined,
				entity_type: 'Person',
				entity_id: selectedId
			});
			reminderForm = {
				title: '',
				due_at: '',
				priority: 'Normal',
				notes: ''
			};
			await Promise.all([loadPeople(), loadSelectedPerson(selectedId)]);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create reminder.';
		}
	}

	async function completeSelectedReminder(reminderId: string) {
		try {
			await completeReminder(reminderId);
			if (selectedId) {
				await loadSelectedPerson(selectedId);
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to complete reminder.';
		}
	}

	async function submitTag() {
		if (!selectedId || !tagForm.name) return;
		try {
			selectedPersonDetail = await addPersonTag(selectedId, { name: tagForm.name });
			tagForm = { name: '' };
			await loadPeople();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to add tag.';
		}
	}

	async function submitLocation() {
		if (!selectedId) return;
		try {
			selectedPersonDetail = await addPersonLocation(selectedId, {
				location: {
					label: null,
					city: locationForm.city || null,
					region: locationForm.region || null,
					country: locationForm.country || null,
					address_line: null,
					latitude: null,
					longitude: null,
					location_type: 'Home',
					notes: null
				},
				is_primary: true
			});
			locationForm = { city: '', region: '', country: '' };
			await loadPeople();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to add location.';
		}
	}

	async function submitRole() {
		if (!selectedId || !roleForm.organization_id) return;
		try {
			selectedPersonDetail = await addPersonOrganizationRole(selectedId, {
				organization_id: roleForm.organization_id,
				title: roleForm.title || undefined,
				role_type: roleForm.role_type || undefined,
				is_current: true
			});
			roleForm = { organization_id: organizations[0]?.id || '', title: '', role_type: '' };
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to add organization role.';
		}
	}

	async function snoozeSelectedReminder(reminderId: string) {
		try {
			const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
			await snoozeReminder(reminderId, tomorrow);
			if (selectedId) {
				await loadSelectedPerson(selectedId);
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to snooze reminder.';
		}
	}

	function formatDateTime(value: string | null) {
		if (!value) return '—';
		return new Date(value).toLocaleDateString([], {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	}
</script>

<svelte:head>
	<title>People | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>People</h1>
		</div>
		<p class="meta">{people.length} records</p>
	</header>

	<section class="toolbar">
		<label>
			<span>Search</span>
			<input bind:value={query} placeholder="Name, location, notes, how you met" />
		</label>
		<label>
			<span>Category</span>
			<select bind:value={category}>
				{#each categoryOptions as option (option)}
					<option value={option}>{option}</option>
				{/each}
			</select>
		</label>
		<label>
			<span>City</span>
			<input bind:value={city} placeholder="Filter city" />
		</label>
		<button type="button" onclick={loadPeople}>Refresh</button>
	</section>

	{#if error}
		<p class="notice error">{error}</p>
	{/if}

	<section class="workspace">
		<section class="panel list-panel">
			<div class="panel-header">
				<h2>Directory</h2>
				<span>{loading ? 'Loading…' : 'Live'}</span>
			</div>
			<div class="table">
				<div class="row head">
					<span>Name</span>
					<span>Category</span>
					<span>Location</span>
				</div>
				{#each people as person (person.id)}
					<button class:selected={selectedPerson?.id === person.id} class="row person-row" onclick={() => (selectedId = person.id)}>
						<span>{person.display_name}</span>
						<span>{person.relationship_category}</span>
						<span>{person.primary_location || '—'}</span>
					</button>
				{/each}
			</div>
		</section>

		<section class="panel detail-panel">
			<div class="panel-header">
				<h2>Selected record</h2>
				<span>{selectedPerson ? selectedPerson.relationship_score : 0} score</span>
			</div>
			{#if selectedPersonDetail}
				<div class="detail-grid">
					<div class="field">
						<span>Name</span>
						<strong>{selectedPersonDetail.display_name}</strong>
					</div>
					<div class="field">
						<span>Location</span>
						<strong>{selectedPersonDetail.primary_location || '—'}</strong>
					</div>
					<div class="field wide">
						<span>Relationship summary</span>
						<p>{selectedPersonDetail.relationship_summary || 'No summary yet.'}</p>
					</div>
					<div class="field wide">
						<span>How we met</span>
						<p>{selectedPersonDetail.how_we_met || 'No origin recorded.'}</p>
					</div>
					<div class="field">
						<span>Last interaction</span>
						<strong>{formatDateTime(selectedPersonDetail.last_interaction_date)}</strong>
					</div>
					<div class="field">
						<span>Next reminder</span>
						<strong>{formatDateTime(selectedPersonDetail.next_reminder_date)}</strong>
					</div>
					<div class="field wide">
						<span>Relationship score</span>
						<p>{selectedPersonDetail.relationship_score} · {selectedPersonDetail.relationship_category}</p>
						<p>{selectedPersonDetail.relationship_score_reason || 'No explanation yet.'}</p>
					</div>
					<div class="field wide">
						<span>Contact methods</span>
						<ul>
							{#if selectedPersonDetail.contact_methods.length}
								{#each selectedPersonDetail.contact_methods as method (method.id)}
									<li>{method.type}: {method.value}</li>
								{/each}
							{:else}
								<li>No contact methods</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>External profiles</span>
						<ul>
							{#if selectedPersonDetail.external_profiles.length}
								{#each selectedPersonDetail.external_profiles as profile (profile.id)}
									<li>{profile.platform}: {profile.url_or_handle}</li>
								{/each}
							{:else}
								<li>No external profiles</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Organizations</span>
						<ul>
							{#if selectedPersonDetail.organization_roles.length}
								{#each selectedPersonDetail.organization_roles as role (role.id)}
									<li>{role.organization_name || 'Organization'} · {role.title || role.role_type || 'Role'}</li>
								{/each}
							{:else}
								<li>No organization roles yet.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Tags</span>
						<p>{selectedPersonDetail.tags.map((item) => item.tag.name).join(', ') || 'No tags yet.'}</p>
					</div>
					<div class="field wide">
						<span>Locations</span>
						<ul>
							{#if selectedPersonDetail.locations.length}
								{#each selectedPersonDetail.locations as item (item.id)}
									<li>{item.location.city || item.location.label || '—'} {item.location.country || ''}</li>
								{/each}
							{:else}
								<li>No structured locations yet.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Timeline</span>
						<ul>
							{#if selectedPersonDetail.recent_events.length}
								{#each selectedPersonDetail.recent_events as event (event.id)}
									<li>{formatDateTime(event.started_at)} · {event.type} · {event.title}</li>
								{/each}
							{:else}
								<li>No events yet.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Active reminders</span>
						<ul class="reminder-list">
							{#if selectedPersonDetail.active_reminders.length}
								{#each selectedPersonDetail.active_reminders as reminder (reminder.id)}
									<li>
										<div>
											<strong>{reminder.title}</strong>
											<p>{formatDateTime(reminder.due_at)}</p>
										</div>
										<div class="inline-actions">
											<button type="button" onclick={() => completeSelectedReminder(reminder.id)}>Done</button>
											<button type="button" onclick={() => snoozeSelectedReminder(reminder.id)}>+1d</button>
										</div>
									</li>
								{/each}
							{:else}
								<li>No active reminders.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Pipeline items</span>
						<ul>
							{#if selectedPersonDetail.pipeline_items.length}
								{#each selectedPersonDetail.pipeline_items as item (item.id)}
									<li>{item.title}</li>
								{/each}
							{:else}
								<li>No pipeline items.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Notes</span>
						<p>{selectedPersonDetail.notes || 'No notes yet.'}</p>
					</div>
				</div>
			{:else}
				<p class="empty">No person selected.</p>
			{/if}
		</section>

		<section class="panel form-panel">
			<div class="panel-header">
				<h2>Actions</h2>
				<span>Capture and follow-up</span>
			</div>
			<form class="stacked" onsubmit={(event) => {
				event.preventDefault();
				submitPerson();
			}}>
				<h3>New person</h3>
				<label>
					<span>Display name</span>
					<input bind:value={form.display_name} required />
				</label>
				<label>
					<span>Given name</span>
					<input bind:value={form.given_name} />
				</label>
				<label>
					<span>Family name</span>
					<input bind:value={form.family_name} />
				</label>
				<label>
					<span>Location</span>
					<input bind:value={form.primary_location} />
				</label>
				<label class="wide">
					<span>Relationship summary</span>
					<textarea bind:value={form.relationship_summary} rows="2"></textarea>
				</label>
				<label class="wide">
					<span>How we met</span>
					<textarea bind:value={form.how_we_met} rows="2"></textarea>
				</label>
				<label>
					<span>Contact type</span>
					<select bind:value={form.contact_type}>
						<option>Email</option>
						<option>Phone</option>
						<option>WeChat</option>
						<option>WhatsApp</option>
						<option>Telegram</option>
						<option>Website</option>
					</select>
				</label>
				<label>
					<span>Contact value</span>
					<input bind:value={form.contact_value} />
				</label>
				<label>
					<span>Contact label</span>
					<input bind:value={form.contact_label} placeholder="Personal, work, assistant" />
				</label>
				<label>
					<span>Profile platform</span>
					<select bind:value={form.profile_platform}>
						<option>LinkedIn</option>
						<option>Website</option>
						<option>X</option>
						<option>GitHub</option>
					</select>
				</label>
				<label class="wide">
					<span>Profile URL or handle</span>
					<input bind:value={form.profile_url} />
				</label>
				<label class="wide">
					<span>Notes</span>
					<textarea bind:value={form.notes} rows="4"></textarea>
				</label>
				<button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Create person'}</button>
			</form>
			<form class="stacked separator" onsubmit={(event) => {
				event.preventDefault();
				submitEvent();
			}}>
				<h3>Log event for selected person</h3>
				<label class="wide">
					<span>Title</span>
					<input bind:value={eventForm.title} />
				</label>
				<label>
					<span>Type</span>
					<select bind:value={eventForm.type}>
						<option>One-on-one</option>
						<option>Meeting</option>
						<option>Call</option>
						<option>Email</option>
						<option>Message</option>
						<option>Meal</option>
						<option>Work session</option>
						<option>Intro</option>
					</select>
				</label>
				<label>
					<span>Started at</span>
					<input bind:value={eventForm.started_at} type="datetime-local" />
				</label>
				<label>
					<span>Duration</span>
					<input bind:value={eventForm.duration_minutes} min="0" type="number" />
				</label>
				<label class="wide">
					<span>Context</span>
					<input bind:value={eventForm.context} />
				</label>
				<label class="wide">
					<span>Summary</span>
					<textarea bind:value={eventForm.summary} rows="3"></textarea>
				</label>
				<button type="submit">Log event</button>
			</form>
			<form class="stacked separator" onsubmit={(event) => {
				event.preventDefault();
				submitReminder();
			}}>
				<h3>Create reminder for selected person</h3>
				<label class="wide">
					<span>Title</span>
					<input bind:value={reminderForm.title} />
				</label>
				<label>
					<span>Due at</span>
					<input bind:value={reminderForm.due_at} type="datetime-local" />
				</label>
				<label>
					<span>Priority</span>
					<select bind:value={reminderForm.priority}>
						<option>Low</option>
						<option>Normal</option>
						<option>High</option>
					</select>
				</label>
				<label class="wide">
					<span>Notes</span>
					<textarea bind:value={reminderForm.notes} rows="3"></textarea>
				</label>
				<button type="submit">Create reminder</button>
			</form>
			<form class="stacked separator" onsubmit={(event) => {
				event.preventDefault();
				submitTag();
			}}>
				<h3>Add tag to selected person</h3>
				<label class="wide">
					<span>Tag</span>
					<input bind:value={tagForm.name} placeholder="friend, investor, supplier" />
				</label>
				<button type="submit">Add tag</button>
			</form>
			<form class="stacked separator" onsubmit={(event) => {
				event.preventDefault();
				submitLocation();
			}}>
				<h3>Add location to selected person</h3>
				<label>
					<span>City</span>
					<input bind:value={locationForm.city} />
				</label>
				<label>
					<span>Region</span>
					<input bind:value={locationForm.region} />
				</label>
				<label>
					<span>Country</span>
					<input bind:value={locationForm.country} />
				</label>
				<button type="submit">Add location</button>
			</form>
			<form class="stacked separator" onsubmit={(event) => {
				event.preventDefault();
				submitRole();
			}}>
				<h3>Link selected person to organization</h3>
				<label class="wide">
					<span>Organization</span>
					<select bind:value={roleForm.organization_id}>
						{#each organizations as organization (organization.id)}
							<option value={organization.id}>{organization.name}</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Title</span>
					<input bind:value={roleForm.title} />
				</label>
				<label>
					<span>Role type</span>
					<input bind:value={roleForm.role_type} />
				</label>
				<button type="submit">Add role</button>
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
	.workspace {
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
	.meta {
		font-size: 0.74rem;
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

	h3 {
		margin: 0;
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.toolbar {
		grid-template-columns: minmax(0, 1.4fr) 10rem 10rem auto;
		align-items: end;
		margin-top: 1rem;
	}

	.workspace {
		grid-template-columns: 1.2fr 1fr 0.95fr;
		margin-top: 1rem;
	}

	.panel {
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
		box-shadow: var(--shadow);
		min-height: 34rem;
	}

	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 0.85rem;
		border-bottom: 1px solid var(--line);
	}

	.panel-header span,
	label span,
	.field span {
		display: block;
		font-size: 0.72rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--muted);
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

	.table {
		display: grid;
		padding: 0 0.85rem 0.85rem;
	}

	.row {
		display: grid;
		grid-template-columns: 1.5fr 0.8fr 0.9fr;
		gap: 0.75rem;
		padding: 0.7rem 0;
		border-bottom: 1px solid var(--line);
		text-align: left;
		background: transparent;
	}

	.row.head {
		font-size: 0.72rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--muted);
	}

	.person-row {
		border-left: 0;
		border-right: 0;
		border-top: 0;
		padding-left: 0;
		padding-right: 0;
	}

	.person-row.selected {
		background: #efefec;
	}

	.detail-grid,
	form {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.85rem;
		padding: 0.85rem;
	}

	.stacked {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.separator {
		border-top: 1px solid var(--line);
	}

	.field {
		display: grid;
		gap: 0.3rem;
		padding-bottom: 0.8rem;
		border-bottom: 1px solid var(--line);
	}

	.field strong,
	.field p,
	.field ul {
		margin: 0;
	}

	.field ul {
		padding-left: 1rem;
	}

	.reminder-list {
		list-style: none;
		padding-left: 0;
	}

	.reminder-list li {
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
		padding: 0.55rem 0;
		border-bottom: 1px solid var(--line);
	}

	.inline-actions {
		display: flex;
		gap: 0.4rem;
		align-items: start;
	}

	.inline-actions button {
		width: auto;
	}

	.wide {
		grid-column: 1 / -1;
	}

	.notice,
	.empty {
		margin-top: 1rem;
		padding: 0.85rem;
		border: 1px solid var(--line-strong);
		background: var(--panel-strong);
	}

	@media (max-width: 960px) {
		.toolbar,
		.workspace,
		.row,
		.detail-grid,
		form {
			grid-template-columns: 1fr;
		}
	}
</style>
