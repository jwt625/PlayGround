<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		addOrganizationLocation,
		addOrganizationTag,
		getOrganization,
		listOrganizations,
		type Organization,
		type OrganizationDetail
	} from '$lib/api';

	let loading = $state(true);
	let error = $state('');
	let query = $state('');
	let industry = $state('');
	let organizations = $state<Organization[]>([]);
	let selectedId = $state('');
	let selectedOrganizationDetail = $state<OrganizationDetail | null>(null);

	let tagForm = $state({ name: '' });
	let locationForm = $state({ city: '', region: '', country: '' });

	const selectedOrganization = $derived(
		organizations.find((organization) => organization.id === selectedId) ?? organizations[0]
	);

	$effect(() => {
		if (selectedId) {
			void loadSelectedOrganization(selectedId);
		}
	});

	onMount(loadOrganizations);

	async function loadOrganizations() {
		loading = true;
		error = '';
		try {
			organizations = await listOrganizations({ q: query, limit: 100, industry: industry || undefined });
			if (!selectedId && organizations[0]) {
				selectedId = organizations[0].id;
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load organizations.';
		} finally {
			loading = false;
		}
	}

	async function loadSelectedOrganization(organizationId: string) {
		try {
			selectedOrganizationDetail = await getOrganization(organizationId);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load organization detail.';
		}
	}

	async function submitTag() {
		if (!selectedId || !tagForm.name) return;
		try {
			selectedOrganizationDetail = await addOrganizationTag(selectedId, { name: tagForm.name });
			tagForm = { name: '' };
			await loadOrganizations();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to add tag.';
		}
	}

	async function submitLocation() {
		if (!selectedId) return;
		try {
			selectedOrganizationDetail = await addOrganizationLocation(selectedId, {
				location: {
					label: null,
					city: locationForm.city || null,
					region: locationForm.region || null,
					country: locationForm.country || null,
					address_line: null,
					latitude: null,
					longitude: null,
					location_type: 'Work',
					notes: null
				},
				is_primary: true
			});
			locationForm = { city: '', region: '', country: '' };
			await loadOrganizations();
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to add location.';
		}
	}
</script>

<svelte:head>
	<title>Organizations | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Organizations</h1>
		</div>
		<p class="meta">{organizations.length} records</p>
	</header>

	<section class="toolbar">
		<label>
			<span>Search</span>
			<input bind:value={query} placeholder="Name, industry, location, notes" />
		</label>
		<label>
			<span>Industry</span>
			<input bind:value={industry} placeholder="Filter industry" />
		</label>
		<button type="button" onclick={loadOrganizations}>Refresh</button>
	</section>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	<section class="workspace">
		<section class="panel">
			<div class="panel-header">
				<h2>Directory</h2>
				<span>{loading ? 'Loading…' : 'Live'}</span>
			</div>
			<div class="table">
				<div class="row head">
					<span>Name</span>
					<span>Type</span>
					<span>Industry</span>
				</div>
				{#each organizations as organization (organization.id)}
					<button
						class:selected={selectedOrganization?.id === organization.id}
						class="row item"
						onclick={() => (selectedId = organization.id)}
					>
						<span>{organization.name}</span>
						<span>{organization.type}</span>
						<span>{organization.industry || '—'}</span>
					</button>
				{/each}
			</div>
		</section>

		<section class="panel">
			<div class="panel-header">
				<h2>Selected record</h2>
				<span>{selectedOrganization?.location || 'No location'}</span>
			</div>
			{#if selectedOrganizationDetail}
				<div class="detail-grid">
					<div class="field">
						<span>Name</span>
						<strong>{selectedOrganizationDetail.name}</strong>
					</div>
					<div class="field">
						<span>Type</span>
						<strong>{selectedOrganizationDetail.type}</strong>
					</div>
					<div class="field">
						<span>Industry</span>
						<strong>{selectedOrganizationDetail.industry || '—'}</strong>
					</div>
					<div class="field">
						<span>Location</span>
						<strong>{selectedOrganizationDetail.location || '—'}</strong>
					</div>
					<div class="field wide">
						<span>Website</span>
						<p>{selectedOrganizationDetail.website || 'No website recorded.'}</p>
					</div>
					<div class="field wide">
						<span>Description</span>
						<p>{selectedOrganizationDetail.description || 'No description recorded.'}</p>
					</div>
					<div class="field wide">
						<span>Tags</span>
						<p>{selectedOrganizationDetail.tags.map((item) => item.tag.name).join(', ') || 'No tags yet.'}</p>
					</div>
					<div class="field wide">
						<span>Locations</span>
						<ul>
							{#if selectedOrganizationDetail.locations.length}
								{#each selectedOrganizationDetail.locations as item (item.id)}
									<li>{item.location.city || item.location.label || '—'} {item.location.country || ''}</li>
								{/each}
							{:else}
								<li>No structured locations yet.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>People</span>
						<ul>
							{#if selectedOrganizationDetail.people.length}
								{#each selectedOrganizationDetail.people as role (role.id)}
									<li>{role.title || role.role_type || 'Role'} · {role.organization_name}</li>
								{/each}
							{:else}
								<li>No linked people yet.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Pipeline items</span>
						<ul>
							{#if selectedOrganizationDetail.pipeline_items.length}
								{#each selectedOrganizationDetail.pipeline_items as item (item.id)}
									<li>{item.title}</li>
								{/each}
							{:else}
								<li>No pipeline items.</li>
							{/if}
						</ul>
					</div>
					<div class="field wide">
						<span>Notes</span>
						<p>{selectedOrganizationDetail.notes || 'No notes yet.'}</p>
					</div>
					<form class="wide inline-form" onsubmit={(event) => {
						event.preventDefault();
						submitTag();
					}}>
						<label>
							<span>Add tag</span>
							<input bind:value={tagForm.name} placeholder="Supplier, school, partner" />
						</label>
						<button type="submit">Add tag</button>
					</form>
					<form class="wide tri-form" onsubmit={(event) => {
						event.preventDefault();
						submitLocation();
					}}>
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
						<button type="submit">Add primary location</button>
					</form>
				</div>
			{:else}
				<p class="notice">No organization selected.</p>
			{/if}
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

	.toolbar {
		grid-template-columns: minmax(0, 1fr) 12rem auto;
		align-items: end;
		margin-top: 1rem;
	}

	.workspace {
		grid-template-columns: 1.1fr 1fr;
		margin-top: 1rem;
	}

	.brand,
	.meta,
	.panel-header span,
	label span,
	.row.head,
	.field span {
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
		grid-template-columns: 1.3fr 0.7fr 1fr;
		gap: 0.75rem;
		padding: 0.7rem 0;
		border-bottom: 1px solid var(--line);
		background: transparent;
		text-align: left;
	}

	.item {
		border-left: 0;
		border-right: 0;
		border-top: 0;
		padding-left: 0;
		padding-right: 0;
	}

	.item.selected {
		background: #efefec;
	}

	.detail-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.85rem;
		padding: 0.85rem;
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

	.wide {
		grid-column: 1 / -1;
	}

	label {
		display: grid;
		gap: 0.35rem;
	}

	input,
	button {
		width: 100%;
		border: 1px solid var(--line);
		background: var(--panel);
		padding: 0.62rem 0.72rem;
		color: var(--text);
	}

	.inline-form {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 0.75rem;
		align-items: end;
	}

	.tri-form {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
		gap: 0.75rem;
		align-items: end;
	}

	.notice {
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
		.inline-form,
		.tri-form {
			grid-template-columns: 1fr;
		}
	}
</style>
