<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		createPipelineItem,
		getPipeline,
		listOrganizations,
		listPeople,
		listPipelines,
		movePipelineItem,
		type Organization,
		type Person,
		type Pipeline,
		type PipelineDetail
	} from '$lib/api';

	let loading = $state(true);
	let error = $state('');
	let pipelines = $state<Pipeline[]>([]);
	let people = $state<Person[]>([]);
	let organizations = $state<Organization[]>([]);
	let selectedId = $state('');
	let selectedPipeline = $state<PipelineDetail | null>(null);

	let form = $state({
		title: '',
		stage_id: '',
		primary_person_id: '',
		primary_organization_id: ''
	});

	$effect(() => {
		if (selectedId) {
			void loadPipelineDetail(selectedId);
		}
	});

	onMount(async () => {
		await Promise.all([loadPipelines(), loadPeople(), loadOrganizations()]);
	});

	async function loadPipelines() {
		loading = true;
		error = '';
		try {
			pipelines = await listPipelines();
			if (!selectedId && pipelines[0]) {
				selectedId = pipelines[0].id;
				form.stage_id = pipelines[0].stages[0]?.id || '';
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load pipelines.';
		} finally {
			loading = false;
		}
	}

	async function loadPipelineDetail(pipelineId: string) {
		try {
			selectedPipeline = await getPipeline(pipelineId);
			if (!form.stage_id) {
				form.stage_id = selectedPipeline.stages[0]?.id || '';
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load pipeline detail.';
		}
	}

	async function loadPeople() {
		people = await listPeople({ limit: 100 });
	}

	async function loadOrganizations() {
		organizations = await listOrganizations({ limit: 100 });
	}

	async function submitItem() {
		if (!selectedId || !form.title || !form.stage_id) return;
		try {
			await createPipelineItem(selectedId, {
				title: form.title,
				stage_id: form.stage_id,
				primary_person_id: form.primary_person_id || undefined,
				primary_organization_id: form.primary_organization_id || undefined
			});
			form = {
				title: '',
				stage_id: selectedPipeline?.stages[0]?.id || '',
				primary_person_id: '',
				primary_organization_id: ''
			};
			await loadPipelineDetail(selectedId);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to create pipeline item.';
		}
	}

	async function advance(itemId: string, currentStageId: string) {
		if (!selectedPipeline) return;
		const stages = [...selectedPipeline.stages].sort((a, b) => a.sort_order - b.sort_order);
		const currentIndex = stages.findIndex((stage) => stage.id === currentStageId);
		const nextStage = stages[currentIndex + 1];
		if (!nextStage) return;
		try {
			await movePipelineItem(itemId, nextStage.id);
			await loadPipelineDetail(selectedPipeline.id);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to move pipeline item.';
		}
	}

	function stageItems(stageId: string) {
		return selectedPipeline?.items.filter((item) => item.stage_id === stageId) || [];
	}
</script>

<svelte:head>
	<title>Pipelines | Kizuna</title>
</svelte:head>

<main class="shell">
	<header class="topbar">
		<div>
			<a class="brand" href={resolve('/')}>Kizuna</a>
			<h1>Pipelines</h1>
		</div>
		<p class="meta">{selectedPipeline?.items.length || 0} active items</p>
	</header>

	{#if error}
		<p class="notice">{error}</p>
	{/if}

	<section class="toolbar">
		<label>
			<span>Pipeline</span>
			<select bind:value={selectedId}>
				{#each pipelines as pipeline (pipeline.id)}
					<option value={pipeline.id}>{pipeline.name}</option>
				{/each}
			</select>
		</label>
		<button type="button" onclick={loadPipelines}>Refresh</button>
	</section>

	{#if selectedPipeline}
		<section class="workspace">
			<section class="panel board">
				<div class="panel-header">
					<h2>Board</h2>
					<span>{loading ? 'Loading…' : selectedPipeline.template_type}</span>
				</div>
				<div class="columns">
					{#each [...selectedPipeline.stages].sort((a, b) => a.sort_order - b.sort_order) as stage (stage.id)}
						<section class="stage">
							<header>
								<strong>{stage.name}</strong>
								<span>{stageItems(stage.id).length}</span>
							</header>
							<ul>
								{#each stageItems(stage.id) as item (item.id)}
									<li>
										<div>
											<strong>{item.title}</strong>
											<p>{item.priority}</p>
										</div>
										<button type="button" onclick={() => advance(item.id, item.stage_id)}>Advance</button>
									</li>
								{/each}
							</ul>
						</section>
					{/each}
				</div>
			</section>

			<section class="panel">
				<div class="panel-header">
					<h2>New item</h2>
					<span>Quick add</span>
				</div>
				<form onsubmit={(event) => {
					event.preventDefault();
					submitItem();
				}}>
					<label class="wide">
						<span>Title</span>
						<input bind:value={form.title} />
					</label>
					<label>
						<span>Stage</span>
						<select bind:value={form.stage_id}>
							{#each selectedPipeline.stages as stage (stage.id)}
								<option value={stage.id}>{stage.name}</option>
							{/each}
						</select>
					</label>
					<label>
						<span>Person</span>
						<select bind:value={form.primary_person_id}>
							<option value="">None</option>
							{#each people as person (person.id)}
								<option value={person.id}>{person.display_name}</option>
							{/each}
						</select>
					</label>
					<label>
						<span>Organization</span>
						<select bind:value={form.primary_organization_id}>
							<option value="">None</option>
							{#each organizations as organization (organization.id)}
								<option value={organization.id}>{organization.name}</option>
							{/each}
						</select>
					</label>
					<button type="submit">Create item</button>
				</form>
			</section>
		</section>
	{/if}
</main>

<style>
	.shell { min-height: 100vh; padding: 1.25rem; }
	.topbar, .toolbar, .workspace, form { display: grid; gap: 1rem; }
	.topbar { grid-template-columns: minmax(0, 1fr) auto; align-items: end; padding-bottom: 1rem; border-bottom: 1px solid var(--line-strong); }
	.toolbar { grid-template-columns: 18rem auto; align-items: end; margin-top: 1rem; }
	.workspace { grid-template-columns: 1.5fr 0.8fr; margin-top: 1rem; }
	.brand, .meta, .panel-header span, label span { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); text-decoration: none; }
	h1 { margin: 0.35rem 0 0; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -0.05em; }
	h2 { margin: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; }
	.panel, .notice { border: 1px solid var(--line-strong); background: var(--panel-strong); box-shadow: var(--shadow); }
	.panel-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0.85rem; border-bottom: 1px solid var(--line); }
	.columns { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; padding: 0.85rem; overflow-x: auto; }
	.stage { border: 1px solid var(--line); background: var(--panel); min-height: 18rem; }
	.stage header { display: flex; justify-content: space-between; padding: 0.65rem 0.75rem; border-bottom: 1px solid var(--line); }
	.stage ul { list-style: none; margin: 0; padding: 0.25rem 0.75rem 0.75rem; }
	.stage li { display: grid; gap: 0.5rem; padding: 0.75rem 0; border-bottom: 1px solid var(--line); }
	.stage li strong, .stage li p { display: block; margin: 0; }
	.stage li p { margin-top: 0.2rem; color: var(--muted); }
	form { grid-template-columns: 1fr; padding: 0.85rem; }
	label { display: grid; gap: 0.35rem; }
	input, select, button { width: 100%; border: 1px solid var(--line); background: var(--panel); padding: 0.62rem 0.72rem; color: var(--text); }
	button { cursor: pointer; }
	.wide { grid-column: 1 / -1; }
	.notice { margin-top: 1rem; padding: 0.85rem; }
	@media (max-width: 960px) { .toolbar, .workspace, .columns { grid-template-columns: 1fr; } }
</style>
