<script lang="ts">
	import Frame from './Frame.svelte';
	import { store } from './state.svelte';
	import { EM_DASH } from './logic';
	import { link } from './paths';
	const sims = $derived(store.atlas?.sims ?? []);
</script>

<Frame letter="j" desc="Simulation configs and recorded validation status. Open a config to run cross-section stages; complete device reproduction is pending. No solver output is stored." height={200} badge={`${sims.length}`} badgeTip="simulation configs available">
	<div class="sc">
		<table>
			<thead>
				<tr><th>Config</th><th>Paper</th><th>Repro</th><th>Validation</th><th class="r">Targets</th><th class="r" title="Complete device reproduction has not been evaluated">Score</th></tr>
			</thead>
			<tbody>
				{#each sims as s (s.id)}
					<tr>
						<td title={s.title ?? ''}><a href={link(`/sim?id=${encodeURIComponent(s.id)}`)}>{s.id}</a></td>
						<td>{s.paper_id}</td>
						<td>{s.repro_grade ?? EM_DASH}</td>
						<td>{s.validation_status ?? EM_DASH}</td>
						<td class="r num">{s.n_targets}</td>
						<td class="r muted" title="Complete device reproduction pending">{EM_DASH}</td>
					</tr>
				{/each}
				{#if !sims.length}<tr><td colspan="6" class="muted">no simulation configs</td></tr>{/if}
			</tbody>
		</table>
	</div>
</Frame>

<style>
	.sc {
		height: 100%;
		overflow: auto;
	}
	table {
		width: 100%;
		border-collapse: collapse;
	}
	th {
		text-align: left;
		color: var(--ink-3);
		font-weight: 600;
		padding: 3px 8px;
		border-bottom: 1px solid var(--line);
		position: sticky;
		top: 0;
		background: var(--surface);
	}
	td {
		padding: 3px 8px;
		border-bottom: 1px solid var(--grid);
	}
	.r {
		text-align: right;
	}
</style>
