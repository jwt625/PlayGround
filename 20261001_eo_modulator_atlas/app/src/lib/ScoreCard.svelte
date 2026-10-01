<script lang="ts">
	import Frame from './Frame.svelte';
	import { store } from './state.svelte';
	import { EM_DASH } from './logic';
	const sims = $derived(store.atlas?.sims ?? []);
</script>

<Frame letter="j" desc="Simulation-reproduction scorecard: paper-reported targets versus engine output. The engine runner is pending; no solver output is stored." height={200} badge={`${sims.length}`} badgeTip="simulation configs available">
	<div class="sc">
		<table>
			<thead>
				<tr><th>Config</th><th>Paper</th><th>Repro</th><th>Validation</th><th class="r">Targets</th><th class="r" title="Fraction of targets within tolerance; filled once the engine runner exists">Score</th></tr>
			</thead>
			<tbody>
				{#each sims as s (s.id)}
					<tr>
						<td title={s.title ?? ''}>{s.id}</td>
						<td>{s.paper_id}</td>
						<td>{s.repro_grade ?? EM_DASH}</td>
						<td>{s.validation_status ?? EM_DASH}</td>
						<td class="r num">{s.n_targets}</td>
						<td class="r muted" title="Engine runner pending">{EM_DASH}</td>
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
