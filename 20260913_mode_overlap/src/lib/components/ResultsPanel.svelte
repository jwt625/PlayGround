<script lang="ts">
	import type { SessionResults } from '$lib/types';

	interface Props {
		results: SessionResults;
		shareId?: string;
		onRestart: () => void;
		onExit: () => void;
		onExport: () => void;
	}

	let { results, shareId, onRestart, onExit, onExport }: Props = $props();

	const bucketLabels = ['[0,.1)', '[.1,.3)', '[.3,.6)', '[.6,.8)', '[.8,.95)', '[.95,.99)', '[.99,1]'];
</script>

<div class="results">
	<div class="score-card">
		<span class="score">{results.score.toFixed(1)}</span>
		<span class="score-label">mean score</span>
		{#if shareId}<span class="share mono">{shareId}</span>{/if}
	</div>

	<div class="metrics">
		<div class="metric"><span class="m-value">{results.mae.toFixed(4)}</span><span class="m-label">MAE η</span></div>
		<div class="metric"><span class="m-value">{results.medianAbsEtaError.toFixed(4)}</span><span class="m-label">median |Δη|</span></div>
		<div class="metric"><span class="m-value">{results.rmse.toFixed(4)}</span><span class="m-label">RMSE η</span></div>
		<div class="metric"><span class="m-value">{results.meanDbError.toFixed(3)}</span><span class="m-label">mean dB error</span></div>
		<div class="metric"><span class="m-value">{results.meanDurationMs.toFixed(0)}</span><span class="m-label">mean ms</span></div>
		<div class="metric"><span class="m-value">{results.medianDurationMs.toFixed(0)}</span><span class="m-label">median ms</span></div>
	</div>

	<div class="tables">
		<div class="table-block">
			<h3>By category</h3>
			<table>
				<thead><tr><th>Category</th><th>n</th><th>MAE</th><th>Score</th></tr></thead>
				<tbody>
					{#each results.categoryBreakdown as row}
						<tr>
							<td>{row.category}</td>
							<td class="mono">{row.count}</td>
							<td class="mono">{row.mae.toFixed(4)}</td>
							<td class="mono">{row.score.toFixed(1)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<div class="table-block">
			<h3>Signed bias by η bucket</h3>
			<table>
				<thead><tr><th>Bucket</th><th>n</th><th>Bias</th></tr></thead>
				<tbody>
					{#each results.biasByBucket as row}
						<tr>
							<td class="mono">{bucketLabels[row.bucket]}</td>
							<td class="mono">{row.count}</td>
							<td class="mono">{row.signedBias >= 0 ? '+' : ''}{row.signedBias.toFixed(4)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="hint">Categories with fewer than five samples are not a confident weakness claim.</p>
		</div>
	</div>

	<div class="actions">
		<button class="primary" onclick={onRestart}>Play again</button>
		<button onclick={onExit}>Change setup</button>
		<button onclick={onExport}>Export JSON</button>
	</div>
</div>

<style>
	.results {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.score-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-lg);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.score {
		font-family: var(--font-mono);
		font-size: 3.5rem;
		font-weight: 700;
		color: var(--accent);
		line-height: 1;
	}

	.score-label {
		font-size: 0.75rem;
		text-transform: uppercase;
		color: var(--text-secondary);
	}

	.share {
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.metrics {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
		gap: var(--spacing-sm);
	}

	.metric {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--spacing-sm);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.m-value {
		font-family: var(--font-mono);
		font-size: 1.1rem;
	}

	.m-label {
		font-size: 0.7rem;
		text-transform: uppercase;
		color: var(--text-secondary);
	}

	.tables {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-lg);
	}

	.table-block h3 {
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-sm);
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.82rem;
	}

	th,
	td {
		text-align: left;
		padding: 4px 8px;
		border-bottom: 1px solid var(--border);
	}

	.mono {
		font-family: var(--font-mono);
	}

	.hint {
		margin-top: var(--spacing-sm);
		font-size: 0.72rem;
		color: var(--text-muted);
	}

	.actions {
		display: flex;
		gap: var(--spacing-md);
	}

	@media (max-width: 700px) {
		.tables {
			grid-template-columns: 1fr;
		}
	}
</style>
