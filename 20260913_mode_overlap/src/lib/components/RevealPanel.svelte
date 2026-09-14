<script lang="ts">
	import { formatLossDB, mismatchLossDB } from '$lib/engine';
	import type { Explanation } from '$lib/explain';
	import type { AnswerRecord, CachedQuestion } from '$lib/types';

	interface Props {
		question: CachedQuestion;
		record: AnswerRecord;
		explanation: Explanation;
		expanded?: boolean;
	}

	let { question, record, explanation, expanded = $bindable(false) }: Props = $props();

	let trueLoss = $derived(formatLossDB(mismatchLossDB(record.trueEta)));
	let guessLoss = $derived(formatLossDB(mismatchLossDB(record.guessEta)));
	let exactZero = $derived(record.trueEta === 0);
</script>

<div class="reveal">
	<div class="verdict">
		<div class="cell">
			<span class="label">Your η</span>
			<span class="value">{record.guessEta.toFixed(4)}</span>
			<span class="sub">{guessLoss}</span>
		</div>
		<div class="cell">
			<span class="label">True η</span>
			<span class="value">{record.trueEta.toFixed(4)}</span>
			<span class="sub">{trueLoss}</span>
		</div>
		<div class="cell">
			<span class="label">Signed error</span>
			<span class="value" class:text-error={record.signedEtaError < 0} class:text-success={record.signedEtaError > 0}>
				{record.signedEtaError >= 0 ? '+' : ''}{record.signedEtaError.toFixed(4)}
			</span>
			<span class="sub">η, guess − true</span>
		</div>
		<div class="cell">
			<span class="label">dB error</span>
			<span class="value">{record.signedDbError >= 0 ? '+' : ''}{record.signedDbError.toFixed(3)}</span>
			<span class="sub">60 dB floor for scoring</span>
		</div>
		<div class="cell">
			<span class="label">Score</span>
			<span class="value">{record.score.toFixed(1)}</span>
			<span class="sub">100·exp(−|ΔdB|/3)</span>
		</div>
	</div>

	<p class="dominant">{explanation.sentence}</p>

	{#if exactZero}
		<p class="note">
			Exact zero overlap: mismatch loss is infinite (∞). The 60 dB floor above is a scoring
			convention only, never the physical answer.
		</p>
	{:else}
		<p class="note">
			Mode-mismatch loss assumes a single forward guided mode and no reflections. It is not an
			exact interface insertion loss.
		</p>
	{/if}

	<button class="toggle" onclick={() => (expanded = !expanded)}>
		{expanded ? 'Hide' : 'Show'} counterfactual breakdown
	</button>
	{#if expanded}
		<table class="candidates">
			<thead>
				<tr><th>Effect removed</th><th>η</th><th>Improvement</th></tr>
			</thead>
			<tbody>
				{#each explanation.candidates as candidate}
					<tr>
						<td>{candidate.effect}</td>
						<td class="mono">{candidate.eta.toFixed(4)}</td>
						<td class="mono">+{candidate.improvement.toFixed(4)}</td>
					</tr>
				{/each}
				{#if explanation.candidates.length === 0}
					<tr><td colspan="3" class="muted">No single perturbation improves η.</td></tr>
				{/if}
			</tbody>
		</table>
	{/if}

	<p class="identifier">Question {question.id} · {question.category}</p>
</div>

<style>
	.reveal {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		padding: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.verdict {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
		gap: var(--spacing-md);
	}

	.cell {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.label {
		font-size: 0.7rem;
		text-transform: uppercase;
		color: var(--text-secondary);
	}

	.value {
		font-family: var(--font-mono);
		font-size: 1.15rem;
	}

	.sub {
		font-family: var(--font-mono);
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.dominant {
		font-size: 0.95rem;
	}

	.note {
		font-size: 0.78rem;
		color: var(--text-secondary);
	}

	.toggle {
		align-self: flex-start;
		font-size: 0.8rem;
		padding: 4px 10px;
	}

	.candidates {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
	}

	.candidates th,
	.candidates td {
		text-align: left;
		padding: 3px 8px;
		border-bottom: 1px solid var(--border);
	}

	.mono {
		font-family: var(--font-mono);
	}

	.muted {
		color: var(--text-muted);
	}

	.identifier {
		font-family: var(--font-mono);
		font-size: 0.7rem;
		color: var(--text-muted);
	}
</style>
