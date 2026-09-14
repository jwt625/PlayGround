<script lang="ts">
	import { mismatchLossDB, formatLossDB } from '$lib/engine';
	import { INTERVALS } from '$lib/scoring';
	import type { AnswerMode } from '$lib/types';

	interface Props {
		mode: AnswerMode;
		disabled: boolean;
		onSubmitContinuous: (guessEta: number) => void;
		onSubmitInterval: (intervalIndex: number) => void;
	}

	let { mode, disabled, onSubmitContinuous, onSubmitInterval }: Props = $props();

	let guess = $state(0.5);
	let fine = $state(false);
	let selectedInterval = $state<number | null>(null);

	let step = $derived(fine ? 0.0001 : 0.001);
	let lossText = $derived(formatLossDB(mismatchLossDB(Math.min(1, Math.max(0, guess)))));

	function submitContinuous() {
		if (disabled) return;
		onSubmitContinuous(Math.min(1, Math.max(0, guess)));
	}

	function submitInterval() {
		if (disabled || selectedInterval === null) return;
		onSubmitInterval(selectedInterval);
	}
</script>

{#if mode === 'continuous'}
	<div class="answer continuous">
		<div class="row">
			<label for="eta-range">η guess</label>
			<input
				id="eta-range"
				type="range"
				min="0"
				max="1"
				{step}
				bind:value={guess}
				disabled={disabled}
			/>
			<input
				class="number"
				type="number"
				min="0"
				max="1"
				{step}
				bind:value={guess}
				disabled={disabled}
				aria-label="eta numeric input"
			/>
		</div>
		<div class="row">
			<label class="fine">
				<input type="checkbox" bind:checked={fine} disabled={disabled} />
				fine step (1e-4)
			</label>
			<span class="readout">η = {Math.min(1, Math.max(0, guess)).toFixed(4)} · {lossText}</span>
		</div>
		<button class="primary" onclick={submitContinuous} disabled={disabled}>Submit</button>
		<p class="hint">Enter submits; arrow keys adjust the slider.</p>
	</div>
{:else if mode === 'interval'}
	<div class="answer interval">
		<div class="intervals">
			{#each INTERVALS as interval}
				<button
					class="chip"
					class:active={selectedInterval === interval.index}
					onclick={() => (selectedInterval = interval.index)}
					disabled={disabled}
				>
					{interval.label}
				</button>
			{/each}
		</div>
		<button class="primary" onclick={submitInterval} disabled={disabled || selectedInterval === null}>
			Submit interval
		</button>
		<p class="hint">Interval scoring is zero when the true η falls inside the chosen bin.</p>
	</div>
{:else}
	<div class="answer">
		<p class="hint">Ranking is available in the Practice ranking drill.</p>
	</div>
{/if}

<style>
	.answer {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		padding: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.row {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	label {
		font-family: var(--font-mono);
		font-size: 0.8rem;
		color: var(--text-secondary);
	}

	input[type='range'] {
		flex: 1;
		accent-color: var(--accent);
	}

	.number {
		width: 96px;
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
		color: var(--text-primary);
		padding: 4px 6px;
		font-family: var(--font-mono);
	}

	.fine {
		display: flex;
		align-items: center;
		gap: var(--spacing-xs);
	}

	.readout {
		font-family: var(--font-mono);
		font-size: 0.85rem;
		color: var(--accent);
	}

	.intervals {
		display: flex;
		flex-wrap: wrap;
		gap: var(--spacing-xs);
	}

	.chip {
		font-family: var(--font-mono);
		font-size: 0.8rem;
	}

	.chip.active {
		border-color: var(--accent);
		color: var(--accent);
	}

	button.primary {
		align-self: flex-start;
	}

	.hint {
		font-size: 0.75rem;
		color: var(--text-muted);
	}
</style>
