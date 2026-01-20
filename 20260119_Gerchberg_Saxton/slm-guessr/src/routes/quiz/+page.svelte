<script lang="ts">
	import type { QuizMode, Difficulty } from '$lib/types';

	let gameStarted = $state(false);
	let selectedMode = $state<QuizMode>('phase-to-intensity');
	let selectedDifficulty = $state<Difficulty>('easy');

	function startQuiz() {
		gameStarted = true;
		// TODO: Initialize quiz with samples from manifest
	}

	function resetQuiz() {
		gameStarted = false;
	}
</script>

<div class="quiz">
	<header class="page-header">
		<h1>SLM-Guessr Quiz</h1>
		<p class="subtitle">Test your phase-intensity pattern recognition</p>
	</header>

	{#if !gameStarted}
		<div class="setup">
			<section class="option-group">
				<h2>Mode</h2>
				<div class="options">
					<button
						class="option-btn"
						class:active={selectedMode === 'phase-to-intensity'}
						onclick={() => selectedMode = 'phase-to-intensity'}
					>
						<span class="option-title">Phase to Intensity</span>
						<span class="option-desc">Given phase mask, predict intensity</span>
					</button>
					<button
						class="option-btn"
						class:active={selectedMode === 'intensity-to-phase'}
						onclick={() => selectedMode = 'intensity-to-phase'}
					>
						<span class="option-title">Intensity to Phase</span>
						<span class="option-desc">Given intensity, predict phase mask</span>
					</button>
				</div>
			</section>

			<section class="option-group">
				<h2>Difficulty</h2>
				<div class="options difficulty-options">
					<button
						class="option-btn small"
						class:active={selectedDifficulty === 'easy'}
						onclick={() => selectedDifficulty = 'easy'}
					>
						<span class="option-title">Easy</span>
						<span class="option-desc">L1-L2</span>
					</button>
					<button
						class="option-btn small"
						class:active={selectedDifficulty === 'medium'}
						onclick={() => selectedDifficulty = 'medium'}
					>
						<span class="option-title">Medium</span>
						<span class="option-desc">L3-L5</span>
					</button>
					<button
						class="option-btn small"
						class:active={selectedDifficulty === 'hard'}
						onclick={() => selectedDifficulty = 'hard'}
					>
						<span class="option-title">Hard</span>
						<span class="option-desc">L6-L7</span>
					</button>
				</div>
			</section>

			<div class="start-section">
				<button class="start-btn primary" onclick={startQuiz}>
					Start Quiz
				</button>
				<p class="hint">10 questions per round</p>
			</div>
		</div>
	{:else}
		<div class="game">
			<div class="game-placeholder">
				<p>Quiz game will be implemented once samples are generated</p>
				<button onclick={resetQuiz}>Back to Setup</button>
			</div>
		</div>
	{/if}
</div>

<style>
	.quiz {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xl);
	}

	.page-header h1 {
		margin-bottom: var(--spacing-xs);
	}

	.subtitle {
		color: var(--text-secondary);
	}

	.setup {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xl);
		max-width: 600px;
	}

	.option-group h2 {
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-md);
	}

	.options {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.difficulty-options {
		flex-direction: row;
	}

	.option-btn {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--spacing-xs);
		padding: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		text-align: left;
		cursor: pointer;
		transition: border-color 0.15s, background-color 0.15s;
	}

	.option-btn:hover {
		background-color: var(--bg-tertiary);
	}

	.option-btn.active {
		border-color: var(--accent);
		background-color: var(--bg-tertiary);
	}

	.option-btn.small {
		flex: 1;
		align-items: center;
		text-align: center;
	}

	.option-title {
		font-weight: 500;
		color: var(--text-primary);
	}

	.option-desc {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	.start-section {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--spacing-sm);
	}

	.start-btn {
		padding: var(--spacing-md) var(--spacing-xl);
		font-size: 1rem;
	}

	.hint {
		font-size: 0.85rem;
		color: var(--text-muted);
	}

	.game-placeholder {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-md);
		min-height: 300px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-secondary);
	}
</style>

