<script lang="ts">
	import { base } from '$app/paths';
	import type { QuizMode, Sample, SampleManifest } from '$lib/types';
	import GifPlayer from '$lib/components/GifPlayer.svelte';
	import { onMount } from 'svelte';

	// Constants
	const QUESTIONS_PER_ROUND = 5;
	const OPTIONS_PER_QUESTION = 4;
	const POINTS_CORRECT = 10;
	const STREAK_BONUS = 5;
	const MAX_STREAK_BONUS = 25;
	const EXCLUDED_SAMPLES = ['cubic_x', 'cubic_y'];

	// Game states
	type GamePhase = 'setup' | 'playing' | 'feedback' | 'results';

	interface QuizQuestion {
		correctSample: Sample;
		options: Sample[];
		correctIndex: number;
	}

	// State
	let gamePhase = $state<GamePhase>('setup');
	let selectedMode = $state<QuizMode>('phase-to-intensity');
	let allSamples = $state<Sample[]>([]);
	let questions = $state<QuizQuestion[]>([]);
	let currentQuestionIndex = $state(0);
	let score = $state(0);
	let streak = $state(0);
	let selectedAnswer = $state<number | null>(null);
	let answers = $state<boolean[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);

	// Derived
	let currentQuestion = $derived(questions[currentQuestionIndex]);
	let isLastQuestion = $derived(currentQuestionIndex === questions.length - 1);
	let streakBonus = $derived(Math.min(streak * STREAK_BONUS, MAX_STREAK_BONUS));

	onMount(async () => {
		try {
			const res = await fetch(`${base}/samples.json`);
			if (!res.ok) throw new Error(`Failed to load samples: ${res.status}`);
			const manifest: SampleManifest = await res.json();
			// Filter to L1-L2 and exclude cubic samples
			allSamples = manifest.samples.filter(
				s => (s.level === 1 || s.level === 2) && !EXCLUDED_SAMPLES.includes(s.id)
			);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	});

	function shuffleArray<T>(array: T[]): T[] {
		const shuffled = [...array];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}
		return shuffled;
	}

	function generateQuestions(): QuizQuestion[] {
		const shuffledSamples = shuffleArray(allSamples);
		const generatedQuestions: QuizQuestion[] = [];

		for (let i = 0; i < QUESTIONS_PER_ROUND; i++) {
			// Pick correct answer (cycle through shuffled samples)
			const correctSample = shuffledSamples[i % shuffledSamples.length];

			// Pick 3 wrong options (different from correct)
			const wrongOptions = shuffleArray(
				allSamples.filter(s => s.id !== correctSample.id)
			).slice(0, OPTIONS_PER_QUESTION - 1);

			// Combine and shuffle options
			const allOptions = shuffleArray([correctSample, ...wrongOptions]);
			const correctIndex = allOptions.findIndex(s => s.id === correctSample.id);

			generatedQuestions.push({
				correctSample,
				options: allOptions,
				correctIndex
			});
		}

		return generatedQuestions;
	}

	function startQuiz() {
		questions = generateQuestions();
		currentQuestionIndex = 0;
		score = 0;
		streak = 0;
		selectedAnswer = null;
		answers = [];
		gamePhase = 'playing';
	}

	function selectAnswer(index: number) {
		if (gamePhase !== 'playing' || selectedAnswer !== null) return;

		selectedAnswer = index;
		const isCorrect = index === currentQuestion.correctIndex;
		answers = [...answers, isCorrect];

		if (isCorrect) {
			const bonus = Math.min(streak * STREAK_BONUS, MAX_STREAK_BONUS);
			score += POINTS_CORRECT + bonus;
			streak += 1;
		} else {
			streak = 0;
		}

		gamePhase = 'feedback';
	}

	function nextQuestion() {
		if (isLastQuestion) {
			gamePhase = 'results';
		} else {
			currentQuestionIndex += 1;
			selectedAnswer = null;
			gamePhase = 'playing';
		}
	}

	function resetQuiz() {
		gamePhase = 'setup';
		questions = [];
		currentQuestionIndex = 0;
		score = 0;
		streak = 0;
		selectedAnswer = null;
		answers = [];
	}

	function resolvePath(path: string): string {
		return `${base}/${path}`;
	}

	function getQuestionGif(sample: Sample): string {
		// Question shows what we're asking FROM
		return selectedMode === 'phase-to-intensity'
			? resolvePath(sample.phase_gif)
			: resolvePath(sample.intensity_gif);
	}

	function getOptionGif(sample: Sample): string {
		// Options show what we're asking TO identify
		return selectedMode === 'phase-to-intensity'
			? resolvePath(sample.intensity_gif)
			: resolvePath(sample.phase_gif);
	}

	function getOptionClass(index: number): string {
		if (gamePhase !== 'feedback') return '';
		if (index === currentQuestion.correctIndex) return 'correct';
		if (index === selectedAnswer) return 'incorrect';
		return 'dimmed';
	}
</script>

<div class="quiz">
	<header class="page-header">
		<h1>SLM-Guessr Quiz</h1>
		<p class="subtitle">Test your phase-intensity pattern recognition</p>
	</header>

	{#if loading}
		<div class="loading-state">
			<p>Loading samples...</p>
		</div>
	{:else if error}
		<div class="error-state">
			<p class="text-error">{error}</p>
		</div>
	{:else if allSamples.length < OPTIONS_PER_QUESTION}
		<div class="error-state">
			<p>Not enough samples available for quiz</p>
			<p class="hint">Need at least {OPTIONS_PER_QUESTION} samples, found {allSamples.length}</p>
		</div>
	{:else if gamePhase === 'setup'}
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
						class="option-btn small active"
					>
						<span class="option-title">Easy</span>
						<span class="option-desc">L1-L2</span>
					</button>
					<button
						class="option-btn small disabled"
						disabled
					>
						<span class="option-title">Medium</span>
						<span class="option-desc">L3-L5</span>
						<span class="option-wip">WIP</span>
					</button>
					<button
						class="option-btn small disabled"
						disabled
					>
						<span class="option-title">Hard</span>
						<span class="option-desc">L6-L7</span>
						<span class="option-wip">WIP</span>
					</button>
				</div>
			</section>

			<div class="start-section">
				<button class="start-btn primary" onclick={startQuiz}>
					Start Quiz
				</button>
				<p class="hint">{QUESTIONS_PER_ROUND} questions per round</p>
			</div>
		</div>
	{:else if gamePhase === 'playing' || gamePhase === 'feedback'}
		<div class="game">
			<!-- Score bar -->
			<div class="score-bar">
				<div class="score-item">
					<span class="score-label">Question</span>
					<span class="score-value">{currentQuestionIndex + 1}/{questions.length}</span>
				</div>
				<div class="score-item">
					<span class="score-label">Score</span>
					<span class="score-value">{score}</span>
				</div>
				<div class="score-item">
					<span class="score-label">Streak</span>
					<span class="score-value">{streak}{streak > 0 ? ` (+${streakBonus})` : ''}</span>
				</div>
			</div>

			<!-- Question -->
			<div class="question-section">
				<p class="question-prompt">
					{#if selectedMode === 'phase-to-intensity'}
						Which intensity pattern does this phase mask produce?
					{:else}
						Which phase mask produces this intensity pattern?
					{/if}
				</p>
				<div class="question-image">
					<GifPlayer
						src={getQuestionGif(currentQuestion.correctSample)}
						alt={selectedMode === 'phase-to-intensity' ? 'Phase mask' : 'Intensity pattern'}
					/>
					<span class="image-label">
						{selectedMode === 'phase-to-intensity' ? 'Phase' : 'Intensity'}
					</span>
				</div>
			</div>

			<!-- Options -->
			<div class="options-section">
				<p class="options-prompt">Select the matching {selectedMode === 'phase-to-intensity' ? 'intensity' : 'phase'}:</p>
				<div class="options-grid">
					{#each currentQuestion.options as option, index}
						<button
							class="option-card {getOptionClass(index)}"
							onclick={() => selectAnswer(index)}
							disabled={gamePhase === 'feedback'}
						>
							<GifPlayer
								src={getOptionGif(option)}
								alt="Option {index + 1}"
							/>
							{#if gamePhase === 'feedback'}
								<span class="option-label">{option.name}</span>
							{/if}
						</button>
					{/each}
				</div>
			</div>

			<!-- Feedback / Next -->
			{#if gamePhase === 'feedback'}
				<div class="feedback-section">
					{#if selectedAnswer === currentQuestion.correctIndex}
						<p class="feedback-text correct">Correct! +{POINTS_CORRECT}{streak > 1 ? ` +${Math.min((streak - 1) * STREAK_BONUS, MAX_STREAK_BONUS)} streak bonus` : ''}</p>
					{:else}
						<p class="feedback-text incorrect">Incorrect. The answer was: {currentQuestion.correctSample.name}</p>
					{/if}
					<button class="next-btn primary" onclick={nextQuestion}>
						{isLastQuestion ? 'See Results' : 'Next Question'}
					</button>
				</div>
			{/if}
		</div>
	{:else if gamePhase === 'results'}
		<div class="results">
			<div class="results-card">
				<h2>Quiz Complete</h2>
				<div class="final-score">
					<span class="score-number">{score}</span>
					<span class="score-label">points</span>
				</div>
				<div class="results-stats">
					<div class="stat">
						<span class="stat-value">{answers.filter(a => a).length}/{answers.length}</span>
						<span class="stat-label">Correct</span>
					</div>
					<div class="stat">
						<span class="stat-value">{Math.round(answers.filter(a => a).length / answers.length * 100)}%</span>
						<span class="stat-label">Accuracy</span>
					</div>
				</div>
				<div class="results-actions">
					<button class="primary" onclick={startQuiz}>Play Again</button>
					<button onclick={resetQuiz}>Change Mode</button>
				</div>
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

	.loading-state,
	.error-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 200px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-secondary);
	}

	.hint {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-top: var(--spacing-sm);
	}

	/* Setup */
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

	.option-btn:hover:not(:disabled) {
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

	.option-btn.disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}

	.option-title {
		font-weight: 500;
		color: var(--text-primary);
	}

	.option-desc {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}

	.option-wip {
		font-size: 0.7rem;
		font-family: var(--font-mono);
		color: var(--text-muted);
		padding: 2px 6px;
		background-color: var(--bg-primary);
		border: 1px solid var(--border);
		margin-top: var(--spacing-xs);
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

	/* Game */
	.game {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.score-bar {
		display: flex;
		gap: var(--spacing-lg);
		padding: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.score-item {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.score-label {
		font-size: 0.75rem;
		text-transform: uppercase;
		color: var(--text-secondary);
	}

	.score-value {
		font-family: var(--font-mono);
		font-size: 1.1rem;
		color: var(--accent);
	}

	.question-section {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-md);
	}

	.question-prompt {
		font-size: 1.1rem;
		text-align: center;
	}

	.question-image {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-xs);
		width: 256px;
	}

	.image-label {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		color: var(--text-secondary);
	}

	.options-section {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-md);
	}

	.options-prompt {
		font-size: 0.9rem;
		color: var(--text-secondary);
	}

	.options-grid {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: var(--spacing-md);
	}

	@media (max-width: 800px) {
		.options-grid {
			grid-template-columns: repeat(2, 1fr);
		}
	}

	.option-card {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		padding: var(--spacing-sm);
		background-color: var(--bg-secondary);
		border: 2px solid var(--border);
		cursor: pointer;
		transition: border-color 0.15s, opacity 0.15s;
	}

	.option-card:hover:not(:disabled) {
		border-color: var(--accent);
	}

	.option-card:disabled {
		cursor: default;
	}

	.option-card.correct {
		border-color: var(--success);
		background-color: rgba(74, 222, 128, 0.1);
	}

	.option-card.incorrect {
		border-color: var(--error);
		background-color: rgba(248, 113, 113, 0.1);
	}

	.option-card.dimmed {
		opacity: 0.5;
	}

	.option-label {
		font-size: 0.75rem;
		color: var(--text-secondary);
		text-align: center;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.feedback-section {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-lg);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
	}

	.feedback-text {
		font-size: 1.1rem;
		font-weight: 500;
	}

	.feedback-text.correct {
		color: var(--success);
	}

	.feedback-text.incorrect {
		color: var(--error);
	}

	.next-btn {
		padding: var(--spacing-sm) var(--spacing-lg);
	}

	/* Results */
	.results {
		display: flex;
		justify-content: center;
	}

	.results-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-lg);
		padding: var(--spacing-xl);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		min-width: 300px;
	}

	.results-card h2 {
		margin: 0;
	}

	.final-score {
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.score-number {
		font-family: var(--font-mono);
		font-size: 4rem;
		font-weight: 700;
		color: var(--accent);
		line-height: 1;
	}

	.results-stats {
		display: flex;
		gap: var(--spacing-xl);
	}

	.stat {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
	}

	.stat-value {
		font-family: var(--font-mono);
		font-size: 1.5rem;
		color: var(--text-primary);
	}

	.stat-label {
		font-size: 0.75rem;
		text-transform: uppercase;
		color: var(--text-secondary);
	}

	.results-actions {
		display: flex;
		gap: var(--spacing-md);
	}
</style>

