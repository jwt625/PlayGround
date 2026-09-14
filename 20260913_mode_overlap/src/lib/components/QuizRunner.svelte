<script lang="ts">
	import { base } from '$app/paths';
	import { loadBank, loadModeGallery } from '$lib/data';
	import { balancedMixedBank } from '$lib/engine';
	import { explainQuestion } from '$lib/explain';
	import { Session } from '$lib/session.svelte';
	import { createStatsStore, shareId, STATS_SCHEMA, type StoredRun } from '$lib/stats';
	import type { AnswerMode, CachedQuestion, ModeGalleryEntry } from '$lib/types';
	import { createWaveguideQuestions } from '$lib/waveguide';
	import { onDestroy, onMount } from 'svelte';
	import AnswerControls from './AnswerControls.svelte';
	import ModePairView from './ModePairView.svelte';
	import ResultsPanel from './ResultsPanel.svelte';
	import RevealPanel from './RevealPanel.svelte';
	import WaveguidePairView from './WaveguidePairView.svelte';

	interface Props {
		title: string;
		subtitle: string;
		limit: number | null;
	}

	let { title, subtitle, limit }: Props = $props();

	let loading = $state(true);
	let error = $state<string | null>(null);
	let questions = $state<CachedQuestion[]>([]);
	let modeEntries = $state<ModeGalleryEntry[]>([]);
	let started = $state(false);
	let answerMode = $state<AnswerMode>('continuous');
	let difficulty = $state<1 | 2>(2);
	let category = $state<string | null>(null);
	let expanded = $state(false);
	let tick = $state(0);
	let runShare = $state<string | undefined>(undefined);
	let nextButton = $state<HTMLButtonElement | null>(null);
	let saved = false;
	let timer: ReturnType<typeof setInterval> | undefined;
	let session = new Session();

	let categories = $derived([...new Set(questions.map((question) => question.category))].sort());

	onMount(async () => {
		try {
			const bank = await loadBank((input) => fetch(input), base);
			questions = bank.questions;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to load the shipped bank';
		} finally {
			loading = false;
		}
		try {
			modeEntries = (await loadModeGallery((input) => fetch(input), base)).entries;
		} catch {
			modeEntries = [];
		}
	});

	onDestroy(() => {
		if (timer) clearInterval(timer);
	});

	function randomSeed(): number {
		if (globalThis.crypto?.getRandomValues) {
			return globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
		}
		return (Date.now() & 0xffffffff) >>> 0;
	}

	function filteredPool(): CachedQuestion[] {
		let pool = questions;
		if (category) pool = pool.filter((question) => question.category === category);
		if (difficulty === 1) pool = pool.filter((question) => question.category === 'gaussian-alignment');
		return pool;
	}

	function buildPool(seed: number): CachedQuestion[] {
		// Level 1 stays pure alignment; level 2 drops the alignment-only
		// duplicates and mixes fresh Gaussian variants with waveguide modes.
		if (difficulty === 2 && category === null) {
			const anchors = questions.filter((question) => question.category !== 'gaussian-alignment');
			const mixed = balancedMixedBank(seed, 1) as unknown as CachedQuestion[];
			const waveguide = modeEntries.length > 0 ? createWaveguideQuestions(modeEntries, { seed, count: 40 }) : [];
			return [...anchors, ...mixed, ...waveguide];
		}
		return filteredPool();
	}

	function categoryCaps(): Record<string, number> | undefined {
		// Level 2 is waveguide-dominant; Gaussian is capped so it cannot dominate.
		return difficulty === 2 && category === null ? { 'gaussian-mixed': 4, 'gaussian-anchor': 2 } : undefined;
	}

	function start() {
		const seed = randomSeed();
		const pool = buildPool(seed);
		if (pool.length === 0) {
			error = 'No analytic questions match this selection.';
			return;
		}
		error = null;
		session.start({ pool, answerMode, limit, category, seed, maxPerCategory: categoryCaps() });
		started = true;
		expanded = false;
		saved = false;
		runShare = undefined;
		if (timer) clearInterval(timer);
		timer = setInterval(() => (tick = Date.now()), 100);
	}

	function submitContinuous(guessEta: number) {
		session.submitContinuous(guessEta);
		expanded = false;
	}

	function submitInterval(intervalIndex: number) {
		session.submitInterval(intervalIndex);
		expanded = false;
	}

	function next() {
		session.next();
		expanded = false;
		if (session.phase === 'results') persistRun();
	}

	function restart() {
		session.restart(session.config.seed);
		expanded = false;
		saved = false;
		runShare = undefined;
	}

	function endSession() {
		session.finish();
		if (session.phase === 'results') persistRun();
	}

	function exit() {
		started = false;
		saved = false;
		runShare = undefined;
	}

	async function persistRun() {
		if (saved) return;
		saved = true;
		const run: StoredRun = {
			schemaVersion: STATS_SCHEMA,
			createdAt: new Date().toISOString(),
			config: { ...session.config },
			results: session.results
		};
		runShare = await shareId(run);
		createStatsStore().save({ ...run, shareId: runShare });
	}

	function exportRun() {
		const run: StoredRun = {
			schemaVersion: STATS_SCHEMA,
			createdAt: new Date().toISOString(),
			config: { ...session.config },
			results: session.results,
			shareId: runShare
		};
		const blob = new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const anchor = document.createElement('a');
		anchor.href = url;
		anchor.download = `${runShare ?? 'mode-overlap-run'}.json`;
		anchor.click();
		URL.revokeObjectURL(url);
	}

	$effect(() => {
		if (session.phase === 'revealed') nextButton?.focus();
	});

	let currentQuestion = $derived(session.current);
	let currentRecord = $derived(session.records[session.records.length - 1]);
	let explanation = $derived(currentQuestion ? explainQuestion(currentQuestion) : null);
	let elapsedLabel = $derived.by(() => {
		void tick;
		const ms = session.elapsedMs();
		return `${(ms / 1000).toFixed(1)} s`;
	});
</script>

<div class="runner">
	<header class="page-header">
		<h1>{title}</h1>
		<p class="subtitle">{subtitle}</p>
	</header>

	{#if loading}
		<div class="state">Loading shipped analytic bank…</div>
	{:else if questions.length === 0}
		<div class="state">
			<p class="text-error">{error ?? 'No analytic questions found.'}</p>
			<p class="hint">Run <span class="mono">npm run cache:build</span> to generate the static bank.</p>
		</div>
	{:else if !started}
		<div class="setup">
			<section class="option-group">
				<h2>Answer mode</h2>
				<div class="options">
					<button class="option-btn" class:active={answerMode === 'continuous'} onclick={() => (answerMode = 'continuous')}>
						<span class="option-title">Continuous η</span>
						<span class="option-desc">Slider and numeric entry, with live dB readout</span>
					</button>
					<button class="option-btn" class:active={answerMode === 'interval'} onclick={() => (answerMode = 'interval')}>
						<span class="option-title">Interval</span>
						<span class="option-desc">Choose the η bin; zero score distance if correct</span>
					</button>
					<button class="option-btn disabled" disabled title="Ranking drill is planned after C3">
						<span class="option-title">Ranking</span>
						<span class="option-desc">Order candidate η values (post-solver)</span>
					</button>
				</div>
			</section>

			<section class="option-group">
				<h2>Difficulty</h2>
				<div class="options row">
					<button class="option-btn small" class:active={difficulty === 1} onclick={() => (difficulty = 1)}>
						<span class="option-title">L1</span>
						<span class="option-desc">Alignment only</span>
					</button>
					<button class="option-btn small" class:active={difficulty === 2} onclick={() => (difficulty = 2)}>
						<span class="option-title">L2</span>
						<span class="option-desc">All analytic effects</span>
					</button>
					{#each [3, 4, 5] as level}
						<button class="option-btn small disabled" disabled title="Requires the validated vector solver (C3)">
							<span class="option-title">L{level}</span>
							<span class="option-desc">solver</span>
						</button>
					{/each}
				</div>
			</section>

			<section class="option-group">
				<h2>Category</h2>
				<select bind:value={category}>
					<option value={null}>All analytic</option>
					{#each categories as item}
						<option value={item}>{item}</option>
					{/each}
				</select>
			</section>

			{#if error}
				<p class="text-error">{error}</p>
			{/if}

			<div class="start-section">
				<button class="primary start" onclick={start}>
					{limit === null ? 'Start practice' : `Start ${limit}-question speedrun`}
				</button>
				<p class="hint">
					{limit === null
						? 'Unlimited questions, drawn from the selected pool.'
						: 'Timer measures visible question to submit; feedback time is excluded.'}
				</p>
			</div>
		</div>
	{:else if session.phase === 'results'}
		<ResultsPanel
			results={session.results}
			shareId={runShare}
			onRestart={restart}
			onExit={exit}
			onExport={exportRun}
		/>
	{:else if currentQuestion}
		{@const question = currentQuestion}
		<div class="play">
			<div class="status-bar">
				<span>Question {session.progressLabel}</span>
				<span>Score so far {(session.results.score).toFixed(1)}</span>
				{#if session.phase === 'playing'}<span>Time {elapsedLabel}</span>{/if}
				<span>{answerMode === 'continuous' ? 'continuous η' : 'interval'}</span>
				{#if limit === null}
					<button class="end" onclick={endSession} disabled={session.records.length === 0}>End session</button>
				{/if}
			</div>

			{#if question.view}
				<WaveguidePairView question={question} revealed={session.phase === 'revealed'} />
			{:else}
				<ModePairView question={question} revealed={session.phase === 'revealed'} />
			{/if}

			{#if session.phase === 'playing'}
				<AnswerControls
					mode={session.config.answerMode}
					disabled={false}
					onSubmitContinuous={submitContinuous}
					onSubmitInterval={submitInterval}
				/>
			{:else if currentRecord && explanation}
				<RevealPanel question={question} record={currentRecord} {explanation} bind:expanded />
				<div class="continue">
					<button class="primary" bind:this={nextButton} onclick={next}>
						{session.isLast && session.config.limit !== null ? 'See results' : 'Next question'}
					</button>
				</div>
			{/if}
		</div>
	{:else}
		<div class="state">Preparing question…</div>
	{/if}
</div>

<style>
	.runner {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.page-header h1 {
		margin-bottom: var(--spacing-xs);
	}

	.subtitle {
		color: var(--text-secondary);
	}

	.state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 200px;
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-secondary);
		gap: var(--spacing-sm);
	}

	.hint {
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.mono {
		font-family: var(--font-mono);
	}

	.setup {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
		max-width: 640px;
	}

	.option-group h2 {
		font-size: 0.85rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-sm);
	}

	.options {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.options.row {
		flex-direction: row;
		flex-wrap: wrap;
	}

	.option-btn {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 2px;
		padding: var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		text-align: left;
		cursor: pointer;
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
	}

	.option-desc {
		font-size: 0.8rem;
		color: var(--text-secondary);
	}

	select {
		background-color: var(--bg-tertiary);
		color: var(--text-primary);
		border: 1px solid var(--border);
		padding: var(--spacing-sm);
		font-family: var(--font-mono);
		max-width: 320px;
	}

	.start-section {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--spacing-sm);
	}

	.start {
		padding: var(--spacing-md) var(--spacing-xl);
		font-size: 1rem;
	}

	.play {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.status-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--spacing-lg);
		padding: var(--spacing-sm) var(--spacing-md);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		font-family: var(--font-mono);
		font-size: 0.8rem;
		color: var(--text-secondary);
	}

	.end {
		margin-left: auto;
		padding: 2px 8px;
		font-size: 0.75rem;
	}

	.continue {
		display: flex;
		justify-content: flex-start;
	}
</style>
