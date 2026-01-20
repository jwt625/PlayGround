<script lang="ts">
	import { base } from '$app/paths';
</script>

<div class="home">
	<section class="hero">
		<h1>SLM-Guessr</h1>
		<p class="tagline">Train your intuition for SLM phase masks and 2D FFT</p>
	</section>

	<section class="modes">
		<a href="{base}/gallery" class="mode-card">
			<div class="mode-icon">
				<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
					<rect x="3" y="3" width="7" height="7" />
					<rect x="14" y="3" width="7" height="7" />
					<rect x="3" y="14" width="7" height="7" />
					<rect x="14" y="14" width="7" height="7" />
				</svg>
			</div>
			<h2>Gallery</h2>
			<p>Browse phase-intensity pairs organized by concept and difficulty</p>
		</a>

		<a href="{base}/quiz" class="mode-card">
			<div class="mode-icon">
				<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
					<circle cx="12" cy="12" r="9" />
					<path d="M12 8v4l2 2" />
				</svg>
			</div>
			<h2>Quiz</h2>
			<p>Test your pattern recognition skills with timed challenges</p>
		</a>
	</section>

	<section class="info">
		<h3>What is this?</h3>
		<p>
			SLM-Guessr is a training tool for developing intuition about Spatial Light Modulator (SLM)
			phase masks and their corresponding Fourier-plane intensity distributions. By viewing
			many phase-intensity pairs, you will learn to predict how phase patterns transform
			under 2D FFT.
		</p>
		<div class="concepts">
			<div class="concept">
				<span class="label">Phase Mask</span>
				<span class="desc">Pattern applied to SLM (input plane)</span>
			</div>
			<div class="concept">
				<span class="label">Intensity</span>
				<span class="desc">Result in Fourier plane after FFT</span>
			</div>
		</div>
	</section>
</div>

<style>
	.home {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xl);
	}

	.hero {
		text-align: center;
		padding: var(--spacing-xl) 0;
	}

	.hero h1 {
		font-family: var(--font-mono);
		font-size: 3rem;
		margin-bottom: var(--spacing-sm);
	}

	.tagline {
		color: var(--text-secondary);
		font-size: 1.1rem;
	}

	.modes {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: var(--spacing-lg);
	}

	.mode-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		text-align: center;
		padding: var(--spacing-xl);
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		color: var(--text-primary);
		text-decoration: none;
		transition: border-color 0.15s, background-color 0.15s;
	}

	.mode-card:hover {
		border-color: var(--accent);
		background-color: var(--bg-tertiary);
	}

	.mode-icon {
		color: var(--accent);
		margin-bottom: var(--spacing-md);
	}

	.mode-card h2 {
		margin-bottom: var(--spacing-sm);
	}

	.mode-card p {
		color: var(--text-secondary);
		font-size: 0.9rem;
	}

	.info {
		background-color: var(--bg-secondary);
		border: 1px solid var(--border);
		padding: var(--spacing-lg);
	}

	.info h3 {
		margin-bottom: var(--spacing-md);
	}

	.info > p {
		color: var(--text-secondary);
		margin-bottom: var(--spacing-lg);
		line-height: 1.7;
	}

	.concepts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: var(--spacing-md);
	}

	.concept {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
		padding: var(--spacing-md);
		background-color: var(--bg-tertiary);
		border: 1px solid var(--border);
	}

	.concept .label {
		font-family: var(--font-mono);
		font-weight: 600;
		color: var(--accent);
	}

	.concept .desc {
		font-size: 0.85rem;
		color: var(--text-secondary);
	}
</style>
