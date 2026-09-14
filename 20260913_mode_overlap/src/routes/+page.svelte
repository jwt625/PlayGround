<script lang="ts">
	import { base } from '$app/paths';
</script>

<div class="home">
	<section class="hero">
		<h1>Mode Overlap</h1>
		<p class="tagline">Train your intuition for photonic mode matching and mismatch loss</p>
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
			<p>Browse mode pairs with physical axes, geometry overlays and overlap integrands</p>
		</a>

		<a href="{base}/speedrun" class="mode-card">
			<div class="mode-icon">
				<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
					<circle cx="12" cy="13" r="8" />
					<path d="M12 9v4l2.5 2.5M9 3h6" />
				</svg>
			</div>
			<h2>Speedrun</h2>
			<p>Twenty questions: estimate overlap, mismatch loss and dominant effect</p>
		</a>

		<a href="{base}/practice" class="mode-card">
			<div class="mode-icon">
				<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
					<path d="M4 19V5m0 14h16M8 15l3-4 3 2 4-6" />
				</svg>
			</div>
			<h2>Practice</h2>
			<p>Unlimited questions from a chosen category with local statistics</p>
		</a>
	</section>

	<section class="info">
		<h3>What is this?</h3>
		<p>
			Mode Overlap is a training tool for developing intuition about how mode size, lateral offset,
			angular tilt, ellipticity and polarization change the overlap integral between two optical
			modes &mdash; and therefore mode-mismatch loss.
		</p>
		<div class="concepts">
			<div class="concept">
				<span class="label">Mode A / Mode B</span>
				<span class="desc">Two normalized fields at a common waist plane</span>
			</div>
			<div class="concept">
				<span class="label">&eta; overlap</span>
				<span class="desc">Squared scalar overlap, from 0 to 1</span>
			</div>
			<div class="concept">
				<span class="label">Mismatch loss</span>
				<span class="desc">&minus;10&thinsp;log&#8321;&#8320;(&eta;); exact zero overlap is &infin; dB</span>
			</div>
		</div>
		<p class="disclaimer">
			Ready questions are scalar paraxial Gaussian overlaps at a common waist plane. Numerical
			guides and vector fields are locked until an externally validated solver exists.
		</p>
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
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
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

	.disclaimer {
		margin-top: var(--spacing-lg);
		font-size: 0.85rem;
		color: var(--text-muted);
		border-top: 1px solid var(--border);
		padding-top: var(--spacing-md);
	}
</style>
