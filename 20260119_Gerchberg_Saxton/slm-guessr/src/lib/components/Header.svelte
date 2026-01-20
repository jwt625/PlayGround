<script lang="ts">
	import { page } from '$app/state';
	import { base } from '$app/paths';

	const navItems = [
		{ href: '/', label: 'Home' },
		{ href: '/gallery', label: 'Gallery' },
		{ href: '/quiz', label: 'Quiz' }
	];

	function isActive(href: string): boolean {
		if (href === '/') {
			return page.url.pathname === base || page.url.pathname === base + '/';
		}
		return page.url.pathname.startsWith(base + href);
	}
</script>

<header>
	<div class="header-content">
		<a href="{base}/" class="logo">
			<span class="logo-text">SLM-Guessr</span>
		</a>
		<nav>
			{#each navItems as item}
				<a 
					href="{base}{item.href}" 
					class="nav-link" 
					class:active={isActive(item.href)}
				>
					{item.label}
				</a>
			{/each}
		</nav>
	</div>
</header>

<style>
	header {
		background-color: var(--bg-secondary);
		border-bottom: 1px solid var(--border);
		padding: var(--spacing-md) var(--spacing-lg);
	}

	.header-content {
		max-width: 1200px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.logo {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
		color: var(--text-primary);
		text-decoration: none;
	}

	.logo-text {
		font-family: var(--font-mono);
		font-size: 1.25rem;
		font-weight: 600;
		letter-spacing: -0.02em;
	}

	nav {
		display: flex;
		gap: var(--spacing-md);
	}

	.nav-link {
		color: var(--text-secondary);
		text-decoration: none;
		padding: var(--spacing-xs) var(--spacing-sm);
		font-size: 14px;
		transition: color 0.15s;
	}

	.nav-link:hover {
		color: var(--text-primary);
	}

	.nav-link.active {
		color: var(--accent);
	}
</style>

