<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<style>
	:global(:root) {
		color-scheme: light;
		--bg: #f3f3f1;
		--panel: #fbfbfa;
		--panel-strong: #ffffff;
		--line: #cbcbc5;
		--line-strong: #151515;
		--text: #111111;
		--muted: #626262;
		--muted-soft: #7c7c77;
		--accent: #111111;
		--radius: 0;
		--shadow: 0 0 0 1px rgba(17, 17, 17, 0.06);
		font-family:
			'IBM Plex Sans', 'Aptos', 'Segoe UI', sans-serif;
		background: var(--bg);
		color: var(--text);
	}

	:global(html) {
		background: var(--bg);
	}

	:global(body) {
		margin: 0;
		background:
			linear-gradient(to bottom, rgba(17, 17, 17, 0.03), rgba(17, 17, 17, 0.03)) 0 0 / 100% 1px
				no-repeat,
			var(--bg);
		color: var(--text);
		font-family: inherit;
	}

	:global(*) {
		box-sizing: border-box;
	}

	:global(a) {
		color: inherit;
	}

	:global(button),
	:global(input),
	:global(select),
	:global(textarea) {
		font: inherit;
	}
	
	:global(button),
	:global(input),
	:global(select),
	:global(textarea) {
		border-radius: 0;
	}
</style>

{@render children()}
