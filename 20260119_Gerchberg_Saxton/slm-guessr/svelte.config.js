import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),

	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: undefined,
			precompress: false,
			strict: true
		}),
		paths: {
			// For GitHub Pages deployment under a subpath
			// Update this if deploying to a different path
			base: process.env.NODE_ENV === 'production' ? '/PlayGround' : ''
		}
	}
};

export default config;
