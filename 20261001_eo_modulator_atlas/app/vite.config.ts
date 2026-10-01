import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [
		sveltekit({
			adapter: adapter({ pages: 'build', assets: 'build', fallback: undefined, precompress: false, strict: true }),
			paths: { base: (process.env.BASE_PATH ?? '') as '' | `/${string}`, relative: false },
			prerender: { handleHttpError: 'fail' }
		})
	],
	test: { include: ['src/**/*.test.ts'], environment: 'node' }
});
