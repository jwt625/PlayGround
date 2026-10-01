import { asset, resolve } from '$app/paths';

/** App-internal link honoring the configured base path (BASE_PATH). */
export function link(path: string): string {
	return resolve(path as '/');
}

/** URL of a file in static/ honoring the base path. */
export function staticUrl(file: string): string {
	return asset(file as 'favicon.svg');
}
