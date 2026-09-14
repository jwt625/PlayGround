import type {
	CachedQuestion,
	Manifest,
	ManifestEntry,
	ModeGalleryManifest,
	NumericalRecipes,
	PendingRecipe
} from './types';

export type Fetcher = (input: string) => Promise<Response>;

export const CACHE_ROOT = 'cache';

function joinPath(basePath: string, ...parts: string[]): string {
	const prefix = basePath.replace(/\/$/, '');
	const joined = [prefix, ...parts].filter(Boolean).join('/');
	return joined.startsWith('/') ? joined : `/${joined}`;
}

export async function loadManifest(fetcher: Fetcher, basePath = ''): Promise<Manifest> {
	const response = await fetcher(joinPath(basePath, CACHE_ROOT, 'manifest.json'));
	if (!response.ok) throw new Error(`Failed to load manifest: ${response.status}`);
	return (await response.json()) as Manifest;
}

export async function loadQuestion(
	fetcher: Fetcher,
	entry: ManifestEntry,
	basePath = ''
): Promise<CachedQuestion> {
	const response = await fetcher(joinPath(basePath, CACHE_ROOT, entry.path));
	if (!response.ok) throw new Error(`Failed to load question ${entry.id}: ${response.status}`);
	return (await response.json()) as CachedQuestion;
}

export async function loadModeGallery(fetcher: Fetcher, basePath = ''): Promise<ModeGalleryManifest> {
	const response = await fetcher(joinPath(basePath, CACHE_ROOT, 'modes', 'manifest.json'));
	if (!response.ok) throw new Error(`Failed to load mode gallery: ${response.status}`);
	return (await response.json()) as ModeGalleryManifest;
}

export async function loadPending(
	fetcher: Fetcher,
	manifest: Manifest,
	basePath = ''
): Promise<PendingRecipe[]> {
	const response = await fetcher(joinPath(basePath, CACHE_ROOT, manifest.pendingPath));
	if (!response.ok) throw new Error(`Failed to load recipes: ${response.status}`);
	return ((await response.json()) as NumericalRecipes).recipes;
}

async function mapLimit<T, R>(items: T[], limit: number, worker: (item: T) => Promise<R>): Promise<R[]> {
	const results = new Array<R>(items.length);
	let cursor = 0;
	const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (cursor < items.length) {
			const index = cursor++;
			results[index] = await worker(items[index]);
		}
	});
	await Promise.all(runners);
	return results;
}

export interface LoadedBank {
	manifest: Manifest;
	questions: CachedQuestion[];
	pending: PendingRecipe[];
}

/**
 * Load the shipped analytic bank. Ready-only by default; the caller can opt
 * into pending recipes for display, but they never carry a computed answer.
 */
export async function loadBank(fetcher: Fetcher, basePath = '', concurrency = 12): Promise<LoadedBank> {
	const manifest = await loadManifest(fetcher, basePath);
	const ready = manifest.entries.filter((entry) => entry.status === 'ready-analytic');
	const questions = await mapLimit(ready, concurrency, (entry) => loadQuestion(fetcher, entry, basePath));
	const pending = await loadPending(fetcher, manifest, basePath);
	return { manifest, questions, pending };
}
