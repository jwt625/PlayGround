import { contentKey } from './engine';
import type { SessionConfig, SessionResults } from './types';

export const STATS_SCHEMA = 'player-stats-1';

export interface StoredRun {
	schemaVersion: string;
	createdAt: string;
	config: SessionConfig;
	results: SessionResults;
	shareId?: string;
}

export interface StatsStore {
	load(): StoredRun[];
	save(run: StoredRun): void;
	clear(): void;
	export(): string;
}

const STORAGE_KEY = 'mode-overlap:runs';

function isStoredRun(value: unknown): value is StoredRun {
	if (!value || typeof value !== 'object') return false;
	const candidate = value as Partial<StoredRun>;
	return (
		candidate.schemaVersion === STATS_SCHEMA &&
		typeof candidate.createdAt === 'string' &&
		!!candidate.config &&
		!!candidate.results
	);
}

export function createStatsStore(storage: Storage | null = globalThis.localStorage ?? null): StatsStore {
	const read = (): StoredRun[] => {
		if (!storage) return [];
		try {
			const parsed = JSON.parse(storage.getItem(STORAGE_KEY) ?? '[]') as unknown;
			if (!Array.isArray(parsed)) return [];
			// Schema versions are kept separate: old-major runs are dropped, not coerced.
			return parsed.filter(isStoredRun);
		} catch {
			return [];
		}
	};
	return {
		load: read,
		save(run: StoredRun) {
			if (!storage) return;
			const runs = read();
			runs.unshift(run);
			try {
				storage.setItem(STORAGE_KEY, JSON.stringify(runs.slice(0, 50)));
			} catch {
				/* quota exceeded: keep in-memory session results only */
			}
		},
		clear() {
			if (!storage) return;
			try {
				storage.removeItem(STORAGE_KEY);
			} catch {
				/* ignore */
			}
		},
		export() {
			return JSON.stringify(read(), null, 2);
		}
	};
}

/** Deterministic share ID over versions and the full persisted result. */
export async function shareId(run: Omit<StoredRun, 'shareId'>): Promise<string> {
	return `mo-${(await contentKey(run)).slice(0, 16)}`;
}
