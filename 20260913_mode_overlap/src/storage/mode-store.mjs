/** C1.4 mode store: persistent binary/JSON mode bundles with a memory fallback.
 *
 * The same ModeStore logic runs against an IndexedDB backend in the browser and
 * a Map backend in tests. Integrity is a canonical SHA-256 over the value; a
 * corrupt entry is evicted rather than returned. Byte-based LRU keeps the store
 * under a configurable target (default 128 MiB). IndexedDB denial, quota errors
 * or absence fall back to memory, where callers use the shipped static bank.
 */
import { canonicalJSON, contentKey } from './cache.mjs';

export const MODE_STORE_SCHEMA = 'mode-store-1';
export const DEFAULT_BYTES_LIMIT = 128 * 1024 * 1024;

const encoder = new TextEncoder();

function byteLengthOf(value) {
	return encoder.encode(canonicalJSON(value)).byteLength;
}

export class MemoryBackend {
	constructor() {
		this.map = new Map();
		this.kind = 'memory';
	}
	async readAll() {
		return [...this.map.values()].map((record) => ({ ...record }));
	}
	async write(record) {
		this.map.set(record.key, { ...record });
	}
	async delete(key) {
		this.map.delete(key);
	}
	async clear() {
		this.map.clear();
	}
	async close() {}
}

export class IdbBackend {
	constructor(factory, name = 'mode-overlap-modes', storeName = 'modes') {
		this.factory = factory;
		this.name = name;
		this.storeName = storeName;
		this.db = null;
		this.kind = 'indexeddb';
	}
	open() {
		return new Promise((resolvePromise, rejectPromise) => {
			const request = this.factory.open(this.name, 1);
			request.onupgradeneeded = () => {
				const db = request.result;
				if (!db.objectStoreNames.contains(this.storeName)) db.createObjectStore(this.storeName, { keyPath: 'key' });
			};
			request.onsuccess = () => {
				this.db = request.result;
				resolvePromise(this);
			};
			request.onerror = () => rejectPromise(request.error ?? new Error('IndexedDB open failed'));
			request.onblocked = () => rejectPromise(new Error('IndexedDB open blocked'));
		});
	}
	#transaction(mode) {
		if (!this.db) throw new Error('IndexedDB backend is not open');
		return this.db.transaction(this.storeName, mode).objectStore(this.storeName);
	}
	readAll() {
		return new Promise((resolvePromise, rejectPromise) => {
			const request = this.#transaction('readonly').getAll();
			request.onsuccess = () => resolvePromise(request.result ?? []);
			request.onerror = () => rejectPromise(request.error ?? new Error('IndexedDB read failed'));
		});
	}
	write(record) {
		return new Promise((resolvePromise, rejectPromise) => {
			const request = this.#transaction('readwrite').put(record);
			request.onsuccess = () => resolvePromise();
			request.onerror = () => rejectPromise(request.error ?? new Error('IndexedDB write failed'));
		});
	}
	delete(key) {
		return new Promise((resolvePromise, rejectPromise) => {
			const request = this.#transaction('readwrite').delete(key);
			request.onsuccess = () => resolvePromise();
			request.onerror = () => rejectPromise(request.error ?? new Error('IndexedDB delete failed'));
		});
	}
	clear() {
		return new Promise((resolvePromise, rejectPromise) => {
			const request = this.#transaction('readwrite').clear();
			request.onsuccess = () => resolvePromise();
			request.onerror = () => rejectPromise(request.error ?? new Error('IndexedDB clear failed'));
		});
	}
	close() {
		this.db?.close();
		this.db = null;
	}
}

export class ModeStore {
	constructor(backend, options = {}) {
		this.backend = backend;
		this.bytesLimit = options.bytesLimit ?? DEFAULT_BYTES_LIMIT;
		this.now = options.now ?? (() => Date.now());
		this.records = new Map();
		this.bytes = 0;
		this.migrated = 0;
		this.dropped = 0;
	}
	async init() {
		const records = await this.backend.readAll();
		for (const record of records) {
			if (record.schemaVersion !== MODE_STORE_SCHEMA) {
				this.dropped++;
				await this.backend.delete(record.key);
				continue;
			}
			this.records.set(record.key, record);
			this.bytes += record.byteLength ?? 0;
		}
		return this.stats();
	}
	async get(key) {
		const record = this.records.get(key);
		if (!record) return null;
		if ((await contentKey(record.value)) !== record.integrity) {
			await this.#remove(record);
			return null;
		}
		record.lastAccess = this.now();
		await this.backend.write(record);
		return record.value;
	}
	async put(key, value) {
		const integrity = await contentKey(value);
		const byteLength = byteLengthOf(value);
		const record = {
			key,
			schemaVersion: MODE_STORE_SCHEMA,
			byteLength,
			integrity,
			value,
			storedAt: this.now(),
			lastAccess: this.now()
		};
		this.records.set(key, record);
		this.bytes += byteLength;
		await this.backend.write(record);
		await this.#evict();
		return record;
	}
	async delete(key) {
		const record = this.records.get(key);
		if (record) await this.#remove(record);
	}
	async clear() {
		this.records.clear();
		this.bytes = 0;
		await this.backend.clear();
	}
	keys() {
		return [...this.records.keys()];
	}
	stats() {
		return { backend: this.backend.kind, entries: this.records.size, bytes: this.bytes, bytesLimit: this.bytesLimit, migrated: this.migrated, dropped: this.dropped };
	}
	async #remove(record) {
		this.records.delete(record.key);
		this.bytes -= record.byteLength ?? 0;
		await this.backend.delete(record.key);
	}
	async #evict() {
		if (this.bytes <= this.bytesLimit) return [];
		const ordered = [...this.records.values()].sort((a, b) => (a.lastAccess ?? 0) - (b.lastAccess ?? 0));
		const evicted = [];
		for (const record of ordered) {
			if (this.bytes <= this.bytesLimit) break;
			await this.#remove(record);
			evicted.push(record.key);
		}
		return evicted;
	}
}

/** Try IndexedDB, fall back to memory on absence, denial, quota or any error. */
export async function createModeStore(options = {}) {
	const factory = options.indexedDB ?? globalThis.indexedDB;
	const bytesLimit = options.bytesLimit ?? DEFAULT_BYTES_LIMIT;
	if (factory) {
		try {
			const backend = new IdbBackend(factory, options.name, options.storeName);
			await backend.open();
			const store = new ModeStore(backend, { bytesLimit, now: options.now });
			await store.init();
			return { backend: 'indexeddb', store };
		} catch (error) {
			if (options.fallbackToMemory === false) throw error;
		}
	}
	const store = new ModeStore(new MemoryBackend(), { bytesLimit, now: options.now });
	await store.init();
	return { backend: 'memory', store };
}

/** Optional persistent-storage request; denial simply leaves the best-effort store. */
export async function requestPersistence(storageManager = globalThis.navigator?.storage) {
	if (!storageManager?.persist) return false;
	try {
		return await storageManager.persist();
	} catch {
		return false;
	}
}
