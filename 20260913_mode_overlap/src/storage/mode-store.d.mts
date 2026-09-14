export declare const MODE_STORE_SCHEMA: string;
export declare const DEFAULT_BYTES_LIMIT: number;

export interface StoredModeRecord<T = unknown> {
	key: string;
	schemaVersion: string;
	byteLength: number;
	integrity: string;
	value: T;
	storedAt: number;
	lastAccess: number;
}

export interface ModeStoreStats {
	backend: string;
	entries: number;
	bytes: number;
	bytesLimit: number;
	migrated: number;
	dropped: number;
}

export declare class MemoryBackend {
	readonly kind = 'memory';
	readAll(): Promise<StoredModeRecord[]>;
	write(record: StoredModeRecord): Promise<void>;
	delete(key: string): Promise<void>;
	clear(): Promise<void>;
	close(): Promise<void>;
}

export declare class ModeStore<T = unknown> {
	constructor(backend: unknown, options?: { bytesLimit?: number; now?: () => number });
	init(): Promise<ModeStoreStats>;
	get(key: string): Promise<T | null>;
	put(key: string, value: T): Promise<StoredModeRecord<T>>;
	delete(key: string): Promise<void>;
	clear(): Promise<void>;
	keys(): string[];
	stats(): ModeStoreStats;
}

export declare function createModeStore(options?: {
	indexedDB?: IDBFactory | null;
	name?: string;
	storeName?: string;
	bytesLimit?: number;
	fallbackToMemory?: boolean;
	now?: () => number;
}): Promise<{ backend: 'indexeddb' | 'memory'; store: ModeStore }>;

export declare function requestPersistence(storageManager?: StorageManager | null): Promise<boolean>;
