import test from 'node:test';
import assert from 'node:assert/strict';
import {
	createModeStore,
	MemoryBackend,
	ModeStore,
	MODE_STORE_SCHEMA,
	requestPersistence
} from '../src/storage/mode-store.mjs';

test('put and get round-trip values with integrity metadata', async () => {
	const store = new ModeStore(new MemoryBackend());
	await store.init();
	const record = await store.put('mode-a', { neff: 2.44, fields: [1, 2, 3] });
	assert.equal(record.schemaVersion, MODE_STORE_SCHEMA);
	assert.ok(record.integrity.length === 64);
	assert.deepEqual(await store.get('mode-a'), { neff: 2.44, fields: [1, 2, 3] });
	assert.equal(await store.get('missing'), null);
});

test('corrupt entries fail integrity and are evicted, not returned', async () => {
	const backend = new MemoryBackend();
	const store = new ModeStore(backend);
	await store.init();
	await store.put('mode-a', { neff: 2.4 });
	const tampered = backend.map.get('mode-a');
	tampered.value = { neff: 9.9 };
	await backend.write(tampered);

	const reopened = new ModeStore(backend);
	await reopened.init();
	assert.equal(await reopened.get('mode-a'), null);
	assert.deepEqual(reopened.keys(), []);
});

test('byte-based LRU evicts the least recently used entry under the target', async () => {
	const value = { blob: 'x'.repeat(256) };
	const probe = new ModeStore(new MemoryBackend());
	await probe.init();
	await probe.put('probe', value);
	const unit = probe.records.get('probe').byteLength;
	assert.ok(unit > 0);

	let clock = 0;
	const store = new ModeStore(new MemoryBackend(), { bytesLimit: Math.floor(unit * 2.5), now: () => clock++ });
	await store.init();
	await store.put('a', value);
	await store.put('b', value);
	await store.put('c', value);
	assert.equal(await store.get('a'), null);
	assert.deepEqual(await store.get('b'), value);
	assert.deepEqual(await store.get('c'), value);
	assert.ok(store.stats().bytes <= Math.floor(unit * 2.5));
});

test('incompatible schema versions are dropped on init', async () => {
	const backend = new MemoryBackend();
	await backend.write({
		key: 'legacy',
		schemaVersion: 'mode-store-0',
		byteLength: 1,
		integrity: 'irrelevant',
		value: { old: true },
		storedAt: 1,
		lastAccess: 1
	});
	const store = new ModeStore(backend);
	const stats = await store.init();
	assert.equal(stats.dropped, 1);
	assert.deepEqual(store.keys(), []);
});

test('createModeStore falls back to memory on IndexedDB denial or absence', async () => {
	const denied = {
		open() {
			throw new Error('persistent storage denied');
		}
	};
	const fallback = await createModeStore({ indexedDB: denied });
	assert.equal(fallback.backend, 'memory');
	await assert.rejects(() => createModeStore({ indexedDB: denied, fallbackToMemory: false }));

	const absent = await createModeStore({ indexedDB: null });
	assert.equal(absent.backend, 'memory');
});

test('requestPersistence reports denial without throwing', async () => {
	assert.equal(await requestPersistence({ persist: async () => true }), true);
	assert.equal(await requestPersistence({ persist: async () => false }), false);
	assert.equal(
		await requestPersistence({
			persist: async () => {
				throw new Error('denied');
			}
		}),
		false
	);
	assert.equal(await requestPersistence(null), false);
});
