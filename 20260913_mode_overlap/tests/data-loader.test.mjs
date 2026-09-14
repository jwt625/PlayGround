import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { loadBank, loadManifest } from '../src/lib/data.ts';

const staticDir = fileURLToPath(new URL('../static', import.meta.url));

/** Fetcher over the shipped static directory, exercising the real URL joins. */
async function fileFetcher(input) {
	const relative = String(input).replace(/^\//, '');
	const path = `${staticDir}/${relative}`;
	try {
		const text = await readFile(path, 'utf8');
		return { ok: true, status: 200, async json() { return JSON.parse(text); } };
	} catch {
		return { ok: false, status: 404, async json() { return null; } };
	}
}

test('manifest loads from the shipped cache with ready and pending counts', async () => {
	const manifest = await loadManifest(fileFetcher);
	assert.equal(manifest.readyCount, 146);
	assert.equal(manifest.entries.length, 146);
	assert.equal(manifest.pendingCount, 14);
});

test('bank loads every ready analytic question and never a pending answer', async () => {
	const bank = await loadBank(fileFetcher);
	assert.equal(bank.questions.length, 146);
	assert.equal(bank.pending.length, 14);
	for (const question of bank.questions) {
		assert.equal(question.status, 'ready-analytic');
		assert.ok(question.answer.eta >= 0 && question.answer.eta <= 1);
		assert.ok(question.transform, question.id);
		assert.ok(question.a.mfdXUm > 0 && question.b.mfdYUm > 0);
	}
	assert.ok(bank.pending.every((recipe) => !('answer' in recipe)));
});

test('missing files surface as an error rather than a silent empty bank', async () => {
	await assert.rejects(() => loadManifest(async () => ({ ok: false, status: 404 })), /404/);
});
