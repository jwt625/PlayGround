import test from 'node:test';
import assert from 'node:assert/strict';
import { createSolveQueue } from '../src/solver/worker-queue.mjs';

function fakeWorkerFactory({ delay = 2, behavior = (message) => ({ ok: true, result: message.payload * 2 }), messages = [] } = {}) {
	return () => {
		const worker = { onmessage: null, onerror: null, terminated: false };
		worker.postMessage = (message) => {
			messages.push(message);
			setTimeout(() => {
				if (worker.terminated) return;
				worker.onmessage?.({ data: behavior(message) });
			}, delay);
		};
		worker.terminate = () => {
			worker.terminated = true;
		};
		return worker;
	};
}

test('runs all jobs with bounded concurrency and records telemetry', async () => {
	const messages = [];
	const queue = createSolveQueue({ workerFactory: fakeWorkerFactory({ messages }), maxWorkers: 3 });
	const jobs = await Promise.all(
		Array.from({ length: 10 }, (_, i) => queue.submit({ key: `k${i}`, method: 'solve', payload: i }))
	);
	await queue.whenIdle();
	assert.deepEqual(jobs, Array.from({ length: 10 }, (_, i) => i * 2));
	const stats = queue.stats();
	assert.equal(stats.completed, 10);
	assert.equal(stats.deduped, 0);
	assert.ok(stats.activePeak <= 3);
	assert.equal(messages.length, 10);
});

test('identical in-flight keys are deduplicated', async () => {
	const messages = [];
	const queue = createSolveQueue({ workerFactory: fakeWorkerFactory({ delay: 20, messages }), maxWorkers: 2 });
	const first = queue.submit({ key: 'same', method: 'solve', payload: 7 });
	const second = queue.submit({ key: 'same', method: 'solve', payload: 7 });
	const [a, b] = await Promise.all([first, second]);
	await queue.whenIdle();
	assert.equal(a, 14);
	assert.equal(b, 14);
	assert.equal(queue.stats().deduped, 1);
	assert.equal(messages.length, 1);
});

test('failed solves reject and increment the failure counter', async () => {
	const queue = createSolveQueue({
		workerFactory: fakeWorkerFactory({ behavior: () => ({ ok: false, error: 'no convergence' }) }),
		maxWorkers: 2
	});
	await assert.rejects(() => queue.submit({ key: 'bad', method: 'solve', payload: 1 }), /no convergence/);
	await queue.whenIdle();
	assert.equal(queue.stats().failed, 1);
});

test('queued jobs can be cancelled before a worker starts them', async () => {
	const queue = createSolveQueue({ workerFactory: fakeWorkerFactory({ delay: 30 }), maxWorkers: 1 });
	const running = queue.submit({ key: 'running', method: 'solve', payload: 1 });
	const waiting = queue.submit({ key: 'waiting', method: 'solve', payload: 2 });
	assert.equal(queue.cancel('waiting'), true);
	await assert.rejects(() => waiting, /cancelled/);
	await running;
	await queue.whenIdle();
	assert.equal(queue.stats().cancelled, 1);
	assert.equal(queue.stats().completed, 1);
});

test('backlog is exposed to the caller for analytic fallback', async () => {
	const queue = createSolveQueue({ workerFactory: fakeWorkerFactory({ delay: 20 }), maxWorkers: 1, lowWater: 1, highWater: 1 });
	const jobs = ['a', 'b', 'c', 'd'].map((key, i) => queue.submit({ key, method: 'solve', payload: i }));
	assert.equal(queue.isBacklogged(), true);
	await Promise.all(jobs);
	await queue.whenIdle();
	assert.equal(queue.isBacklogged(), false);
});

test('rejects invalid worker factories and water marks', () => {
	assert.throws(() => createSolveQueue({}), /workerFactory/);
	assert.throws(() => createSolveQueue({ workerFactory: () => ({}), lowWater: 5, highWater: 2 }), /water marks/);
});
