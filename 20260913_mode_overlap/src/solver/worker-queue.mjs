/** Concurrent solve queue with deduplication, cancellation and telemetry (C4.3).
 *
 * The queue owns a small pool of worker-like endpoints. It never blocks the UI
 * thread itself: callers inject a real Web Worker factory in the browser and a
 * plain fake in tests. Identical in-flight keys share one solve. When the
 * backlog is above `highWater` or a solve fails, the caller falls back to the
 * eligible analytic/cached pool; the queue exposes `stats()` so that policy can
 * be decided outside it.
 */
export const QUEUE_VERSION = 'solve-queue-1';

export function createSolveQueue({
	workerFactory,
	lowWater = 5,
	highWater = 20,
	maxWorkers = 20,
	onTelemetry
} = {}) {
	if (typeof workerFactory !== 'function') throw new TypeError('workerFactory required');
	if (!(lowWater >= 1) || !(highWater >= lowWater)) throw new RangeError('invalid water marks');
	if (!(maxWorkers >= 1)) throw new RangeError('invalid maxWorkers');
	const queue = [];
	const inflight = new Map();
	const active = new Map();
	const idle = [];
	const workers = new Set();
	let counter = 0;
	const telemetry = {
		queueVersion: QUEUE_VERSION,
		submitted: 0,
		deduped: 0,
		queued: 0,
		started: 0,
		completed: 0,
		failed: 0,
		cancelled: 0,
		activePeak: 0
	};

	const notify = () => onTelemetry?.({ ...telemetry });

	function spawn() {
		if (workers.size >= maxWorkers) return null;
		const worker = workerFactory();
		worker.onmessage = (event) => handleMessage(worker, event.data);
		worker.onerror = (event) => handleMessage(worker, { ok: false, error: event?.message ?? 'worker error' });
		workers.add(worker);
		return worker;
	}

	function dispatch() {
		while (queue.length > 0) {
			const worker = idle.pop() ?? spawn();
			if (!worker) break;
			const entry = queue.shift();
			telemetry.started++;
			telemetry.activePeak = Math.max(telemetry.activePeak, active.size + 1);
			active.set(worker, entry);
			entry.worker = worker;
			worker.postMessage({ id: entry.job.id, key: entry.job.key, method: entry.job.method, payload: entry.job.payload });
		}
	}

	function settle(entry, outcome) {
		inflight.delete(entry.job.key);
		if (outcome.cancelled) {
			telemetry.cancelled++;
			for (const waiter of entry.waiters) waiter.reject(new Error('solve cancelled'));
		} else if (outcome.ok) {
			telemetry.completed++;
			for (const waiter of entry.waiters) waiter.resolve(outcome.result);
		} else {
			telemetry.failed++;
			for (const waiter of entry.waiters) waiter.reject(new Error(outcome.error ?? 'solve failed'));
		}
		notify();
	}

	function handleMessage(worker, data) {
		const entry = active.get(worker);
		if (!entry) return;
		active.delete(worker);
		if (entry.cancelled) settle(entry, { cancelled: true });
		else if (data?.ok === false) settle(entry, { ok: false, error: data.error });
		else settle(entry, { ok: true, result: data?.result });
		idle.push(worker);
		dispatch();
	}

	function submit({ key, method, payload }) {
		if (key === undefined || method === undefined) throw new TypeError('key and method required');
		telemetry.submitted++;
		if (inflight.has(key)) {
			telemetry.deduped++;
			const entry = inflight.get(key);
			return new Promise((resolve, reject) => entry.waiters.push({ resolve, reject }));
		}
		const entry = { job: { id: ++counter, key, method, payload }, waiters: [] };
		inflight.set(key, entry);
		queue.push(entry);
		telemetry.queued++;
		notify();
		dispatch();
		return new Promise((resolve, reject) => entry.waiters.push({ resolve, reject }));
	}

	function cancel(key) {
		const entry = inflight.get(key);
		if (!entry) return false;
		if (entry.worker) {
			entry.cancelled = true;
			return true;
		}
		const index = queue.indexOf(entry);
		if (index >= 0) queue.splice(index, 1);
		settle(entry, { cancelled: true });
		return true;
	}

	function stats() {
		return { ...telemetry, pending: queue.length, active: active.size, workers: workers.size, lowWater, highWater };
	}

	function whenIdle() {
		if (queue.length === 0 && active.size === 0) return Promise.resolve();
		return new Promise((resolve) => {
			const check = () => {
				if (queue.length === 0 && active.size === 0) {
					clearInterval(timer);
					resolve();
				}
			};
			const timer = setInterval(check, 1);
		});
	}

	function close() {
		for (const worker of workers) worker.terminate?.();
		workers.clear();
		idle.length = 0;
		for (const entry of queue) settle(entry, { cancelled: true });
		queue.length = 0;
	}

	return { submit, cancel, stats, whenIdle, close, isBacklogged: () => queue.length > highWater };
}
