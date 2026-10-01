import { runCrossSection } from '../../../engine/src/run.mjs';

self.onmessage = (event: MessageEvent<{ text: string; section: string; optical: boolean; meshScale: number }>) => {
  try {
    const { text, ...options } = event.data;
    const result = runCrossSection(text, {
      ...options, onProgress: (message) => self.postMessage({ type: 'progress', message })
    });
    self.postMessage({ type: 'result', result });
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
