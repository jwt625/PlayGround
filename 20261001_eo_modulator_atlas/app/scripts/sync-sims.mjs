// Publish input configs into the static app; no solver outputs or reference PDFs.
import { readdir, mkdir, copyFile, rm } from 'node:fs/promises';
const source = new URL('../../sims/', import.meta.url);
const dest = new URL('../static/sims/', import.meta.url);
await rm(dest, { recursive: true, force: true });
for (const entry of await readdir(source, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  try {
    await mkdir(new URL(`${entry.name}/`, dest), { recursive: true });
    await copyFile(new URL(`${entry.name}/config.yaml`, source), new URL(`${entry.name}/config.yaml`, dest));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}
