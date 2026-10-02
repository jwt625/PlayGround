// Runs against a production build with its own temporary preview server and Chrome profile.
// No results are saved unless SMOKE_SCREENSHOTS points to a local scratch directory.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright-core';
import { runCrossSection } from '../../engine/src/run.mjs';
import { parseConfig } from '../../engine/src/config.mjs';

const appDir = fileURLToPath(new URL('../', import.meta.url));
const base = process.env.BASE_PATH ?? '';
const chen = await readFile(new URL('../../sims/chen2022/config.yaml', import.meta.url), 'utf8');
const analytic = await readFile(new URL('../../engine/tests/fixtures/parallel-plates.yaml', import.meta.url), 'utf8');
const expected = runCrossSection(chen);
const expectedOptical = runCrossSection(analytic, { optical: true });
const preview = spawn(process.execPath, [fileURLToPath(new URL('../node_modules/vite/bin/vite.js', import.meta.url)), 'preview', '--host', '127.0.0.1', '--port', '0'], { cwd: appDir, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
let browser;
let serverOutput = '';

async function until(predicate, label, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (await predicate()) return;
    await delay(50);
  }
  throw new Error(`Timed out: ${label}`);
}

try {
  const origin = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Preview startup timed out: ${serverOutput}`)), 15000);
    const collect = chunk => {
      serverOutput += chunk.toString().replace(/\x1b\[[0-9;]*m/g, '');
      const match = serverOutput.match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    };
    preview.stdout.on('data', collect); preview.stderr.on('data', collect);
    preview.on('error', error => { clearTimeout(timer); reject(error); });
    preview.once('exit', code => { clearTimeout(timer); reject(new Error(`Preview exited ${code}: ${serverOutput}`)); });
  });
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  page.setDefaultTimeout(30000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`); });
  const url = path => `${origin}${base}/${path}`;
  const runButton = page.getByRole('button', { name: 'Run cross-section', exact: true });
  const editor = page.getByLabel('Input YAML · edits apply to this session');
  const complete = () => page.getByText('Cross-section solve complete', { exact: true }).waitFor();
  const loadSim = async (query = 'id=chen2022-c') => {
    await page.goto(url(`sim?${query}`));
    await until(() => runButton.isEnabled(), 'config loaded');
  };
  const screenshot = async name => {
    if (!process.env.SMOKE_SCREENSHOTS) return;
    await mkdir(process.env.SMOKE_SCREENSHOTS, { recursive: true });
    await page.screenshot({ path: join(process.env.SMOKE_SCREENSHOTS, `${name}.png`), fullPage: true });
  };

  for (const path of ['', 'table', 'explore', 'about']) {
    await page.goto(url(path));
    await page.waitForLoadState('networkidle');
    if (path === 'table') await page.locator('main table').first().waitFor();
    if (path === 'explore') await page.locator('.js-plotly-plot').first().waitFor();
    if (path === 'about') await page.getByRole('heading', { name: 'About the atlas' }).waitFor();
  }
  console.log('PASS dashboard/table/explore/about render');

  await page.goto(url('sim?id=deng2026-a'));
  await page.getByRole('alert').filter({ hasText: 'Preview only. Solve blocked: materials.barium_titanate.eps_r' }).waitFor();
  await page.getByRole('img', { name: 'Configured cross-section geometry' }).waitFor();
  await page.getByRole('heading', { name: 'Missing or inferred inputs' }).waitFor();
  await page.getByText('Config validation status: unvalidated', { exact: true }).waitFor();
  assert.equal(await runButton.isEnabled(), false, 'incomplete physical inputs must not reach the worker');
  assert.equal(await page.locator('[data-metric]').count(), 0);
  assert.equal(page.workers().length, 0);
  await screenshot('sim-incomplete');
  console.log('PASS incomplete paper config previews geometry and disclosures with solve disabled');

  await loadSim();
  const published = await page.request.get(url('sims/chen2022/config.yaml'));
  assert.equal(await published.text(), chen, 'published config must equal the tracked input');
  await runButton.click(); await complete();
  for (const [metric, value] of Object.entries(expected.metrics)) {
    const actual = Number(await page.locator(`[data-metric="${metric}"]`).getAttribute('data-value'));
    assert.ok(Math.abs(actual - value) <= Math.abs(value) * 1e-8, `browser/Node disagreement: ${metric}`);
  }
  assert.match(await page.locator('.target-summary').innerText(), /0 of 11 targets evaluated/);
  await screenshot('sim-desktop');
  console.log('PASS real Chen worker run, browser/Node agreement, unsupported targets excluded');

  await editor.fill(`${chen}\n# Edited locally for smoke validation\n`);
  await page.getByText('Inputs changed. These results belong to the previous run.').waitFor();
  await page.getByLabel('Cross-section', { exact: true }).selectOption('unloaded');
  await runButton.click(); await complete();
  await page.getByText(/chen2022-c · unloaded/).waitFor();
  assert.equal(await page.locator('.notice').count(), 0);
  console.log('PASS stale-result detection and alternate cross-section');

  await loadSim();
  await runButton.click();
  await page.getByText('Meshing and solving electrostatics', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.getByText('Cancelled', { exact: true }).waitFor();
  await until(() => page.workers().length === 0, 'cancelled worker terminated');
  assert.equal(await page.locator('[data-metric]').count(), 0);
  assert.ok(await runButton.isEnabled());
  console.log('PASS cancellation clears the active worker and discards the run');

  await page.getByLabel('Include optical mode').check();
  await runButton.click();
  await page.getByRole('alert').filter({ hasText: 'metal optical modes are unsupported' }).waitFor();
  await until(() => page.workers().length === 0, 'failed worker terminated');
  assert.equal(await page.locator('[data-metric]').count(), 0);
  await editor.fill('schema: invalid');
  await page.getByRole('alert').filter({ hasText: 'expected eo-atlas.sim/v1' }).waitFor();
  assert.equal(await runButton.isEnabled(), false);
  console.log('PASS unsupported optical error and invalid-input guard');

  await editor.fill(analytic);
  await runButton.click(); await complete();
  const neff = Number(await page.locator('[data-metric="n_eff"]').getAttribute('data-value'));
  assert.ok(Math.abs(neff - expectedOptical.metrics.n_eff) < 1e-8);
  assert.match(await page.locator('.target-summary').innerText(), /1 of 2 targets evaluated · 1 passed · 0 failed/);
  const miss = parseConfig(analytic).raw;
  miss.targets[0].value = 20;
  await editor.fill(JSON.stringify(miss));
  await runButton.click(); await complete();
  assert.match(await page.locator('.target-summary').innerText(), /1 of 2 targets evaluated · 0 passed · 1 failed/);
  console.log('PASS retry, optical browser/Node agreement and failed-target display');

  await page.goto(url('sim?id=unknown-config'));
  await page.getByText('The requested config is not in this atlas.').waitFor();
  assert.equal(await runButton.isEnabled(), false);
  assert.equal(await editor.inputValue(), '');
  await loadSim('config=sims/chen2022/config.yaml');
  await runButton.click();
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await page.getByRole('heading', { name: 'About the atlas' }).waitFor();
  await until(() => page.workers().length === 0, 'navigation terminated the worker');
  console.log('PASS unknown config, legacy config URL and navigation cleanup');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url('sim?id=deng2026-a'));
  await page.getByRole('alert').filter({ hasText: 'Preview only. Solve blocked:' }).waitFor();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'incomplete draft overflows narrow viewport');
  assert.ok(await page.locator('.sim-page').evaluate(el => el.scrollWidth <= el.clientWidth + 1), 'incomplete draft content overflows narrow viewport');
  await loadSim();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'app shell overflows narrow viewport');
  assert.ok(await page.locator('.sim-page').evaluate(el => el.scrollWidth <= el.clientWidth + 1), 'sim content overflows narrow viewport');
  await editor.fill(analytic); await runButton.click(); await complete();
  await page.locator('.sim-page').evaluate(el => el.scrollTop = 0);
  await screenshot('sim-mobile');
  console.log('PASS narrow-viewport controls and worker run');
  assert.deepEqual(errors, [], 'browser/network errors');
  console.log(`PASS smoke suite at base path ${base || '/'}`);
} finally {
  await browser?.close();
  preview.kill('SIGTERM');
}
