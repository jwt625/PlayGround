import { test, expect } from "@playwright/test";

test.describe("CBC browser app", () => {
  test("renders the bench, schematic, and live metrics", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await expect(page.locator("#view")).toBeVisible();
    // The saved MaleCNS run auto-loads so the neuron cloud is always on.
    await page.waitForFunction(
      () => (window as unknown as { __cbc: { getMetrics: () => { graphSource: string } } }).__cbc.getMetrics().graphSource === "malecns",
      undefined,
      { timeout: 30_000 },
    );
    await expect(page.locator("#m-mode")).toHaveText("connectome");
    await page.waitForTimeout(1500);

    const metrics = await page.evaluate(() => (window as unknown as { __cbc: { getMetrics: () => unknown } }).__cbc.getMetrics());
    expect(metrics).toBeTruthy();
    expect(errors, errors.join("\n")).toEqual([]);

    await page.screenshot({ path: "test-results/app-analytic.png" });
  });

  test("mode switching and channel selection work", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(500);

    await page.getByRole("button", { name: "spgd" }).click();
    await expect(page.locator("#m-mode")).toHaveText("spgd");
    await page.waitForTimeout(800);

    await page.getByRole("button", { name: "connectome" }).click();
    await expect(page.locator("#m-mode")).toHaveText("connectome");
    await page.waitForTimeout(500);

    // Select CH01 in the schematic by clicking the first cell region.
    const canvas = page.locator("#schematic");
    const box = await canvas.boundingBox();
    if (box) {
      await canvas.click({ position: { x: 30, y: 75 } });
    }
    await page.waitForTimeout(300);
    await page.screenshot({ path: "test-results/app-connectome-selected.png" });

    const selected = await page.evaluate(
      () => (window as unknown as { __cbc: { getMetrics: () => { selected: number } } }).__cbc.getMetrics().selected,
    );
    expect(selected).toBeGreaterThanOrEqual(0);
  });

  test("target curriculum options are selectable", async ({ page }) => {
    await page.goto("/");
    for (const kind of ["lissajous", "circle", "fly", "static"]) {
      await page.getByRole("button", { name: kind }).click();
      await page.waitForTimeout(200);
    }
    await page.screenshot({ path: "test-results/app-fly-target.png" });
  });

  test("articulated fly GLB loads for the target instance", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "fly" }).click();
    await page.waitForFunction(
      () =>
        (window as unknown as { __cbc: { getMetrics: () => { flyLoaded: boolean; flyError: string | null } } })
          .__cbc.getMetrics().flyLoaded,
      undefined,
      { timeout: 45_000 },
    );
    const metrics = await page.evaluate(
      () => (window as unknown as { __cbc: { getMetrics: () => { flyLoaded: boolean; flyError: string | null } } }).__cbc.getMetrics(),
    );
    expect(metrics.flyError).toBeNull();
    expect(metrics.flyLoaded).toBe(true);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: "test-results/app-flybody-loaded.png" });
  });
});

test("inspector pause, cameras, layer toggles and section range", async ({ page }) => {
  await page.goto("/");
  await page.getByRole('button', {name:'Pause simulation'}).click();
  const step=await page.evaluate(()=> (window as any).__cbc.getMetrics().step);
  await page.getByRole('button', {name:'array',exact:true}).click();
  await page.waitForTimeout(250);
  expect(await page.evaluate(()=> (window as any).__cbc.getMetrics().step)).toBe(step);
  await page.getByLabel('Selected channel').selectOption('9');
  await page.getByLabel('piston', {exact:true}).press('ArrowRight');
  await expect(page.getByLabel('Selected channel')).toHaveValue('9');
  await page.getByLabel('Section range', {exact:true}).fill('1.4');
  await expect(page.locator('#section-range-value')).toHaveText('1.40 m');
  await page.getByLabel('Dome', {exact:true}).uncheck();
  await page.getByRole('button', {name:'Resume simulation'}).click();
  await expect.poll(()=>page.evaluate(()=> (window as any).__cbc.getMetrics().step)).toBeGreaterThan(step);
  expect(await page.evaluate(()=> (window as any).__cbc.getMetrics().graphSource)).toBe('malecns');
});

test('saved MaleCNS before/after drives the scene and 3D activation',async({page})=>{
  await page.goto('/');
  await page.getByRole('button',{name:'Load real training run + 3D neurons',exact:true}).click();
  await page.selectOption('#replay-speed','1');
  await page.waitForFunction(()=>(window as any).__cbc?.getMetrics().replayFinished);
  const before=await page.evaluate(()=>(window as any).__cbc.getMetrics());
  expect(before.graphSource).toBe('malecns');expect(before.neural3dNodes).toBe(4268);
  await page.getByRole('button',{name:'After training',exact:true}).click();
  await page.waitForFunction(()=>(window as any).__cbc.getMetrics().replayFinished);
  const after=await page.evaluate(()=>(window as any).__cbc.getMetrics());
  expect(after.replayMeanIntensity).toBeGreaterThan(before.replayMeanIntensity+.3);
  expect(after.connectome.fractionActive).toBeGreaterThan(0);
  await expect(page.locator('#graph-provenance')).toContainText('MaleCNS v1.0');
});

test('Inspect 3D neurons loads the run standalone and focuses the mapped somata', async ({ page }) => {
  await page.goto('/');
  // No prior "Load real training run" click: the button must self-load.
  await page.getByRole('button', { name: 'Inspect 3D neurons', exact: true }).click();
  await page.waitForFunction(
    () => (window as any).__cbc?.getMetrics().neural3dNodes > 0,
    undefined,
    { timeout: 30_000 },
  );
  const metrics = await page.evaluate(() => (window as any).__cbc.getMetrics());
  expect(metrics.graphSource).toBe('malecns');
  expect(metrics.neural3dNodes).toBeGreaterThan(1000);
});

test('console operator fly works the panel', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => (window as any).__cbc?.getMetrics().flyLoaded);
  await page.waitForTimeout(600);
  const metrics = await page.evaluate(() => (window as any).__cbc.getMetrics());
  const [x, y, z] = metrics.operatorDisplay as number[];
  // The fly now hovers over the console to work every knob, so assert it stays
  // within the panel/console region rather than at one fixed spot.
  expect(x).toBeGreaterThan(-70);
  expect(x).toBeLessThan(-46);
  expect(z).toBeGreaterThan(-24);
  expect(z).toBeLessThan(8);
  expect(y).toBeGreaterThan(-13.5);
  expect(y).toBeLessThan(-5);
});

test('operator forelegs work all 43 console knobs', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => (window as any).__cbc?.getMetrics().flyLoaded);
  await page.waitForFunction(() => {
    const m = (window as any).__cbc.getMetrics();
    return (m.activeKnobs as string[]).length >= 43 && (m.flyForelegMotion as number) > 0.002;
  }, undefined, { timeout: 20_000 });
  const metrics = await page.evaluate(() => (window as any).__cbc.getMetrics());
  expect((metrics.activeKnobs as string[]).length).toBe(43);
  expect(metrics.flyForelegMotion).toBeGreaterThan(0);
  // Over a short window the claws must actually land on console knobs.
  await expect.poll(async () => {
    const m = await page.evaluate(() => (window as any).__cbc.getMetrics());
    const tips = m.forelegDiagnostics.tips as number[][];
    const targets = m.forelegDiagnostics.targets as number[][];
    return Math.min(...tips.map((tip) =>
      Math.min(...targets.map((t) => Math.hypot(tip[0] - t[0], tip[1] - t[1], tip[2] - t[2]))),
    ));
  }, { timeout: 10_000 }).toBeLessThan(0.9);
});

test('camera pose readout, copy and trajectory playback', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/');
  await page.waitForTimeout(1000);
  await expect(page.locator('#camera-pose')).toContainText('pos');
  const options = await page.locator('#cam-trajectory option').allTextContents();
  expect(options).toContain('authored 5-key fly-in');
  expect(options).toContain('authored 6-key orbit');
  await page.getByRole('button', { name: 'Copy pose' }).click();
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  expect(clip).toContain('"position"');
  expect(clip).toContain('"yaw_deg"');

  const before = await page.evaluate(() => (window as any).__cbc.getMetrics().camera.position as number[]);
  await page.locator('#cam-trajectory').selectOption('1');
  await page.getByRole('button', { name: 'Play' }).click();
  await page.waitForTimeout(1500);
  const mid = await page.evaluate(() => (window as any).__cbc.getMetrics().camera.position as number[]);
  expect(Math.hypot(mid[0] - before[0], mid[1] - before[1], mid[2] - before[2])).toBeGreaterThan(2);
  await page.getByRole('button', { name: 'Stop' }).click();
});

test('authored cubic trajectory eases in and out', async ({ page }) => {
  await page.goto('/');
  await page.waitForTimeout(500);
  await page.locator('#cam-trajectory').selectOption({ label: 'authored 5-key fly-in' });
  await page.getByRole('button', { name: 'Play' }).click();
  const pos = () =>
    page.evaluate(() => (window as any).__cbc.getMetrics().camera.position as number[]);
  const p0 = await pos();
  const yaw0 = await page.evaluate(() => (window as any).__cbc.getMetrics().camera.yaw_deg as number);
  await page.waitForTimeout(250);
  const p1 = await pos();
  const startSpeed = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
  expect(startSpeed).toBeLessThan(6);

  // The curve passes through the authored first and last keyframes.
  expect(Math.hypot(p0[0] + 56.097, p0[1] - 17.169, p0[2] + 7.702)).toBeLessThan(0.5);

  await page.waitForTimeout(11800);
  const pA = await pos();
  await page.waitForTimeout(250);
  const pB = await pos();
  const endSpeed = Math.hypot(pB[0] - pA[0], pB[1] - pA[1], pB[2] - pA[2]);
  expect(endSpeed).toBeLessThan(6);
  expect(Math.hypot(pB[0] + 77.992, pB[1] - 21.694, pB[2] + 133.973)).toBeLessThan(1.5);
  // Orientation must be re-aimed during playback, not frozen.
  const yaw1 = await page.evaluate(() => (window as any).__cbc.getMetrics().camera.yaw_deg as number);
  expect(Math.abs(yaw1 - yaw0)).toBeGreaterThan(30);
});

test('viewer panels collapse toward their anchored edge', async ({ page }) => {
  await page.goto('/');
  for (const id of ['panel', 'hud', 'run-comparison']) {
    await page.locator(`[data-collapse="${id}"]`).click();
    await expect(page.locator(`#${id}`)).toHaveClass(/panel-collapsed/);
  }
});

test('target fly moves across the scene while the learner stays fixed', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'fly' }).click();
  await page.waitForFunction(() => (window as any).__cbc?.getMetrics().flyLoaded);
  const first = await page.evaluate(() => (window as any).__cbc.getMetrics().targetDisplay as number[]);
  await page.waitForTimeout(2500);
  const second = await page.evaluate(() => (window as any).__cbc.getMetrics().targetDisplay as number[]);
  const moved = Math.hypot(second[0] - first[0], second[1] - first[1]);
  expect(moved).toBeGreaterThan(1);
});

test('hardware bench loads on demand and routes cables', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Hardware bench (v2)', { exact: true }).check();
  await page.waitForFunction(
    () => (window as any).__cbc?.getMetrics().hardwareLoaded,
    undefined,
    { timeout: 60_000 },
  );
  const metrics = await page.evaluate(() => (window as any).__cbc.getMetrics());
  expect(metrics.hardwareError).toBeNull();
  expect(metrics.hardwareVisible).toBe(true);
  expect(metrics.hardwareNodes).toBeGreaterThan(20);
  expect(metrics.hardwareCables).toBeGreaterThan(200);
  // Seven GLB kinds are used. fc-bulkhead is intentionally unused because the
  // splitter already models its own sockets, and the superseded five-knob
  // fly-console prefab is retired in favor of the procedural 43-knob console.
  expect(metrics.hardwareComponentKinds).toBe(7);
  expect(metrics.hardwareAssetInstances).toBeGreaterThan(70);
  expect(metrics.hardwareConnectorInstances).toBeGreaterThan(100);
  expect(metrics.hardwareKnobs).toBe(43);
  expect(metrics.hardwareSelectors).toBe(19);
  expect(Number.isFinite(metrics.hardwareMinRadiusMm)).toBe(true);
  expect(metrics.hardwareMinRadiusMm).toBeGreaterThan(0);
  expect(Number.isFinite(metrics.hardwareClearanceViolations)).toBe(true);
  expect(metrics.hardwareFailures).toBe(0);
  await page.screenshot({ path: 'test-results/hardware-bench.png' });
});
