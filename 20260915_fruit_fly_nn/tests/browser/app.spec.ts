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
  expect(metrics.hardwareCables).toBeGreaterThan(0);
  // All nine generated asset kinds are instantiated (including bulkheads).
  expect(metrics.hardwareComponentKinds).toBe(9);
  expect(metrics.hardwareAssetInstances).toBeGreaterThan(70);
  expect(metrics.hardwareConnectorInstances).toBeGreaterThan(100);
  expect(metrics.hardwareFailures).toBe(0);
  await page.screenshot({ path: 'test-results/hardware-bench.png' });
});
