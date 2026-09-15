import { test, expect } from "@playwright/test";

test.describe("CBC browser app", () => {
  test("renders the bench, schematic, and live metrics", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await expect(page.locator("#view")).toBeVisible();
    await expect(page.locator("#m-mode")).toHaveText("analytic");
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
