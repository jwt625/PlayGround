/**
 * Capture a short demo clip of the running app (T10). Builds the client, starts
 * a preview server, records the scene across controller modes and a moving fly
 * target, and writes a webm under outputs/videos/.
 *
 * Usage: npm run make-video
 */
import { execSync, spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { chromium } from "@playwright/test";

const PORT = 4175;
const URL = `http://localhost:${PORT}/`;

console.log("[make-video] building client...");
execSync("npm run build", { stdio: "inherit" });

const server = spawn("npm", ["run", "preview", "--", "--port", String(PORT), "--strictPort"], {
  stdio: "ignore",
});

async function waitForServer() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(URL);
      if (res.ok) return;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("preview server did not start");
}

try {
  await waitForServer();
  mkdirSync("outputs/videos", { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: { dir: "outputs/videos", size: { width: 1280, height: 720 } },
  });
  const page = await context.newPage();
  await page.goto(URL);
  await page.waitForTimeout(2000);

  const scene = async (label, button, ms) => {
    console.log(`[make-video] scene: ${label}`);
    await page.getByRole("button", { name: button }).click();
    await page.waitForTimeout(ms);
  };

  await scene("analytic phase lock", "analytic", 3000);
  await scene("spgd convergence", "spgd", 4000);
  await scene("fly target (tracking)", "fly", 7000);
  await scene("connectome control", "connectome", 4000);

  const video = page.video();
  await context.close();
  await browser.close();
  if (video) console.log(`[make-video] wrote ${await video.path()}`);
} finally {
  server.kill("SIGTERM");
}
