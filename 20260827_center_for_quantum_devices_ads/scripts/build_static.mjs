#!/usr/bin/env node

import { cp, mkdir, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "dist");

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

for (const file of ["index.html", "styles.css", "app.js", "data.js"]) {
  await cp(path.join(root, file), path.join(output, file));
}
await cp(path.join(root, "public"), path.join(output, "public"), { recursive: true });

console.log("Built public-only site in dist/ (source and extraction files excluded)");
