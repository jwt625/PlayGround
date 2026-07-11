import { loadMeshArtifact } from "./workspace_io.js";

self.addEventListener("message", async (event) => {
  try {
    const text = await event.data.file.text();
    const artifact = loadMeshArtifact(text, event.data.file.name);
    self.postMessage({ ok: true, artifact });
  } catch (error) {
    self.postMessage({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
