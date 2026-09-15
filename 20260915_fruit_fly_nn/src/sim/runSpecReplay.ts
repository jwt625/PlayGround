import { defaultRunSpec, type RunSpec, type TaskKind } from "./runner";

/** Restore the actual training configuration; legacy manifests retain known fields. */
export function replayRunSpec(manifest: Record<string, unknown>, savedSpec?: RunSpec): RunSpec {
  if (savedSpec) return savedSpec;
  const spec = defaultRunSpec(manifest.task as TaskKind);
  if (manifest.connectomeSource === "provided-json") {
    throw new Error("Legacy JSON-connectome run lacks its graph; supply a saved run-spec.json before replay.");
  }
  if (manifest.reservoir) spec.reservoir = { ...spec.reservoir, ...manifest.reservoir as RunSpec["reservoir"] };
  if (manifest.environment) spec.environment = { ...spec.environment, ...manifest.environment as RunSpec["environment"] };
  if (manifest.train) spec.train = manifest.train as RunSpec["train"];
  if (manifest.malecns) spec.malecns = manifest.malecns as RunSpec["malecns"];
  return spec;
}
