import { defaultRunSpec, type RunSpec, type TaskKind } from "./runner";
import type { ConnectomeJson } from "../connectome/graph";

/**
 * Reconstruct a runnable `RunSpec` from a saved run manifest for reproducible
 * checkpoint replay (T10).
 *
 * Manifests written by `src/cli.ts` record the task, reservoir, environment and
 * training overrides, the connectome source, and the MaleCNS selection. A full
 * spec, when embedded, is authoritative. A `provided-json` source must carry its
 * graph JSON; otherwise replay fails loudly rather than silently substituting a
 * synthetic graph.
 */
export const MANIFEST_SCHEMA_VERSION = 1;

export interface RunManifest {
  manifestSchemaVersion?: number;
  task?: string;
  arrayOverrides?: RunSpec["arrayOverrides"];
  array?: { id?: string };
  reservoir?: Partial<RunSpec["reservoir"]>;
  environment?: Partial<RunSpec["environment"]>;
  train?: Partial<RunSpec["train"]>;
  malecns?: RunSpec["malecns"] | null;
  connectomeSource?: string;
  connectomeJson?: ConnectomeJson;
  /** Optional embedded full spec for exact replay. */
  spec?: RunSpec;
}

export function replayRunSpec(manifest: RunManifest, savedSpec?: RunSpec): RunSpec {
  if (savedSpec) return deepCloneSpec(savedSpec);
  if (manifest.spec) return deepCloneSpec(manifest.spec);

  const task = (manifest.task as TaskKind) ?? "phase_lock";
  const spec = defaultRunSpec(task);
  if (manifest.arrayOverrides) Object.assign(spec.arrayOverrides, manifest.arrayOverrides);
  if (manifest.array?.id) spec.arrayOverrides.id = manifest.array.id;
  if (manifest.reservoir) Object.assign(spec.reservoir, manifest.reservoir);
  if (manifest.environment) Object.assign(spec.environment, manifest.environment);
  if (manifest.train) Object.assign(spec.train, manifest.train);

  if (manifest.malecns) {
    spec.malecns = {
      maxNeurons: manifest.malecns.maxNeurons,
      seed: manifest.malecns.seed,
      minSynapses: manifest.malecns.minSynapses,
      maxEdges: manifest.malecns.maxEdges,
      cacheDir: manifest.malecns.cacheDir,
    };
  }

  if (manifest.connectomeSource === "provided-json") {
    if (!manifest.connectomeJson) {
      throw new Error(
        "replayRunSpec: manifest declares connectomeSource 'provided-json' but lacks its graph JSON",
      );
    }
    spec.connectomeJson = manifest.connectomeJson;
  }

  return spec;
}

function deepCloneSpec(spec: RunSpec): RunSpec {
  return JSON.parse(JSON.stringify(spec)) as RunSpec;
}
