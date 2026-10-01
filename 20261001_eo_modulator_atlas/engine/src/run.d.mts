export interface TargetResult {
  metric: string; value: number; at_ghz?: number; actual: number | null;
  tolerance: number; status: 'pass' | 'fail' | 'not_evaluated'; reason: string | null;
  source?: Record<string, unknown>;
}
export interface SimulationResult {
  id: string; section: string; scope: string; stages: string[]; pendingStages: string[];
  metrics: Record<string, number>; targets: TargetResult[]; warnings: string[];
  targetSummary: { total: number; evaluated: number; passed: number; failed: number };
  meshScale: number; elapsed_ms: number;
  electrostatics: { nNodes: number; nTris: number; energyChargeRelativeError: number };
  optical: { nNodes: number; nTris: number; iterations: number; form: string } | null;
}
export function runCrossSection(text: string, options?: {
  section?: string; optical?: boolean; meshScale?: number; onProgress?: (message: string) => void;
}): SimulationResult;
