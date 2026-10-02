export interface Geometry {
  domain: { x: number[]; y: number[] };
  optical_window: { x: number[]; y: number[] } | null;
  regions: { name: string; material: string; poly: number[][] }[];
  electrodes: { name: string; material: string; weight: number; poly: number[][] }[];
}
export interface ParsedConfig {
  raw: {
    id: string; title?: string; paper_id?: string; validation_status?: string;
    chain?: string[]; targets?: Record<string, unknown>[];
    provenance?: Record<string, { class: string; locator?: string; citation?: string; note?: string }>;
    missing?: unknown[]; limitations?: string[];
  };
  geometries: Record<string, Geometry>;
}
export function parseConfig(text: string): ParsedConfig;
/** Geometry and disclosures only; never use preview as solver input. */
export function inspectConfig(text: string): { preview: ParsedConfig; solveError: string };
