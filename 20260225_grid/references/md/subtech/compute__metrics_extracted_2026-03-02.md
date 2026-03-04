---
category: compute
key: compute_metrics_extracted_2026_03_02
retrieved_at_utc: 2026-03-02T00:00:00Z
source_type: derived_summary_from_local_references
sources:
  - references/md/subtech/compute__coreweave_s1a1_2025.md
  - references/md/subtech/compute__nvidia_enterprise_support.md
  - references/md/subtech/compute__nvidia_warranty.md
  - references/md/subtech/compute__google_dram_errors.md
  - references/md/subtech/compute__fail_slow_scale_arxiv.md
---

# Extracted Metrics for Compute CAPEX/OPEX/RMA Modeling

This note consolidates source-backed signals used for compute economics assumptions in:
`config/assumptions_2026_us_ai_datacenter.yaml`.

## 1) Compute asset life / refresh signals

- CoreWeave S-1/A provides useful-life context for technology equipment and depreciation policy changes.
- Practical planning implication for AI clusters: use a 3-5 year useful-life range with explicit refresh events.

Model mapping:
- `compute_asset_useful_life_years`: 3 / 4 / 5
- `compute_refresh_fraction_at_useful_life`: 0.40 / 0.60 / 0.85

## 2) Compute OPEX signals (service + software ops)

- NVIDIA enterprise support materials indicate paid enterprise support paths, onsite engineer services, and onsite spare programs.
- Warranty + support structure implies recurring hardware service overhead and additional software/platform ops cost.

Model mapping:
- `annual_compute_software_ops_fraction`: 0.03 / 0.08 / 0.15
- `annual_compute_hw_service_fraction`: 0.02 / 0.05 / 0.10

## 3) RMA / repair reserve signals

- NVIDIA enterprise support and warranty paths establish RMA/repair operational workflow and entitlement boundaries.
- Google fleet DRAM error study shows non-trivial annual hardware fault incidence at scale.
- Fail-slow-at-scale literature supports adding reserve for non-catastrophic but throughput-impacting failures.

Model mapping:
- `annual_compute_rma_repair_fraction`: 0.01 / 0.03 / 0.07
- `compute_spares_pool_fraction`: 0.02 / 0.06 / 0.12

## 4) Compute CAPEX scale signal

- AI cluster deployments imply compute CAPEX dominates power/BOP CAPEX for GPU-heavy sites.
- Planning range captures generation variance and procurement dispersion.

Model mapping:
- `compute_capex_usd_per_mw_it`: 15M / 28M / 50M

## Notes

- These are planning ranges, not fixed market quotes.
- For strict project underwriting, replace with vendor quote packs and current-term service contracts by sub-technology generation.
