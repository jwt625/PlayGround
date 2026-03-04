# DevLog-001: Interactive Financial Timeline and Uncertainty Visuals

## Objective
Build an interactive HTML visualization that shows project cash behavior over time with:
- uncertainty bands derived from parameter ranges,
- thick highlighted mean-parameter trajectory,
- hoverable statistics,
- toggleable assumptions/parameter text.

## Scope (V1)
- Input source: `config/assumptions_2026_us_ai_datacenter.yaml`
- Scales: `edge_10`, `regional_100`, `hyperscale_1000`
- Designs: `legacy_grid`, `offgrid_sofc`, `offgrid_turbine`, `green_microgrid`
- Primary visual: cumulative cashflow over monthly timeline with P10/P90 uncertainty band + mean line.

## Implementation Plan
1. Create simulation engine that consumes YAML ranges and runs Monte Carlo using triangular distributions.
2. Implement design-specific cost/revenue logic using assumptions in config.
3. Aggregate monthly trajectories to P10/P50/P90 plus mean-parameter deterministic line.
4. Generate interactive HTML (Plotly) with scenario selector and rich hover stats.
5. Add toggleable assumptions panel (text/JSON summary used for the run).
6. Save outputs to `outputs/` with timestamped artifacts.

## Progress
- [x] Planning document created.
- [x] Simulation engine implemented.
- [x] Interactive chart generation implemented.
- [x] Assumptions panel toggle implemented.
- [x] CLI and reproducible run command documented.

## Current Artifacts
- Generator script: `scripts/generate_interactive_financial_timeline.py`
- First output: `outputs/interactive_financial_timeline.html`
- Repro command:
  - `. .venv/bin/activate && python scripts/generate_interactive_financial_timeline.py --scale regional_100 --months 120 --samples 500 --out outputs/interactive_financial_timeline.html`

## V1 Notes
- Model equations are intentionally simplified to make uncertainty behavior visible first.
- After V1 visual validation, V2 will harden equations per formal model spec.

## Progress Update (2026-03-03)
### New work completed
- Added compute-revenue reference ingestion pipeline and artifacts:
  - `scripts/fetch_revenue_references.py`
  - `references/manifest.revenue.json`
  - `references/md/revenue/revenue__*.md`
  - `references/raw/revenue/*`
- Added derived benchmark note for vertical-integration revenue normalization:
  - `references/md/revenue/vertical_integration_revenue_metrics_2026-03-02.md`
- Updated assumptions config with profile-based revenue model:
  - `revenue_and_sla.selected_revenue_profile`
  - `revenue_and_sla.gross_compute_revenue_profiles`
    - `colocation_power_hosting`
    - `ai_infra_cloud_gpu_capacity`
    - `integrated_ai_platform`
- Updated simulator to support revenue profile selection:
  - config-selected default profile
  - CLI override: `--revenue-profile <profile>`
- Generated three profile-specific outputs:
  - `outputs/interactive_financial_timeline_colocation_power_hosting.html`
  - `outputs/interactive_financial_timeline_ai_infra_cloud_gpu_capacity.html`
  - `outputs/interactive_financial_timeline_integrated_ai_platform.html`

### New analysis (sanity check)
- Per-profile, per-design monthly component analysis (regional_100 mean assumptions) was run to validate economics.
- Key result: current colocation profile is economically inconsistent in the model:
  - Revenue (~$9M-$10M/month) is far below modeled operator-side compute OPEX+RMA (~$30M/month) plus infra OPEX.
  - This causes structurally negative operating cash even before CAPEX timing.
- Root cause:
  - Current equations always assign full compute CAPEX and compute service/RMA burden to the operator for all business models.
  - This conflicts with reported hosting structures where customer funds significant capex and power is pass-through.

### Accuracy gaps identified
- Missing business-model-specific cost-bearing logic (ownership split for compute CAPEX/OPEX/RMA).
- Missing pass-through accounting behavior for colocation contracts.
- Revenue equation too coarse:
  - not linked to utilization/sold capacity mix,
  - no decomposition into fixed capacity fee vs variable usage/services.
- Refresh lifecycle under-modeled (single refresh event only).
- Financing/tax/depreciation not yet included in cash model.

## Proposed V2 improvements
1. **Cost-bearing by revenue profile (highest priority)**
   - Add profile parameters for:
     - compute capex borne by operator fraction,
     - compute service/RMA borne by operator fraction,
     - power pass-through fraction.
   - Apply these directly in simulation cash equations.

2. **Revenue decomposition**
   - Split revenue into:
     - fixed reserved-capacity fees,
     - usage-driven fees,
     - managed-service/platform uplift.
   - Tie realized revenue to sold/active utilization curve and ramp.

3. **Lifecycle realism**
   - Replace one-time refresh with periodic refresh schedule over full horizon (e.g., 48/96/144 months).
   - Add option for staggered fleet refresh rather than step-function replacement.

4. **Contract mechanics**
   - Add contract term, escalators, take-or-pay, and prepaid credit amortization toggles for hosting-style deals.

5. **Finance layer**
   - Add debt/equity structure, interest, principal schedule, tax effects, and depreciation.
   - Produce unlevered and levered views side-by-side.

6. **Visual outputs**
   - Add a profile comparison dashboard view (three profiles on same canvas with aligned assumptions).
   - Add component waterfall and breakeven diagnostics to explain curve behavior.
