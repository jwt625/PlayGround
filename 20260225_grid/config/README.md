# Modeling Assumptions Config

## File
- `assumptions_2026_us_ai_datacenter.yaml`

## Design
- Every numeric assumption is encoded as `min/mean/max`.
- Default uncertainty model is triangular (`min`, `mode=mean`, `max`).
- Sections cover both financial and engineering modeling.

## How to use
1. Select `scale_cases_mw` (`edge_10`, `regional_100`, `hyperscale_1000`).
2. Select `scenario_templates` (`base_case`, `low_cost_fast_track`, `stressed_supply_chain`, `carbon_constrained`).
3. For deterministic runs, take `mean` values.
4. For Monte Carlo, sample from triangular distributions and apply scenario overrides.

## Notes
- Parameters with policy sensitivity (e.g., SOFC incentive eligibility) are intentionally wide.
- Supply-chain lead times and multipliers are informed by the market players reference pack.
