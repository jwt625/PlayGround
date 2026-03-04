---
title: Revenue Benchmarks by Vertical Integration (Derived)
as_of_date: 2026-03-02
method: derived_from_reported_company_disclosures
sources:
  - references/md/revenue/revenue__core_scientific_coreweave_update_2025_10_30.md
  - references/md/revenue/revenue__coreweave_q4_fy2025_results.md
  - references/md/revenue/revenue__nebius_q2_2025_results_ex99_1.md
  - references/md/revenue/revenue__nebius_q3_2025_shareholder_letter_ex99_2.md
---

# Purpose
Create traceable, company-anchored revenue assumption bands for `gross_compute_revenue` and business-model profiles.

# Benchmarks (reported inputs + implied calculations)

## 1) Colocation / powered-shell hosting (low vertical integration)
- Company: Core Scientific (8-K Exhibit 99.1, Oct 30, 2025)
- Reported:
  - `~$850M` average annual run-rate revenue
  - `~590 MW` infrastructure in CoreWeave contract summary
- Implied:
  - `$850M / 590 MW / 12 = ~$120k per MW-month`
- Notes:
  - Company also references `~800 MW gross`, which would imply a lower bound near `$88k per MW-month`.

## 2) AI infrastructure cloud / GPU capacity (medium vertical integration)
- Company: CoreWeave (Q4/FY2025 results)
- Reported:
  - FY2025 revenue: `$5,131M`
  - Active power capacity: `>850 MW`
- Implied:
  - `$5,131M / 850 MW / 12 = ~$503k per MW-month` (using 850 MW as conservative denominator)
- Notes:
  - Because active power is stated as "more than 850 MW", this estimate is directionally conservative.

## 3) Integrated AI infra platform (higher vertical integration, partial normalization)
- Company: Nebius (Q2 2025 results + 6-K disclosures)
- Reported:
  - ARR guidance (end-2025): `$900M - $1.1B`
  - Q2 revenue: `$105.1M`
  - In process of securing `>1 GW` of power by end-2026
  - Large dedicated-capacity contracts disclosed (including multi-year hyperscaler commitments)
- Interpretation:
  - Nebius provides strong evidence of downstream value-capture, but does not provide a same-period, directly comparable active-MW denominator for strict `$ / MW-month` normalization in the same disclosure set.
  - Therefore, this profile is anchored primarily by CoreWeave normalized metrics and expanded upward with contract-evidence context.

# Modeling translation used in config
- `colocation_power_hosting`: `90k / 120k / 170k` USD per MW-month
- `ai_infra_cloud_gpu_capacity`: `350k / 500k / 750k` USD per MW-month
- `integrated_ai_platform`: `600k / 950k / 1,600k` USD per MW-month

# Caveats
- These are planning ranges, not contract prices.
- Differences in denominator definitions (gross MW, billable MW, active MW, contracted MW) materially affect normalization.
- A tighter model should ingest customer-level price sheets and realized utilization by workload mix.
