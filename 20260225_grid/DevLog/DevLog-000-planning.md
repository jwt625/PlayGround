# Financial Model Framework: AI Data Center Scenarios (2026)
### Capacity Cases: 10 MW (Edge/Inference) | 100 MW (Regional/Fine-Tuning) | 1 GW (Hyperscale Training)

## 1. Design Scenarios for Comparison
| Scenario | Power Source | Infrastructure | Primary Market Target |
| :--- | :--- | :--- | :--- |
| **A: Legacy Grid** | Utility Substation + Diesel Backup | AC Distribution (480V) | Wholesale Colocation / Cloud |
| **B: Off-Grid SOFC** | NatGas SOFC + DC Buffer BESS | 800V DC + Trigeneration | High-Density AI / Sovereign Nodes |
| **C: Off-Grid Turbine** | Gas Turbines (Aeroderivative) | AC Distribution | Industrial / Rapid Deploy clusters |
| **D: Green Microgrid** | Solar + BESS + SOEC + SOFC | 800V DC / Hydrogen Loop | Net-Zero / Frontier Training |

---

## 2. Capacity Scaling Factors & Logic
The model must treat the three capacity tiers with different logic for land, utility, and networking:

* **10 MW (The "Fast-Track" Node):** Focus on **Speed-to-Market**. Usually fits on existing industrial plots. Permitting is local/state.
* **100 MW (The "Regional" Hub):** Focus on **Grid Congestion**. Likely requires a dedicated substation if grid-tied. Off-grid SOFC requires major high-pressure gas trunk lines.
* **1 GW (The "Hyperscale" Campus):** Focus on **Power Generation**. Off-grid at this scale behaves like a utility-scale power plant. Requires significant land for trigeneration heat rejection and potentially on-site LNG storage.

---

## 3. Technical & Operational Input Variables
* **IT Load Allocation:** Assume 80% of facility nameplate for IT (e.g., 800 MW IT load for a 1 GW site).
* **Compute Density:** Model next-gen racks (NVIDIA GB200 NVL72 or equivalent) at **120 kW - 150 kW per rack**.
* **PUE Assumptions:**
    * Grid/Turbine: 1.25 – 1.40 (Standard Air/Liquid mix).
    * SOFC + Trigeneration (Absorption Chillers): 1.05 – 1.15 (Waste heat to cooling).
* **Energy Costs:** $/kWh (Grid) vs. $/MMBtu (NatGas Pipeline) + Pipeline Tariff.
* **800V DC Efficiency:** Factor a **5-8% TCO reduction** via reduced conversion stages and copper savings.

---

## 4. Critical Financial Levers
### A. The "Time-to-Market" (TTM) Revenue Delta
* **SOFC Timeline:** 6–12 months (Site dependent).
* **Grid Timeline:** 24–48 months (Scaling with capacity; 1 GW can take 5+ years for transmission).
* **Lost Revenue:** Model ~$0.7M - $1M in gross compute revenue per MW/Month for Blackwell-class clusters.

### B. CAPEX & Subsidies (US - 2026)
* **SOFC CAPEX:** $4M–$7M per MW (Pre-subsidy). 
* **ITC (Investment Tax Credit):** Apply **30%–40% tax credit** to SOFC and BESS under IRA 2022/2025.
* **Depreciation:** 5-year MACRS for power; 3-year for GPU hardware.

### C. Lifecycle & Scaling Costs
* **Stack Replacement:** Recurring CAPEX every 4–5 years for SOFC chemical modules.
* **Economies of Scale:** Apply a **10-15% discount** on power hardware CAPEX when moving from 10 MW to 1 GW.
* **Land Use:** Solar Scenario (D) requires ~5-7 acres per MW. Model real estate costs accordingly.

---

## 5. Risks & Constraints (The "Gotchas")
* **The Pipeline Filter:** 1 GW requires a major interstate gas transmission line. Include cost/time for "Last Mile" gas interconnection.
* **The ESG Penalty:** Model the cost of **Renewable Natural Gas (RNG)** offsets for tenants requiring Net-Zero at SOFC sites.
* **The BESS Buffer:** SOFCs are base-load; include BESS CAPEX to handle "Step Loads" from AI inference/training cycles.
* **Geopolitical Tariffs:** If using non-US/Non-FTA compliant SOFCs, apply **25%+ Section 301 tariffs**.

---

## 6. High-Impact Factors to Explicitly Model
The following are the highest-impact variables on IRR, NPV, and payback and should be first-class inputs (not fixed assumptions):

* **Schedule Risk Engine (TTM Distribution):** Model COD as a probability distribution by critical path (utility interconnection, gas interconnect, permits, EPC labor, OEM slot).
* **Tariff Stack Decomposition:** Separate energy, demand, capacity, transmission, ancillary, and standby charges (for partial grid-connected scenarios).
* **Natural Gas Delivered Cost Stack:** Henry Hub + basis + firm transport + utility tariff + pressure/compression upgrades.
* **SOFC Lifecycle Curve:** Include degradation, forced outage rate, startup/ramp penalty, stack replacement cycle, and LTSA/O&M structure.
* **Balance-of-Plant (BOP) + Site Costs:** Substation/switchyard, gas receiving/compression, heat rejection/water, controls/cyber, fire-life safety, and permits.
* **Reliability Value / SLA Penalty:** Translate outages and derates into compute-revenue at-risk and unserved energy cost.
* **Workload Flexibility Coupling:** Explicitly model deferrable load, interruptible tranche, and workload migration value in dispatch economics.
* **Tax + Financing Structure:** Debt terms, DSCR covenants, depreciation, and incentive eligibility should be scenario-based.
* **Carbon / Compliance Exposure:** Include RNG/offset costs, emissions permit path, and future carbon price sensitivity.
* **Scale Nonlinearity (10 MW vs 100 MW vs 1 GW):** Use tier-specific curves for cost, schedule, and permitting complexity.

---

## 7. Market Players & Supply-Chain Reference (Added)
Detailed supplier/integrator reference and extracted metrics are captured in:

* `references/md/market_players_supply_chain_metrics_2026-03-01.md`
* `references/data/market_players_supply_chain_metrics_2026-03-01.csv`

High-level findings to incorporate into assumptions:

* **Turbine market is tight:** OEM commitments/backlogs and reservation-fee behavior imply schedule risk and potential prepayment requirements.
* **BESS has better volume visibility but still ramp risk:** Large shipment volumes exist, but integrator manufacturing ramp and commissioning bottlenecks remain key.
* **SOFC has credible data-center references but limited vendor depth:** De-risked at tens-to-100+ MW deployments, but concentration risk and service coverage matter.
* **SOEC is scaling but still early for core-load economics:** Treat as optional pathway in green microgrid scenario with conservative maturity assumptions.
* **EPC labor and backlog are material:** Large EPC backlogs should be translated into schedule multipliers and regional cost adders.

Modeling translation:

* Convert backlog/commitment signals into scenario multipliers for lead time (`base`, `tight`, `stressed`).
* Add supplier concentration penalty at 100 MW and 1 GW where single-OEM dependency is high.
* Run sensitivity on LTSA/O&M, stack replacement cadence, and EPC labor availability.

---

## 8. Model Output Metrics
* **Project IRR & NPV:** 5, 10, and 15-year horizons.
* **Payback Period:** Calculate the "Breakeven Month" comparing the early SOFC revenue vs. late Grid revenue.
* **Levelized Cost of Compute (LCOC):** Total TCO divided by estimated TFLOPS/Tokens delivered over the hardware lifecycle.

---

## 9. Progress Update (as of 2026-03-02)
### Completed
* Created a reproducible reference ingestion pipeline and captured raw + extracted sources in `references/`.
* Added browser-assisted extraction for blocked/dynamic pages (Playwright) and generated run manifests.
* Built supplier/integrator market reference pack with financial-model-relevant metrics:
  * `references/md/market_players_supply_chain_metrics_2026-03-01.md`
  * `references/data/market_players_supply_chain_metrics_2026-03-01.csv`
* Built core modeling assumptions config covering engineering + finance factors:
  * `config/assumptions_2026_us_ai_datacenter.yaml`
  * `config/README.md`
* Refined assumptions using sub-technology research (SOFC/MCFC variants, turbine variants, BESS/PV ranges) and added traceable source pointers in config.
* Added sub-technology extracted reference pack and range notes:
  * `references/md/subtech_parameter_ranges_2026-03-02.md`
  * `references/data/subtech_parameter_ranges_2026-03-02.csv`

### Data/Artifact Status
* Reference run manifests are available for audit/repro:
  * `references/manifest.json`
  * `references/manifest.playwright.json`
  * `references/manifest.subtech.json`
  * `references/manifest.subtech.additional.json`
  * `references/manifest.subtech.retries.json`
  * `references/manifest.subtech.playwright_retries.json`
* Sub-technology source extraction exists under:
  * `references/md/subtech/`
  * `references/raw/subtech/`

### Key Improvements Made Since Initial Draft
* Moved from single SOFC/turbine assumptions to **sub-technology-specific** parameterization.
* Tightened several parameter bands using recent primary-source values (e.g., BESS cost trajectories, turbine startup behavior, efficiency envelopes).
* Added explicit supply-chain and execution-risk factors as first-class model inputs.
* Added source traceability into assumption blocks to reduce “black box” estimates.

### Remaining Gaps / Next Work
* Add automated assumptions validator (`min <= mean <= max`, missing source tags, unit consistency checks).
* Add region-specific tariff and gas-delivery templates (PJM/ERCOT/CAISO variants) for site-level modeling.
* Add full scenario matrices for each scale (`10 MW`, `100 MW`, `1 GW`) with base/tight/stressed presets.
* Link assumptions config directly to simulation notebooks/scripts for deterministic + Monte Carlo runs.
