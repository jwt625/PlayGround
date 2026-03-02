---
title: Market Players and Supply-Chain Metrics for AI Data Center Power Designs
as_of_date: 2026-03-01
prepared_for: SOFC / off-grid-behind-the-meter financial modeling
coverage: Solar, BESS, Gas Turbine, SOFC/SOEC, Integrators
---

## Scope and method
- Focused on primary sources (company IR releases, company annual/quarterly publications, SEC pages, OEM official sites).
- Extracted metrics most relevant to financial modeling: `manufacturing/output scale`, `backlog/commitments`, `execution signals`, and `lead-time proxies`.
- Note: many OEMs do not publish explicit delivery lead times; backlog + reservation structures are used as proxies.

## Core players and extracted metrics

### Solar (modules + storage-adjacent suppliers)
- First Solar (module OEM)
  - 2025 guidance volume sold: `18–20 GW`; capex guidance: `$1.3B–$1.5B`.
  - 2024 net bookings: `4.4 GW` at avg selling price `30.5 c/W` (ex adjustments).
  - Modeling use: ASP trajectory, booked-volume certainty, expansion capex burden.
  - Source: https://www.businesswire.com/news/home/20250225031936/en/First-Solar-Inc.-Announces-Fourth-Quarter-and-Full-Year-2024-Financial-Results-and-2025-Guidance

- Trina Solar (module + storage supplier)
  - 2023 module shipments: `65.21 GW`.
  - Cumulative shipments by Q1 2024: `>205 GW`; storage shipments by end-2023: `5 GWh`.
  - Modeling use: supplier scale and shipment continuity.
  - Source: https://www.trinasolar.com/us/resources/newsroom/2023-annual-report/

- Canadian Solar / e-STORAGE (module + BESS)
  - 2024 module shipments: `31.1 GW`.
  - 2024 battery storage shipped volume: `6.6 GWh`.
  - As-of Dec 31, 2024: BESS manufacturing capacity `20.0 GWh`, target `30.0 GWh` by Dec 31, 2025.
  - Modeling use: integrated supplier optionality and expansion execution risk.
  - Source: https://www.sec.gov/Archives/edgar/data/1375877/000141057825001046/csiq-20241231x20f.htm
  - Source: https://investors.canadiansolar.com/news-releases/news-release-details/canadian-solar-files-annual-report-form-20-f-year-ended-10

### BESS (systems + integrators)
- Tesla Energy (Megapack)
  - 2025 storage deployments: `46.7 GWh`; Q4 2025 deployments: `14.2 GWh`.
  - Lathrop Megafactory stated capacity: `10,000 Megapacks/year` (`40 GWh/year`).
  - Modeling use: large-scale supply availability and benchmark deployment velocity.
  - Source: https://ir.tesla.com/press-release/tesla-fourth-quarter-2025-production-deliveries-deployments
  - Source: https://www.tesla.com/megafactory

- Fluence (BESS integrator/platform)
  - As of Jun 30, 2025: deployed `16.7 GWh`; contracted backlog `8.2 GW`; pipeline `114.3 GWh`.
  - Management commentary indicates slower-than-expected production ramp at newer U.S. facilities (execution risk signal).
  - Modeling use: delivery-risk multipliers and EPC/commissioning schedule assumptions.
  - Source: https://ir.fluenceenergy.com/news-releases/news-release-details/fluence-energy-inc-reports-third-quarter-2025-results-reaffirms/

### Gas turbines (dispatchable generation OEMs)
- Siemens Energy (Gas Services + Grid)
  - Q1 FY26: Gas Services highest order intake ever; `102 gas turbines`, `13 GW booked`, `80 GW total commitments`, `22 GW` data-center-related commitments.
  - Gas Services backlog: `€60bn`; group backlog: `€146bn`.
  - Includes advance payments/reservation-fee dynamics in disclosure.
  - Modeling use: turbine supply tightness, reservation/prepayment needs, long equipment queues.
  - Source: https://www.siemens-energy.com/global/en/home/investor-relations.html
  - Source (Q1 FY26 earnings PDF): https://p3.aprimocdn.net/siemensenergy/f8b7f2d2-e4f2-4fe5-8295-b3ee005765a4/2026-02-11_Earnings-Release-Q1-FY26-pdf_Original%20file.pdf
  - Source (Q1 FY26 analyst PDF): https://p3.aprimocdn.net/siemensenergy/d5e24758-5d19-44f7-b57d-b3ee0057e335/2026-02-11_Q1_Analyst_presentation-pdf_Original%20file.pdf

- GE Vernova
  - Installed base: `>7,000 gas turbines` (largest in world per company statement).
  - Services backlog: `$86bn`; total backlog up to `$150bn` in 2025.
  - Modeling use: strong incumbency in services/LTSA and evidence of constrained high-demand market.
  - Source: https://www.gevernova.com/investors/annual-report/ceo-letter

- Mitsubishi Power / MHI
  - Reported 2023 global gas turbine market share: `36%` (McCoy report cited by company).
  - Company outlook emphasizes substantially increased >100 MW gas turbine demand forecast and supply-chain pressure.
  - Modeling use: supplier concentration and OEM concentration risk.
  - Source: https://power.mhi.com/news/240315.html
  - Source: https://power.mhi.com/regions/amer/insights/us-power-outlook-and-long-term-trends

### SOFC / SOEC
- Bloom Energy (SOFC)
  - Equinix collaboration exceeds `100 MW`; `~75 MW` operational and `30 MW` under construction; across `19` data centers.
  - 10-year collaboration structure.
  - Modeling use: bankable reference for data-center BTM deployment and contract tenor assumptions.
  - Source: https://investor.bloomenergy.com/press-releases/press-release-details/2025/Bloom-Energy-Expands-Data-Center-Power-Agreement-with-Equinix-Surpassing-100MW/default.aspx

- FuelCell Energy (carbonate fuel cells)
  - Backlog: `$1.19bn` as of Oct 31, 2025.
  - Entered 20-year PPA for `7.4 MW` Hartford project; stated contract revenue `~$167.4m` over term.
  - Ongoing module replacement cadence disclosed into FY2026.
  - Modeling use: long-term contract revenue profile, module replacement cadence assumptions.
  - Source: https://investor.fce.com/press-releases/press-release-details/2025/FuelCell-Energy-Ends-FY2025-with-Revenue-Growth-and-a-Focus-on-Data-Center-Opportunities/default.aspx

- Topsoe (SOEC)
  - Herning SOEC facility: `500 MW/year` initial stack manufacturing capacity, scalable; `23,000 m²` facility.
  - Claimed efficiency uplift: ~`20%` vs alkaline/PEM (up to `30%` with steam integration).
  - Modeling use: SOEC technology maturity and expansion path for hydrogen loop scenarios.
  - Source: https://www.topsoe.com/news/topsoe-inaugurates-europes-largest-soec-manufacturing-facility

### EPC / system integration layer (site + interconnect execution)
- Quanta Services (grid/power EPC)
  - Q4/FY2025 record backlog: `$44.0bn`.
  - Modeling use: transmission/substation/EPC availability as COD driver.
  - Source: https://investors.quantaservices.com/news-events/press-releases/detail/390/quanta-services-reports-fourth-quarter-and-full-year-2025-results

- Primoris (utilities + energy EPC)
  - Backlog as of Jun 30, 2025: `$11.5bn` (`$6.0bn` Utilities, `$5.5bn` Energy).
  - Modeling use: EPC market depth and schedule competition in utility + energy builds.
  - Source: https://ir.prim.com/news-and-events/news-releases/2025/08-04-2025-211559739

- Mortenson (large storage EPC examples)
  - Nova Power Bank phases announced: `620 MW / 2,480 MWh` with expansion phase under construction.
  - Modeling use: reference complexity for utility-scale BESS EPC timelines.
  - Source: https://www.mortenson.com/news-insights/decommissioned-gas-plant-energy-storage-facility

## Financial-model variables this evidence supports
- `Lead-time proxies`: OEM backlog, commitments, reservation fees, and under-construction vs operational split.
- `Production-volume proxies`: annual manufacturing capacity and annual/quarterly deployment/shipments.
- `Supplier diversity`: concentration by technology class (especially gas turbines and SOFC), plus EPC depth.
- `Execution risk`: manufacturing ramp comments, backlog conversion risk, and commissioning cadence.

## Suggested parameterization bands (first pass)
- Gas turbine equipment schedule risk multiplier: `1.15x–1.45x` base schedule in high-demand regions.
- BESS schedule risk multiplier: `1.05x–1.30x` depending on domestic-content constraints and integrator queue.
- SOFC project ramp risk multiplier: `1.10x–1.40x` depending on fuel interconnect + permitting path.
- EPC scarcity adder (large campuses): `+5% to +18%` on labor/EPC package costs depending on region.

## Data gaps to close next
- Explicit OEM quoted lead times by frame/class (especially >100 MW turbines) from latest call transcripts.
- SOFC stack replacement cost curves by operating regime (cycling vs baseload).
- Region-specific BOP cost libraries (substation, gas receiving, water/thermal rejection, controls).
