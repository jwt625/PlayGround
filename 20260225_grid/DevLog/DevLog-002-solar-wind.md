# Paper Summary: Hosting Capacity of Combined Wind + Solar DER

**Paper:** Hosting capacity considerations for the combination of wind and solar on distribution electric power systems subject to different levels of coincident operations  
**Source:** Sandia / Journal of Renewable & Sustainable Energy (2025)

---

# 1. Datasets Used (concise + links)

## Weather / Renewable resource datasets
- **NREL WIND Toolkit (Wind Integration National Dataset)**
  - Wind speed, direction, etc. at high spatial/temporal resolution
  - Used for wind generation modeling
  - Link: https://www.nrel.gov/grid/wind-toolkit.html  
  - Direct data portal: https://developer.nrel.gov/docs/wind/wind-toolkit/

- **NREL National Solar Radiation Database (NSRDB)**
  - Solar irradiance + temperature time series
  - Used for PV modeling
  - Link: https://nsrdb.nrel.gov/
  - API access: https://developer.nrel.gov/docs/solar/nsrdb/

---

## Load datasets
- **OpenEI End-Use Load Profiles for U.S. Building Stock**
  - Hourly residential / commercial / industrial load demand
  - Link: https://openei.org/datasets/dataset/end-use-load-profiles-for-the-us-building-stock  
  - OEDI portal: https://data.openei.org/submissions/4520

---

## Grid / system datasets
- **SMART-DS Synthetic Distribution Feeder Models**
  - Used Santa Fe feeder “uhs01247–udt4776”
  - Link: https://github.com/NREL/SMART-DS  
  - Documentation: https://www.nrel.gov/grid/smart-ds.html

- **Open Energy Data Initiative (OEDI)**
  - Additional load + energy datasets
  - Link: https://www.nrel.gov/data/oedi.html

---

## Modeling tools (not datasets but required)
- **PVWatts / PVLIB**
  - PV generation modeling
  - PVWatts: https://pvwatts.nrel.gov/  
  - PVLIB: https://pvlib-python.readthedocs.io/

- **windpowerlib**
  - Wind turbine generation simulation
  - Link: https://windpowerlib.readthedocs.io/

- **OpenDSS**
  - Distribution power-flow + QSTS simulation
  - Link: https://www.epri.com/pages/sa/opendss

---

# 2. Methodology (concise)

## Step 1 — Generate PV + Wind time-series
- Use **NSRDB weather** → PVWatts/PVLIB → hourly PV output
- Use **WIND Toolkit wind speed** → windpowerlib → hourly WT output

---

## Step 2 — Correlation analysis
For each location:

\[
R = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}
\]

Where:
- X = PV hourly generation
- Y = wind hourly generation

Interpretation:
- **R > 0** → concurrent generation → higher grid stress
- **R < 0** → complementary generation → smoother net output
- **R ≈ 0** → weak interaction

---

## Step 3 — Spatial clustering
- Features: seasonal correlation values  
  `[R_winter, R_spring, R_summer, R_fall]`
- Algorithm: **K-means++**
- Optimal clusters: **k = 8**
- Cluster centroids → representative sites for simulation  
  (reduces computational cost vs nationwide simulation)

---

## Step 4 — Hosting Capacity (HC) simulation
For each representative site:

1. Generate full-year PV + wind + load profiles
2. Apply DER penetration scenarios:
   - Scenario 0: no DER
   - Scenario 1–3: increasing PV + WT deployment
3. Run **QSTS (quasi-static time series)** simulation in OpenDSS using SMART-DS feeder
4. Evaluate grid performance metrics:
   - Voltage
   - Line loading
   - Transformer loading

---

# 3. Key Calculations

## PV generation model (PVWatts)
\[
P_{dc} = \frac{G_{poa}}{1000} P_{dc0}\left(1 + c_{pdc}(T_{cell}-T_{ref})\right)
\]

Inputs:
- Plane-of-array irradiance
- Module temperature
- Nameplate rating
- Temperature coefficient

---

## Wind generation model
- Windpowerlib converts wind speed → power via **turbine power curve**
- Turbine used: **Bergey Excel 10 (8.9 kW)**

---

## Clustering objective
\[
J = \sum_{i=1}^{k} \sum_{x_j \in C_i} ||x_j - \mu_i||^2
\]

Minimizes within-cluster correlation variance.

---

# 4. Findings (short)

- **Concurrent PV + wind (positive R)** → higher line & transformer loading → lower hosting capacity.
- **Complementary generation (negative R)** → spreads generation across 24h → reduces grid stress.
- Voltage violations are driven mostly by **PV**, not correlation alone.
- Correlation is a good **predictor of thermal loading**, but weaker predictor of voltage issues.

---

# 5. Reproducible Workflow (practical interpretation)

You can reproduce the paper with this pipeline:

1. Download weather data:
   - NSRDB + WIND Toolkit
2. Generate hourly PV + wind time series
3. Compute seasonal correlation per location
4. Cluster locations → select representative sites
5. Load SMART-DS feeder into OpenDSS
6. Run QSTS with increasing DER penetration
7. Extract voltage / line / transformer violations
8. Plot correlation vs hosting capacity metrics

---

# 6. My Technical Assessment

## Strengths
- Uses **real high-resolution datasets (NREL)** → realistic results
- Smart reduction of compute via **correlation clustering**
- Demonstrates interaction effects that PV-only studies miss

## Limitations
- Single feeder topology → results not fully generalizable
- No storage, controls, or inverter VAR support modeled
- Extreme weather coincidence events not evaluated

## Research insight
The most important takeaway:

> Hosting capacity is not only about DER penetration level —  
> **temporal correlation between resources is a first-order driver of thermal limits.**

This suggests:
- Hybrid PV + wind siting should consider **correlation maps**
- Negative-correlation regions can host significantly more DER without upgrades

---

# 7. If you want to extend this work

Possible improvements:
- Add **battery storage dispatch optimization**
- Include **volt-VAR inverter control**
- Run Monte-Carlo DER placement instead of fixed allocation
- Use multiple feeder topologies for generalization

---

# End