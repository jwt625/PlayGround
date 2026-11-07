---
source: Irrational Analysis, "Avicena Deep-Dive" (substack post, captured as 153-page browser PDF on 2026-11-06)
url: https://irrationalanalysis.substack.com/p/avicena-deep-dive
accessed: 2026-11-06
note: Raw PDF not stored in the repo (subscriber content, 94 MB). Page numbers refer to the captured PDF.
---

# Notes relevant to MMF GRIN profile design

Facts as stated in the article or on Avicena slides photographed in it. Not independently verified.

| Item | Value / statement | Page |
|---|---|---|
| Operating wavelength | 430 nm ("430nm operating wavelength"; table "@430nm") | 70, 77 |
| Dispersion penalty method | Simple ISI calculation, ref IEEE 802.3z link model; no CTLE/DFE credit | 77 |
| Loss at 430 nm | fused silica 0.09 dB/m; borosilicate and PMMA 0.6-1 dB/m | 70 |
| Chromatic penalty slide | assumes 10 nm FWHM; author: "10nm FWHM is wrong", real penalty ~2.5x higher | 76 |
| Step-index modal penalty | penalty grows quickly with NA (0.2 .. 0.46 curves); "GRIN structure can lower effective NA" | 72 |
| OM3 measured profile | peak delta n ~0.015, core ~50 um, trench ~-0.006 outside core; compared to 0.22 NA step | 73 |
| Custom fibers | "cheaper" and "MUCH CHEAPER" custom GRIN profiles (redacted); smooth = many layers = expensive | 74 |
| Key design problem | "maximize coupling efficiency while minimizing modal dispersion" | 74 |
| Launch | underfilled launch: more launch loss, lower dispersion; overfilled: higher dispersion | 77 |
| Coupling loss | Tx lens to GRIN fiber bundle ~4.2-4.7 dB; total link -6.2 to -7.2 dB | 82 |
| Alignment tolerance | +/- 5 um all axes (more in z) | 65 |
| Reach target | 1.6T gearboxed AOC, 10 m | 131 |
| uVCSEL | needs OM3-class fiber; 850 nm AMS OSRAM uVCSEL, 1060 nm alternative | 85-86 |
| Profile precision | alpha_opt is wavelength dependent; alpha 2.100 peaks at 850 nm, 2.076 at 950 nm | 84 |

Paper shown on pp. 84-86: Photonics 13(5), 408 (2026), doi:10.3390/photonics13050408 (MDPI returned HTTP 403 to
automated fetch on 2026-11-06; content above read from the screenshots in the article only).
