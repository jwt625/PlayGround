# Research: pluggable form-factor timeline, XPO, companies, supply-chain choke points

Date: 2026-10-09. Method: research subagent, read-only. "2026 status" is the agent's assessment, not a sourced claim. [2nd] = secondary source only. LightCounting pages returned 403 (snippets only).

## A. Form factors

| Item | First spec / MSA | Rate x lanes | 2026 status | Source | Conf. |
|---|---|---|---|---|---|
| 1x9 soldered | N/A (no formal spec date found) | 155M-1G x1 | extinct | N/A | low |
| GBIC (SFF-8053) | Rev 5.5 dated 2000-09-27; ~1995 origin [2nd] | 1G x1 | extinct | snia.org/technology-communities/sff/specifications | med |
| SFP (INF-8074i) | 2001-05-01 | 1G x1 | legacy, shipping | en.wikipedia.org/wiki/Small_Form-factor_Pluggable [2nd] | med |
| SFP+ (SFF-8431) | Rev 4.1 2009-07-06; first published 2006-05-09 [2nd] | 10G x1 | mainstream (enterprise) | same; SFF-8431 copy | high |
| SFP28 (SFF-8402) | 2014-09-13 [2nd] | 25G x1 | mainstream | Wikipedia | med |
| SFP-DD / SFP112 / SFP-DD112 | SFP-DD MSA Rev 5.0 announced 2021-11-05 | 100G x1; 2x112G | niche | businesswire.com/news/home/20211105005655/en | high |
| XENPAK | MSA announced 2001-03 (Agere, Agilent) | 10G, 4x3.125G XAUI | extinct | lightwaveonline.com/home/article/16668974/msas-vie-for-10gbit-s | high |
| X2 | announced 2002-07; spec 2003-02-13 [2nd] | 10G XAUI | extinct | same + Wikipedia | med |
| XPAK | announced 2002-03-19; "not a commercial success" [2nd] | 10G XAUI | extinct | Wikipedia | med |
| XFP | MSA launched OFC 2002-03 | 10G serial | legacy | Lightwave | high |
| CFP | Rev 1.4 2010-06-07 | 100G (10x10G or 4x25G) | extinct for new designs | approvednetworks.com/content/msa/CFP_MSA.pdf | high |
| CFP2 / CFP4 / CFP8 | 2013-07-31 / 2014-08-28 / 2017-03-17 | 100-400G | CFP4, CFP8 extinct; CFP2 survives as DCO | approvednetworks.com/content/msa/CFP4_MSA.pdf, CFP8_MSA.pdf | high |
| CFP2-DCO | OIF-CFP2-DCO-01.0 2018-10; announced 2019-05-21 | 100-400G coherent | mainstream metro | oiforum.com | high |
| QSFP+ (SFF-8436) | 2012-04-01; QSFP INF-8438 2006-11 [2nd] | 4x10G | legacy | Wikipedia | med |
| QSFP28 (SFF-8665) | 2014-09-13 [2nd] | 4x25G | mainstream | Wikipedia | med |
| QSFP56 / QSFP112 | dates doubtful [2nd] | 4x50G / 4x100G | mainstream | Wikipedia | low |
| QSFP-DD | MSA formed 2016-03 (13 companies) | 8x50G | mainstream | lightwaveonline.com/articles/2016/03/qsfp-dd-msa-targets-200g-400g-optical-transceivers.html | high |
| QSFP-DD800 | MSA 2019-09 | 8x100G | mainstream | lightwaveonline.com (14040268) | high |
| QSFP-DD1600 | announced 2023-03-14 | 8x200G | ramping | lightwaveonline.com (14291021) | high |
| OSFP | MSA doc 2016-11-04; Arista-led, 49 members | 8x (16x later) | mainstream for AI | osfpmsa.org/assets/pdf/OSFP_MSA_20211004.pdf | high |
| OSFP-XD | spec targeted Q4 2022; Rev 1.1 | 16x100G / 16x200G | niche | osfpmsa.org/assets/pdf/OSFP1600_and_OSFP-XD.pdf | med |
| OSFP-RHS | optional riding heat sink, OSFP spec Rev 4.0 (2021-05), 9.5 mm tall | OSFP lanes | mainstream (NVIDIA NICs [2nd]) | osfpmsa.org/assets/pdf/OSFP_Module_Specification_Rev5_2.pdf | med |
| 400ZR | OIF-400ZR-01.0 2020-03; released 2020-04-29 | 400G DP-16QAM | mainstream | oiforum.com | high |
| OpenZR+ | Rev 1.0 2020-09/10 | 100-400G | mainstream | lightwaveonline.com (14184990) | high |
| LPO MSA | 2024-03-21; 12 founders (Accelink, AMD, Arista, Broadcom, Cisco, Eoptolink, Hisense, Innolight, Intel, MACOM, NVIDIA, Semtech) | 100G/lane | growing | businesswire.com/news/home/20240321794629/en | high |
| LRO | not an MSA; half-retimed (DSP on TX only); Credo Dove 850 2023-11-29 | 8x112G | niche | businesswire.com/news/home/20231129341796/en | high |
| AEC | Credo HiWire production 2019-09-04 | to 400G (2019), 800G later | mainstream in AI racks | Credo 2019 PDFs | high |

## B. XPO

- Name: eXtra-dense Pluggable Optics (expansion from nextplatform.com 2026-04-17 [2nd]; not spelled out in Arista PR or xpomsa.com).
- MSA announced by Arista 2026-03-11/12; debuted at OFC 2026 (Los Angeles). Founders Arista, Microsoft, Marvell, Broadcom, Ciena [2nd]; 100+ backers, 20+ module vendors [2nd].
- Per module 12.8 Tb/s (Arista PR); 64 lanes (xpomsa.com) x 200G (Eoptolink PR). Integrated liquid cold plate, up to 400 W. 204.8 Tb/s per OCP rack unit (= 4x 1.6T OSFP per Arista). About two OSFP footprints [2nd]; no primary dimensions. Demos at OFC 2026; volume expected 2027 [2nd].
- URLs: arista.com/en/company/news/press-release/23697-pr-20260311 ; xpomsa.com ; technode.global/prnasia/eoptolink-joins-xpo-msa-and-unveils-industry-first-12-8-tbps-liquid-cooled-pluggable-optics-for-ai-data-centers/

## C. Company roles (parody names in `../DevLog/DevLog-000-*.md`)

| Company | Role (2024-26) | Source | Conf. |
|---|---|---|---|
| Broadcom | PAM4 DSPs (Sian2, Sian3 3 nm 2025-03-25); EML, VCSEL, CW lasers (200G EML production, 200G VCSEL demo 2024-03-13) | investors.broadcom.com | high |
| Marvell | PAM4 DSPs (Nova 2 5 nm; Ara 3 nm 1.6T 2024-12-03; Spica Gen2); XPO founder | investor.marvell.com | high |
| Innolight | Module maker, #1 by revenue: $3.3B (2024), $5.3B (2025) [2nd]; operating entity InnoLight Technology (Suzhou); HK listing 2026-07-30 | lightcounting snippets; innolight.com; hkex.com.hk | med |
| Eoptolink | Module maker, Chengdu; #3 on LightCounting 2024 list (from #7) [2nd]; XPO 12.8T module | Eoptolink 2026-03-12 release | high / med |
| Coherent Corp | GaAs VCSELs, InP DML/EML/CW lasers, transceivers; 6-inch InP | sec.gov 10-K FY2025 | high |
| Lumentum | EML, CW, pump lasers; modules via Cloud Light (acq. 2023-11) | sec.gov 10-K FY2025 | high |
| MaxLinear | Keystone 5 nm 800G PAM4 DSP (production 2023-03) | maxlinear.com | high |
| Credo | AECs; optical DSPs (Dove, Bluebird 3 nm 1.6T; Robin 400 for 400G FR4) | credosemi.com | high |
| Fabrinet | Contract manufacturer, Thailand (Pinehurst, Chonburi); FY2025 revenue NVIDIA 27.6%, Cisco 18.2% | sec.gov fn-20250627 | high |
| Mitsubishi Electric | 200G EML mass production 2024-04 | mitsubishielectric.co.jp/ja/pr/2024/0301/ | high |
| Sumitomo Electric | InP substrates (3.1x FY2024 capacity by FY2028), EMLs | trendforce.com/news/?p=60600 [2nd] | med |
| Semtech | FiberEdge TIAs/drivers; 200G TIA GN1834D 2025-09 | semtech.com | high |
| MACOM | PURE DRIVE linear drivers, TIAs, 200G VCSEL driver | ir.macom.com | high |
| Hisense Broadband | Module maker, Qingdao [2nd] | itiger.com | med |
| Accelink | Modules and components, Wuhan, CICT-controlled [2nd] | filings aggregators | med |
| Source Photonics | Module maker; HQ and owner unclear | pitchbook [2nd] | low |
| Tower Semiconductor | SiPh foundry; 2025 SiPh revenue $228M (2024 $106M) [2nd transcript] | Q4 2025 call | med |
| GlobalFoundries | Fotonix (2022-03), Malta NY | investors.gf.com | high |
| TSMC | COUPE SiPh engine [2nd] | - | med |

Rankings beyond the above: N/A (paywalled).

## D. Choke points

| Event | Date | Source | Conf. |
|---|---|---|---|
| China Ga/Ge export licensing (MOFCOM 2023 No. 23) | announced 2023-07-03, effective 2023-08-01 | iea.org/policies/17893 | high |
| Ga/Ge/Sb export ban in principle to US (MOFCOM 2024 No. 46) | 2024-12-03 | fastmarkets.com | high |
| Indium items incl. InP licensing, plus W, Te, Bi, Mo (MOFCOM 2025 No. 10) | effective 2025-02-04 | iea.org/policies/26795 | high |
| US-specific ban suspended (licences still required) | 2025-11-09 to 2026-11-27 | fastmarkets.com | high |
| AXT first InP export permits; Q3 2025 InP revenue >+250% QoQ | 2025-06 | sec.gov AXT exhibit | high |
| Lumentum EML capacity +50% by end CY26 vs CY25 | OFC 2026 deck; FY26 Q4 results 2026-08-11 | Lumentum OFC 2026 PDF | high |
| EML supply ~30% short of demand | 2025-26 | secondary only | med |
| NVIDIA $2B each into Lumentum and Coherent | 2026-03 | secondary only, unverified | low |

Notes: QSFP-DD800 MSA is 2019 not 2021; LRO is an architecture not an MSA; MOFCOM primary texts not fetched (IEA / law-firm summaries).
