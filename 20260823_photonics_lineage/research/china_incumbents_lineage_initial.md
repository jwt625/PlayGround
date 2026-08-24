# China optical-component incumbents: initial lineage tranche

**Status:** evidence-backed staging dataset  
**As of:** 2026-08-23  
**Structured data:** `research/china_incumbents_people_initial.yaml`

## Result

This pass adds **20 named founders or technically relevant leaders** and **60 atomic
person–organization edges** across the six priority incumbent clusters. It deliberately
does not turn corporate ownership, acquisition history, or “team came from X” language
into person-level migration claims.

| Target cluster | People | High-confidence person-level lineage |
|---|---:|---|
| InnoLight / 中际旭创 | 7 | Opnext, Pine Photonics, Agere, SDL, Intel, Oplink, WaveSplitter, and Princeton Photonics antecedents are individually stated in the SEC F-1. |
| Eoptolink / 新易盛 | 3 | Leshan Radio Factory's optical-communications branch, Fiberxon Chengdu, Guangsheng, Chengdu Kanghe, and Sichuan Guangheng are issuer-reported paths. |
| Accelink / 光迅科技 | 4 | Product/R&D leaders trace into the Wuhan posts-and-telecom solid-state-device institute and Wuhan Telecommunication Devices Company. |
| Source Photonics | 2 | Yu-Heng Jan's MRV/UCSB path is in a primary paper; Frank Chang's Inphi/Vitesse/JDSU path is retained at grade B from Optica. |
| Broadex / 博创科技 | 1 | Tom Tang's YOFC optical-module product-line role and earlier semiconductor hardware roles are on Broadex's official management page. |
| Hisense Broadband / 海信宽带 | 3 | Huang Weiping is a technically relevant founder; David Li led optical-technology industrialization; Jianying Zhou provides the named Intel/NeoPhotonics/Oclaro/Finisar/JDS path. |

## Strongest lineage paths

- **Opnext / Pine Photonics → InnoLight.** Sheng Liu and Xiangzhong Wang have direct
  Opnext antecedents; Sheng Liu, Hsing Kung, Osa Mok, Weilong Lee, and Xiangzhong Wang
  have individually described Pine roles. This supports multiple person paths, not a
  single inferred team-transfer event.
- **Leshan optical communications → Eoptolink.** Gao Guangrong and Luo Yuming both have
  issuer-reported service in the optical-communications branch, with different roles and
  periods. Huang Xiaolei supplies a separate Fiberxon/CTO path into Eoptolink's operating
  leadership.
- **Wuhan solid-state devices → Accelink.** Hu Qianggao and Bu Qinlian have explicit
  institute antecedents; Zhang Jun's path runs through Wuhan Telecommunication Devices.
  These are person-level employment paths, not a blanket institutional spinout claim.
- **Intel / NeoPhotonics / legacy U.S. optics → Hisense Broadband.** Jianying Zhou is
  the strongest direct bridge, but the source is a co-hosted technical-event biography,
  therefore grade B. Exact dates and the grouped Oclaro/Finisar/JDS titles remain open.

## Evidence discipline

- SEC, HKEX, Shenzhen Stock Exchange filings, issuer annual reports, official company
  biographies, and a primary technical paper are grade A.
- Optica and event-host speaker biographies are grade B under `source_policy.md`; they
  are not promoted to A merely because they are detailed.
- Every edge carries a direct URL, evidence grade, photonics relevance, and uncertainty.
- Education edges prove only the degree/training stated. UCSB affiliation does not infer
  an advisor or lab, and David Li's unusually compressed joint-PhD wording is preserved.
- Chinese names are canonical where the primary source prints them. Researcher-supplied
  pinyin or Chinese rendering is labeled and must not drive identity merges by itself.
- Chairman/director entries are included only when the same person has directly evidenced
  optical engineering, chief-engineer, product, R&D, or founder relevance.

## Gaps and next diligence queue

1. **Broadex founders:** issuer-grade biographies for Zhu Wei and Ding Yong are still
   needed before asserting the frequently repeated Bell Labs/Intel antecedents.
2. **Source Photonics formation:** this pass found strong technical-leader evidence but
   not a clean primary founder/formation biography.
3. **Jianying Zhou chronology:** Intel, NeoPhotonics, Oclaro, Finisar, Nortel, and JDS are
   explicitly named, but only Intel and NeoPhotonics have titles. Patents, papers, or
   archived employer biographies should establish sequence and dates.
4. **InnoLight Chinese aliases:** only aliases tied to a reliable artifact should be
   canonicalized; romanization-based guesses are intentionally absent.
5. **Accelink institutional lineage:** legal succession among the historical ministry
   institute, Wuhan Research Institute, Wuhan Telecommunication Devices, and Accelink
   should be represented with sourced organization events, not inferred from biographies.
6. **Eoptolink founder labels:** long tenure and control do not by themselves prove the
   formal founder relation. The staging edges retain the precise issuer-reported roles.

## Suitable use

The A/B edges are suitable for a conservative named-person graph after canonical entity
normalization. Historical filings should be rendered as dated role snapshots. The three
X-grade non-edges are diligence prompts only and must not contribute to migration counts,
team-transfer claims, or Sankey weights.
