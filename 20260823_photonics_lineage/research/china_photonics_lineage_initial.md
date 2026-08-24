# China photonics / Huawei / overseas-returnee lineage: initial tranche

**Status:** evidence-backed staging dataset  
**As of:** 2026-08-23  
**Structured data:** `research/china_photonics_people_initial.yaml`

## Result

This first pass identifies **20 named founders or technical leaders** and **25 atomic person–organization edges**: 13 grade A and 12 grade B. Ten people are attached to the six requested startup clusters and ten are named Huawei/HiSilicon optical-interconnect leaders. Five generic team-origin statements are explicitly retained as discovery non-edges.

| Cluster | Named people | Defensible finding |
|---|---|---|
| 摩尔芯光 / LightIC | 孙杰, 孙天博, 王建胜 | Founder/CEO, founder/CTO, and deputy-GM roles are directly named; no individual Intel edge is asserted from the generic team biography. |
| 映讯芯光 / Inxun | 李若林, 杨光华 | Both founders are named. 李若林's Intel silicon-photonics design role is supported by a trade-press event bio; 杨光华's HKU training is self-described in an authoritative interview. |
| 英伟芯 / NVISION | 聂辉 | Official company material establishes CEO identity; reputable secondary reporting supports the named Bell Labs and Intel antecedents. |
| 量引科技 / Liangyin | 李耀基, 赵京雄 | Founder and co-founder/CTO roles are supported by industry-association/interview sources. Generic Intel/Cisco team claims are not converted into edges. |
| 傲科光电 / Aluksen | 商松泉 | Official company pages establish chairman/CEO leadership. The much-repeated Intel/Broadcom/Huawei team-origin claim remains a non-edge. |
| 孛璞半导体 / Beeplux | 邢宇飞 | Official company material establishes deputy-GM status and direct silicon-photonic OCS technical leadership. The unnamed Intel/imec/Xanadu team statement remains a lead only. |
| Huawei / HiSilicon | 侯康, 吕东, 庄四祥, 满江伟, 周立兵, 梁亦铂, 罗军, 张乐伟, 程远兵, 董晓文 | Official or professional-society conference programs provide point-in-time affiliation, printed roles where available, and direct optical-interconnect relevance. |

## Highest-confidence lineage paths

- **李若林: Intel silicon photonics → 映讯芯光.** Both edges are individually named in the same EET China event biography. The source is grade B, so an Intel patent, paper affiliation, or archived Intel bio remains the preferred corroboration.
- **聂辉: Lucent Bell Labs → Intel → 英伟芯.** The downstream CEO identity is grade A from NVISION. The two antecedents are grade B from a Securities Times profile and retain uncertainty around exact titles, legal entities, and dates.
- **杨光华: HKU optoelectronics training → 映讯芯光.** The evidence is an edited first-person industry interview. It supports the person-level path but not an advisor/lab edge.
- **Huawei/HiSilicon optical leadership cohort.** The conference records show a named set spanning chief optical-transport architecture, advanced optoelectronics laboratory leadership, silicon-photonics planning, optical-device standards, PON components, and photonic computing. They establish a technical cohort, not a team transfer or diaspora event.

## Evidence discipline

The staging YAML follows `source_policy.md` at claim level:

- Company, university, government, and official conference artifacts are grade A.
- Trade interviews, industry associations, and professional-society conference pages are grade B.
- No startup database was used as validating evidence.
- A paper or patent affiliation would prove only that explicit relationship; it would not be silently expanded into an employment interval.
- Chinese names are canonical in this file. English forms are marked as official usage or researcher-supplied pinyin, and every person has employer/title/technical-subject identity discriminators.

Most importantly, “the team came from Intel/Huawei/Broadcom/Cisco” is not treated as evidence about any named person. The LightIC, NVISION, Beeplux, Aluksen, and Inxun variants of that pattern are stored only under `discovery_non_edges` with grade X.

## Gaps and next diligence queue

1. **Primary antecedent artifacts:** locate Intel papers/patents for 李若林 and 聂辉, and Bell Labs publications or archived staff pages for 聂辉.
2. **Founder verification:** obtain official founder pages or filings for 孙天博, 李耀基, 赵京雄, and 李若林; current founder evidence is grade B.
3. **Beeplux founder identity:** 郭鹏 appears in corporate/legal artifacts, but this pass found no clean primary artifact tying him to a founder or technical-leadership claim, so he is excluded.
4. **Named team decomposition:** company claims connecting LightIC, Liangyin, Aluksen, and Beeplux to Intel/Cisco/Broadcom/Huawei should be pursued through patents, papers, conference biographies, and official filings one person at a time.
5. **HiSilicon entity normalization:** programs alternate among Huawei, HiSilicon, 海思光电, and 海思光电子. Preserve the printed entity until corporate-unit relationships are sourced.
6. **Name collision risk:** 梁亦铂 should not be merged with similarly rendered 梁亦瑜; romanization alone is insufficient. The conference topic/title is the current discriminator.
7. **Uncovered antecedents:** this tranche did not yet validate named Futurewei, NeoPhotonics, InnoLight, Cisco, or Broadcom returnee paths beyond the individually named Intel/Bell Labs paths above.

## Suitable use

The A/B edges are suitable as a conservative seed layer. Conference-derived Huawei/HiSilicon claims should be visualized as point-in-time affiliation/technical-leadership evidence, not dated employment spans. The five X-grade items are research prompts only and must never contribute to a Sankey weight or a team-transfer edge.
