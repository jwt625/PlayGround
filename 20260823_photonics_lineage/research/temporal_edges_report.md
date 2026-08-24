# Temporal Evidence Improvement Report

As of 2026-08-23, this pass proposes ten date-field updates across ten canonical edge IDs. Applied to the `generated/graph_ab.json` snapshot identified in the patch, **four of the 79 current sequence exclusions become strictly orderable** under the existing rule (`latest possible prior end < earliest possible later start`). The patch is research staging only; canonical files were not changed.

## Strict-order impact

| Person | Newly orderable flow | Dated boundary |
|---|---|---|
| Mario Paniccia | Intel Silicon Photonics → ANELLO | Intel end 2016; ANELLO cofounder/executive role observed 2019-11 |
| Peter Winzer | Bell Labs → Nubis | Bell Labs end 2019; Nubis founded 2020-09 |
| Guilhem de Valicourt | Alcatel-Lucent/Bell Labs → Nubis | Bell Labs end 2017; Nubis founded 2020-09 |
| Chong Zhang | UCSB Bowers Group → Nexus Photonics | dissertation 2017-03; Nexus founded 2018 |

The four unlocked person-pairs aggregate to three institution-level flows because Winzer and de Valicourt contribute to the same Bell Labs → Nubis flow.

Two additional training dates improve temporal resolution but do not unlock a flow: Alexander Fang's Bowers-group dissertation is dated 2008-03-01, while Aurrion remains only year-dated to 2008; Minh Tran's dissertation is dated 2019-03, while his Nexus employment remains only year-dated to 2019. Ronnen Lovinger's Credo role can be dated to the 2026-05-28 DustPhotonics acquisition close, but the same-day continuity is intentionally not a strict-before talent flow.

## Evidence decisions

- Only A/B artifacts are used. Official company/university records and transaction materials are grade A; the conference and technical-publication biographies are grade B.
- A degree or dissertation date is used as the end of the corresponding research-training edge only when the thesis directly identifies the Bowers relationship.
- Acquisition timing is not silently converted into a person's prior-employer end date. A close-date announcement can directly date a named post-acquirer role, but equality on the handoff day still fails the graph's strict-before test.
- Biography ordering such as “prior to Acacia” is preserved as relationship evidence, not converted into an invented month or day.

## High-value paths still unresolved

| Target path | What A/B evidence establishes | Why it remains excluded |
|---|---|---|
| Intel → DustPhotonics (Yoel Chetrit) | Dust's official bio calls Intel his last role; official acquisition material dates Dust's founding to 2017 | No direct Intel end date or exact founder start for Chetrit |
| Intel/Rockley/Broadcom → Xscape (Vivek Raghunathan) | Xscape's official bio supplies the sequence | No role boundary dates; list order is not a date |
| Intel/Luxtera → Xscape (Vikram Sadagopan) | Xscape officially dates his appointment to 2025-04-18 | Intel and Luxtera endpoints remain undated |
| Bell Labs/Mintera → Acacia (Benny Mikkelsen) | Acacia's SEC S-1 says Bell Labs, then Mintera, then Acacia; Acacia role begins 2009-06 | “Prior to joining” does not support an exact Mintera or Bell Labs endpoint |
| Mintera → Acacia (Christian Rasmussen, Mehrdad Givehchi) | SEC biographies establish prior Mintera roles and Acacia service from 2009-06 | No direct Mintera end date |
| UCSB → Aurrion (Alexander Fang) | UCSB records PhD/dissertation in 2008; UCSB TIA says Aurrion founded 2008 | Same-year overlap; older company fact sheet also reports a conflicting 2007 founding year |
| UCSB → Quintessent (Alan Liu) | Quintessent says PhD completed in 2016 and describes two intervening years before reconvening | The article does not directly state a founding year; deriving 2018 would violate the no-date-inference policy |
| UCSB → Nexus (Minh Tran) | Dissertation is 2019-03; Nexus says he joined in 2019 | Month of joining is absent, so intervals overlap |
| MIT → Ayar (Chen Sun) | MIT PhD and Ayar founding are both supported in 2015 | No directly supported month boundary separates them |
| MIT → Lightmatter (Nicholas Harris) | MIT PhD is 2017; company launch is 2017-12 | Year-precision PhD endpoint overlaps the launch month |
| MIT → Lightmatter (Darius Bunandar, Thomas Graham) | They were still MIT students when the company formed | This is concurrent university/startup activity, not a strictly ordered departure flow |
| Columbia → Xscape founders | Official bios support Columbia faculty and Xscape founder roles | Faculty roles are concurrent; forcing a departure would be incorrect |
| DustPhotonics → Credo (Ronnen Lovinger) | Credo identifies him as VP on the 2026-05-28 close date | Same-day handoff is continuity, not strict-before ordering |
| Nubis → Ciena (Peter Winzer) | Nubis acquisition and Ciena role are both dated 2025-10-07 | Same-day handoff is continuity, not strict-before ordering |

## Conflict notes

ANELLO uses November 2019 for Paniccia's CEO/chairman appointment and 2020 language for “starting” ANELLO. The patch treats 2019-11 narrowly as the first dated observation of the combined founder/executive edge, not as a legal-incorporation claim. Aurrion has a more material 2007-versus-2008 founding-year conflict across available artifacts, so no founder-date refinement is proposed.

## Reproduction

The YAML records the exact canonical SHA-256 values embedded in the graph snapshot, all proposed field values, evidence locators, and the four affected exclusion pairs. Re-run the graph builder only after review and canonical merge; this pass deliberately did not edit canonical or generated outputs.
