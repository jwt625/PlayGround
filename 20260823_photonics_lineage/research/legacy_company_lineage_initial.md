# Legacy Commercial-Company Diaspora

As of 2026-08-23. The companion YAML contains 25 named people, 39 atomic person edges, and 17 A/B sources. It is an initial person-lineage tranche, not a complete workforce census.

## High-signal flows

```text
Luxtera ── Vikram Sadagopan ───────────────> Xscape
         └ Ron Horan ─────────────────────> Cisco

Bookham ─ Andrew Rickman ─ Kotura ─────────> Rockley
Rockley ─ Vivek Raghunathan ───────────────> Xscape

Elenion ─ Matt Streshinsky ────────────────> Enosemi
        └ Ari Novack ──────────────────────> Enosemi

NeoPhotonics ─ Nicolas Herriau ────────────> Xscape
              Wupen Yuen ─────────────────> Lumentum

Finisar ─ Juthika Basak ──────────────────> AMD

JDSU ─ Greg Dougherty ────────────────────> Oclaro

Inphi ─ Loi Nguyen ───────────────────────> Marvell
      ├ Nariman Yousefi ──────────────────> Marvell
      └ Radha Nagarajan ──────────────────> Marvell
```

## What the evidence supports

- **Xscape is the clearest cross-company mixing node in this tranche.** Its own biographies directly place Vivek Raghunathan at Rockley, Vikram Sadagopan at Luxtera, and Nicolas Herriau at NeoPhotonics before their Xscape roles. These are three individual paths, not evidence of a collective transfer.
- **Enosemi is a documented Elenion downstream branch.** Enosemi explicitly calls Matt Streshinsky and Ari Novack members of Elenion's founding team and identifies their Enosemi leadership. The source supports two named paths; it does not establish that the wider Elenion team moved together.
- **Andrew Rickman is a serial-company bridge, not an acquisition-retention edge.** A Rockley regulatory biography supports Bookham founder/CEO, Kotura chairman, and Rockley founder/CEO roles. It does not support employment at Oclaro, which formed years after he left Bookham leadership.
- **The Inphi-to-Marvell paths are acquisition-linked but individually documented.** Marvell biographies and filings directly support Loi Nguyen, Nariman Yousefi, and Radha Nagarajan in both companies. Ford Tamer's Inphi CEO path terminates at the acquisition but is not modeled as Marvell employment.
- **Legacy incumbents also create executive and technical circulation without startup formation.** Greg Dougherty links JDSU to Oclaro; Wupen Yuen links NeoPhotonics to Lumentum; Juthika Basak links Finisar/Coherent product development to AMD silicon-photonics leadership.

## Corporate events kept separate from person migration

| Target/history | Acquirer or successor | Announced | Effective/closed | Treatment here |
|---|---|---:|---:|---|
| Luxtera | Cisco | 2018-12-18 | 2019-02-06 | Corporate acquisition is not a blanket employee edge. Ron Horan is the only directly documented Luxtera-to-Cisco person in this tranche. |
| Kotura | Mellanox | 2013 | 2013 | Rickman's chairmanship through the sale is supported; no named Kotura-to-Mellanox workforce transfer is asserted. |
| Elenion | Nokia | 2020-02-19 | 2020-03-25 | Nokia's [announcement](https://www.nokia.com/newsroom/nokia-to-acquire-elenion-to-improve-economics-of-advanced-optical-connectivity-solutions/) and [completion release](https://www.nokia.com/newsroom/nokia-completes-acquisition-of-elenion-technologies/) are separate artifacts. This dataset does not convert the acquisition into person edges. |
| NeoPhotonics | Lumentum | 2021-11-04 | 2022-08-03 | [Announcement](https://investor.lumentum.com/financial-news-releases/news-details/2021/Lumentum-To-Acquire-NeoPhotonics-to-Accelerate-Optical-Network-Speed-And-Scalability/default.aspx) and [completion evidence](https://investor.lumentum.com/financial-news-releases/news-details/2022/Lumentum-Announces-Fiscal-Fourth-Quarter-and-Full-Year-2022-Results/default.aspx) remain corporate facts; only Yuen has a verified post-close role here. |
| Finisar | II-VI, later Coherent brand | 2018-11-08 | 2019-09-24 | The [effective date](https://www.coherent.com/content/dam/coherent/site/en/documents/investors/annual-filings/2019/ii-vi-proxy-statement-2019.pdf) does not establish every Finisar employee's successor tenure. |
| JDS Fitel + Uniphase | JDS Uniphase formation | 1999 | 1999-06-30 | Modeled as corporate formation context; Straus, Kalkhoven, Leonberger and later Scifres have independent person evidence. |
| Bookham + Avanex | Oclaro formation | 2009 | 2009 | Corporate succession is not used to place pre-2009 Bookham alumni at Oclaro. Couder's continuous leadership is separately supported. |
| Oclaro | Lumentum | 2018-03-12 | 2018-12-10 | Lumentum's [announcement](https://investor.lumentum.com/financial-news-releases/news-details/2018/Lumentum-To-Acquire-Oclaro-For-1-8B-In-Cash-And-Stock-2018-3-12/default.aspx) and [completion](https://investor.lumentum.com/financial-news-releases/news-details/2018/Lumentum-Announces-Completion-Of-Oclaro-Acquisition-2018-12-10/default.aspx) are not person-retention evidence. |
| Inphi | Marvell | 2020-10-29 | 2021-04-20 | Three post-close Marvell roles are represented only because person-level biographies support them. |

Rockley is treated as a company/founder node. Its later restructuring is not silently represented as an acquisition or successor event.

## Boundary decisions

1. No `team_transfer` edge is included. Multiple people reaching the same downstream company remains a set of auditable atomic edges.
2. Jozef Straus's founder edge uses the 1999 JDSU combination year; the 1981 date belongs to JDS Fitel's origin and is retained only in the rationale.
3. The Finisar/Coherent wording in Juthika Basak's OFC biography is not split into invented employer dates.
4. Andrew Rickman's Bookham history and Oclaro's corporate ancestry are shown separately; there is no Rickman-to-Oclaro person edge.
5. Nariman Yousefi's ClariPhy founder history and Donald Scifres's SDL founder history are source-backed discovery leads, but those organizations are outside the current seed catalog, so they are noted rather than assigned orphan endpoints.

## Source gaps and next expansion

- Add primary founder/technical-leader records for Luxtera's original founding group and Kotura's operating leadership; current evidence is strongest for their downstream alumni and Rickman's chairmanship.
- Add Elenion CEO Larry Schwerin and technical founders from regulatory, patent, or archived-company sources; do not infer founding status from Nokia acquisition copy alone.
- Resolve NeoPhotonics' original NanoGram-era founders versus later operating leadership.
- Expand Finisar beyond its 2005 executive snapshot and separate Finisar-to-II-VI-to-Coherent employment intervals with individual evidence.
- Add JDS Fitel, Uniphase, SDL, Avanex, and ClariPhy as normalized organizations before importing the omitted founder edges.
- Verify current 2026 roles for historical biographies; mutable-page roles are not projected forward when the artifact is old.

## Bottom line

The validated initial graph has **25 people and 39 person edges**: 11 founder, 14 executive-role, 12 technical-leadership, and 2 employment edges. The strongest downstream startup creation is Elenion→Enosemi and Rockley→Xscape; the strongest serial-founder bridge is Bookham→Kotura→Rockley through Andrew Rickman. Corporate acquisitions are contextual events, not proxies for employee migration.
