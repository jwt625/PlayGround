# IEEE 802.3 Subpages Relevant to AI Scale-Up and Scale-Out

Browsed: 2026-06-13

Decision supported: which IEEE 802.3 public areas are worth caching and mining for AI cluster physical-layer, packaging, cabling, and Ethernet service-interface evidence.

## Highest Priority

| Page | URL | Why it matters for AI |
|---|---|---|
| IEEE 802.3 NEA Ethernet for AI Assessment | https://www.ieee802.org/3/ad_hoc/E4AI/public/index.html | Best cross-cutting AI-specific source. Contains scale-up/scale-out framing, 400 Gb/s+ per-lane C2C/C2M/CR/channel material, fiber-for-AI workshop decks, OCP/UALink/UEC context, and channel data. Already cached in `ieee802_e4ai_cache/`. |
| IEEE P802.3dj 200 Gb/s, 400 Gb/s, 800 Gb/s, and 1.6 Tb/s Ethernet Task Force | https://www.ieee802.org/3/dj/ | Main current high-speed Ethernet PMD/electrical project already cached in `ieee802_3dj_cache/`. Core for 200G/lane optics, C2M/AUI, DR/FR/SR, FEC, latency, and 1.6T architecture. |
| IEEE 802.3 400 Gb/s/Lane Signaling Study Group | https://www.ieee802.org/3/400GPL/index.html | Chartered on 2026-03-13 for 400 Gb/s per lane electrical interconnects and SMF optical interconnects up to 500 m, which maps directly to AI radix pressure and 3.2T-class Ethernet timing. Cached in `ieee802_3_ai_related_cache/400GPL/`. |
| IEEE P802.3ds 200 Gb/s per Wavelength MMF PHYs Task Force | https://www.ieee802.org/3/ds/ | Current MMF/VCSEL short-reach project. Relevant to short scale-out links, multimode installed-base questions, 850 nm vs 1060 nm, OM3/OM4/OM5/optimized-MMF reach, MPO MDI, and low-cost high-density optical attach. Cached in `ieee802_3_ai_related_cache/ds/`. |
| IEEE 802.3 200 Gb/s per Wavelength MMF PHYs Study Group | https://www.ieee802.org/3/200GMMF/index.html | Predecessor to P802.3ds. Useful for CFI, objectives, PAR/CSD, early 1060 nm and 850 nm feasibility arguments, and scale-up/very-short-reach objective debates. Cached in `ieee802_3_ai_related_cache/200GMMF/`. |

## High-Value Supporting Pages

| Page | URL | Relevance | Cache priority |
|---|---|---|---|
| IEEE P802.3dq Pin-Optimized PHY Interface Task Force | https://www.ieee802.org/3/dq/index.html | Host/package interface architecture. Relevant when AI scale-up drives dense electrical interfaces, pin efficiency, and packaging constraints. | High |
| IEEE 802.3 Channel Operating Margin (COM) Open Source Project Ad Hoc | https://www.ieee802.org/3/ad_hoc/COM/index.html | Electrical channel feasibility and reference code path. Important for C2C/C2M/CR analysis at 200G and 400G per lane. | High |
| IEEE P802.3df 400 Gb/s and 800 Gb/s Ethernet Task Force | https://www.ieee802.org/3/df/ | Completed baseline for 400G/800G Ethernet and predecessor lineage for 802.3dj. Useful history for 100G/200G lane tradeoffs, early 800G optics, FEC, and objectives. | Medium-high |
| IEEE 802.3 Beyond 400 Gb/s Ethernet Study Group | https://www.ieee802.org/3/B400G/ | Predecessor study group that fed P802.3df. Useful for market-need, objective formation, and early beyond-400G architectural assumptions. | Medium |
| IEEE P802.3dt Ethernet Metadata Services Task Force | https://www.ieee802.org/3/dt/index.html | Not a PHY bandwidth project, but potentially relevant to AI fabrics if per-packet metadata, telemetry, scheduling, congestion, or accelerator-network services become part of Ethernet semantics. | Medium |

## Situational / Lower Priority

| Page | URL | Why lower priority |
|---|---|---|
| IEEE P802.3dk Greater than 50 Gb/s Bidirectional Optical Access PHYs | https://www.ieee802.org/3/dk/index.html | Optical PHY work, but access/bidirectional focus is less directly tied to AI cluster scale-up/scale-out than datacenter short-reach and high-radix links. |
| IEEE 802.3 New Ethernet Applications Ad Hoc | https://www.ieee802.org/3/ad_hoc/ngrates/index.html | Parent NEA context can matter for new-CFI flow, but E4AI is the relevant child area already cached. |
| IEEE 802.3 FMP Ethernet Interoperability Study Group | https://www.ieee802.org/3/FMP/index.html | Potentially relevant only if FMP maps to AI fabric interoperability concerns; otherwise lower signal than PHY/link pages. |
| IEEE 802.3 PDCC Ad Hoc | https://www.ieee802.org/3/ad_hoc/PDCC/index.html | Possible data-center coordination context, but not obviously central to scale-up/scale-out PHY decisions without a specific question. |

## Practical Cache Order

1. Already cached: `dj`, `E4AI`, `200GMMF`, `ds`, `400GPL`.
2. Cache next: `dq` and COM, because these support electrical feasibility and package/interface constraints.
3. Then cache: `df` and `B400G`, for historical objective formation and earlier 400G/800G/1.6T tradeoffs.
4. Only cache `dt`, `dk`, FMP, or PDCC when a specific AI-fabric metadata, access-optics, interoperability, or datacenter-coordination question requires them.

## Notes From Browsing

- The IEEE 802.3 working-group index lists active/current projects and ad hocs including P802.3dj, P802.3ds, P802.3dq, P802.3dt, the 400 Gb/s/Lane Signaling Study Group, NEA, E4AI, and COM.
- `200GMMF` says the study group transitioned to P802.3ds after P802.3ds PAR approval on 2025-12-10.
- `400GPL` says it was chartered on 2026-03-13 for 400 Gb/s per lane signaling for electrical interconnects and SMF optical interconnects with reaches up to 500 m.
- `P802.3ds` public material is especially rich in 850 nm vs 1060 nm, optimized MMF, OM3/OM4/OM5 reach, 2-row MPO MDI, launch condition, receiver reflectance/ORL, and TDECQ/DFE evidence.
