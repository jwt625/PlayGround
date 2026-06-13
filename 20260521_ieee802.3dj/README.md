# IEEE 802.3 AI / High-Speed Ethernet Cache

Local cache and browsing workspace for IEEE 802.3 public materials relevant to AI scale-up, AI scale-out, and high-speed datacenter Ethernet.

## Processed 802.3 Areas

| Area | Source | Local cache / metadata | Checklist | Status |
|---|---|---|---|---|
| IEEE P802.3dj 200 Gb/s, 400 Gb/s, 800 Gb/s, and 1.6 Tb/s Ethernet | https://www.ieee802.org/3/dj/ | `ieee802_3dj_cache/`, `ieee802_3dj_browser/metadata/` | `IEEE-802p3dj-metadata-fix-checklist.md` | Cached and browser-ready. Core high-speed Ethernet material. |
| IEEE 802.3 NEA Ethernet for AI Assessment | https://www.ieee802.org/3/ad_hoc/E4AI/public/index.html | `ieee802_e4ai_cache/`, `ieee802_e4ai_metadata/` | `IEEE-802p3-e4ai-cache-checklist.md` | Cached. Best AI-specific source for scale-up/scale-out framing, 400G+ per-lane feasibility, channel data, and fiber-for-AI material. |
| IEEE 802.3 200 Gb/s per Wavelength MMF PHYs Study Group | https://www.ieee802.org/3/200GMMF/index.html | `ieee802_3_ai_related_cache/200GMMF/`, `ieee802_3_ai_related_metadata/` | `IEEE-802p3-ai-related-cache-checklist.md` | Cached. Predecessor to P802.3ds; useful for CFI, objectives, 850 nm / 1060 nm VCSEL-MMF feasibility, and short-reach scale-up/scale-out objectives. |
| IEEE P802.3ds 200 Gb/s per Wavelength MMF PHYs Task Force | https://www.ieee802.org/3/ds/ | `ieee802_3_ai_related_cache/ds/`, `ieee802_3_ai_related_metadata/` | `IEEE-802p3-ai-related-cache-checklist.md` | Cached. Current MMF/VCSEL short-reach project; relevant to 200G/lane SR optics, OM3/OM4/OM5 reach, optimized MMF, MPO MDI, and link test details. |
| IEEE 802.3 400 Gb/s/Lane Signaling Study Group | https://www.ieee802.org/3/400GPL/index.html | `ieee802_3_ai_related_cache/400GPL/`, `ieee802_3_ai_related_metadata/` | `IEEE-802p3-ai-related-cache-checklist.md` | Cached. Post-E4AI 400G/lane electrical and up-to-500 m SMF path for AI-driven next-lane-rate Ethernet. |

## Browser

The static browser currently targets the P802.3dj cache:

- Web UI: `ieee802_3dj_browser/`
- Cached PDFs and meeting index pages: `ieee802_3dj_cache/`
- Extracted markdown/text and metadata: `ieee802_3dj_browser/extracted_text/`, `ieee802_3dj_browser/metadata/`

The root `index.html` redirects to the browser UI. ZIP bundles and the local Marker/PyMuPDF environment are intentionally not committed.

## Maps

- `ieee802_3_ai_map/ieee8023-ai-relationships.svg` gives an outsider-facing dark-theme map of relevant IEEE 802.3 task forces, study groups, support ad hocs, topic relationships, timeline, officers, and visible contributor ecosystems.
- `ieee802_3_ai_map/high_level_pages/` caches the high-level source pages used by the map.
- `ieee802_3_ai_map/participant_summary.json` records the visible affiliation counts used for the per-node company labels in the SVG.
- `ieee802_3_ai_map/timeline_data.json` records the generated timeline ranges, lanes, source files, and precision notes.

Expected Pages URL:

```text
https://jwt625.github.io/ieee802_3dj_browser/
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/cache_8023dj.py` | Cache IEEE P802.3dj public meeting materials. |
| `scripts/cache_e4ai.py` | Cache IEEE 802.3 E4AI public materials and write E4AI metadata/checklist. |
| `scripts/cache_8023_ai_related.py` | Cache selected adjacent AI-relevant 802.3 project pages. Currently covers `200GMMF`, `ds`, and `400GPL`. |
| `scripts/cache_8023_high_level.py` | Cache high-level IEEE 802.3 source pages used by the relationship map. |
| `scripts/summarize_8023_participants.py` | Summarize visible public affiliation signals for map company labels. |
| `scripts/generate_8023_timeline.py` | Generate non-overlapping timeline lanes from cached meeting/public-index dates and update the SVG. |
| `tools/extract_pdf_text.py` | Extract text from cached P802.3dj PDFs. |
| `tools/extract_e4ai_text.py` | Extract text from cached E4AI PDFs. |

## Proposed Next 802.3 Areas

See `IEEE-802p3-ai-relevant-subpages.md` for the browsed rationale. Practical next order:

1. `dq` - IEEE P802.3dq Pin-Optimized PHY Interface Task Force: package/host interface constraints for dense AI systems.
2. COM - IEEE 802.3 Channel Operating Margin Open Source Project Ad Hoc: reference electrical channel feasibility tooling.
3. `df` and `B400G` - historical 400G/800G/Beyond-400G material for objective formation, FEC, and lane-rate tradeoffs.
4. Conditional: `dt`, `dk`, FMP, and PDCC only when investigating Ethernet metadata services, access/bidirectional optics, interoperability, or data-center coordination questions.
