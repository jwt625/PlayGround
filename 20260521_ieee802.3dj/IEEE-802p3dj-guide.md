# How to Navigate IEEE 802.3 Public Pages for Short-Reach Datacom Transceivers

## 1. IEEE 802.3 URL naming convention

Example:

https://www.ieee802.org/3/dj/public/24_01/

Breakdown:

| URL part | Meaning |
|---|---|
| ieee802.org/3 | IEEE 802.3 Ethernet working group |
| dj | Project / task force code, e.g. P802.3dj |
| public | Public meeting materials |
| 24_01 | Meeting date folder: Jan 2024 |
| index.html | Meeting index page, usually implicit |

So:

https://www.ieee802.org/3/df/public/22_11/

means IEEE 802.3df public materials from the Nov 2022 meeting.

https://www.ieee802.org/3/dj/public/24_01/

means IEEE 802.3dj public materials from the Jan 2024 meeting.

The master index for a project is usually:

https://www.ieee802.org/3/[project]/public/index.html

For P802.3dj:

https://www.ieee802.org/3/dj/public/index.html

## 2. Relevant IEEE 802.3 project folders

For short-reach datacom transceivers, especially intra-data-center and sub-500 m reach, focus mainly on these:

| Project | Relevance |
|---|---|
| 802.3df | Earlier 200G/lane, 800G, and 1.6T Ethernet work. Useful historical decks, including cost and power discussions. |
| 802.3dj | Main newer continuation for 200G/lane, 400G, 800G, and 1.6T Ethernet. Search here first. |
| 802.3cw | Mostly coherent / DCI-ish context. Less relevant for sub-500 m IMDD, but joint meetings may contain useful overlap. |

Useful master page:

https://www.ieee802.org/3/dj/public/index.html

## 3. Keywords for intra-data-center, sub-500 m optics

Search for these PMD and architecture terms:

| Keyword | Meaning / relevance |
|---|---|
| DR | Usually 500 m single-mode fiber class. Very relevant. |
| DR1 / DR2 / DR4 / DR8 | 1, 2, 4, or 8 optical lanes. Relevant to 200G, 400G, 800G, and 1.6T. |
| DR4-2 / DR8-2 | 200G/lane variants / newer 802.3dj naming. Very relevant. |
| FR4-500 | 500 m 4-wavelength WDM variant. Very relevant. |
| 500m | Direct reach keyword. |
| SR / SR4 / SR8 | Short-reach multimode fiber. Relevant for MMF/VCSEL transceivers. |
| VCSEL | Short-reach multimode ecosystem. |
| C2M | Chip-to-module electrical interface. Relevant to pluggables, LPO, and LRO. |
| AUI / 200G/lane electrical | Host/module electrical-side interface. |
| IM-DD / IMDD | Intensity modulation / direct detection. Core datacom architecture. |
| FECi / inner FEC / FEC bypass / low latency | Power and latency-heavy part of short-reach 200G/lane debate. |
| TDECQ / COM / reference receiver | Link budget, test, and spec-closure details for IMDD optics. |

## 4. Terms to deprioritize for sub-500 m intra-DC work

These are useful for comparison, but usually less relevant to short-reach datacom:

| Keyword | Usually means |
|---|---|
| LR / LR1 / LR4 | Around 10 km single-mode links. Less intra-DC. |
| ER / ER1 | Around 40 km links. Not intra-DC. |
| ZR / 800ZR | Coherent DCI. Not short-reach datacom. |
| coherent | Mostly DCI/LR/ER, unless the deck compares coherent against IMDD. |

## 5. IEEE 802.3dj folders to search first

Start from:

https://www.ieee802.org/3/dj/public/index.html

Then prioritize these folders:

| Folder | Why it matters |
|---|---|
| /3/dj/public/24_01/ | Very useful. Includes 800G over 4 wavelengths up to at least 500 m, low-latency inner FEC, C2M, advanced packaging, and 500 m CWDM TDECQ material. |
| /3/dj/public/23_11/ | Good for baseline optical PMD proposals: DR1, DR2, DR4, FR4, DR8, and 200G/lane optics. |
| /3/dj/public/23_05/ | Good for FEC bypass, latency, economics, and 200G/lane IMDD discussions. |
| /3/dj/public/24_03/ | Follow-on refinements after Jan 2024 baselines. |
| /3/dj/public/24_05/ | Follow-on refinements after Jan 2024 baselines. |
| /3/dj/public/24_07/ | Follow-on refinements after Jan 2024 baselines. |
| /3/dj/public/25_*/ | Later consensus/spec evolution. Useful for seeing what survived. |
| /3/dj/public/26_*/ | Most recent consensus/spec evolution. Useful for current status. |

## 6. Search queries to use

Use web search with site filters:

site:ieee802.org/3/dj/public DR4 500m 800GBASE

site:ieee802.org/3/dj/public "FR4-500"

site:ieee802.org/3/dj/public "800GBASE-DR4"

site:ieee802.org/3/dj/public "1.6TBASE-DR8"

site:ieee802.org/3/dj/public "200GBASE-DR1"

site:ieee802.org/3/dj/public "FEC Inner Code Bypass"

site:ieee802.org/3/dj/public "Low Latency Mode" FEC

site:ieee802.org/3/dj/public "C2M" "200 Gb/s per lane"

site:ieee802.org/3/dj/public "TDECQ" "500m"

site:ieee802.org/3/dj/public "IM-DD" power

## 7. Priority order for short-reach transceiver architecture research

For intra-data-center, sub-500 m optical transceivers, prioritize in this order:

1. DR / DR4 / DR8 / DR4-2 / DR8-2
2. FR4-500
3. FECi / inner FEC / FEC bypass / low latency
4. C2M / AUI / advanced packaging
5. TDECQ / link budget / reference receiver

## 8. Mental model

DR and FR4-500 usually describe the optical PMD.

C2M and AUI describe the electrical host-to-module bottleneck.

FECi, inner FEC, and DSP discussions usually point to the power and latency bottleneck.

TDECQ, COM, and reference receiver discussions usually point to spec/test closure.

For short-reach intra-DC optics, the most relevant combination is:

DR / FR4-500 + IMDD + 200G/lane + C2M/AUI + FECi/FEC bypass + TDECQ/reference receiver