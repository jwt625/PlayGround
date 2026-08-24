# Bell Labs / Mintera / Acacia / Nubis Lineage — Initial Pass

As of 2026-08-23. The normalized person and edge records are in `bell_labs_people_initial.yaml`.

## Bottom line

Two high-confidence founder paths emerge, but neither supports a wholesale `team_transfer` claim:

```text
Bell Labs ── Benny Mikkelsen ──> Mintera ──> Acacia ──> Cisco Acacia
Mintera ──── Christian Rasmussen ───────────> Acacia
Mintera ──── Mehrdad Givehchi ─────────────> Acacia

Bell Labs ── Peter Winzer ─────────────────> Nubis ──> Ciena
Bell Labs ── Guilhem de Valicourt ─────────> Nubis ──> Ciena
```

The dataset validates 16 named people and 27 atomic edges with A/B evidence. It deliberately models person-level paths rather than converting repeated movement into a collective-transfer assertion.

## Strongest lineage findings

| Path | Person-level evidence | Grade | Interpretation |
|---|---|---|---|
| Bell Labs → Mintera → Acacia | Benny Mikkelsen held Bell Labs engineering roles, co-founded Mintera as VP Technology, then co-founded Acacia as CTO | A | Direct three-node founder lineage |
| Mintera → Acacia | Christian Rasmussen was Mintera Principal Optical Engineer, then Acacia founder and DSP/Optics leader | A | Direct person migration; not by itself team transfer |
| Mintera → Acacia | Mehrdad Givehchi was Mintera Consulting Optical Engineer, then Acacia founder and Hardware/Software leader | A | Direct person migration; not by itself team transfer |
| Bell Labs → Nubis → Ciena | Peter Winzer spent two decades at Bell Labs, founded Nubis, and became Ciena VP Systems Architecture after acquisition | A | Direct founder and post-acquisition continuity |
| Bell Labs → Nubis | Guilhem de Valicourt spent nine years at Bell Labs and became Nubis co-founder/VP Engineering | A | Direct founder lineage; exact Bell Labs dates unresolved |
| Lucent Bell Labs → ETH | Jürg Leuthold worked at Bell Labs/Lucent during 1999–2004, later becoming ETH IEF head | A | Important academic dissemination branch |

## Bell Labs research-school context

The startup paths sit within a broader coherent-optics and optical-networking school represented here by René-Jean Essiambre, Roland Ryf, David Neilson, Qian Hu, Robert Tkach, and Sethumadhavan Chandrasekhar. Their edges establish technical employment at Bell Labs entities without implying startup formation.

- Essiambre joined Bell Labs in 1997 and developed foundational nonlinear optical-transmission models.
- Ryf joined in 2000 and advanced optical switching and space-division multiplexing.
- Neilson has been at Bell Laboratories since 1998, working on optical switches, integrated optoelectronics, interconnects, and network energy.
- Hu joined Nokia Bell Labs in 2015 and works on coherent DSP, high-speed transceivers, and datacenter links.
- Tkach has two distinct Bell Labs intervals, separated by AT&T Labs and Celion Networks; these are modeled as separate edges.
- Chandrasekhar has a directly supported Alcatel-Lucent Bell Labs affiliation and coherent-transmission contribution, but the reviewed evidence does not establish an employment interval.

Phil Winterbottom is retained as an adjacent Bell Labs lineage: his Bell Labs work was computing/network systems rather than coherent optics, while his later Celestial AI CTO role is directly photonics-relevant. The dataset marks the Bell Labs edge `enabling`, not `direct`.

## Evidence discipline

- Acacia's SEC S-1 is the core source for Mintera→Acacia. It separately names Mikkelsen, Rasmussen, and Givehchi and their prior roles.
- Ciena's transaction deck is the core source for Bell Labs→Nubis, naming Winzer and de Valicourt with their Bell Labs tenures.
- Current Nokia, ETH, and Optica biographies support research-employment edges.
- Corporate-era identity is preserved: `org_bell_labs`, `org_lucent_bell_labs`, `org_alcatel_lucent_bell_labs`, and `org_nokia_bell_labs` are not collapsed.

## Known gaps

1. Exact Bell Labs dates and time-aware corporate-era allocation remain unresolved for Benny Mikkelsen and Guilhem de Valicourt.
2. Peter Winzer's 2000–2020 span is supported, but a CV-quality source is needed to split it precisely among Lucent, Alcatel-Lucent, and Nokia Bell Labs.
3. Mintera employment dates for Mikkelsen, Rasmussen, and Givehchi are not disclosed in the Acacia S-1.
4. The full early Acacia technical roster and later post-Cisco roles require expansion from patents, papers, and archived team pages.
5. Nubis has 50+ engineers, but this pass validates only the two founders and CEO; named early engineers should be mined from patents and conference papers.
6. No source reviewed justifies either a Mintera→Acacia or Bell Labs→Nubis `team_transfer` edge.
7. Sethumadhavan Chandrasekhar's employment dates and Phil Winterbottom's detailed Bell Labs interval require stronger primary biographies or CVs.

## Recommended next pass

Mine Acacia's patent corpus and pre-IPO paper affiliations for additional Mintera-linked engineers; mine Nubis patents and OFC/ECOC author lists for named Bell Labs alumni; and obtain CV-quality histories for Winzer, de Valicourt, Mikkelsen, and Chandrasekhar before generating time-resolved Sankey weights.
