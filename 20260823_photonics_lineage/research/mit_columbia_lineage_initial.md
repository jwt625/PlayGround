# MIT / Columbia Photonic-Compute and Optical-I/O Lineage

As of 2026-08-23. The normalized companion dataset contains 20 people and atomic source-backed edges.

## Core flows

```text
MIT Dirk Englund group ── Nicholas Harris ──┐
                         Darius Bunandar ───┼──> Lightmatter
MIT Sloan ─────────────── Thomas Graham ────┘

MIT/UC Berkeley/Colorado POEM collaboration
  ├── Chen Sun ──────────┐
  ├── Mark Wade ─────────┤
  ├── Vladimir Stojanovic├──> Ayar Labs
  ├── Rajeev Ram ────────┤
  └── Milos Popovic ─────┘

MIT Soljacic group ── Yichen Shen ──> Lightelligence
MIT enrollment ────── Huaiyu Meng ──> Lightelligence

Columbia Bergman/Lipson/Gaeta collaboration
  ├── Keren Bergman ─────┐
  ├── Michal Lipson ─────┤
  ├── Alexander Gaeta ───┼──> Xscape Photonics
  ├── Yoshi Okawachi ────┤
  └── Vivek Raghunathan ─┘
```

## Findings

- **Lightmatter:** MIT directly identifies Nicholas Harris, Darius Bunandar, and Thomas Graham as the three founders. It explicitly places Harris and Bunandar in Dirk Englund's lab; Graham's evidence supports an MIT Sloan MBA only.
- **Ayar Labs:** the origin is a cross-institutional POEM collaboration rather than one PI lab. MIT names the collaboration among Vladimir Stojanovic, Rajeev Ram, and Milos Popovic and the later participation of Chen Sun and Mark Wade. Ayar's current leadership page independently calls all five co-founders.
- **Lightelligence:** Yichen Shen has a clean Soljacic-group-to-founder path. Huaiyu Meng is a verified MIT graduate and co-founder, but no reviewed source places him in a specific PI group.
- **Xscape:** Columbia explicitly attributes the technology to collaboration among the Bergman, Lipson, and Gaeta groups and names five co-founders. Vivek Raghunathan is an industry-to-startup branch; Yoshi Okawachi is supported as a Columbia research scientist, but this pass does not assign him to a specific PI group.
- **Related Englund alumni:** Saumil Bandyopadhyay, Ryan Hamerly, and Alexander Sludds are included only at the group-membership level supported by Englund's MIT presentation; no startup edges are inferred.

## Boundary decisions

1. Yichen Shen appears on MIT's 2017 Lightmatter competition team, but MIT's 2024 company history names only Harris, Bunandar, and Graham as founders. No Lightmatter founder edge is assigned to Shen.
2. MIT states that Shen teamed with Marin Soljacic when founding Lightelligence, but the reviewed wording does not explicitly call Soljacic a founder. His role remains research PI/founding collaborator, not founder.
3. University degrees for Thomas Graham, Chen Sun, Milos Popovic, and Huaiyu Meng do not become PI-group edges without direct group evidence.
4. The Ayar origin is represented with individual founder and research/employment edges, not a formal MIT spinout or a team-transfer edge.

## Gaps

- Identify Lightelligence's complete legal founding team from its listing prospectus and reconcile English/Chinese name order.
- Resolve Mark Wade's exact MIT research title/group and Chen Sun's MIT advisor/group from theses or CVs.
- Add Alex Wright-Gladstein and other early Ayar commercial founders in the next expansion pass.
- Obtain primary dates for the Englund-group alumni and Columbia faculty/group appointments.
- Expand Xscape's early technical employees through patents and Columbia project papers; do not treat paper coauthorship alone as employment.
- Determine whether Lightmatter's licensed MIT IP creates a formal `ip_license` edge and capture the agreement from MIT Technology Licensing Office records.
